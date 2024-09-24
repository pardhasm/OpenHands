import asyncio
import threading
from datetime import datetime
from enum import Enum
from typing import Callable, ClassVar, Iterable

from openhands.controller.state.state import State
from openhands.core.logger import openhands_logger as logger
from openhands.core.utils import json
from openhands.events.action.action import Action
from openhands.events.action.agent import (
    AgentFinishAction,
    ChangeAgentStateAction,
)
from openhands.events.action.empty import NullAction
from openhands.events.action.message import MessageAction
from openhands.events.event import Event, EventSource
from openhands.events.observation import Observation
from openhands.events.observation.agent import AgentStateChangedObservation
from openhands.events.observation.commands import CmdOutputObservation
from openhands.events.observation.delegate import AgentDelegateObservation
from openhands.events.observation.empty import NullObservation
from openhands.events.serialization.event import event_from_dict, event_to_dict
from openhands.runtime.utils.shutdown_listener import should_continue
from openhands.storage import FileStore


class EventStreamSubscriber(str, Enum):
    AGENT_CONTROLLER = 'agent_controller'
    SECURITY_ANALYZER = 'security_analyzer'
    SERVER = 'server'
    RUNTIME = 'runtime'
    MAIN = 'main'
    TEST = 'test'


class EventStream:
    sid: str
    file_store: FileStore
    state: State | None
    # For each subscriber ID, there is a stack of callback functions - useful
    # when there are agent delegates
    _subscribers: dict[str, list[Callable]]
    _cur_id: int
    _lock: threading.Lock
    filter_out: ClassVar[tuple[type[Event], ...]] = (
        NullAction,
        NullObservation,
        ChangeAgentStateAction,
        AgentStateChangedObservation,
    )

    def __init__(self, sid: str, file_store: FileStore):
        self.sid = sid
        self.file_store = file_store
        self.state = None
        self._subscribers = {}
        self._cur_id = 0
        self._lock = threading.Lock()
        self._reinitialize_from_file_store()

    def set_state(self, state: State):
        self.state = state

    def _reinitialize_from_file_store(self) -> None:
        try:
            events = self.file_store.list(f'sessions/{self.sid}/events')
        except FileNotFoundError:
            logger.debug(f'No events found for session {self.sid}')
            self._cur_id = 0
            return

        # if we have events, we need to find the highest id to prepare for new events
        for event_str in events:
            id = self._get_id_from_filename(event_str)
            if id >= self._cur_id:
                self._cur_id = id + 1

    def _get_filename_for_id(self, id: int) -> str:
        return f'sessions/{self.sid}/events/{id}.json'

    @staticmethod
    def _get_id_from_filename(filename: str) -> int:
        try:
            return int(filename.split('/')[-1].split('.')[0])
        except ValueError:
            logger.warning(f'get id from filename ({filename}) failed.')
            return -1

    def get_events(
        self,
        start_id=None,
        end_id=None,
        reverse=False,
        include_delegates=False,
        filter_out_types: tuple[type[Event], ...] | None = None,
    ) -> Iterable[Event]:
        """
        Retrieve events from the stream, optionally filtering out certain event types and delegate events.

        Args:
            start_id: The starting event ID. Defaults to the state's start_id or 0.
            end_id: The ending event ID. Defaults to the state's end_id or the latest event.
            reverse: Whether to iterate in reverse order. Defaults to False.
            filter_out_types: Event types to filter out. Defaults to the built-in filter_out.

        Yields:
            Event: Events from the stream that match the criteria.
        """
        if self.state is None:
            raise ValueError('State has not been set for EventStream')

        if start_id is None:
            start_id = self.state.start_id if self.state.start_id != -1 else 0
        if end_id is None:
            end_id = self.state.end_id if self.state.end_id != -1 else self._cur_id - 1

        exclude_types = (
            filter_out_types if filter_out_types is not None else self.filter_out
        )

        event_range = (
            range(end_id, start_id - 1, -1) if reverse else range(start_id, end_id + 1)
        )

        for event_id in event_range:
            # check if we're shutting down
            if not should_continue():
                break

            try:
                event = self.get_event(event_id)

                # Filter out excluded event types
                if isinstance(event, exclude_types):
                    continue

                # Filter out delegate events
                if not include_delegates and any(
                    delegate_start < event.id < delegate_end
                    for delegate_start, (
                        _,
                        _,
                        delegate_end,
                    ) in self.state.delegates.items()
                ):
                    continue

                yield event
            except FileNotFoundError:
                logger.debug(f'No event found for ID {event_id}')
                if not reverse:
                    break

    def get_event(self, id: int) -> Event:
        filename = self._get_filename_for_id(id)
        content = self.file_store.read(filename)
        data = json.loads(content)
        return event_from_dict(data)

    def get_latest_event(self) -> Event:
        return self.get_event(self._cur_id - 1)

    def get_latest_event_id(self) -> int:
        return self._cur_id - 1

    def subscribe(self, id: EventStreamSubscriber, callback: Callable, append=False):
        if id in self._subscribers:
            if append:
                self._subscribers[id].append(callback)
            else:
                raise ValueError('Subscriber already exists: ' + id)
        else:
            self._subscribers[id] = [callback]

    def unsubscribe(self, id: EventStreamSubscriber):
        if id not in self._subscribers:
            logger.warning('Subscriber not found during unsubscribe: ' + id)
        else:
            self._subscribers[id].pop()
            if len(self._subscribers[id]) == 0:
                del self._subscribers[id]

    def add_event(self, event: Event, source: EventSource):
        with self._lock:
            event._id = self._cur_id  # type: ignore [attr-defined]
            self._cur_id += 1
        logger.debug(f'Adding {type(event).__name__} id={event.id} from {source.name}')
        event._timestamp = datetime.now().isoformat()
        event._source = source  # type: ignore [attr-defined]
        data = event_to_dict(event)
        if event.id is not None:
            self.file_store.write(self._get_filename_for_id(event.id), json.dumps(data))

        for key in sorted(self._subscribers.keys()):
            stack = self._subscribers[key]
            callback = stack[-1]
            asyncio.create_task(callback(event))

    def filtered_events_by_source(self, source: EventSource):
        for event in self.get_events():
            if event.source == source:
                yield event

    def clear(self):
        self.file_store.delete(f'sessions/{self.sid}')
        self._cur_id = 0
        # self._subscribers = {}
        self._reinitialize_from_file_store()

    def get_last_action(self, end_id: int = -1) -> Action | None:
        """Return the last action from the event stream, filtered to exclude unwanted events."""
        if self.state is None:
            raise ValueError('State has not been set for EventStream')

        end_id = end_id if end_id != -1 else self.state.end_id
        if end_id == -1:
            end_id = self._cur_id - 1

        last_action = next(
            (
                event
                for event in self.get_events(end_id=end_id, reverse=True)
                if isinstance(event, Action)
            ),
            None,
        )

        return last_action

    def get_last_observation(self, end_id: int = -1) -> Observation | None:
        """Return the last observation from the event stream, filtered to exclude unwanted events."""
        if self.state is None:
            raise ValueError('State has not been set for EventStream')

        end_id = end_id if end_id != -1 else self.state.end_id
        if end_id == -1:
            end_id = self._cur_id - 1

        last_observation = next(
            (
                event
                for event in self.get_events(end_id=end_id, reverse=True)
                if isinstance(event, Observation)
            ),
            None,
        )

        return last_observation

    def get_last_user_message(self) -> str:
        """Return the content of the last user message from the event stream."""
        last_user_message = next(
            (
                event.content
                for event in self.get_events(reverse=True)
                if isinstance(event, MessageAction) and event.source == EventSource.USER
            ),
            None,
        )

        return last_user_message if last_user_message is not None else ''

    def get_last_agent_message(self) -> str:
        """Return the content of the last agent message from the event stream."""
        last_agent_message = next(
            (
                event.content
                for event in self.get_events(reverse=True)
                if isinstance(event, MessageAction)
                and event.source == EventSource.AGENT
            ),
            None,
        )

        return last_agent_message if last_agent_message is not None else ''

    def get_last_events(self, n: int) -> list[Event]:
        """Return the last n events from the event stream."""
        # dummy agent is using this
        # it works, but it's not great to store temporary lists now just for a test
        end_id = self._cur_id - 1
        start_id = max(0, end_id - n + 1)

        return list(
            event for event in self.get_events(start_id=start_id, end_id=end_id)
        )

    def get_current_user_intent(self):
        """Returns the latest user message and image(if provided) that appears after a FinishAction, or the first (the task) if nothing was finished yet."""
        last_user_message = None
        last_user_message_image_urls: list[str] | None = []
        for event in self.get_events(reverse=True):
            if isinstance(event, MessageAction) and event.source == 'user':
                last_user_message = event.content
                last_user_message_image_urls = event.images_urls
            elif isinstance(event, AgentFinishAction):
                if last_user_message is not None:
                    return last_user_message

        return last_user_message, last_user_message_image_urls

    def has_delegation(self) -> bool:
        for event in self.get_events():
            if isinstance(event, AgentDelegateObservation):
                return True
        return False

    def get_events_as_list(self, include_delegates: bool = False) -> list[Event]:
        """Return the history as a list of Event objects."""
        return list(self.get_events(include_delegates=include_delegates))

    # history is now available as a filtered stream of events, rather than list of pairs of (Action, Observation)
    # we rebuild the pairs here
    # for compatibility with the existing output format in evaluations
    def compatibility_for_eval_history_pairs(self) -> list[tuple[dict, dict]]:
        history_pairs = []

        for action, observation in self.get_pairs():
            history_pairs.append((event_to_dict(action), event_to_dict(observation)))

        return history_pairs

    def get_pairs(self) -> list[tuple[Action, Observation]]:
        """Return the history as a list of tuples (action, observation)."""
        tuples: list[tuple[Action, Observation]] = []
        action_map: dict[int, Action] = {}
        observation_map: dict[int, Observation] = {}

        # runnable actions are set as cause of observations
        # (MessageAction, NullObservation) for source=USER
        # (MessageAction, NullObservation) for source=AGENT
        # (other_action?, NullObservation)
        # (NullAction, CmdOutputObservation) background CmdOutputObservations

        for event in self.get_events_as_list():
            if event.id is None or event.id == -1:
                logger.debug(f'Event {event} has no ID')

            if isinstance(event, Action):
                action_map[event.id] = event

            if isinstance(event, Observation):
                if event.cause is None or event.cause == -1:
                    logger.debug(f'Observation {event} has no cause')

                if event.cause is None:
                    # runnable actions are set as cause of observations
                    # NullObservations have no cause
                    continue

                observation_map[event.cause] = event

        for action_id, action in action_map.items():
            observation = observation_map.get(action_id)
            if observation:
                # observation with a cause
                tuples.append((action, observation))
            else:
                tuples.append((action, NullObservation('')))

        for cause_id, observation in observation_map.items():
            if cause_id not in action_map:
                if isinstance(observation, NullObservation):
                    continue
                if not isinstance(observation, CmdOutputObservation):
                    logger.debug(f'Observation {observation} has no cause')
                tuples.append((NullAction(), observation))

        return tuples.copy()
