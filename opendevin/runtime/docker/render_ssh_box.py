import atexit
import os
import re
import sys
import tarfile
import tempfile
import time
import uuid
from collections import namedtuple
from glob import glob

import requests
import websockets
from pexpect import exceptions, pxssh
from tenacity import retry, stop_after_attempt, wait_fixed

from opendevin.core.config import config
from opendevin.core.exceptions import SandboxInvalidBackgroundCommandError
from opendevin.core.logger import opendevin_logger as logger
from opendevin.core.schema import CancellableStream
from opendevin.runtime.sandbox import Sandbox

# FIXME: these are not used, can we remove them?
InputType = namedtuple('InputType', ['content'])
OutputType = namedtuple('OutputType', ['content'])


class SSHExecCancellableStream(CancellableStream):
    def __init__(self, ssh, cmd, timeout):
        super().__init__(self.read_output())
        self.ssh = ssh
        self.cmd = cmd
        self.timeout = timeout

    def close(self):
        self.closed = True

    def exit_code(self):
        self.ssh.sendline('echo $?')
        success = self.ssh.prompt(timeout=self.timeout)
        if not success:
            return -1

        _exit_code = self.ssh.before.strip()
        return int(_exit_code)

    def read_output(self):
        st = time.time()
        buf = ''
        crlf = '\r\n'
        lf = '\n'
        prompt_len = len(self.ssh.PROMPT)
        while True:
            try:
                if self.closed:
                    break
                _output = self.ssh.read_nonblocking(timeout=1)
                if not _output:
                    continue

                buf += _output

                if len(buf) < prompt_len:
                    continue

                match = re.search(self.ssh.PROMPT, buf)
                if match:
                    idx, _ = match.span()
                    yield buf[:idx].replace(crlf, lf)
                    buf = ''
                    break

                res = buf[:-prompt_len]
                if len(res) == 0 or res.find(crlf) == -1:
                    continue
                buf = buf[-prompt_len:]
                yield res.replace(crlf, lf)
            except exceptions.TIMEOUT:
                if time.time() - st < self.timeout:
                    match = re.search(self.ssh.PROMPT, buf)
                    if match:
                        idx, _ = match.span()
                        yield buf[:idx].replace(crlf, lf)
                        break
                    continue
                else:
                    yield buf.replace(crlf, lf)
                break
            except exceptions.EOF:
                break


def split_bash_commands(commands):
    # States
    NORMAL = 0
    IN_SINGLE_QUOTE = 1
    IN_DOUBLE_QUOTE = 2
    IN_HEREDOC = 3

    state = NORMAL
    heredoc_trigger = None
    result = []
    current_command: list[str] = []

    i = 0
    while i < len(commands):
        char = commands[i]

        if state == NORMAL:
            if char == "'":
                state = IN_SINGLE_QUOTE
            elif char == '"':
                state = IN_DOUBLE_QUOTE
            elif char == '\\':
                # Check if this is escaping a newline
                if i + 1 < len(commands) and commands[i + 1] == '\n':
                    i += 1  # Skip the newline
                    # Continue with the next line as part of the same command
                    i += 1  # Move to the first character of the next line
                    continue
            elif char == '\n':
                if not heredoc_trigger and current_command:
                    result.append(''.join(current_command).strip())
                    current_command = []
            elif char == '<' and commands[i : i + 2] == '<<':
                # Detect heredoc
                state = IN_HEREDOC
                i += 2  # Skip '<<'
                while commands[i] == ' ':
                    i += 1
                start = i
                while commands[i] not in [' ', '\n']:
                    i += 1
                heredoc_trigger = commands[start:i]
                current_command.append(commands[start - 2 : i])  # Include '<<'
                continue  # Skip incrementing i at the end of the loop
            current_command.append(char)

        elif state == IN_SINGLE_QUOTE:
            current_command.append(char)
            if char == "'" and commands[i - 1] != '\\':
                state = NORMAL

        elif state == IN_DOUBLE_QUOTE:
            current_command.append(char)
            if char == '"' and commands[i - 1] != '\\':
                state = NORMAL

        elif state == IN_HEREDOC:
            current_command.append(char)
            if (
                char == '\n'
                and heredoc_trigger
                and commands[i + 1 : i + 1 + len(heredoc_trigger) + 1]
                == heredoc_trigger + '\n'
            ):
                # Check if the next line starts with the heredoc trigger followed by a newline
                i += (
                    len(heredoc_trigger) + 1
                )  # Move past the heredoc trigger and newline
                current_command.append(
                    heredoc_trigger + '\n'
                )  # Include the heredoc trigger and newline
                result.append(''.join(current_command).strip())
                current_command = []
                heredoc_trigger = None
                state = NORMAL
                continue

        i += 1

    # Add the last command if any
    if current_command:
        result.append(''.join(current_command).strip())

    # Remove any empty strings from the result
    result = [cmd for cmd in result if cmd]

    return result


class RenderSSHBox(Sandbox):
    instance_id: str
    service_name: str
    _ssh_password: str
    ssh: pxssh.pxssh

    base_url = config.render_base_url
    api_key = config.render_api_key
    image_name = config.sandbox_image
    service_name_prefix = config.service_name_prefix

    cur_background_id = 0
    background_commands: dict[int, str] = {}

    def __init__(
        self,
        service_name: str | None = None,
        timeout: int = config.sandbox_timeout,
        sid: str | None = None,
    ):
        logger.info(
            f'SSHBox is running as "opendevin" user with USER_ID={self.user_id}'
        )
        self.timeout = timeout
        self.service_name = service_name or self.service_name_prefix + str(uuid.uuid4())

        self._ssh_password = str(uuid.uuid4())

        try:
            service = self.find_service()
            if service:
                self.is_initial_session = False
                logger.info('Using existing Render service')
                self.service_url = service['service']['serviceDetails']['url']
            else:
                self.is_initial_session = True
                logger.info('Creating new Render service')
                self.service_url = self.deploy_service()
        except Exception as e:
            logger.exception('Error creating Render service', exc_info=False)
            raise e

        try:
            self.start_ssh_session()
        except Exception as e:
            self.close()
            raise e

        atexit.register(self.close)
        super().__init__()

    def get_headers(self):
        return {'accept': 'application/json', 'authorization': f'Bearer {self.api_key}'}

    def list_services(self):
        url = f'{self.base_url}services?limit=20'
        response = requests.get(url, headers=self.get_headers())
        if response.status_code == 200:
            return response.json()
        else:
            logger.error(f'Error listing services: {response.status_code}')
            return None

    def find_service(self):
        services = self.list_services()
        if services:
            for service in services:
                if service['service']['name'] == self.service_name:
                    return service
        return None

    def deploy_service(self):
        url = f'{self.base_url}services'
        payload = {
            'type': 'webservice',
            'autoDeploy': True,
            'serviceDetails': {'imagePath': self.image_name},
        }
        response = requests.post(url, json=payload, headers=self.get_headers())
        if response.status_code == 201:
            service_url = response.json()['service']['serviceDetails']['url']
            logger.info(f'Service deployed at: {service_url}')
            return service_url
        else:
            logger.error(f'Error deploying service: {response.status_code}')
            return None

    def start_ssh_session(self):
        self.__ssh_login()

        # Fix: https://github.com/pexpect/pexpect/issues/669
        self.ssh.sendline("bind 'set enable-bracketed-paste off'")
        self.ssh.prompt()
        self.ssh.sendline('cd /home/opendevin')
        self.ssh.prompt()

    @retry(stop=stop_after_attempt(5), wait=wait_fixed(5))
    def __ssh_login(self):
        try:
            self.ssh = pxssh.pxssh(
                echo=False,
                timeout=self.timeout,
                encoding='utf-8',
                codec_errors='replace',
            )
            hostname = self.service_url
            username = 'opendevin'
            logger.info('Connecting to SSH session...')
            self.ssh.login(hostname, username, self._ssh_password)
            logger.info('Connected to SSH session')
        except pxssh.ExceptionPxssh as e:
            logger.exception(
                'Failed to login to SSH session, retrying...', exc_info=False
            )
            raise e

    def execute(
        self, cmd: str, stream: bool = False, timeout: int | None = None
    ) -> tuple[int, str | CancellableStream]:
        timeout = timeout or self.timeout
        commands = split_bash_commands(cmd)
        if len(commands) > 1:
            all_output = ''
            for command in commands:
                exit_code, output = self.execute(command)
                all_output += str(output)
                if exit_code != 0:
                    return exit_code, all_output
            return 0, all_output

        self.ssh.sendline(cmd)
        if stream:
            return 0, SSHExecCancellableStream(self.ssh, cmd, self.timeout)
        success = self.ssh.prompt(timeout=timeout)
        if not success:
            return self._send_interrupt(cmd)
        command_output = self.ssh.before

        while True:
            self.ssh.sendline('\n')
            timeout_not_reached = self.ssh.prompt(timeout=1)
            if not timeout_not_reached:
                break
            output = self.ssh.before
            if isinstance(output, str) and output.strip() == '':
                break
            command_output += output
        command_output = command_output.removesuffix('\r\n')

        self.ssh.sendline('echo $?')
        self.ssh.prompt()
        exit_code_str = self.ssh.before.strip()
        try:
            exit_code = int(exit_code_str)
        except ValueError:
            exit_code = -1
        return exit_code, command_output

    def execute_in_background(self, cmd: str) -> int:
        self.ssh.sendline(cmd)
        self.ssh.prompt()
        bg_id = self.cur_background_id
        self.background_commands[bg_id] = cmd
        self.cur_background_id += 1
        return bg_id

    def read_logs(self, id: int) -> str:
        if id not in self.background_commands:
            raise SandboxInvalidBackgroundCommandError()
        command = self.background_commands[id]
        return f'Logs for command {command}'

    def kill_background(self, id: int) -> None:
        if id not in self.background_commands:
            raise SandboxInvalidBackgroundCommandError()
        self.ssh.sendline(f'kill {id}')
        self.ssh.prompt()
        self.background_commands.pop(id)

    def copy_to(self, host_src: str, sandbox_dest: str, recursive: bool = False):
        exit_code, logs = self.execute(f'mkdir -p {sandbox_dest}')
        if exit_code != 0:
            raise Exception(f'Failed to create directory {sandbox_dest}: {logs}')
        with tempfile.TemporaryDirectory() as tmp_dir:
            if recursive:
                assert os.path.isdir(
                    host_src
                ), 'Source must be a directory when recursive is True'
                files = glob(host_src + '/**/*', recursive=True)
                srcname = os.path.basename(host_src)
                tar_filename = os.path.join(tmp_dir, srcname + '.tar')
                with tarfile.open(tar_filename, mode='w') as tar:
                    for file in files:
                        tar.add(
                            file,
                            arcname=os.path.relpath(file, os.path.dirname(host_src)),
                        )
            else:
                assert os.path.isfile(
                    host_src
                ), 'Source must be a file when recursive is False'
                srcname = os.path.basename(host_src)
                tar_filename = os.path.join(tmp_dir, srcname + '.tar')
                with tarfile.open(tar_filename, mode='w') as tar:
                    tar.add(host_src, arcname=srcname)

            with open(tar_filename, 'rb') as f:
                self.ssh.sendline(f'tar xvf - -C {sandbox_dest}')
                self.ssh.send(f.read())
                self.ssh.prompt()

    def get_exec_cmd(self, cmd: str) -> list[str]:
        return ['/bin/bash', '-c', cmd]

    def _send_interrupt(
        self, cmd: str, prev_output: str = '', ignore_last_output: bool = False
    ) -> tuple[int, str]:
        logger.exception(
            f'Command "{cmd}" timed out, killing process...', exc_info=False
        )
        self.ssh.sendintr()
        self.ssh.prompt()
        command_output = prev_output
        if not ignore_last_output:
            command_output += '\n' + self.ssh.before
        return (
            -1,
            f'Command: "{cmd}" timed out. Sent SIGINT to the process: {command_output}',
        )

    async def subscribe_to_logs(self):
        url = 'wss://api.render.com/v1/logs/subscribe?direction=backward&limit=20'
        headers = {
            'accept': 'application/json',
            'authorization': f'Bearer {self.api_key}',
        }

        async with websockets.connect(url, extra_headers=headers) as websocket:
            logger.info('Subscribed to logs. Listening for log messages...')
            while True:
                try:
                    log_message = await websocket.recv()
                    logger.info(f'Log message: {log_message}')
                except websockets.ConnectionClosed:
                    logger.error('WebSocket connection closed')
                    break

    def setup_user(self):
        logger.info('Setting up the user in Render container')

    def add_to_env(self, key: str, value: str):
        logger.info(f'Added {key} to environment with value {value}')

    def close(self):
        if self.ssh:
            self.ssh.logout()
            logger.info('SSH connection closed.')

    @property
    def user_id(self):
        return config.sandbox_user_id

    @property
    def sandbox_user_id(self):
        return config.sandbox_user_id

    @property
    def run_as_devin(self):
        return config.run_as_devin

    @property
    def sandbox_workspace_dir(self):
        return config.workspace_mount_path_in_sandbox

    @property
    def ssh_hostname(self):
        return self.service_url

    @property
    def use_host_network(self):
        return config.use_host_network

    def is_container_running(self):
        # Assuming Render's API provides a status endpoint for the service
        service = self.find_service()
        if service and service['service']['serviceDetails']['url'] == self.service_url:
            return True
        return False

    @property
    def volumes(self):
        mount_dir = config.workspace_mount_path
        logger.info(f'Mounting workspace directory: {mount_dir}')
        return {
            mount_dir: {'bind': self.sandbox_workspace_dir, 'mode': 'rw'},
            config.cache_dir: {
                'bind': '/home/opendevin/.cache'
                if self.run_as_devin
                else '/root/.cache',
                'mode': 'rw',
            },
        }

    def restart_docker_container(self):
        try:
            self.remove_docker_container()
        except Exception as ex:
            logger.exception('Failed to remove container', exc_info=False)
            raise ex

        try:
            logger.info(f'Mounting volumes: {self.volumes}')
            # Restart logic using the Render API
            self.deploy_service()
        except Exception as ex:
            logger.exception('Failed to start container', exc_info=False)
            raise ex


if __name__ == '__main__':
    try:
        ssh_box = RenderSSHBox()
    except Exception as e:
        logger.exception('Failed to start Render service: %s', e)
        sys.exit(1)

    logger.info(
        "Interactive Render service started. Type 'exit' or use Ctrl+C to exit."
    )

    bg_cmd = ssh_box.execute_in_background(
        "while true; do echo -n '.' && sleep 10; done"
    )

    try:
        while True:
            try:
                user_input = input('$ ')
            except EOFError:
                logger.info('Exiting...')
                break
            if user_input.lower() == 'exit':
                logger.info('Exiting...')
                break
            if user_input.lower() == 'kill':
                ssh_box.kill_background(bg_cmd)
                logger.info('Background process killed')
                continue
            exit_code, output = ssh_box.execute(user_input)
            logger.info(f'exit code: {exit_code}')
            logger.info(output)
            if bg_cmd in ssh_box.background_commands:
                logs = ssh_box.read_logs(bg_cmd)
                logger.info(f'background logs: {logs}')
    except KeyboardInterrupt:
        logger.info('Exiting...')
    ssh_box.close()
