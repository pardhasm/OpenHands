FROM ubuntu:22.04

# Install basic packages
RUN apt-get update && apt-get install -y \
    curl \
    wget \
    git \
    vim \
    nano \
    unzip \
    zip \
    python3 \
    python3-pip \
    python3-venv \
    python3-dev \
    build-essential \
    openssh-server \
    sudo \
    bash \
    gcc \
    jq \
    g++ \
    make \
    iproute2 \
    libgl1-mesa-glx \
    && rm -rf /var/lib/apt/lists/*

# Symlink python3 to python
RUN ln -s /usr/bin/python3 /usr/bin/python

# Unlock the root account as per Render.com limitations
RUN passwd -u root

# Create a non-root user with sudo access
RUN useradd -m -s /bin/bash yourusername && \
    echo "yourusername:yourpassword" | chpasswd && \
    adduser yourusername sudo

# Make sure the new user can run sudo without a password
RUN echo "yourusername ALL=(ALL) NOPASSWD: ALL" >> /etc/sudoers

# === OpenDevin Runtime Client ===
RUN mkdir -p /opendevin && mkdir -p /opendevin/logs && chmod 777 /opendevin/logs
WORKDIR /opendevin

RUN wget "https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-$(uname)-$(uname -m).sh"
RUN bash Miniforge3-$(uname)-$(uname -m).sh -b -p /opendevin/miniforge3
RUN bash -c ". /opendevin/miniforge3/etc/profile.d/conda.sh && conda config --set changeps1 False && conda config --append channels conda-forge"
RUN echo "export PATH=/opendevin/miniforge3/bin:$PATH" >> /home/yourusername/.bashrc

# Install agent skills dependencies
RUN /opendevin/miniforge3/bin/pip install --upgrade pip
RUN /opendevin/miniforge3/bin/pip install jupyterlab notebook jupyter_kernel_gateway flake8
RUN /opendevin/miniforge3/bin/pip install python-docx PyPDF2 python-pptx pylatexenc openai opencv-python

# Set up SSH: create the .ssh directory and set the correct permissions
RUN mkdir -p /home/yourusername/.ssh && \
    chmod 0700 /home/yourusername/.ssh && \
    chown yourusername:yourusername /home/yourusername/.ssh

# Set up root user .ssh directory for potential root login (if needed for SSH)
RUN mkdir -p /root/.ssh && \
    chmod 0700 /root/.ssh && \
    chown root:root /root/.ssh

# Set up OpenSSH
RUN mkdir -p -m0755 /var/run/sshd

# Expose SSH port
EXPOSE 22

# Start the SSH service
CMD ["/usr/sbin/sshd", "-D"]
