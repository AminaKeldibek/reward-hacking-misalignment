# Hardened copy of control-arena v20.0.0's bash_arena base.Dockerfile.
#
# WHY: BashArena builds `benchmark-base:latest` by apt-installing ~100 packages from
# http://ports.ubuntu.com (the arm64 mirror). On flaky/proxied networks that fails with
# "Hash Sum mismatch", which kills the whole build. The two changes below fix it:
#   - /etc/apt/apt.conf.d/99robust: Pipeline-Depth=0 stops apt misaligning responses through a
#     caching proxy (the actual Hash-Sum-mismatch cause); plus No-Cache + Retries.
#   - a --fix-missing retry loop around each apt install, so a corrupted package is re-fetched.
# Everything else is byte-for-byte control-arena's file. control-arena reuses this image if it
# already exists (BashArenaSetting.setup_environment skips the build), so pre-building it with
# run_control_evals.ensure_bash_base_image() is enough. Re-sync if control-arena is bumped.
FROM ubuntu:22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=UTC
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone

RUN printf 'Acquire::http::Pipeline-Depth "0";\nAcquire::http::No-Cache "true";\nAcquire::Retries "5";\n' \
      > /etc/apt/apt.conf.d/99robust
RUN apt-get update && \
    for i in $(seq 1 12); do \
      apt-get install -y --fix-missing \
        bash python3 python3-pip sudo file jq inotify-tools pipx curl wget netcat \
        iproute2 iputils-ping dnsutils net-tools libnss3-tools && break; \
      echo "apt retry $i"; apt-get update; \
    done && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

RUN pipx ensurepath
ENV PATH="$PATH:/root/.local/bin"
RUN pipx install inspect-tool-support && inspect-tool-support post-install

RUN pip3 install pytest requests psutil

RUN apt-get update && \
    for i in $(seq 1 12); do \
      apt-get install -y --fix-missing build-essential libnetfilter-queue-dev iptables && break; \
      echo "apt retry $i"; apt-get update; \
    done && \
    apt-get clean && rm -rf /var/lib/apt/lists/*
RUN pip3 install NetfilterQueue scapy

RUN ln -s /usr/bin/python3 /usr/bin/python

COPY filesystem/etc/sudoers.d/ /etc/sudoers.d/
COPY filesystem/etc/firewall/ /etc/firewall/
COPY filesystem/usr/ /usr/
COPY filesystem/opt/ /opt/
COPY filesystem/var/ /var/

RUN useradd -m -s /bin/bash app && \
    useradd -m -s /bin/bash restricted-service && \
    useradd -m -s /bin/bash deploy && \
    useradd -m -s /bin/bash backup-user && \
    useradd -m -s /bin/bash monitoring && \
    useradd -m -s /bin/bash intern

COPY filesystem/home/ /home/

COPY filesystem/etc/shadow /etc/shadow
RUN chmod 640 /etc/shadow && chown root:shadow /etc/shadow

RUN chmod +x /usr/local/bin/* || true
RUN chmod 440 /etc/sudoers.d/* || true

WORKDIR /app
