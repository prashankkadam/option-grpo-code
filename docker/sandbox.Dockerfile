# Execution sandbox for one SDK.
#
#   docker build -f docker/sandbox.Dockerfile \
#       --build-arg SDK_REQUIREMENT="daft==<version>" -t option-grpo-sandbox:daft .
#
# The image only needs Python and the SDK: the harness is a single standard-library file
# that is mounted into the container at run time, together with the program under test.
# Containers are started with --network none, a memory limit and a read-only root.
ARG PYTHON_VERSION=3.11
FROM python:${PYTHON_VERSION}-slim

ARG SDK_REQUIREMENT
ARG EXTRA_REQUIREMENTS=""

ENV PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

RUN test -n "${SDK_REQUIREMENT}" || (echo "SDK_REQUIREMENT build argument is required" && false)
RUN pip install "${SDK_REQUIREMENT}" ${EXTRA_REQUIREMENTS}

# Unprivileged user; the working directory is bind-mounted per program.
RUN useradd --create-home --uid 1000 sandbox && mkdir /work && chown sandbox /work
USER sandbox
WORKDIR /work

# Record the exact environment the benchmark ran against.
RUN pip freeze > /home/sandbox/requirements.lock
