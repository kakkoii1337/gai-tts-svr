# syntax=docker/dockerfile:1.2

FROM torch2.2.0-cuda12.1-ubuntu22.04 as build

# Build Final ----------------------------------------------------------------------------------------

FROM pytorch/pytorch:2.2.0-cuda12.1-cudnn8-runtime AS base
ARG CATEGORY=tts
ARG DEVICE=cuda
ENV DEBIAN_FRONTEND=noninteractive PIP_PREFER_BINARY=1

# Step 1: Install poetry
RUN apt update && apt install -y curl \
    && curl -sSL https://install.python-poetry.org | python3 - \
    && apt remove -y curl \
    && apt autoremove -y \
    && apt clean \
    && rm -rf /var/lib/apt/lists/*
ENV PATH="/root/.local/bin:${PATH}"
RUN poetry config virtualenvs.create false

# Step 2: Install prebuilt wheels

## deepspeed
# WORKDIR /app/wheels
# COPY --from=build /app/DeepSpeed/dist/* /app/wheels/
# RUN --mount=type=cache,target=/root/.cache/pip \
#     pip install /app/wheels/*.whl

# Step 3: Copy Source Code
WORKDIR /app
COPY src/gai/tts src/gai/tts
COPY pyproject.toml.docker ./pyproject.toml
COPY poetry.lock .

# Step 4: Install from wheel
RUN poetry build -f wheel
RUN pip install dist/*.whl

# Step 5: Startup
RUN echo '{"app_dir":"/app/.gai"}' > /root/.gairc
VOLUME /app/.gai
ENV MODEL_PATH="/app/.gai/models"
ENV CATEGORY=${CATEGORY}
WORKDIR /app/src/gai/tts/server/api
CMD ["bash","-c","python main.py"]

