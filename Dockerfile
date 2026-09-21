FROM python:3.12-slim

ARG PROCESSOR_MODEL_ID
ARG CLASSIFICATION_MODEL_ID
ARG S3_URI="s3://kaneel-sagemaker-testing/model-artifacts/$CLASSIFICATION_MODEL_ID/output/model.tar.gz"
ARG AWS_REGION

ENV PROCESSOR_MODEL_ID=$PROCESSOR_MODEL_ID

ENV CLASSIFICATION_MODEL_ID=$CLASSIFICATION_MODEL_ID

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

RUN pip install --no-cache-dir huggingface_hub
ENV HF_HOME=/app/.models
RUN hf download $PROCESSOR_MODEL_ID

RUN apt-get update && apt-get install -y awscli tar && rm -rf /var/lib/apt/lists/*
RUN --mount=type=secret,id=aws_creds,target=/root/.aws/credentials \
    mkdir -p /app/.models \
    && aws s3 cp $S3_URI /app/.models/model.tar.gz \
    && tar -xzf /app/.models/model.tar.gz -C /app/.models \
    && rm /app/.models/model.tar.gz

# Install dependencies first for better layer caching
COPY src/requirements.txt /app/requirements.txt
RUN python -m pip install --upgrade pip \
    && pip install -r /app/requirements.txt

# Copy source
COPY src /app/src

# Create writable runtime dirs used by the script
RUN mkdir -p /app/.cache /app/tmp \
    && useradd --create-home --shell /bin/bash appuser \
    && chown -R appuser:appuser /app

USER appuser

CMD ["python", "src/main.py"]