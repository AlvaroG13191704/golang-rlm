# syntax=docker/dockerfile:1

# Build stage: generate Python gRPC stubs from the proto definitions.
FROM python:3.11-slim AS builder

RUN pip install --no-cache-dir grpcio-tools

WORKDIR /proto
COPY proto/rlm/v1/*.proto /proto/rlm/v1/

RUN mkdir -p /out

RUN python -m grpc_tools.protoc \
    --proto_path=/proto \
    --python_out=/out \
    --grpc_python_out=/out \
    rlm/v1/types.proto \
    rlm/v1/lm_service.proto \
    rlm/v1/rlm_service.proto

# Final stage: slim Python 3.11 runtime with dill and grpcio pre-installed.
FROM python:3.11-slim

RUN pip install --no-cache-dir dill grpcio protobuf

COPY --from=builder /out/rlm /usr/local/lib/python3.11/site-packages/rlm
COPY container/exec_script.py /usr/local/bin/rlm_exec_script.py

WORKDIR /workspace
