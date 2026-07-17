FROM python:3.11-slim AS builder

RUN apt-get -o Acquire::Retries=5 update \
    && apt-get -o Acquire::Retries=5 install -y --no-install-recommends build-essential cargo \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /build
COPY . /build
RUN python -m pip wheel --no-cache-dir --timeout 300 --retries 10 --wheel-dir /wheels .

FROM python:3.11-slim

COPY --from=builder /wheels /wheels
RUN python -m pip install --no-cache-dir /wheels/*.whl \
    && rm -rf /wheels

WORKDIR /workspace
COPY . /workspace

CMD ["ted", "--help"]
