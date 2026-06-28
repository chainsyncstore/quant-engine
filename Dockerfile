# syntax=docker/dockerfile:1.7
ARG PYTHON_BASE="python:3.11.9-slim-bookworm@sha256:8fb099199b9f2d70342674bd9dbccd3ed03a258f26bbd1d556822c6dfc60c317"
ARG DEBIAN_SNAPSHOT="20250201T000000Z"

FROM ${PYTHON_BASE} AS os-runtime
ARG DEBIAN_SNAPSHOT
RUN rm -f /etc/apt/sources.list /etc/apt/sources.list.d/debian.sources \
    && printf '%s\n' \
      "deb [check-valid-until=no] https://snapshot.debian.org/archive/debian/${DEBIAN_SNAPSHOT}/ bookworm main" \
      "deb [check-valid-until=no] https://snapshot.debian.org/archive/debian-security/${DEBIAN_SNAPSHOT}/ bookworm-security main" \
      > /etc/apt/sources.list \
    && apt-get -o Acquire::Check-Valid-Until=false update \
    && apt-get install -y --no-install-recommends libgomp1=12.2.0-14+deb12u1 \
    && rm -rf /var/lib/apt/lists/*

FROM os-runtime AS dependencies
ENV PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1
RUN python -m venv /opt/build-tools \
    && python -m venv /opt/runtime-venv
COPY requirements/build.lock requirements/runtime.lock /build/requirements/
RUN /opt/build-tools/bin/python -m pip install --require-hashes --no-deps \
        -r /build/requirements/build.lock \
    && /opt/build-tools/bin/uv pip install --python /opt/runtime-venv/bin/python \
        --require-hashes --no-deps \
        --torch-backend cpu -r /build/requirements/runtime.lock

FROM os-runtime AS runtime
ARG VCS_REF="unknown"
ARG SOURCE_MANIFEST_SHA256="unknown"
ARG LOCK_SHA256="unknown"
ARG BUILD_DATE="unknown"
ARG CI_RUN_ID="local"
LABEL org.opencontainers.image.revision="$VCS_REF" \
      org.opencontainers.image.created="$BUILD_DATE" \
      org.opencontainers.image.source-manifest-sha256="$SOURCE_MANIFEST_SHA256" \
      org.opencontainers.image.dependency-lock-sha256="$LOCK_SHA256" \
      org.opencontainers.image.ci-run-id="$CI_RUN_ID"
ENV PATH="/opt/runtime-venv/bin:$PATH" \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    HF_HOME=/home/quantbot/.cache/huggingface \
    HOME=/home/quantbot
WORKDIR /app
COPY --from=dependencies /opt/runtime-venv /opt/runtime-venv
COPY quant/ ./quant/
COPY quant_v2/ ./quant_v2/
COPY bootstrap_registry.py pyproject.toml ./
COPY tools/image_smoke.py ./tools/image_smoke.py
COPY .build/wp02-manifest.json ./build-manifest.json
RUN groupadd --gid 10001 quantbot \
    && useradd --uid 10001 --gid 10001 --no-create-home --home-dir /home/quantbot \
        --shell /usr/sbin/nologin quantbot \
    && mkdir -p /home/quantbot/.cache/huggingface /app/models /state /tmp \
    && chown -R 10001:10001 /home/quantbot /app/models /state
USER 10001:10001
CMD ["python", "-m", "quant_v2.execution.main"]

FROM runtime AS test
USER 0:0
COPY requirements/test.lock /build/requirements/test.lock
RUN /opt/runtime-venv/bin/python -m pip install --require-hashes --no-deps \
        -r /build/requirements/test.lock
COPY --chown=10001:10001 . /app/
USER 10001:10001
RUN pytest -q /app/tests && ruff check /app/tools /app/tests/infra \
    && python /app/tools/image_smoke.py \
    && touch /tmp/wp02-tests-passed

# A release cannot be produced without completing the test stage above.
FROM runtime AS release
COPY --from=test --chown=10001:10001 /tmp/wp02-tests-passed /app/.wp02-tests-passed
