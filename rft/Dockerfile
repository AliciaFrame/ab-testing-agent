# RFT tool server + endpoint grader (see rft/README.md). Build from the repository root:
#   az acr build -r <registry> -t abtest-rft-tools:<tag> -f rft/Dockerfile .
FROM python:3.13-slim

WORKDIR /app
COPY rft/requirements.txt ./rft/requirements.txt
RUN pip install --no-cache-dir -r rft/requirements.txt

COPY agent/domain.py agent/abstats.py agent/grader.py ./agent/
COPY agent/data/arms.parquet ./agent/data/arms.parquet
COPY agent/specialists ./agent/specialists
COPY rft/__init__.py rft/common.py rft/grader.py rft/tool_server.py ./rft/

ENV PYTHONPATH=/app PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 RFT_STATE_DB=/app/state/traces.db
RUN useradd --create-home --uid 10001 appuser && mkdir -p /app/state && chown appuser /app/state
USER appuser
EXPOSE 8000
CMD ["hypercorn", "rft.tool_server:app", "--bind", "0.0.0.0:8000", "--workers", "4", "--keep-alive", "75"]
