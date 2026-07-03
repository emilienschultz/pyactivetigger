#!/bin/bash
set -e

# The container starts as root so it can fix ownership of the named volumes
# (venv, uv cache, HF cache), then drops privileges via gosu. The chown is
# guarded: a recursive chown over downloaded models can take minutes, so it
# only runs when the venv is not yet owned by the target user (first boot).
if [ "$(id -u)" = "0" ]; then
  if [ "$(stat -c %u /home/python/venv 2>/dev/null)" != "${USER_ID:-1000}" ]; then
    chown -R "${USER_ID:-1000}:${GROUP_ID:-1000}" /home/python
  fi
  exec gosu "${USER_ID:-1000}:${GROUP_ID:-1000}" "$0" "$@"
fi

# Fail fast if the compose overlay declared a required MODE and .env disagrees.
# The prod overlay sets EXPECTED_MODE=prod; the base/dev stack leaves it unset.
if [ -n "$EXPECTED_MODE" ] && [ "$EXPECTED_MODE" != "$MODE" ]; then
  echo "ERROR: this compose overlay expects MODE=$EXPECTED_MODE but MODE=$MODE." >&2
  echo "       Edit docker/.env and set MODE=$EXPECTED_MODE (see DEPLOY.md)." >&2
  exit 1
fi

if [ "$MODE" = "dev" ]; then
  echo "/!\\ Mode is set to DEV /!\\"
else
  echo "/!\\ Mode is set to PROD /!\\"
fi
echo "(i) Python version is $(python3 --version)"
echo "(i) uv version is $(uv --version)"

if [ ! -f "${UV_PROJECT_ENVIRONMENT}/pyvenv.cfg" ]; then
  echo
  echo " ~"
  echo " ~ Creating python venv"
  echo " ~"
  echo
  rm -rf "${UV_PROJECT_ENVIRONMENT:?}"/* "${UV_PROJECT_ENVIRONMENT:?}"/.[!.]* 2>/dev/null || true
  uv venv "${UV_PROJECT_ENVIRONMENT}"
fi
source "${UV_PROJECT_ENVIRONMENT}/bin/activate"

echo
echo " ~"
echo " ~ Install dependencies"
echo " ~"
echo
# install python deps
cd /api

if [ "$CPU_ONLY" = "true" ]; then
  echo "CPU-only mode: installing PyTorch without CUDA (~5GB savings)"
  uv sync --extra cpu
else
  echo "GPU mode: installing PyTorch with CUDA (~5GB extra) and cuml-cu12"
  uv sync --extra gpu
  uv pip install --extra-index-url https://pypi.nvidia.com cuml-cu12
fi

# Check for a config.yaml file in the api directory
if [ ! -f /api/config.yaml ]; then
    if [ -f /api/config.yaml.sample ]; then
        echo "No config.yaml found. Copying config.yaml.sample to config.yaml..."
        cp /api/config.yaml.sample /api/config.yaml
        echo "You can now edit activetigger/api/config.yaml to modify paths for static files and database."
    else
        echo "No config.yaml.sample found in activetigger/api. Please ensure your configuration is set as needed."
    fi
fi

# Launch workers
uv run watchmedo auto-restart \
    --directory="./task_manager/src/task_manager/tasks" \
    --pattern="*.py" \
    --recursive \
    -- uv run celery-worker-cpu &

# Launch the server
echo "Launching API on port $API_PORT..."

if [ "$MODE" = "dev" ]; then
  echo
  echo " ~"
  echo " ~ Start api dev"
  echo " ~"
  echo
  uv run --no-sync python3 -m activetigger -p "$API_PORT"
else
  echo
  echo " ~"
  echo " ~ Start api production"
  echo " ~"
  echo
  # Single worker is intentional: the orchestrator holds in-memory project state
  # and runs a background asyncio update loop. Heavy compute is offloaded to
  # multiprocessing workers via queue_manager. --proxy-headers is required
  # because nginx forwards X-Forwarded-* in front of the API.
  exec uv run --no-sync uvicorn activetigger.app.main:app \
    --host 0.0.0.0 \
    --port "$API_PORT" \
    --proxy-headers \
    --forwarded-allow-ips='*' \
    --log-level info
fi
