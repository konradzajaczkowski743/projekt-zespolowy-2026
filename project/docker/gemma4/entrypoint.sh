#!/bin/sh
set -e

ollama serve &
PID=$!

until curl -sf http://localhost:11434/api/tags >/dev/null; do
  sleep 2
done

if ! ollama list | grep -q "${MODEL_NAME}"; then
  echo "Pulling ${MODEL_NAME}..."
  ollama pull "${MODEL_NAME}"
fi

echo "Gemma ready"
wait $PID