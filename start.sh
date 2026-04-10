#!/bin/bash
# Databricks Apps startup script — runs FastAPI (background) + Streamlit (foreground)
# FastAPI listens on localhost:8080 (internal only)
# Streamlit listens on $DATABRICKS_APP_PORT (public, assigned by Databricks Apps)

set -e

echo "Starting FastAPI on 127.0.0.1:8080..."
gunicorn api.main:app \
  -w 2 \
  -k uvicorn.workers.UvicornWorker \
  --bind 127.0.0.1:8080 \
  --timeout 120 \
  --access-logfile - &

FASTAPI_PID=$!
echo "FastAPI started (PID $FASTAPI_PID)"

# Brief wait to let FastAPI initialise before Streamlit starts making calls
sleep 3

echo "Starting Streamlit on 0.0.0.0:${DATABRICKS_APP_PORT:-8000}..."
exec streamlit run ui/app.py \
  --server.port "${DATABRICKS_APP_PORT:-8000}" \
  --server.address "0.0.0.0" \
  --server.headless true \
  --server.enableCORS false \
  --server.enableXsrfProtection false
