#!/bin/bash
uvicorn worker:app --host 0.0.0.0 --port ${PORT:-8080}
