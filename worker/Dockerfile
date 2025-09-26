# Use an official Python runtime as a parent image
FROM python:3.10-slim-bullseye

# Set the working directory in the container
WORKDIR /app

# Install ffmpeg and its dependencies
RUN apt-get update && apt-get install -y ffmpeg libsm6 libxext6 && rm -rf /var/lib/apt/lists/*

# Install any needed packages specified in requirements.txt
COPY worker/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the worker script and the shared backend module into the container
COPY worker/worker.py .
COPY backend/ ./backend/

# Command to run the application using uvicorn
CMD ["uvicorn", "worker:app", "--host", "0.0.0.0", "--port", "8080"]
