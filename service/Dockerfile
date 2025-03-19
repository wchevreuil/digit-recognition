# Backend Dockerfile
FROM python:3.9-slim

WORKDIR /app

# Install dependencies
COPY ./requirements.txt requirements.txt
# Install OpenCV dependencies
RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0

RUN pip install -r requirements.txt

# Copy backend code and model
COPY . .

# Expose FastAPI port
EXPOSE 8000

# Run FastAPI server
CMD ["uvicorn", "model_service:app", "--host", "0.0.0.0", "--port", "8000"]
