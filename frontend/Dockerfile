# Frontend Dockerfile
FROM python:3.9-slim

WORKDIR /app

# Install dependencies
COPY ./requirements.txt requirements.txt

RUN apt-get update && apt-get install -y \
    libpq-dev gcc

# Install PostgreSQL client
RUN apt-get update && apt-get install -y postgresql

RUN pip install -r requirements.txt


# Copy frontend code
COPY . .

# Expose Streamlit port
EXPOSE 8501

# Run Streamlit app
CMD ["streamlit", "run", "canvas.py", "--server.port=8501", "--server.address=0.0.0.0"]
