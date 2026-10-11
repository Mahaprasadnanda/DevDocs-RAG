FROM python:3.12-slim

WORKDIR /app

# Pre-install CPU-only PyTorch to prevent downloading multi-gigabyte CUDA wheels
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu

# Install remaining backend dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application source and persisted data
COPY app/ ./app/
COPY data/ ./data/

ENV PYTHONUNBUFFERED=1

EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
