FROM python:3.11-slim

WORKDIR /app
COPY common /app/common
COPY fog_service/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY fog_service/ .
ENV PYTHONPATH=/app

EXPOSE 8002
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8002"]
