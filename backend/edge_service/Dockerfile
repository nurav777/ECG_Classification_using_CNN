FROM python:3.11-slim

WORKDIR /app
COPY common /app/common
COPY edge_service/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY edge_service/ .
ENV PYTHONPATH=/app

EXPOSE 8001
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8001"]
