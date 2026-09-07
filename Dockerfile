FROM python:3.11-slim

WORKDIR /app
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PIP_NO_CACHE_DIR=1

COPY requirements_deploy.txt .
RUN pip install --upgrade pip && pip install -r requirements_deploy.txt

COPY . .

CMD ["sh","-c","exec uvicorn run_app:app --host 0.0.0.0 --port ${PORT:-8000}"]
