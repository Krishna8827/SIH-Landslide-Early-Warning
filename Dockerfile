FROM python:3.11-slim

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PIP_NO_CACHE_DIR=1

# Required by LightGBM / XGBoost OpenMP runtime on Debian slim.
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements_deploy.txt /app/requirements_deploy.txt
RUN pip install --upgrade pip && pip install -r /app/requirements_deploy.txt

COPY . /app

CMD ["sh", "-c", "exec uvicorn run_app:app --host 0.0.0.0 --port ${PORT:-8000}"]
