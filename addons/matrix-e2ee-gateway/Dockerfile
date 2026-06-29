FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
       build-essential \
       ffmpeg \
       libolm-dev \
       libolm3 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt

COPY gateway.py /app/gateway.py
COPY run.sh /run.sh
RUN chmod a+x /run.sh

RUN mkdir -p /data/store

EXPOSE 18081
CMD ["/run.sh"]
