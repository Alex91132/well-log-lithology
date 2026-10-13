FROM python:3.11-slim

# libgomp — OpenMP для LightGBM
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*
WORKDIR /app
COPY requirements-service.txt .
RUN pip install --no-cache-dir -r requirements-service.txt
COPY src/ src/
COPY models/ models/

RUN useradd --create-home app
USER app
ENV PYTHONPATH=/app/src
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health')"
CMD ["uvicorn", "litho.service:app", "--host", "0.0.0.0", "--port", "8000"]
