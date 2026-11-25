FROM python:3.11-slim
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 && rm -rf /var/lib/apt/lists/*
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY almanac ./almanac
# mount or copy a trained model and the history panel at runtime:
#   docker run -p 8000:8000 -v $PWD/models:/app/models -v $PWD/data:/app/data -v $PWD/reports:/app/reports almanac
EXPOSE 8000
CMD ["uvicorn", "almanac.api:app", "--host", "0.0.0.0", "--port", "8000"]
