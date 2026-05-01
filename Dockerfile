FROM python:3.12-slim

WORKDIR /app

COPY pyproject.toml .
RUN pip install --no-cache-dir .

COPY src/ src/
COPY data/processed/ data/processed/
COPY data/docs/ data/docs/
COPY models/ models/
COPY .env .env

EXPOSE 8000

CMD ["uvicorn", "src.serving.app:app", "--host", "0.0.0.0", "--port", "8000"]
