FROM python:3.14-slim

# Lightweight Dockerfile for the rag app image used by docker-compose.
WORKDIR /app

# Install runtime deps first for better caching. Copy pyproject only to
# avoid copying the whole repo before installing deps.
COPY pyproject.toml /app/
RUN python -m pip install --upgrade pip && pip install -e .

# Copy application code.
COPY . /app

EXPOSE 8000 8501

# Default command runs the FastAPI app; override in compose for UI.
CMD ["uvicorn", "rag.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
