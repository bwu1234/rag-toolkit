FROM python:3.14-slim

# Lightweight Dockerfile for the rag app image used by docker-compose.
WORKDIR /app

# Install runtime deps first for better caching: this layer only rebuilds when
# pyproject.toml changes. The stub package is what makes it work -- installing
# with no `rag/` present registers an editable install that maps nothing, and
# `rag` is then importable only from /app (so `streamlit run rag/ui/app.py`,
# which doesn't put the working directory on sys.path, fails). With the stub,
# the install maps `rag` to /app/rag, which the COPY below then fills in.
COPY pyproject.toml /app/
RUN mkdir -p rag && touch rag/__init__.py \
    && python -m pip install --upgrade pip \
    && pip install -e .

# Copy application code.
COPY . /app

EXPOSE 8000 8501

# Default command runs the FastAPI app; override in compose for UI.
CMD ["uvicorn", "rag.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
