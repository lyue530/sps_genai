FROM python:3.12-slim

WORKDIR /app

# Copy project files, including trained CNN weights
COPY . .

# Install uv
RUN pip install --no-cache-dir uv

# Install project dependencies
RUN uv sync

# Download spaCy English language model
RUN uv run python -m spacy download en_core_web_lg

# Expose FastAPI port
EXPOSE 8000

# Start FastAPI application
CMD ["uv", "run", "uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
