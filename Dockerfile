FROM python:3.11-slim

# Install uv
RUN pip install uv

# Set working directory
WORKDIR /app

# Copy project files
COPY pyproject.toml .
COPY uv.lock .

# Install dependencies using uv
RUN uv sync --frozen

# Copy application code
COPY app/ ./app/
COPY data/ ./data/

# Expose port
EXPOSE 8080

# Start FastAPI app
CMD ["uv", "run", "uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8080"]