# Multi-stage Dockerfile using Alpine (smaller and often more reliable)
FROM python:3.12-alpine AS builder

# Install build dependencies
RUN apk add --no-cache \
    build-base \
    git \
    curl \
    gcc \
    musl-dev \
    libffi-dev

# Copy project files
COPY requirements.txt /app/requirements.txt
COPY generate-mr-summary.py /app/generate-mr-summary.py
COPY prompt_template.txt /app/prompt_template.txt

# Install Python packages
WORKDIR /app
RUN pip install --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Final stage (runtime environment)
FROM python:3.12-alpine

# Install minimal runtime dependencies
RUN apk add --no-cache curl ca-certificates

# Create non-root user for security
RUN addgroup -g 1001 -S appuser && \
    adduser -S -D -H -u 1001 -h /app -s /sbin/nologin -G appuser appuser

# Copy only the necessary files from the builder stage
COPY --from=builder /usr/local/lib/python3.12/site-packages /usr/local/lib/python3.12/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin
COPY --from=builder /app /app

# Set working directory and ownership
WORKDIR /app
RUN chown -R appuser:appuser /app

# Switch to non-root user
USER appuser

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONPATH=/app

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "import sys; print('Python OK')" || exit 1

# Command to run the application
CMD ["python", "generate-mr-summary.py"]
