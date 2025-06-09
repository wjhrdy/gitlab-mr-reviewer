# Dockerfile
FROM python:3.11-alpine

# Install system dependencies including git and build tools
RUN apk add --no-cache \
    git \
    curl \
    build-base \
    libffi-dev \
    openssl-dev \
    python3-dev \
    musl-dev \
    gcc \
    g++ \
    make \
    && rm -rf /var/cache/apk/*

# Install dbt-core and Snowflake adapter (since all projects use Snowflake)
RUN pip install --no-cache-dir \
    dbt-core \
    dbt-snowflake

# Configure git to disable SSL verification globally
RUN git config --global http.sslVerify false && \
    git config --global http.sslCAInfo "" && \
    git config --global http.sslCAPath "" && \
    git config --global http.sslCert "" && \
    git config --global http.sslKey "" && \
    git config --global http.sslCertPasswordProtected false

# Set environment variables to disable SSL verification
ENV GIT_SSL_NO_VERIFY=true \
    PYTHONHTTPSVERIFY=0 \
    CURL_CA_BUNDLE="" \
    REQUESTS_CA_BUNDLE="" \
    SSL_VERIFY=false

# Set working directory
WORKDIR /app

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Make /app and /tmp writable for any user (OpenShift compatibility)
RUN chmod -R g+rwX /app && \
    chmod -R g+rwX /tmp

# Don't create specific user - let OpenShift assign UID
# OpenShift will automatically assign a UID in the allowed range

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Run the application
CMD ["python", "webhook_handler.py"]
