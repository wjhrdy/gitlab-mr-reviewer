FROM registry.access.redhat.com/ubi9/python-39:latest

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

# Switch to root to install system packages
USER 0

# Install system dependencies
RUN yum update -y && \
    yum install -y git && \
    yum clean all && \
    rm -rf /var/cache/yum

# Switch back to the default user
USER 1001

# Set working directory
WORKDIR /app

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy the application code and prompt template
COPY generate-mr-summary.py .
COPY prompt_template.txt .

# Create a non-root user for security
RUN chmod +x generate-mr-summary.py

# Set the default command
CMD ["python", "generate-mr-summary.py"]
