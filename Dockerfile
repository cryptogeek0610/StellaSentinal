FROM python:3.11-slim

# Install system dependencies for pyodbc and SQL Server
RUN apt-get update && apt-get install -y \
    curl \
    gnupg \
    unixodbc \
    unixodbc-dev \
    && rm -rf /var/lib/apt/lists/*

# Install Microsoft ODBC Driver for SQL Server
RUN mkdir -p /usr/share/keyrings \
    && curl https://packages.microsoft.com/keys/microsoft.asc | gpg --dearmor -o /usr/share/keyrings/microsoft-prod.gpg \
    && curl https://packages.microsoft.com/config/debian/12/prod.list > /etc/apt/sources.list.d/mssql-release.list \
    && apt-get update \
    && ACCEPT_EULA=Y apt-get install -y msodbcsql18 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy dependency files and source code
COPY pyproject.toml ./
COPY src/ ./src/

# Install Python dependencies (editable install)
RUN pip install --no-cache-dir --upgrade pip setuptools wheel && \
    pip install --no-cache-dir -e .

# Set Python path (redundant but explicit)
ENV PYTHONPATH=/app/src

# Default command (can be overridden in docker-compose)
CMD ["python", "-m", "device_anomaly.cli.main"]

