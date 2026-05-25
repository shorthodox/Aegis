FROM python:3.12-slim

WORKDIR /app

# Build dependencies for native packages (cryptography, grpcio, llvmlite, etc.)
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    g++ \
    libffi-dev \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies first (layer cached unless requirements change)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application source
COPY . .

EXPOSE 8080

RUN chmod +x start.sh
CMD ["./start.sh"]
