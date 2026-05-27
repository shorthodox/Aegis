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

# Guard: fail loudly if compiled CSS was excluded by .dockerignore instead of silently
# serving broken pages. Run: cd web && npm run build:css and commit the result.
RUN test -f web/dist/styles.css || { echo "FATAL: web/dist/styles.css missing from image. Run 'cd web && npm run build:css' and commit the output."; exit 1; }

EXPOSE 8080

RUN chmod +x start.sh
CMD ["./start.sh"]
