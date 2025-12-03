FROM pytorch/pytorch:2.0.1-cuda11.7-cudnn8-devel

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy project files
COPY . .

# Install dependencies and the package
RUN pip install --upgrade pip && \
    pip install -r requirements.txt && \
    pip install -e .

# Default command runs the test suite
CMD ["bash", "scripts/test_all.sh"]
