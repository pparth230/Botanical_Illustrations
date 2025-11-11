# Dockerfile for Botanical Illustration API
# Based on working Colab notebook

FROM runpod/pytorch:2.1.0-py3.10-cuda11.8.0-devel-ubuntu22.04

# Set working directory
WORKDIR /app

# Copy requirements and handler
COPY requirements.txt /app/
COPY handler.py /app/

# Install Python dependencies from your Colab requirements
RUN pip install --no-cache-dir -r requirements.txt && \
    pip install --no-cache-dir runpod

# Clean up to reduce image size
RUN pip cache purge
COPY handler.py /app/

# Expose port (not strictly necessary for serverless but good practice)
EXPOSE 8000

# Set the command to run the handler
CMD ["python", "-u", "handler.py"]
