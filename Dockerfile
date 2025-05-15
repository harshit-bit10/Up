FROM python:3.10-slim

# Set workdir
WORKDIR /app

# Install dependencies
RUN apt update && apt install -y ffmpeg libgl1 wget git

# Install Python packages
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Download RealESRGAN model
RUN mkdir -p models && \
    wget https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth -O models/RealESRGAN_x4plus.pth

# Copy bot code
COPY bot.py .
COPY models models

# Expose nothing (bot will call Telegram API)
CMD ["python", "bot.py"]

