mkdir -p models
wget https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth -O RealESRGAN_x4plus.pth




✅ Requirements
Install these packages:

mkdir -p models
wget https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth -O RealESRGAN_x4plus.pth


bash
Copy
Edit
pip install pyrogram tgcrypto torch torchvision numpy realesrgan opencv-python
Also download the Real-ESRGAN model (x4 scale):

bash
Copy
Edit
mkdir models
wget https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth -O models/RealESRGAN_x4plus.pth
✅ bot.py (Pyrogram Real-ESRGAN Bot)
📝 Notes:
Replace API_ID, API_HASH, and BOT_TOKEN with your actual values from https://my.telegram.org.

This script uses RRDBNet with the RealESRGAN x4 model.

Metadata (SharkToonsIndia) is added in the caption. If you want actual image metadata embedded, let me know—I can use PIL’s PngInfo.


✅ Option 1: Adjust wget to save in bot.py directory
Assuming bot.py is in the current directory:

bash
Copy
Edit
mkdir -p models
wget https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth -O models/RealESRGAN_x4plus.pth
To make bot.py compatible with this change, update the weights path in bot.py:

python
Copy
Edit
WEIGHTS_PATH = "models/RealESRGAN_x4plus.pth"




# Install basicsr first
pip install git+https://github.com/xinntao/BasicSR.git

# Install realesrgan from official repo
pip install git+https://github.com/xinntao/Real-ESRGAN.git

