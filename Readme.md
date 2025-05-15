✅ Requirements
Install these packages:

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
