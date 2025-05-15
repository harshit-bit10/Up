Here's a complete, well-structured Python Telegram bot using Pyrogram that performs real image upscaling using the Real-ESRGAN model. It processes images, sends real-time updates, and returns the final upscaled image with metadata.

✅ FEATURES:
/start command.

Accepts image uploads.

Sends progress updates (uploading → processing → done).

Uses Real-ESRGAN for upscaling (via realesrgan).

Adds metadata: Processed by SharkToonsIndia.

✅ REQUIREMENTS:
Install dependencies:

bash
Copy
Edit
pip install pyrogram tgcrypto pillow realesrgan
Also install Real-ESRGAN if not installed via pip:

bash
Copy
Edit
pip install realesrgan
If this fails, use:

bash
Copy
Edit
pip install git+https://github.com/xinntao/Real-ESRGAN

🧠 NOTES:
Model file RealESRGAN_x4.pth must be in the script's directory. You can download it from:
https://github.com/xinntao/Real-ESRGAN/blob/master/weights/RealESRGAN_x4.pth

Bot supports both CPU and GPU (CUDA).

Metadata is embedded using PIL's PngInfo.

🔐 Example Metadata on final image:
text
Copy
Edit
Comment: Upscaled by SharkToonsIndia
Would you like this bot to support videos or other formats too in future updates?







