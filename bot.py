import os
import torch
import numpy as np
from PIL import Image
from pyrogram import Client, filters
from pyrogram.types import Message
import cv2

from realesrgan import RealESRGANer
from basicsr.archs.rrdbnet_arch import RRDBNet

# Setup
API_ID = 16501053  # Replace with your own
API_HASH = "d8c9b01c863dabacc484c2c06cdd0f6e"
BOT_TOKEN = "7038431984:AAG5FNQMVcKm_lQv_ebQ0VVtyRU5IeCvCRM"
METADATA_CREDIT = "SharkToonsIndia"

# File paths
TEMP_DIR = "temp"
MODEL_PATH = "RealESRGAN_x4plus.pth"
os.makedirs(TEMP_DIR, exist_ok=True)

# Load model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=4)
upsampler = RealESRGANer(
    scale=4,
    model_path=MODEL_PATH,
    model=model,
    tile=0,
    tile_pad=10,
    pre_pad=0,
    half=True if torch.cuda.is_available() else False,
    device=device
)

# Start bot
app = Client("upscaler_bot", api_id=API_ID, api_hash=API_HASH, bot_token=BOT_TOKEN)

# Helper: Check image file
def is_image(path):
    return any(path.lower().endswith(ext) for ext in [".jpg", ".jpeg", ".png", ".bmp", ".webp"])

@app.on_message(filters.command("start"))
async def start(client, message: Message):
    await message.reply("👋 Send an image (as photo or image document) to upscale it using Real-ESRGAN.")

@app.on_message(filters.photo | filters.document)
async def upscale(client, message: Message):
    if message.photo:
        # Download photo - returns file path
        image_path = await message.photo.download(file_name=os.path.join(TEMP_DIR, f"{message.id}_input.jpg"))
    elif message.document:
        # Check if the document is an image
        if not is_image(message.document.file_name):
            await message.reply("❌ Only image documents are supported.")
            return
        image_path = await message.document.download(file_name=os.path.join(TEMP_DIR, f"{message.id}_input.jpg"))
    else:
        await message.reply("❌ Please send a photo or image document.")
        return

    output_path = os.path.join(TEMP_DIR, f"{message.id}_upscaled.jpg")

    try:
        # Read image with OpenCV
        img = cv2.imread(image_path, cv2.IMREAD_COLOR)
        if img is None:
            await message.reply("❌ Could not read the image.")
            os.remove(image_path)
            return

        output, _ = upsampler.enhance(img, outscale=4)
        cv2.imwrite(output_path, output)

        await message.reply_photo(photo=output_path, caption="✅ Upscaled using Real-ESRGAN.")

    except Exception as e:
        await message.reply(f"❌ Error: {e}")

    finally:
        if os.path.exists(image_path):
            os.remove(image_path)
        if os.path.exists(output_path):
            os.remove(output_path)

# Run the bot
app.run()
