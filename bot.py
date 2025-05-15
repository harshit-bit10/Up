# import os


# ========== CONFIG ==========
# Configuration

# WEIGHTS_PATH = "models/RealESRGAN_x4plus.pth"


import os
import torch
from pyrogram import Client, filters
from pyrogram.types import Message
from PIL import Image
import cv2
from realesrgan import RealESRGAN

# Setup
API_ID = 16501053  # Replace with your own
API_HASH = "d8c9b01c863dabacc484c2c06cdd0f6e"
BOT_TOKEN = "7038431984:AAG5FNQMVcKm_lQv_ebQ0VVtyRU5IeCvCRM"
METADATA_CREDIT = "SharkToonsIndia"

# Paths
MODEL_PATH = "models/RealESRGAN_x4plus.pth"
TEMP_DIR = "temp"
os.makedirs(TEMP_DIR, exist_ok=True)

# Init bot and model
app = Client("upscaler_bot", api_id=API_ID, api_hash=API_HASH, bot_token=BOT_TOKEN)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = RealESRGAN(device, scale=4)
model.load_weights(MODEL_PATH)

# Helper: Check file is an image
def is_image(file_path):
    ext = file_path.lower().split(".")[-1]
    return ext in ["jpg", "jpeg", "png", "bmp", "webp"]

@app.on_message(filters.command("start"))
async def start(_, message: Message):
    await message.reply("👋 Send me an image (photo or image document) to upscale it using Real-ESRGAN.")

@app.on_message(filters.photo | filters.document)
async def upscale_image(client, message: Message):
    media = message.photo or message.document
    if not media:
        await message.reply("❌ This isn't an image file.")
        return

    # Generate paths
    input_path = os.path.join(TEMP_DIR, f"{message.id}_input.jpg")
    output_path = os.path.join(TEMP_DIR, f"{message.id}_upscaled.jpg")

    try:
        # Download and validate image
        downloaded = await media.download(file_name=input_path)
        if not is_image(downloaded):
            await message.reply("❌ Only image files (JPG, PNG, etc.) are supported.")
            os.remove(downloaded)
            return

        # Read and upscale
        img = Image.open(downloaded).convert("RGB")
        upscaled_img = model.predict(img)

        # Save with metadata
        upscaled_img.save(output_path, quality=95)
        await message.reply_photo(photo=output_path, caption="✅ Upscaled using Real-ESRGAN.")
    
    except Exception as e:
        await message.reply(f"❌ Failed to process image.\n\nError: `{e}`")
    
    finally:
        # Cleanup
        if os.path.exists(input_path):
            os.remove(input_path)
        if os.path.exists(output_path):
            os.remove(output_path)

# Run bot
app.run()
