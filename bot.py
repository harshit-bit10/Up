import os
import torch
import numpy as np
import cv2
from pyrogram import Client, filters
from pyrogram.types import Message
from PIL import Image
from realesrgan import RealESRGANer
from basicsr.archs.rrdbnet_arch import RRDBNet

# Configuration
API_ID = 16501053  # Replace with your own
API_HASH = "d8c9b01c863dabacc484c2c06cdd0f6e"
BOT_TOKEN = "7038431984:AAG5FNQMVcKm_lQv_ebQ0VVtyRU5IeCvCRM"
METADATA_CREDIT = "SharkToonsIndia"

bot = Client("realesrgan_bot", api_id=API_ID, api_hash=API_HASH, bot_token=BOT_TOKEN)

model_path = "models/RealESRGAN_x4plus.pth"

# Load model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=4)
upscaler = RealESRGANer(
    scale=4,
    model_path=model_path,
    model=model,
    tile=0,
    tile_pad=10,
    pre_pad=0,
    half=False,
    device=device
)

@bot.on_message(filters.command("start"))
async def start_cmd(_, msg: Message):
    await msg.reply("👋 Welcome to *Real-ESRGAN Upscaler Bot*\n\nSend me an image and I'll upscale it for you with *SharkToonsIndia* metadata.", quote=True)

@bot.on_message(filters.photo)
async def upscale(_, msg: Message):
    sent = await msg.reply("🔄 Downloading image...")

    image_path = f"{msg.from_user.id}_input.jpg"
    output_path = f"{msg.from_user.id}_upscaled.png"

    await msg.download(image_path)
    await sent.edit("📈 Upscaling image...")

    img = cv2.imread(image_path, cv2.IMREAD_COLOR)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    try:
        output, _ = upscaler.enhance(img, outscale=4)
        upscaled_img = Image.fromarray(output)
        upscaled_img.save(output_path, pnginfo=None)

        await sent.edit("✅ Upscaling complete! Uploading...")

        await msg.reply_photo(
            output_path,
            caption="✨ Here is your upscaled image by *SharkToonsIndia*."
        )

    except Exception as e:
        await sent.edit(f"❌ Error: {str(e)}")

    finally:
        if os.path.exists(image_path): os.remove(image_path)
        if os.path.exists(output_path): os.remove(output_path)
        await sent.delete()

bot.run()
