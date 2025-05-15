import os
from pyrogram import Client, filters
from pyrogram.types import Message
from PIL import Image, PngImagePlugin
from realesrgan import RealESRGAN
import torch
from io import BytesIO

# Configuration
API_ID = 16501053  # Replace with your own
API_HASH = "d8c9b01c863dabacc484c2c06cdd0f6e"
BOT_TOKEN = "7038431984:AAG5FNQMVcKm_lQv_ebQ0VVtyRU5IeCvCRM"
METADATA_CREDIT = "SharkToonsIndia"

# Create bot instance
app = Client("ImageUpscalerBot", api_id=API_ID, api_hash=API_HASH, bot_token=BOT_TOKEN)

# Start command
@app.on_message(filters.command("start"))
async def start_cmd(_, message: Message):
    await message.reply_text(
        "👋 Welcome to the *Real-ESRGAN Image Upscaler Bot*!\n\n"
        "📤 Send me a photo and I will upscale it for you using AI.\n"
        f"🖋️ Metadata will be tagged as `{METADATA_CREDIT}`.",
        quote=True
    )

# Handle images
@app.on_message(filters.photo)
async def upscale_image(_, message: Message):
    status = await message.reply_text("📥 Downloading image...", quote=True)

    # Download image
    image_path = await app.download_media(message.photo.file_id, file_name="input.jpg")

    await status.edit("⚙️ Upscaling using Real-ESRGAN...")

    try:
        # Load image
        img = Image.open(image_path).convert("RGB")

        # Initialize Real-ESRGAN model
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = RealESRGAN(device, scale=4)
        model.load_weights('RealESRGAN_x4.pth')  # Download model if not present from official repo

        # Upscale image
        sr_image = model.predict(img)

        # Add metadata
        meta = PngImagePlugin.PngInfo()
        meta.add_text("Comment", f"Upscaled by {METADATA_CREDIT}")

        output_path = "upscaled.png"
        sr_image.save(output_path, pnginfo=meta)

        await status.edit("📤 Uploading final image...")

        await app.send_photo(
            chat_id=message.chat.id,
            photo=output_path,
            caption=f"✅ Image upscaled successfully by `{METADATA_CREDIT}`",
            reply_to_message_id=message.message_id
        )

        await status.delete()

    except Exception as e:
        await status.edit(f"❌ Error: {str(e)}")

    finally:
        # Cleanup
        if os.path.exists("input.jpg"):
            os.remove("input.jpg")
        if os.path.exists("upscaled.png"):
            os.remove("upscaled.png")

# Run the bot
app.run()
