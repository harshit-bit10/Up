import os
import torch
import cv2
from pyrogram import Client, filters
from pyrogram.types import Message
from PIL import Image
from realesrgan import RealESRGAN
from datetime import datetime
from io import BytesIO

# ========== CONFIG ==========
# Configuration
API_ID = 16501053  # Replace with your own
API_HASH = "d8c9b01c863dabacc484c2c06cdd0f6e"
BOT_TOKEN = "7038431984:AAG5FNQMVcKm_lQv_ebQ0VVtyRU5IeCvCRM"
METADATA_CREDIT = "SharkToonsIndia"
WEIGHTS_PATH = "RealESRGAN_x4.pth"

# ========== INIT ==========
bot = Client("upscaler_bot", api_id=API_ID, api_hash=API_HASH, bot_token=BOT_TOKEN)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = RealESRGAN(device, scale=4)
model.load_weights(WEIGHTS_PATH)

# ========== UTILS ==========
def add_metadata(image_path: str, credit: str) -> str:
    img = Image.open(image_path)
    metadata = img.info
    metadata["Author"] = credit
    output_path = image_path.replace(".png", "_meta.png")
    img.save(output_path, pnginfo=Image.PngImagePlugin.PngInfo(metadata))
    return output_path

def is_valid_image(mime: str) -> bool:
    return mime.startswith("image/")

# ========== HANDLERS ==========
@bot.on_message(filters.command("start"))
async def start(_, msg: Message):
    await msg.reply("👋 Welcome to the Real-ESRGAN Upscaler Bot!\nSend an image (as photo or document) to upscale.\n\nMetadata will include: SharkToonsIndia")

@bot.on_message(filters.photo | filters.document)
async def upscale_image(_, msg: Message):
    file = msg.photo or msg.document
    mime = file.mime_type if hasattr(file, "mime_type") else None

    if not is_valid_image(mime):
        await msg.reply("❌ Only image files are supported. Please send a valid photo or image document.")
        return

    await msg.reply("📥 Downloading image...")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    input_path = f"input_{timestamp}.png"
    output_path = f"output_{timestamp}.png"

    # Download image
    await msg.download(file_name=input_path)

    # Read image
    try:
        img = cv2.imread(input_path)
        if img is None:
            raise ValueError("Image is empty or unreadable.")
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    except Exception as e:
        await msg.reply(f"❌ Failed to read image. Error: {e}")
        return

    await msg.reply("🧠 Upscaling in progress...")

    # Upscale
    try:
        upscaled = model.predict(img)
        upscaled = cv2.cvtColor(upscaled, cv2.COLOR_RGB2BGR)
        cv2.imwrite(output_path, upscaled)
    except Exception as e:
        await msg.reply(f"❌ Failed during upscaling. Error: {e}")
        return

    # Add metadata
    try:
        final_path = add_metadata(output_path, CREDIT_NAME)
    except Exception:
        final_path = output_path  # fallback if metadata fails

    await msg.reply("✅ Upscaling complete. Sending image...")

    await msg.reply_document(final_path, caption=f"Upscaled by SharkToonsIndia")

    # Clean up
    for path in [input_path, output_path, final_path]:
        if os.path.exists(path):
            os.remove(path)

# ========== START BOT ==========
print("🤖 Bot is running...")
bot.run()
