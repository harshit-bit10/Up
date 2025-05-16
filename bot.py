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
BOT_TOKEN = "7038431984:AAFnFttAC4gN1I2-ISw0UE8kC_4Ya9mHKsQ"
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
    tile=128,
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

# Helper: Check video file
def is_video(path):
    return any(path.lower().endswith(ext) for ext in [".mp4", ".mov", ".avi", ".mkv", ".webm", ".flv", ".ts"])

@app.on_message(filters.command("start"))
async def start(client, message: Message):
    await message.reply("👋 Send an image (photo or document) or a video file to upscale it using Real-ESRGAN.")

@app.on_message(filters.photo | filters.document | filters.video)
async def upscale(client, message: Message):
    try:
        if message.photo:
            # Download photo - returns file path
            media_path = await client.download_media(message.photo.file_id, file_name=os.path.join(TEMP_DIR, f"{message.id}_input.jpg"))
            is_video_file = False
        elif message.video:
            # Download video
            media_path = await client.download_media(message.video.file_id, file_name=os.path.join(TEMP_DIR, f"{message.id}_input{os.path.splitext(message.video.file_name or 'video.mp4')[1]}"))
            is_video_file = True
        elif message.document:
            filename = message.document.file_name or ""
            if is_image(filename):
                media_path = await client.download_media(message.document.file_id, file_name=os.path.join(TEMP_DIR, f"{message.id}_input{os.path.splitext(filename)[1]}"))
                is_video_file = False
            elif is_video(filename):
                media_path = await client.download_media(message.document.file_id, file_name=os.path.join(TEMP_DIR, f"{message.id}_input{os.path.splitext(filename)[1]}"))
                is_video_file = True
            else:
                await message.reply("❌ Unsupported file type. Please send an image or video file.")
                return
        else:
            await message.reply("❌ Please send a photo, image document, or video file.")
            return

        if not is_video_file:
            output_path = os.path.join(TEMP_DIR, f"{message.id}_upscaled.jpg")
            # Read image with OpenCV
            img = cv2.imread(media_path, cv2.IMREAD_COLOR)
            if img is None:
                await message.reply("❌ Could not read the image. Please ensure the file is a valid image.")
                return

            # Upscale the image
            output, _ = upsampler.enhance(img, outscale=4)
            cv2.imwrite(output_path, output)

            await message.reply_photo(photo=output_path, caption="✅ Upscaled using Real-ESRGAN.")

        else:
            # Process video
            output_path = os.path.join(TEMP_DIR, f"{message.id}_upscaled.mp4")

            # Open video capture
            cap = cv2.VideoCapture(media_path)
            if not cap.isOpened():
                await message.reply("❌ Could not open the video file.")
                return

            fps = cap.get(cv2.CAP_PROP_FPS)
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

            # Output video dimensions (4x upscale)
            out_width = width * 4
            out_height = height * 4

            # Define video writer
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(output_path, fourcc, fps, (out_width, out_height))

            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            current_frame = 0

            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                # Upscale frame
                output_frame, _ = upsampler.enhance(frame, outscale=4)

                # Convert output frame from RGB to BGR for OpenCV
                output_frame_bgr = cv2.cvtColor(output_frame, cv2.COLOR_RGB2BGR)

                # Write frame to output video
                out.write(output_frame_bgr)

                current_frame += 1
                # Optional: You can add progress reporting here by editing the message or logging

            cap.release()
            out.release()

            await message.reply_video(video=output_path, caption="✅ Video upscaled using Real-ESRGAN.")

    except Exception as e:
        await message.reply(f"❌ An error occurred: {str(e)}")

    finally:
        # Clean up temporary files
        if 'media_path' in locals() and os.path.exists(media_path):
            os.remove(media_path)
        if 'output_path' in locals() and os.path.exists(output_path):
            os.remove(output_path)

# Run the bot
app.run()
