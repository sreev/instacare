from flask import Flask, render_template, request, jsonify
import whisper
import openai
import os
import base64
import subprocess
import requests
from pathlib import Path
from openai import OpenAI
import time
client = OpenAI()

app = Flask(__name__)

# Function to encode the image
def encode_image(image_path):
  with open(image_path, "rb") as image_file:
    return base64.b64encode(image_file.read()).decode('utf-8')

openai.api_key = os.getenv("OPENAI_API_KEY")

if not openai.api_key:
    raise ValueError("OpenAI API key not found. Make sure you have set the OPENAI_API_KEY environment variable.")

# Setup paths for saving images and audio
image_folder = 'uploads/images'
audio_folder = 'uploads/audio'
transcription_folder = 'uploads/transcriptions'
speech_folder = 'uploads/speech'

if not os.path.exists(image_folder):
    os.makedirs(image_folder)
if not os.path.exists(audio_folder):
    os.makedirs(audio_folder)
if not os.path.exists(transcription_folder):
    os.makedirs(transcription_folder)
if not os.path.exists(speech_folder):
    os.makedirs(speech_folder)

whisper_model = whisper.load_model("base")

def tts(text):
    # add timestamp to the file name
    response_file_name = f"response_{str(time.time())}.mp3"
    speech_file_path = Path(speech_folder) / response_file_name
    response = client.audio.speech.create(
        model="tts-1",
        voice="alloy",
        input=text
    )
    response.stream_to_file(speech_file_path)
    return speech_file_path

def call_model(transcription_text, images):

    headers = {
    "Content-Type": "application/json",
    "Authorization": f"Bearer {openai.api_key}"
    }

    payload = {
    "model": "gpt-4o-mini",
    "messages": [
        {
        "role": "user",
        "content": [
            {
            "type": "text",
            "text": "You are a First aid Personal Doctor. You are tasked with diagnosing the patient from the description of the problem and the image provided."
                    "The patient is located at address 1868 Floribunda Ave, Hillsborough, CA 94010. Please provide a diagnosis and first aid plan such as ordering medicines and so on."
                    f"Patient's name is Poorna. Patient described the problem as {transcription_text}." 
            },
        ]
        }
    ],
    }

    for image in images:
        base64_image = encode_image(image)
        print("Image is being added")
        payload["messages"][0]["content"].append({
            "type": "image_url",
            "image_url": {
                "url": f"data:image/jpeg;base64,{base64_image}"
            }
            
        })
    print("Calling model...")
    response = requests.post("https://api.openai.com/v1/chat/completions", headers=headers, json=payload)
    res_msg = response.json()['choices'][0]['message']['content']
    # print(response.json())
    return res_msg

@app.route('/')
def index():
    return render_template('index.html')

# Route to handle image uploads
@app.route('/upload_image', methods=['POST'])
def upload_image():
    image = request.files['image']
    image.save(os.path.join(image_folder, image.filename))
    return 'Image uploaded successfully'

# Route to handle audio upload
@app.route('/upload_media', methods=['POST'])
def upload_audio():
    # Handle the uploaded media file
    media_file = request.files['media']
    if not media_file:
        return jsonify({"error": "No media file uploaded."}), 400

    # Save the uploaded media (webm)
    media_path = os.path.join(audio_folder, media_file.filename)
    media_file.save(media_path)

    # Define WAV file name and path
    wav_filename = f"{os.path.splitext(media_file.filename)[0]}.wav"
    wav_path = os.path.join(audio_folder, wav_filename)
    # Extract audio from the media and convert it to WAV using FFmpeg
    try:
        # Use FFmpeg to extract only the audio from the media file and convert to WAV
        subprocess.run(
            ['ffmpeg', '-i', media_path, '-vn', '-ac', '1', '-ar', '16000', wav_path],
            check=True
        )
        print(f"Audio extracted and converted to WAV: {wav_path}")
    except subprocess.CalledProcessError as e:
        return jsonify({"error": f"FFmpeg audio extraction failed: {str(e)}"}), 500

    # Transcribe the audio using Whisper
    print("Transcribing audio...")
    transcription = whisper_model.transcribe(wav_path)
    transcription_text = transcription['text']

    # Save the transcription to a file
    transcription_filename = f"{os.path.splitext(wav_filename)[0]}.txt"
    transcription_path = os.path.join(transcription_folder, transcription_filename)
    with open(transcription_path, 'w') as f:
        f.write(transcription_text)

    print(f"Transcription saved to {transcription_path}")

    # Get all the image files in the images folder
    print("Getting all images...")
    images = [os.path.join(image_folder, image) for image in os.listdir(image_folder)]

    response = call_model(transcription_text, images)
    print("Response is ", response)
    audio_response = tts(response)
    return jsonify({"response": audio_response})

if __name__ == '__main__':
    app.run(debug=True)