from flask import Flask, render_template, request, jsonify, send_from_directory
import whisper
import openai
import os
import base64
import subprocess
import requests
import json
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

# add complete absolute path to the image folder
# Setup paths for saving images and audio
image_folder = 'uploads/images'
image_folder = os.path.join(os.getcwd(), image_folder)
audio_folder = 'uploads/audio'
audio_folder = os.path.join(os.getcwd(), audio_folder)
transcription_folder = 'uploads/transcriptions'
transcription_folder = os.path.join(os.getcwd(), transcription_folder)
speech_folder = 'uploads/speech'
speech_folder = os.path.join(os.getcwd(), speech_folder)

# Define the path to your uploads directory (this should be relative to your project directory)
UPLOAD_FOLDER = speech_folder
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

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
    return response_file_name

message_history = []
message_history.append({
    "role": "system",
    "content": [
        {
            "type": "text",
            "text": "You are a First aid Personal Doctor. You are tasked with diagnosing the patient from the description of the problem and the image provided."
                    "The patient is located at address 1868 Floribunda Ave, Hillsborough, CA 94010. Please provide a diagnosis and first aid plan such as ordering medicines and so on."
                    "Patient's name is Poorna. Please interact with the patient to get more information."
        }
    ]
})

def call_model(transcription_text, images):

    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {openai.api_key}"
    }
    message_history.append({
        "role": "user",
        "content": [
            {
                "type": "text",
                "text": f"{transcription_text}"
            }
        ]
    })
    
    payload = {
        "model": "gpt-4o-mini",
        "messages": message_history
    }

    for image in images:
        base64_image = encode_image(image)
        print("Image is being added")
        payload["messages"][-1]["content"].append({
            "type": "image_url",
            "image_url": {
                "url": f"data:image/jpeg;base64,{base64_image}"
            }
            
        })
    print("Calling model... ", transcription_text)
    response = requests.post("https://api.openai.com/v1/chat/completions", headers=headers, json=payload)
    res_msg = response.json()['choices'][0]['message']['content']
    message_history.append({
        "role": "assistant",
        "content": res_msg
    })
    # print(response.json())
    return res_msg

def modelResponse(prompt):
    url = "http://192.168.3.251:11434/api/generate"
    headers = {
        "Content-Type": "application/json"
    }
    data = {
        "model": "llama3",
        "prompt": prompt,
        "stream": False
    }

    response = requests.post(url, headers=headers, data=json.dumps(data))

    if response.status_code == 200:
        response_text = response.text
        data = json.loads(response_text)
        actual_response = data["response"]
        print(actual_response)
        return actual_response
    else:
        print("Error: ", response.status_code, response.text)


@app.route('/')
def index():
    return render_template('index.html')

# Route to handle image uploads
@app.route('/upload_image', methods=['POST'])
def upload_image():
    image = request.files['image']
    image.save(os.path.join(image_folder, image.filename))
    return 'Image uploaded successfully'

@app.route('/uploads/speech/<filename>')
def serve_audio(filename):
    # Serve the file from the uploads directory
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

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
    return jsonify({"file_url": f"/uploads/speech/{audio_response}"})


@app.route('/actual_response')
def actual_response():
    modelResponse("I have a headache")

if __name__ == '__main__':
    app.run(debug=True)