from flask import Flask, request, jsonify
import subprocess

app = Flask(__name__)

@app.route('/transcribe', methods=['POST'])
def animate():
    # Get the audio file from the request
    audio_file = request.files['audio']
    
    # Save the audio file to a temporary location
    audio_path = 'temp_audio.wav'
    audio_file.save(audio_path)

    # Call the animate.py script with the audio file as an argument
    result = subprocess.run(['python', 'animate.py', audio_path], capture_output=True, text=True)

    # Return the output of the animate.py script
    return jsonify({'output': result.stdout, 'error': result.stderr})

if __name__ == '__main__':
    app.run(debug=True)