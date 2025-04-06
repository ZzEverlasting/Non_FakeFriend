from flask import Flask, request, jsonify
import subprocess

app = Flask(__name__)
@app.route('/transcribe', methods=['POST'])

def transcribe():

    audio_file = request.files['audio']
    audio_path = 'temp_audio.wav'
    audio_file.save(audio_path)

    subprocess.Popen(['python', 'transcribe_log.py', audio_path], shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    return jsonify({'message': 'Transcription started'})
if __name__ == '__main__':
    app.run(debug=True)