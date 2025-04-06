from flask import Flask, render_template, jsonify, Response
import time
from threading import Thread
from lsd import lsd, lsd_run

# Set the template_folder to point to the frontend/pages directory relative to app.py.
app = Flask(__name__, template_folder='../frontend/pages')

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/app')
def app_page():
    return render_template('app.html')

@app.route('/video_stream')
def video_feed():
    def generate_video():
        while True:
            frame = lsd.get_frame()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
            time.sleep(1 / 30)
    return Response(generate_video(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/transcript')
def get_transcript():
    return jsonify({"transcript": lsd.transcript})
    
def flask_thread():
    app.run(debug=True, use_reloader=False)

if __name__ == '__main__':
    t = Thread(target=flask_thread)
    t.daemon = True     #external run
    t.start()
    lsd_run()