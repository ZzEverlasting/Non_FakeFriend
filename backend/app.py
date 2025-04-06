from flask import Flask, render_template

# Set the template_folder to point to the frontend/pages directory relative to app.py.
app = Flask(__name__, template_folder='../frontend/pages')

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/app')
def app_page():
    return render_template('app.html')

if __name__ == '__main__':
    app.run(debug=True)