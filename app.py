from flask import Flask, render_template, request, jsonify
from flask_cors import CORS

app = Flask(__name__)
CORS(app)


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/upload', methods=['POST'])
def upload_video():
    return jsonify({'status': 'success'})


if __name__ == '__main__':
    app.run(debug=True, port=5000)

