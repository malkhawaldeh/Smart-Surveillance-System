from flask import Flask, render_template, request, jsonify
from PIL import Image
import io
import base64
from pipeline import pipeline  # the shared pipeline instance

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')


def pil_image_from_base64(data_url: str):
    header, encoded = data_url.split(",", 1)
    img_bytes = base64.b64decode(encoded)
    return Image.open(io.BytesIO(img_bytes)).convert("RGB")


@app.route('/predict', methods=['POST'])
def predict():
    img = None
    if 'file' in request.files:
        f = request.files['file']
        try:
            img = Image.open(f.stream).convert("RGB")
        except Exception as e:
            return jsonify({'error': 'invalid uploaded image', 'details': str(e)}), 400
    else:
        data = request.get_json(silent=True) or {}
        snapshot = data.get('snapshot')
        if snapshot and snapshot.startswith("data:image"):
            try:
                img = pil_image_from_base64(snapshot)
            except Exception as e:
                return jsonify({'error': 'bad snapshot', 'details': str(e)}), 400

    if img is None:
        return jsonify({'error': 'no input image provided'}), 400

    try:
        result = pipeline.infer(img)
        return jsonify({'result': result})
    except Exception as e:
        return jsonify({'error': 'inference failed', 'details': str(e)}), 500


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5001, debug=True)