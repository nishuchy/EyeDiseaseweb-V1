import os
import numpy as np
from flask import Flask, request, render_template, url_for, send_from_directory
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from werkzeug.utils import secure_filename
from PIL import Image

# Flask app setup
app = Flask(__name__)
UPLOAD_FOLDER = 'uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Load Keras model
model = load_model('densenet201_model_improved.h5')

# Class names (update as per your dataset)
class_names = [
    'Not DR', 'NORMAL', 'R1-MO Mild DR', 'R1-M1-Mild DR',
    'R2-M0 Moderate DR', 'R2-M0 Severe DR', 'R2-M1- Moderate DR',
    'R2-M1-Severe DR', 'R3-M0-PDR', 'R3-M1-PDR'
]

IMG_SIZE = (224, 224)

# Route for serving uploaded images
@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)
# Main route
@app.route('/', methods=['GET', 'POST'])
def index():
    results = []
    if request.method == 'POST':
        files = request.files.getlist('images')
        for file in files:
            if file:
                filename = secure_filename(file.filename)
                filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
                file.save(filepath)

                # Preprocess image
                img = Image.open(filepath).convert('RGB')
                img = img.resize(IMG_SIZE)
                img_array = image.img_to_array(img) / 255.0
                img_array = np.expand_dims(img_array, axis=0)

                # Predict
                result = model.predict(img_array)[0]
                top_indices = np.argsort(result)[::-1][:3]
                predictions = [(class_names[i], round(float(result[i]) * 100, 2)) for i in top_indices]

                results.append({
                    "image": url_for('uploaded_file', filename=filename),
                    "predictions": predictions
                })

    return render_template('index.html', results=results)

# Run the app
if __name__ == '__main__':
    app.run(debug=True)
