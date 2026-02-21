import os
import numpy as np
from flask import Flask, render_template, request
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.applications.vgg19 import preprocess_input

app = Flask(__name__)

# ================= UPLOAD FOLDER =================
UPLOAD_FOLDER = "static/uploads"
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Create uploads folder automatically if not exists
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

# ================= LOAD MODEL =================
model = load_model("dogbreed.h5")

# ================= LOAD CLASS NAMES =================
CLASS_FOLDER = "Training/content/train_data"
class_names = sorted(os.listdir(CLASS_FOLDER))


# ================= HOME =================
@app.route('/')
def index():
    return render_template("index.html")


# ================= PREDICT =================
@app.route('/predict', methods=['POST'])
def predict():

    file = request.files['file']

    if file.filename == '':
        return render_template("index.html")

    filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
    file.save(filepath)

    return render_template("predict.html", image_name=file.filename)


# ================= OUTPUT =================
@app.route('/output', methods=['POST'])
def output():

    image_name = request.form['image_name']

    image_path = os.path.join(app.config['UPLOAD_FOLDER'], image_name)

    img = load_img(image_path, target_size=(224, 224))
    img_array = img_to_array(img)
    img_array = preprocess_input(img_array)
    img_array = np.expand_dims(img_array, axis=0)

    prediction = model.predict(img_array)
    predicted_index = np.argmax(prediction)
    predicted_class = class_names[predicted_index]

    return render_template("output.html",
                           prediction=predicted_class,
                           image_name=image_name)


if __name__ == "__main__":
    app.run(debug=True)
