import os
from flask import Flask, request, render_template
import numpy as np
from tensorflow import keras
from PIL import Image
import cv2

app = Flask(__name__)

# Load the saved model
model = keras.models.load_model('mask_detection_model.h5')

@app.route('/', methods=['GET', 'POST'])
def index():
    prediction = None
    if request.method == 'POST':
        # Get the uploaded file from the form
        uploaded_file = request.files['file']

        if uploaded_file.filename != '':
            # Save the uploaded file to the static folder
            image_path = os.path.join('static', uploaded_file.filename)
            uploaded_file.save(image_path)

            # Load and preprocess the uploaded image
            image = Image.open(image_path)
            image = image.resize((128, 128))
            image = image.convert('RGB')
            image = np.array(image)
            image = image / 255.0  # Normalize
            image = np.reshape(image, (1, 128, 128, 3))

            # Make the prediction
            prediction = model.predict(image)
            prediction_label = np.argmax(prediction)

            if prediction_label == 1:
                result = 'The person in the image is wearing a mask.'
            else:
                result = 'The person in the image is not wearing a mask.'

            return render_template('index.html', prediction=result, image_path=image_path)

    return render_template('index.html', prediction=None, image_path=None)

if __name__ == '__main__':
    app.run(debug=True)
