import os
import numpy as np
from flask import Flask, request, render_template
from tensorflow.keras.models import load_model
from werkzeug.utils import secure_filename
from PIL import Image
import cv2  # Added for grayscale conversion

import logging
logging.getLogger('werkzeug').setLevel(logging.WARNING)

# Suppress TensorFlow warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

app = Flask(__name__, static_folder='static')

# Define the folder to save uploaded images
app.config['UPLOAD_FOLDER'] = r'C:\Users\ADMIN\Desktop\AI WEB APP\upload'

# Define path to your trained model
model_path = r"C:\Users\ADMIN\Desktop\AI WEB APP\model\tumor_classifier_model.h5"
model = load_model(model_path)  # Load the model

# Define the labels corresponding to the model's output classes
labels = ['glioma_tumor', 'meningioma_tumor', 'no_tumor', 'pituitary_tumor']

# Route to render the upload form
@app.route('/')
def index():
    return render_template('index.html')

# Route to handle file upload and processing
@app.route('/predict', methods=['POST'])
def predict():
    # Check if the file is part of the request
    if 'file' not in request.files:
        return 'No file part'
    
    file = request.files['file']
    if file.filename == '':
        return 'No selected file'
    
    if file:
        try:
            # Save the uploaded file
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)

            # Step 1: Open the image as grayscale
            img = cv2.imread(filepath, cv2.IMREAD_GRAYSCALE)  # Read as grayscale (1 channel)
            if img is None:
                return "Failed to load the image. Please try again with a valid image."

            # Step 2: Resize the image
            img = cv2.resize(img, (150, 150))  # Resize to match the model's input size

            # Step 3: Add channel dimension to the image (from shape (150, 150) to (150, 150, 1))
            img = np.expand_dims(img, axis=-1)

            # Step 4: Normalize pixel values to [0, 1]
            img = img / 255.0

            # Step 5: Expand dimensions to match batch input (shape becomes (1, 150, 150, 1))
            img = np.expand_dims(img, axis=0)  # Shape: (1, 150, 150, 1)

            # Predict using the trained model
            prediction = model.predict(img)

            # Debugging: Log prediction probabilities and result
            print("Prediction probabilities:", prediction)

            # Get the predicted class index
            result = np.argmax(prediction, axis=1)  # Index of the max probability
            predicted_label = labels[result[0]]  # Map index to class name
            print("Predicted label:", predicted_label)  # Debugging: Log predicted label

            # Remove the uploaded file after prediction
            os.remove(filepath)

            # Render the result in the `result.html` template
            return render_template('result.html', result=predicted_label)

        except Exception as e:
            return f"Error making prediction: {str(e)}"  # Return the error message if something fails

if __name__ == '__main__':
    app.run(debug=True)
