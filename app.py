import base64
import io

import numpy as np
from PIL import Image
from flask import Flask
from flask import jsonify
from flask import request
from keras.models import load_model
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array

app = Flask(__name__)

def get_model():
    global model
    model = load_model('BMS_classification.h5')
    print(" * Model loaded!")

def preprocess_image(image, target_size):
    if image.mode != "RGB":
        image = image.convert("RGB")
    image = image.resize(target_size)
    image = img_to_array(image)
    image = np.expand_dims(image, axis=0)
    return image
print(" * Loading Keras model...")
get_model()

@app.route("/predict", methods=["GET","POST"])
def predict():
    message = request.get_json(force=True)
    encoded = message['image']
    decoded = base64.b64decode(encoded)
    image = Image.open(io.BytesIO(decoded))
    processed_image = preprocess_image(image, target_size=(224, 224))

    prediction = model.predict(processed_image).tolist()
    print(prediction)

    response = {
        'prediction': {
            'baby_sleeping': prediction[0][0],
            'baby_not_sleeping': prediction[0][1]
        }
    }
    return jsonify(response)


if __name__ == '__main__':
    app.run(debug=True)