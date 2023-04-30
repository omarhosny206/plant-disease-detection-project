import numpy as np
from PIL import Image
from tensorflow.keras.models import load_model

class_names_map = {"pepper_model": ['Pepper_Bacterial_spot', 'Pepper_healthy'],
                   "tomato_model": ['Tomato_Bacterial_spot', 'Tomato_healthy'],
                   "potato_model": ['Potato___Early_blight', 'Potato___Late_blight', 'Potato___healthy']}

model_path = {"pepper_model": "./models/pepper_model.h5",
              "tomato_model": "./models/tomato_model.h5",
              "potato_model": "./models/potato_model.h5"}


def predict(model: str, image_path: str):
    class_names = class_names_map[model]
    model = load_model(model_path[model])
    input_image = Image.open(image_path)
    input_image = np.array(input_image)
    input_image = input_image.reshape(
        1, input_image.shape[0], input_image.shape[1], input_image.shape[2])
    prediction = model.predict(input_image)
    return class_names[np.argmax(prediction[0])]
