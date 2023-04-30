import os
from flask import Flask, request, render_template

from prediction import predict

app = Flask(__name__)
app.template_folder = '.'
uploads_directory = os.path.join('static', '.')
app.config['UPLOAD_FOLDER'] = uploads_directory

model_name_map = {"potato": "potato_model",
                  "tomato": "tomato_model",
                  "pepper": "pepper_model", }


@app.route('/')
def home():
    return render_template('./ui/index.html')


@app.route('/api/predictions', methods=['POST'])
def get_prediction():
    delete_all_photos("./static")
    model = model_name_map[request.form['model']]
    image = request.files['image']
    image_path = os.path.join(os.getcwd(), "static", image.filename)
    image.save(image_path)
    prediction = predict(model, image_path)
    return render_template('./ui/show_prediction.html', prediction=prediction, image_name=image.filename)


def delete_all_photos(path):
    for file_name in os.listdir(path):
        if file_name.endswith(".JPG"):
            file_path = os.path.join(path, file_name)
            os.remove(file_path)


if __name__ == '__main__':
    app.run(debug=True)
