from flask import Flask, render_template, request
from werkzeug.utils import secure_filename
import os
import joblib

import pandas as pd
import tensorflow as tf
import numpy as np
import PIL
import os
import matplotlib.pyplot as plt
import joblib

from tensorflow.keras.applications.mobilenet import preprocess_input
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.mobilenet import MobileNet
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler
from PIL import Image
from sklearn.metrics import classification_report,accuracy_score,precision_score
from sklearn.metrics import accuracy_score, confusion_matrix, recall_score
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.utils import shuffle

app = Flask(__name__)

# Папка для загрузки файлов
UPLOAD_FOLDER = 'uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Разрешенные расширения файлов
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}

def my_model(file):
    file = 'uploads/' + file
    path = os.path.join('..', 'data', 'flowers')
    dirs = os.listdir(path)

    # функция для парсинга путей картинок, позволяет преобразовать все картинки
    # в пути и запихнуть в один датафрейм
    def get_photo_names(directory):
        name = []
        for root, dirs, files in os.walk(directory):
            for file in files:
                if file.endswith('.jpg') or file.endswith('.jpeg'):
                    name.append([os.path.join(root, file), root.split('\\')[-1]])
        return name
    photo_names = get_photo_names(path)
    photo_names.append([file, 'unknown'])
    print(photo_names)

    df = pd.DataFrame(photo_names, columns=['photo_name', 'class'])

    # хотим получать вектор признаков для каждого изображения, для
    # этого загрузим уже обученную модель MobileNet
    model = MobileNet(input_shape=(224, 224, 3), include_top=True)

    # с помощью слоя reshape_2, который является вектором из 1000 эл-тов
    # хотим, чтобы слой предсказаний MobileNet принял в себя этот вектор, чтобы мы могли классифицировать
    vector = model.get_layer("reshape_2").output
    feature_extractor = tf.keras.Model(model.input, vector)

    x = []  # массив всех картинок(векторов)

    for image_path in df['photo_name']:
        img = image.load_img(image_path, target_size=(224, 224), color_mode='rgb')
        img_arr = image.img_to_array(img)
        img_arr_b = np.expand_dims(img_arr, axis=0)

        # обработка картинки
        input_img = preprocess_input(img_arr_b)
        # извлечение фичей
        feature_vec = feature_extractor.predict(input_img)
        x.append(feature_vec.ravel())

    dirs = {
        'unknown': 0,
        'daisy': 1,
        'dandelion': 2,
        'rose': 3,
        'sunflower': 4,
        'tulip': 5
    }

    y = df['class'].to_list()
    for flower in range(len(y)):
        y[flower] = dirs[y[flower]]

    X = np.asarray(x, dtype=np.float32)
    Y = np.asarray(y, dtype=np.float32)

    my_pair = (X[-1], Y[-1])

    X = X[:-1]
    Y = Y[:-1]

    for s in range(100):
        X, Y = shuffle(X, Y)

    x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=42)
    x_test, y_test = my_pair

    model_svm_r = SVC(kernel='poly', max_iter=1000)
    model_svm_r.fit(x_train, y_train)
    y_pred = model_svm_r.predict(x_test)

    return y_pred

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return 'No file part'
    file = request.files['file']
    if file.filename == '':
        return 'No selected file'
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)

        return my_model(file.filename)
    else:
        return 'Allowed file types are png, jpg, jpeg, gif'

if __name__ == '__main__':
    app.run(debug=True)