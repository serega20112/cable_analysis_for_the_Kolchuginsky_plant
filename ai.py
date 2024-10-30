import os
import numpy as np
import cv2
from skimage.feature import hog
from sklearn.neighbors import NearestNeighbors
import tkinter as tk
from tkinter import filedialog

# Функция для извлечения признаков изображения с использованием HOG
def extract_features(image_path):
    img = cv2.imread(image_path)
    img = cv2.resize(img, (128, 128))  # Изменение размера изображения
    features = hog(img, pixels_per_cell=(16, 16), cells_per_block=(2, 2), visualize=False)
    return features

# Функция для нахождения похожих изображений
def find_similar_images(target_image_path, images_folder, output_folder):
    target_features = extract_features(target_image_path)

    # Извлечение признаков для всех изображений в папке
    features_list = []
    image_paths = []
    for filename in os.listdir(images_folder):
        if filename.endswith('.jpg') or filename.endswith('.png'):
            image_path = os.path.join(images_folder, filename)
            features = extract_features(image_path)
            features_list.append(features)
            image_paths.append(image_path)

    # Поиск похожих изображений
    nbrs = NearestNeighbors(n_neighbors=5, algorithm='ball_tree').fit(features_list)
    distances, indices = nbrs.kneighbors([target_features])

    # Сохранение найденных изображений
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    for index in indices[0]:
        similar_image_path = image_paths[index]
        similar_image = cv2.imread(similar_image_path)
        output_path = os.path.join(output_folder, os.path.basename(similar_image_path))
        cv2.imwrite(output_path, similar_image)

    print(f"Найдено и сохранено {len(indices[0])} похожих изображений в '{output_folder}'.")

# Функция для выбора изображения и запуска поиска
def select_image_and_find():
    target_image_path = filedialog.askopenfilename()
    output_folder = filedialog.askdirectory()
    find_similar_images(target_image_path, 'images_for_educate', output_folder)

# Создание интерфейса
root = tk.Tk()
root.title("Поиск похожих изображений")

button = tk.Button(root, text="Выберите изображение", command=select_image_and_find)
button.pack()

root.mainloop()