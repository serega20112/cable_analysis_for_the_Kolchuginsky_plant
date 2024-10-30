import os
import cv2
import numpy as np
from PIL import Image
import io

class DeepLearningModel:
    def __init__(self, model_path):
        self.model = self.load_model(model_path)

    def load_model(self, model_path):
        """Загрузка модели глубокого обучения из файла."""
        try:
            net = cv2.dnn.readNetFromCaffe(model_path, model_path)
            return net
        except Exception as e:
            print(f"Ошибка при загрузке модели: {e}")
            return None

    def apply_model(self, img, threshold_value):
        """Применение модели глубокого обучения к изображению."""
        img_resized = cv2.resize(img, (512, 512))
        blob = cv2.dnn.blobFromImage(img_resized, 1/255, (512, 512), (0, 0, 0), True, crop=False)
        self.model.setInput(blob)
        output = self.model.forward()
        output = output[0]
        output = cv2.resize(output, (img.shape[1], img.shape[0]))
        _, thresh = cv2.threshold(output, threshold_value, 255, cv2.THRESH_BINARY)
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for contour in contours:
            cv2.drawContours(img, [contour], -1, (0, 255, 0), 2)

        return img

class CableAnalyzer:
    def __init__(self, model_path=None, pixels_per_mm=None):
        self.model = DeepLearningModel(model_path) if model_path else None
        self.pixels_per_mm = pixels_per_mm
        self.roi = []

    def analyze_image_bytes(self, image_bytes, known_diameter_mm=None, upscale_factor=1, blur_size=(5, 5), threshold_value=0.5):
        """Анализ изображения, переданного в байтовом формате."""
        try:
            img = self._load_image(image_bytes)
            if upscale_factor > 1:
                img = self._upscale_image(img, upscale_factor)

            # Предобработка изображения
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            blur = cv2.GaussianBlur(gray, blur_size, 0)
            thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]

            # Морфологические операции для улучшения контуров
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
            thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
            thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)

            # Найдем контуры
            contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cable_contours = self._filter_cable_contours(contours)

            cable_diameter_px = self._calculate_cable_diameter(cable_contours)

            if known_diameter_mm and cable_diameter_px > 0:
                self.pixels_per_mm = cable_diameter_px / known_diameter_mm

            cable_diameter_mm = cable_diameter_px / self.pixels_per_mm if self.pixels_per_mm else None

            # Применяем модель глубокого обучения, если она загружена
            if self.model:
                img = self.model.apply_model(img, threshold_value)

            # Рисуем контуры на изображении
            for contour in cable_contours:
                cv2.drawContours(img, [contour], -1, (0, 255, 0), 2)

            return {
                "num_cores": len(cable_contours),
                "diameter_px": cable_diameter_px,
                "diameter_mm": cable_diameter_mm,
                "processed_image": img,
            }
        except Exception as e:
            print(f"Ошибка при анализе изображения: {e}")
            return None

    def _load_image(self, image_bytes):
        """Загрузка изображения из байтового формата."""
        img = Image.open(io.BytesIO(image_bytes))
        return cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)

    def _upscale_image(self, image, factor):
        """Увеличение изображения на заданный коэффициент."""
        height, width = image.shape[:2]
        new_size = (int(width * factor), int(height * factor))
        return cv2.resize(image, new_size, interpolation=cv2.INTER_CUBIC)

    def _calculate_cable_diameter(self, contours):
        """Расчет диаметра кабеля на основе контуров."""
        if contours:
            cable_contour = max(contours, key=cv2.contourArea)
            (x, y), radius = cv2.minEnclosingCircle(cable_contour)
            diameter = 2 * radius
            return diameter
        return 0

    def _filter_cable_contours(self, contours):
        """Фильтрация контуров на основе площади и соотношения сторон."""
        cable_contours = []
        for contour in contours:
            area = cv2.contourArea(contour)
            x, y, w, h = cv2.boundingRect(contour)
            aspect_ratio = float(w) / h
            if area > 100 and aspect_ratio > 2:
                cable_contours.append(contour)
        return cable_contours

    def save_processed_image(self, processed_image, file_path):
        """Сохранение обработанного изображения в файл."""
        cv2.imwrite(file_path, processed_image)

    def show_processed_image(self, processed_image):
        """Отображение обработанного изображения."""
        cv2.imshow("Processed Image", processed_image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()