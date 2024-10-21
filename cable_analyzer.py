import cv2
import numpy as np
import io
from PIL import Image

class CableAnalyzer:
    def __init__(self, model_path=None, pixels_per_mm=None):
        """Инициализирует анализатор кабеля.

        Args:
            model_path (str, optional): Путь к файлу модели (не используется).
            pixels_per_mm (float, optional): Количество пикселей на миллиметр.
                                              Если None, диаметр будет в пикселях.
        """
        self.model = model_path  # (не используется)
        self.pixels_per_mm = pixels_per_mm

    def analyze_image_bytes(self, image_bytes, known_diameter_mm=None):
        """Анализирует изображение, полученное в виде байтов.

        Args:
            image_bytes (bytes): Байтовое представление изображения.
            known_diameter_mm (float, optional): Известный диаметр кабеля в мм
                                                (для калибровки).

        Returns:
            dict: Информация о кабеле и изображение с выделением жил.
        """
        # Чтение изображения из байтов
        img = Image.open(io.BytesIO(image_bytes))
        img = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)

        # --- Предобработка ---
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (5, 5), 0)
        thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]

        # --- Поиск контуров ---
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # --- Анализ контуров ---
        cable_diameter_px = self._calculate_cable_diameter(contours)

        # --- Калибровка pixels_per_mm, если известен диаметр ---
        if known_diameter_mm and cable_diameter_px > 0:
            self.pixels_per_mm = cable_diameter_px / known_diameter_mm

        # --- Перевод пикселей в мм/см ---
        cable_diameter_mm = cable_diameter_px / self.pixels_per_mm if self.pixels_per_mm else None

        # --- Анализ количества жил ---
        num_cores, core_contours, processed_image = self._detect_cores(thresh, img.copy())

        # --- Дополнительная информация о жилах (опционально) ---
        core_diameters_mm = []
        if self.pixels_per_mm:
            for core_contour in core_contours:
                core_diameter_px = 2 * cv2.minEnclosingCircle(core_contour)[1]
                core_diameters_mm.append(core_diameter_px / self.pixels_per_mm)

        return {
            "num_cores": num_cores,
            "diameter_px": cable_diameter_px,
            "diameter_mm": cable_diameter_mm,
            "core_diameters_mm": core_diameters_mm,
            "processed_image": processed_image,
        }

    def _calculate_cable_diameter(self, contours):
        """Вычисляет диаметр кабеля по контурам.

        Args:
            contours (list): Список контуров.

        Returns:
            float: Диаметр кабеля в пикселях (или 0, если контуры не найдены).
        """
        if contours:
            cable_contour = max(contours, key=cv2.contourArea)
            (x, y), radius = cv2.minEnclosingCircle(cable_contour)
            diameter = 2 * radius
            return diameter
        else:
            return 0

    def _detect_cores(self, thresh_image, original_image):
        """Определяет количество жил на предобработанном изображении.

        Args:
            thresh_image (np.ndarray): Бинаризованное изображение.
            original_image (np.ndarray): Оригинальное изображение для отрисовки.

        Returns:
            tuple: Количество жил, список контуров жил и изображение с выделением.
                   Если жи ры не обнаружены, возвращает (0, [], оригинальное изображение).
        """
        # Дополнительная обработка для выделения жил
        kernel = np.ones((3,3),np.uint8)
        opening = cv2.morphologyEx(thresh_image, cv2.MORPH_OPEN, kernel, iterations=2)

        # Поиск контуров жил
        contours, _ = cv2.findContours(opening, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # --- Улучшенная фильтрация контуров ---
        core_contours = []
        min_core_area = 10  # Минимальная площадь контура жилы (настройте)

        for i, contour in enumerate(core_contours):
            # Находим центр контура для размещения номера
            M = cv2.moments(contour)
            cX = int(M["m10"] / M["m00"]) if M["m00"] != 0 else 0
            cY = int(M["m01"] / M["m00"]) if M["m00"] != 0 else 0

            # Рисуем контур и номер жилы
            cv2.drawContours(original_image, [contour], -1, (0, 255, 0), 2)
            cv2.putText(original_image, str(i+1), (cX, cY), cv2.FONT_HERSHEY_SIMPLEX,
                        0.8, (0, 0, 255), 2)

        return len(core_contours), core_contours, original_image