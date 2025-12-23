"""
Утилиты для визуализации
"""
import cv2
import numpy as np
from config import CLASS_COLORS


class Visualizer:
    @staticmethod
    def draw_violation_box(frame, violation):
        """
        Отрисовка рамки нарушения

        Args:
            frame: кадр
            violation: информация о нарушении

        Returns:
            frame: кадр с рамкой
        """
        bbox = violation['bbox'].astype(int)
        class_name = violation['class_name']
        confidence = violation['confidence']

        # Цвет рамки
        color = CLASS_COLORS.get(class_name, (255, 255, 255))

        # Рамка
        cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]),
                      color, 2)

        # Текст
        label = f"{class_name}: {confidence:.2%}"
        (w, h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX,
                                    0.6, 2)
        cv2.rectangle(frame, (bbox[0], bbox[1] - h - 10),
                      (bbox[0] + w, bbox[1]), color, -1)
        cv2.putText(frame, label, (bbox[0], bbox[1] - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        return frame

    @staticmethod
    def add_info_panel(frame, info_text):
        """
        Добавление информационной панели

        Args:
            frame: кадр
            info_text: текст для отображения

        Returns:
            frame: кадр с панелью
        """
        panel_height = 80
        panel = np.zeros((panel_height, frame.shape[1], 3), dtype=np.uint8)
        panel[:] = (50, 50, 50)

        # Текст
        y_offset = 30
        for line in info_text.split('\n'):
            cv2.putText(panel, line, (10, y_offset),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            y_offset += 25

        # Объединение
        result = np.vstack([panel, frame])
        return result
