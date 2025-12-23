"""
Класс для работы с YOLO моделью
"""

import cv2
from ultralytics import YOLO
from config import YOLO_MODEL_PATH, VIOLATION_CLASSES


class YOLODetector:
    def __init__(self, model_path: str = YOLO_MODEL_PATH):
        """Инициализация YOLO детектора."""
        self.model = YOLO(model_path)
        self.violation_classes = VIOLATION_CLASSES

    def detect(self, frame, conf_threshold: float = 0.5, iou_threshold: float = 0.45):
        """
        Детекция нарушений на одном кадре.

        Args:
            frame: входной кадр (numpy.ndarray, BGR).
            conf_threshold: порог уверенности.
            iou_threshold: порог IoU для NMS.

        Returns:
            results: объект результатов детекции Ultralytics.
        """
        results = self.model(
            frame,
            conf=conf_threshold,
            iou=iou_threshold,
            verbose=False,
        )
        return results

    def get_violations(self, results):
        """
        Извлечение информации о нарушениях из результата YOLO.

        Returns:
            list[dict]: список словарей с информацией о нарушениях.
        """
        violations = []

        for result in results:
            boxes = result.boxes
            for box in boxes:
                class_id = int(box.cls[0])
                confidence = float(box.conf[0])
                bbox = box.xyxy[0].cpu().numpy()

                violation = {
                    "class_id": class_id,
                    "class_name": self.violation_classes.get(class_id, "unknown"),
                    "confidence": confidence,
                    "bbox": bbox,
                }
                violations.append(violation)

        return violations

    def annotate_frame(self, frame, results):
        """
        Ручная отрисовка bbox и меток с использованием VIOLATION_CLASSES.
        """
        annotated = frame.copy()

        for result in results:
            boxes = result.boxes
            for box in boxes:
                class_id = int(box.cls[0])
                conf = float(box.conf[0])
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)

                class_name = self.violation_classes.get(class_id, "unknown")

                # рисуем прямоугольник
                color = (0, 255, 255)  # можно поменять цвет
                cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)

                # текст: имя класса + уверенность
                label = f"{class_name} {conf:.2f}"
                (tw, th), baseline = cv2.getTextSize(
                    label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2
                )
                # подложка под текст
                cv2.rectangle(
                    annotated,
                    (x1, y1 - th - baseline),
                    (x1 + tw, y1),
                    color,
                    -1,
                )
                cv2.putText(
                    annotated,
                    label,
                    (x1, y1 - baseline),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 0, 0),
                    2,
                )

        return annotated
