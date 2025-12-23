"""
Класс для распознавания лиц с помощью InsightFace.
"""

import cv2
import numpy as np
from insightface.app import FaceAnalysis
from config import INSIGHTFACE_MODEL


class FaceRecognizer:
    def __init__(self, model_name: str = INSIGHTFACE_MODEL):
        """Инициализация модели распознавания лиц."""
        self.app = FaceAnalysis(name=model_name)
        # ctx_id=0 — использовать GPU, если есть, иначе CPU
        self.app.prepare(ctx_id=0, det_size=(640, 640))
        # Простая in-memory база эмбеддингов
        self.known_faces: dict[str, np.ndarray] = {}

    def detect_faces(self, frame):
        """
        Детекция лиц на кадре.

        Args:
            frame: кадр (numpy.ndarray, BGR).

        Returns:
            list: список объектов Face (InsightFace).
        """
        faces = self.app.get(frame)
        return faces

    def register_face(self, frame, name: str) -> bool:
        """
        Регистрация нового лица в базе.

        Args:
            frame: кадр с лицом (numpy.ndarray, BGR).
            name: имя человека.

        Returns:
            bool: True, если лицо успешно добавлено.
        """
        faces = self.detect_faces(frame)
        if faces:
            embedding = faces[0].embedding  # берём первое найденное лицо
            self.known_faces[name] = embedding
            return True
        return False

    def recognize_face(self, face_embedding: np.ndarray, threshold: float = 0.5):
        """
        Распознавание лица по эмбеддингу.

        Args:
            face_embedding: эмбеддинг лица (np.ndarray).
            threshold: порог расстояния для совпадения.

        Returns:
            str | None: имя распознанного человека или None.
        """
        if not self.known_faces:
            return None

        min_distance = float("inf")
        recognized_name = None

        for name, known_embedding in self.known_faces.items():
            distance = np.linalg.norm(face_embedding - known_embedding)
            if distance < min_distance and distance < threshold:
                min_distance = distance
                recognized_name = name

        return recognized_name

    def draw_faces(self, frame, faces):
        """
        Отрисовка рамок вокруг лиц.

        Args:
            frame: кадр (numpy.ndarray, BGR).
            faces: список объектов Face.

        Returns:
            frame: кадр с нарисованными рамками.
        """
        for face in faces:
            bbox = face.bbox.astype(int)
            cv2.rectangle(
                frame,
                (bbox[0], bbox[1]),
                (bbox[2], bbox[3]),
                (0, 255, 0),
                2,
            )
        return frame
