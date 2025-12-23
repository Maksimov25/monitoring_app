from ultralytics import YOLO

# Загрузка модели
model = YOLO( "D:/PyCharm 2025.1/PycharmProjects/monitoring_app/best.pt")

# Вывод информации о модели
print(model.info())
print(model.names)  # Список классов
