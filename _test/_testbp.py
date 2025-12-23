if __name__ == '__main__':
    from ultralytics import YOLO
    import cv2

    model = YOLO("best.pt")
    img = cv2.imread("frame_phone.jpg")

    results = model(img)[0]  # без conf_threshold
    boxes = results.boxes

    print("total boxes:", len(boxes))
    for b in boxes:
        cls = int(b.cls[0])
        conf = float(b.conf[0])
        x1, y1, x2, y2 = map(int, b.xyxy[0])
        print(cls, results.names[cls], f"{conf:.3f}", x1, y1, x2, y2)
