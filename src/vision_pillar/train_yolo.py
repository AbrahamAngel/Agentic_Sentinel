from ultralytics import YOLO


def main():
    model = YOLO("yolo26n.pt")

    model.train(
        data="src/vision_pillar/data_fold5.yaml",
        epochs=50,
        imgsz=640,
        batch=4,
        device=0,
        workers=2,
        project="runs/vision",
        name="dfire_fold5",
        exist_ok=True
    )


if __name__ == "__main__":
    main()