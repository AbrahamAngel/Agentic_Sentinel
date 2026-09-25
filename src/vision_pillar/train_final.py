from ultralytics import YOLO


def main():
    model = YOLO("yolo26n.pt")

    model.train(
        data="src/vision_pillar/data_final.yaml",
        epochs=50,
        imgsz=640,
        batch=4,
        device=0,
        workers=2,
        project="runs/vision",
        name="dfire_final",
        exist_ok=True
    )


if __name__ == "__main__":
    main()