from ultralytics import YOLO


MODEL_PATH = r"runs\detect\runs\vision\dfire_fold1\weights\best.pt"


def predict_image(image_path):
    model = YOLO(MODEL_PATH)

    results = model.predict(
        source=image_path,
        device=0,
        conf=0.25,
        verbose=False
    )

    result = results[0]

    confidence = {
        "smoke": 0.0,
        "fire": 0.0
    }

    for cls, conf in zip(result.boxes.cls, result.boxes.conf):
        class_id = int(cls)
        score = float(conf)

        if class_id == 0:
            confidence["smoke"] = max(confidence["smoke"], score)

        elif class_id == 1:
            confidence["fire"] = max(confidence["fire"], score)

    return confidence


if __name__ == "__main__":
    image = input("Enter image path: ").strip()

    output = predict_image(image)

    print("\nVision Pillar Output")
    print("--------------------")
    print(f"Smoke confidence: {output['smoke']:.4f}")
    print(f"Fire confidence:  {output['fire']:.4f}")
