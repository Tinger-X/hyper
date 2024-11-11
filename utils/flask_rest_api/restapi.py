import io
from torch import hub
from PIL import Image
from flask import Flask, request

app = Flask(__name__)
model = hub.load(
    "../../yolov5",
    "custom",
    path="../../yolo_weight/yolov5s",
    source="local",
    force_reload=True  # force_reload to recache
)


@app.route("/", methods=["POST"])
def predict():
    if request.files.get("image"):
        image_file = request.files["image"]
        image_bytes = image_file.read()

        img = Image.open(io.BytesIO(image_bytes))

        results = model(img, size=640)  # reduce size=320 for faster inference
        print(results)
        return results.pandas().xyxy[0].to_json(orient="records")


if __name__ == "__main__":
    app.run(host="0.0.0.0")
