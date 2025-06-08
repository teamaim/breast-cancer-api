import os
import numpy as np
import pandas as pd
import gdown  # تحميل الملفات من Google Drive
from PIL import Image
from datetime import datetime
from flask import Flask, request, jsonify
from flask_cors import CORS
from tensorflow.keras.models import load_model

# روابط Google Drive
SIMPLE_MODEL_ID = "1TMv74n2wQt-qaagHFTxc204HUM9bHLb5"
DETAILED_MODEL_ID = "1sVk7loso0ZIwmLIpYP2DkElDPJdjKpZf"
# المسارات المحلية
SIMPLE_MODEL_PATH = "Xception.h5"
DETAILED_MODEL_PATH = "detailed_model.h5"
CSV_PATH = "data/modified_dataset.csv"

# تحميل النماذج إذا لم تكن موجودة
if not os.path.exists(SIMPLE_MODEL_PATH):
    print("⬇️ جاري تحميل نموذج الكشف البسيط...")
    gdown.download(f"https://drive.google.com/uc?id={SIMPLE_MODEL_ID}", SIMPLE_MODEL_PATH, quiet=False)

if not os.path.exists(DETAILED_MODEL_PATH):
    print("⬇️ جاري تحميل نموذج التحليل المفصل...")
    gdown.download(f"https://drive.google.com/uc?id={DETAILED_MODEL_ID}", DETAILED_MODEL_PATH, quiet=False)

# التأكد من وجود ملفات البيانات
if not os.path.exists(SIMPLE_MODEL_PATH):
    raise FileNotFoundError(f"❌ نموذج الكشف البسيط غير موجود في: {SIMPLE_MODEL_PATH}")
if not os.path.exists(DETAILED_MODEL_PATH):
    raise FileNotFoundError(f"❌ نموذج التحليل المفصل غير موجود في: {DETAILED_MODEL_PATH}")
if not os.path.exists(CSV_PATH):
    raise FileNotFoundError(f"❌ ملف CSV غير موجود في: {CSV_PATH}")

# تحميل النماذج وبيانات CSV
simple_model = load_model(SIMPLE_MODEL_PATH)
detailed_model = load_model(DETAILED_MODEL_PATH)
csv_data = pd.read_csv(CSV_PATH)

print("✅ تم تحميل النموذجين وملف CSV بنجاح!")

app = Flask(__name__)
CORS(app)

IMG_SIZE = 100
CATEGORIES = ["Cancer", "Non-Cancer"]

def preprocess_image(img, IMG_SIZE):
    img = img.resize((IMG_SIZE, IMG_SIZE))
    img = np.array(img) / 255.0
    img = np.expand_dims(img, axis=0)
    return img

def decode_prediction(encoded_value, column_name):
    unique_values = csv_data[column_name].unique()
    return str(unique_values[int(encoded_value)])

@app.route("/")
def home():
    return "🚀 Flask API تعمل بنجاح! - استخدم /predict/simple أو /predict/detailed"

@app.route("/predict/simple", methods=["POST"])
def predict_simple():
    if "file" not in request.files:
        return jsonify({"error": "❌ لم يتم رفع أي صورة"}), 400

    file = request.files["file"]
    img = Image.open(file).convert("RGB")
    img = preprocess_image(img, 100)

    prediction = simple_model.predict(img)
    predicted_class = np.argmax(prediction)
    confidence = float(np.max(prediction))

    return jsonify({
        "class": CATEGORIES[predicted_class],
        "confidence": confidence,
        "file": file.filename
    })

@app.route("/predict/detailed", methods=["POST"])
def predict_detailed():
    if "file" not in request.files:
        return jsonify({"error": "❌ لم يتم رفع أي صورة"}), 400

    file = request.files["file"]
    img = Image.open(file).convert("RGB")
    img = preprocess_image(img, 128)

    predictions = detailed_model.predict(img)

    results = {
        "mass_shape": decode_prediction(np.argmax(predictions[0], axis=1)[0], 'mass_shape'),
        "mass_margins": decode_prediction(np.argmax(predictions[1], axis=1)[0], 'mass_margins'),
        "calc_type": decode_prediction(np.argmax(predictions[2], axis=1)[0], 'calc_type'),
        "calc_distribution": decode_prediction(np.argmax(predictions[3], axis=1)[0], 'calc_distribution'),
        "pathology": decode_prediction(np.argmax(predictions[4], axis=1)[0], 'pathology'),
        "breast_density": decode_prediction(np.argmax(predictions[5], axis=1)[0], 'breast_density'),
        "left_or_right_breast": decode_prediction(np.argmax(predictions[6], axis=1)[0], 'left_or_right_breast'),
        "image_view": decode_prediction(np.argmax(predictions[7], axis=1)[0], 'image_view'),
        "abnormality_id": decode_prediction(np.argmax(predictions[8], axis=1)[0], 'abnormality_id'),
        "abnormality_type": decode_prediction(np.argmax(predictions[9], axis=1)[0], 'abnormality_type')
    }

    report = {
        "report_date": datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        "processed_image": file.filename,
        "predictions": results,
        "Notes": "بناءً على النتائج، يُنصح بمراجعة أخصائي لتقييم الحالة بدقة."
    }

    return jsonify(report)

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port, debug=True, use_reloader=False)

