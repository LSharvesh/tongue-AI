import os
from flask import Flask, render_template, request
from werkzeug.utils import secure_filename

from segmentation.predict import segment_tongue
from ai.image_analyzer import analyze_tongue_image

app = Flask(__name__)

# ---------------- CONFIG ----------------
UPLOAD_ORIGINAL = "uploads/original"
UPLOAD_SEGMENTED = "uploads/segmented"

ALLOWED_EXTENSIONS = {"bmp", "jpg", "jpeg", "png", "webp"}

os.makedirs(UPLOAD_ORIGINAL, exist_ok=True)
os.makedirs(UPLOAD_SEGMENTED, exist_ok=True)
# ----------------------------------------

def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route("/", methods=["GET", "POST"])
def index():
    report = None
    error = None

    if request.method == "POST":
        name = request.form.get("name")
        age = request.form.get("age")
        symptoms = request.form.get("symptoms")
        file = request.files.get("image")

        if not file or file.filename == "":
            error = "No image uploaded"
            return render_template("index.html", report=report, error=error)

        if not allowed_file(file.filename):
            error = "Unsupported file format"
            return render_template("index.html", report=report, error=error)

        filename = secure_filename(file.filename)

        original_path = os.path.join(UPLOAD_ORIGINAL, filename)
        segmented_path = os.path.join(
            UPLOAD_SEGMENTED, f"segmented_{os.path.splitext(filename)[0]}.png"
        )

        # 1️⃣ Save original image
        try:
            file.save(original_path)
            if not os.path.exists(original_path):
                error = "Failed to save uploaded image"
                return render_template("index.html", report=report, error=error)
        except Exception as e:
            error = f"Failed to save image: {str(e)}"
            return render_template("index.html", report=report, error=error)

        try:
            # 2️⃣ Tongue segmentation (YOUR ML MODEL)
            segment_tongue(original_path, segmented_path)

            # 3️⃣ AI analysis (pass both paths so it can use original if segmentation fails)
            ai_result = analyze_tongue_image(segmented_path, original_path)

            # 4️⃣ Final report
            report = f"""
Name: {name}
Age: {age}
Symptoms: {symptoms}

AI Tongue Analysis:
{ai_result}
            """

        except Exception as e:
            import traceback
            error = f"Processing failed: {str(e)}"
            # Print full error to console for debugging
            print(f"ERROR: {error}")
            print(traceback.format_exc())

    return render_template("index.html", report=report, error=error)


if __name__ == "__main__":
    app.run(debug=True)
