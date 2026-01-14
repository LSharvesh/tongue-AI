# Tongue-AI 👅🧠

Tongue-AI is a **simple deep learning–based tongue image segmentation project** built using **UNet**. The project takes a tongue image as input and generates a segmented tongue mask through a trained model. A lightweight **Flask web app** is used for uploading images and viewing results.

> ⚠️ This project is for **learning, research, and demonstration purposes only**. It is **not** intended for medical diagnosis.

---

## ✨ What This Project Does

* Upload a tongue image via a web interface
* Run **UNet-based segmentation** on the image
* Generate and save segmented output
* Display original and segmented images

---

## 🧠 Model Used

* **Architecture:** UNet
* **Task:** Image Segmentation
* **Framework:** PyTorch
* **Trained Model:** `model/tongue_unet.pth`

---

## 📂 Project Structure

```
TONGUE-AI/
│── ai/
│   └── image_analyzer.py
│
│── data/
│   ├── dataset/
│   └── groundtruth/
│       ├── images/
│       └── mask/
│
│── model/
│   └── tongue_unet.pth
│
│── segmentation/
│   └── predict.py
│
│── training/
│   ├── dataset_loader.py
│   ├── train.py
│   └── unet.py
│
│── templates/
│   └── index.html
│
│── static/
│   ├── css/style.css
│   └── js/script.js
│
│── uploads/
│   ├── original/
│   └── segmented/
│
│── app.py
│── requirements.txt
│── .gitignore
│── venv/
```

---

## ⚙️ Installation

1. Clone the repository

```bash
git clone https://github.com/your-username/tongue-ai.git
cd tongue-ai
```

2. Create and activate virtual environment

```bash
python -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate
```

3. Install dependencies

```bash
pip install -r requirements.txt
```

---

## ▶️ Run the Application

```bash
python app.py
```

Then open your browser and go to:

```
http://127.0.0.1:5000
```

---

## 🧪 Training (Optional)

If you want to train the model again:

```bash
python training/train.py
```

Make sure your dataset and ground truth masks are placed correctly inside `data/`.

---

## 📥 Input & Output

* **Input:** Tongue image (JPG / PNG)
* **Output:** Segmented tongue image
* Saved in:

  * `uploads/original/`
  * `uploads/segmented/`

---

