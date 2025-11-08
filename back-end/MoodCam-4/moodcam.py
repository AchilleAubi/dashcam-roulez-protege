from deepface import DeepFace
import datetime
import json
import pyttsx3
import random
import cv2
import pandas as pd
from fpdf import FPDF
import shutil
import os
from pathlib import Path
import base64
import mimetypes
from typing import Optional

# 1. Analyse de l’émotion depuis une image
def analyser_emotion(image_path):
    try:
        result = DeepFace.analyze(img_path=image_path, actions=['emotion'], enforce_detection=False)
        return result[0]['dominant_emotion']
    except Exception as e:
        print("Erreur d'analyse :", e)
        return "inconnue"

# 2. Simuler style de conduite
def simuler_conduite():
    acceleration = random.choice(["normale", "forte", "faible"])
    freinage = random.choice(["léger", "brutal", "normal"])
    return acceleration, freinage

# 3. Générer journal émotionnel (+ image Base64)
def generer_journal(emotion, accel, freinage, image_path: Optional[str] = None):
    now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    message = f"Émotion détectée : {emotion}. Accélération : {accel}, Freinage : {freinage}."

    # Encodage Base64 si une image est fournie
    image_base64 = None
    image_name = None
    image_size_bytes = None

    if image_path and os.path.exists(image_path):
        try:
            with open(image_path, "rb") as imgf:
                raw = imgf.read()
                image_size_bytes = len(raw)
                b64 = base64.b64encode(raw).decode("utf-8")
                mime, _ = mimetypes.guess_type(image_path)
                if mime is None:
                    mime = "image/jpeg"  # valeur par défaut
                image_base64 = f"data:{mime};base64,{b64}"
                image_name = os.path.basename(image_path)
        except Exception as e:
            print("⚠️ Impossible d’encoder l’image en Base64 :", e)

    journal_entry = {
        "timestamp": now,
        "emotion": emotion,
        "acceleration": accel,
        "freinage": freinage,
        "message": message,
        "image_base64": image_base64,       # data URL directement utilisable dans <img src="..."/>
        "image_name": image_name,
        "image_size_bytes": image_size_bytes
    }

    # Lire l'existant ou démarrer une nouvelle liste
    try:
        with open("journal_emotionnel.json", "r", encoding="utf-8") as f:
            data = json.load(f)
            if not isinstance(data, list):
                data = []
    except (json.JSONDecodeError, FileNotFoundError):
        data = []

    # Ajouter la nouvelle entrée
    data.append(journal_entry)

    # (Optionnel) conserver seulement les N dernières entrées pour éviter un fichier trop lourd
    N_MAX = 500
    if len(data) > N_MAX:
        data = data[-N_MAX:]

    # Réécrire proprement
    with open("journal_emotionnel.json", "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=4)

    return message

# 4. Lecture vocale
def parler(message):
    try:
        engine = pyttsx3.init()
        engine.say(message)
        engine.runAndWait()
    except Exception as e:
        print("Erreur vocale :", e)

# 5. Capture image depuis la webcam
def capturer_image():
    cam = cv2.VideoCapture(0)
    if not cam.isOpened():
        print("❌ Webcam non accessible")
        return None

    print("📸 Capture en cours... Regarde la caméra...")
    ret, frame = cam.read()
    cam.release()

    if ret:
        img_path = "capture.jpg"
        cv2.imwrite(img_path, frame)
        print("✅ Image capturée :", img_path)
        return img_path
    else:
        print("❌ Échec de capture")
        return None

# 6. Export 
def exporter_csv_et_pdf():
    try:
        with open("journal_emotionnel.json", "r", encoding="utf-8") as f:
            data = json.load(f)  # liste d'objets

        # Export CSV (on exclut l'image Base64 pour garder un CSV léger)
        df = pd.DataFrame([{k: v for k, v in d.items() if k != "image_base64"} for d in data])
        df.to_csv("journal_emotionnel.csv", index=False)
        print("✅ Export CSV : journal_emotionnel.csv")

        # Export PDF (texte seulement pour l’instant)
        pdf = FPDF()
        pdf.add_page()
        pdf.set_font("Arial", size=10)
        pdf.cell(200, 10, txt="Journal Émotionnel", ln=True, align="C")
        pdf.ln(5)

        for entry in data:
            pdf.multi_cell(
                0, 10,
                txt=(
                    f"{entry.get('timestamp','')} | Émotion: {entry.get('emotion','')} | "
                    f"Accél.: {entry.get('acceleration','')} | Freinage: {entry.get('freinage','')}\n"
                    f">> {entry.get('message','')}"
                ),
                border=0
            )
            pdf.ln(2)

        pdf.output("journal_emotionnel.pdf")
        print("✅ Export PDF : journal_emotionnel.pdf")

    except Exception as e:
        print("❌ Erreur export CSV/PDF :", e)
    
# 7. Copie auto vers React (../../front-end/public/journal_emotionnel.json)
def copier_json_vers_react():
    try:
        base_dir = Path(__file__).resolve().parent
        src = base_dir / "journal_emotionnel.json"
        dest_dir = (base_dir / ".." / ".." / "front-end" / "public").resolve()
        dest = dest_dir / "journal_emotionnel.json"

        if not src.exists():
            raise FileNotFoundError(f"Fichier source introuvable : {src}")

        dest_dir.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(src, dest)

        print("✅ journal_emotionnel.json copié dans React ✔️")
        return True
    except Exception as e:
        print("❌ Erreur lors de la copie vers React :", e)
        return False
        
# === Programme principal ===
if __name__ == "__main__":
    image_path = capturer_image()
    if image_path:
        emotion = analyser_emotion(image_path)
        accel, freinage = simuler_conduite()
        message = generer_journal(emotion, accel, freinage, image_path=image_path)
        print("📝", message)
        parler(message)
        exporter_csv_et_pdf()
        copier_json_vers_react()
