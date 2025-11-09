# moodcam.py
# -*- coding: utf-8 -*-

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
from typing import Optional, Tuple, List

# =========================
# 1) ANALYSE D'ÉMOTION
# =========================
def analyser_emotion(image_path: str) -> str:
    """
    Analyse l'émotion dominante à partir d'une image (DeepFace).
    Retourne une chaîne en anglais (happy, sad, angry, fear, disgust, neutral...) ou "inconnue".
    """
    try:
        result = DeepFace.analyze(img_path=image_path, actions=['emotion'], enforce_detection=False)
        # DeepFace >= 0.0.79 renvoie une liste
        dominant = result[0]['dominant_emotion'] if isinstance(result, list) else result['dominant_emotion']
        return dominant or "inconnue"
    except Exception as e:
        print("Erreur d'analyse :", e)
        return "inconnue"


# =========================
# 2) STYLE DE CONDUITE (SIMULÉ)
# =========================
def simuler_conduite() -> Tuple[str, str]:
    """
    Retourne (accélération, freinage) ∈ {normale/forte/faible} × {léger/brutal/normal}
    """
    acceleration = random.choice(["normale", "forte", "faible"])
    freinage = random.choice(["léger", "brutal", "normal"])
    return acceleration, freinage


# =========================
# 3) MOTEUR DE RÈGLES : RISQUE, CONSEILS, ALERTE
# =========================
def evaluer_risque(emotion: str, accel: str, freinage: str) -> Tuple[str, List[str]]:
    """
    Retourne (niveau_risque, motifs) où niveau_risque ∈ {"faible","modéré","élevé"}.
    Heuristiques explicites et faciles à enrichir.
    """
    e = (emotion or "").lower()
    a = (accel or "").lower()
    f = (freinage or "").lower()

    motifs: List[str] = []

    # profils émotionnels
    if e in {"angry", "fear", "disgust"}:
        motifs.append("émotion négative")
    elif e in {"sad"}:
        motifs.append("abattement/fatigue possible")
    elif e in {"inconnue"}:
        motifs.append("émotion non déterminée")

    # style de conduite
    if a == "forte":
        motifs.append("accélérations fortes")
    if f == "brutal":
        motifs.append("freinages brusques")

    # Agrégation simple
    score = 0
    for m in motifs:
        if m in {"accélérations fortes", "freinages brusques"}:
            score += 2
        elif m in {"émotion négative"}:
            score += 2
        elif m in {"abattement/fatigue possible", "émotion non déterminée"}:
            score += 1

    if score >= 4:
        niveau = "élevé"
    elif score >= 2:
        niveau = "modéré"
    else:
        niveau = "faible"

    return niveau, motifs


def generer_conseils(emotion: str, accel: str, freinage: str, niveau_risque: str) -> List[str]:
    """
    Génère une liste priorisée de conseils (max 6).
    """
    e = (emotion or "").lower()
    a = (accel or "").lower()
    f = (freinage or "").lower()

    conseils: List[str] = []

    # Sécurité immédiate
    if niveau_risque == "élevé":
        conseils.append("Levez le pied et augmentez la distance de sécurité.")
        conseils.append("Si vous vous sentez tendu, changez de voie prudemment et stabilisez la vitesse.")
    if a == "forte":
        conseils.append("Accélérez plus progressivement pour réduire le stress et la consommation.")
    if f == "brutal":
        conseils.append("Anticipez davantage pour éviter les freinages brusques.")

    # Emotionnel
    if e in {"angry", "disgust"}:
        conseils.append("Respirez 4 s, relâchez 6 s (×5).")
        conseils.append("Mettez une playlist douce ou un podcast calme.")
    if e in {"fear"}:
        conseils.append("Gardez une vitesse stable, évitez les dépassements non nécessaires.")
    if e in {"sad"}:
        conseils.append("Faites une courte pause hydratation/étirements dès que possible.")
        conseils.append("Mettez une musique relaxante à faible volume.")
    if e in {"happy"}:
        conseils.append("Restez vigilant : l’euphorie peut réduire l’anticipation.")
    if e in {"inconnue"}:
        conseils.append("Surveillez vos signaux corporels ; adaptez la vitesse si vous vous sentez distrait.")

    # Génériques
    conseils.append("Balayez la route régulièrement (rétros toutes les 5–8 s).")
    conseils.append("Vérifiez la posture : épaules détendues, mains à 9h15, respiration calme.")

    # Dé-duplication + limite
    seen, out = set(), []
    for c in conseils:
        if c not in seen:
            seen.add(c)
            out.append(c)
    return out[:6]


def construire_alerte_vocale(emotion: str, accel: str, freinage: str, niveau_risque: str) -> Optional[str]:
    """
    Détecte somnolence probable et propose une alerte vocale.
    """
    e = (emotion or "").lower()
    a = (accel or "").lower()
    f = (freinage or "").lower()

    somnolence_probable = (e in {"sad"} and a in {"faible", "normale"} and f in {"léger", "normal"}) \
                          or (e in {"inconnue"} and a == "faible" and f == "léger")

    if somnolence_probable or niveau_risque == "élevé":
        return 'Attention, somnolence détectée. Faites une pause dans un endroit sécurisé.'
    return None


# =========================
# 4) JOURNALISATION (+ IMAGE EN BASE64)
# =========================
def generer_journal(emotion: str, accel: str, freinage: str, image_path: Optional[str] = None):
    """
    Crée une entrée dans journal_emotionnel.json et retourne (message, conseils, alerte).
    """
    now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    niveau_risque, motifs = evaluer_risque(emotion, accel, freinage)
    conseils = generer_conseils(emotion, accel, freinage, niveau_risque)
    alerte = construire_alerte_vocale(emotion, accel, freinage, niveau_risque)

    # Message court (console/voix)
    message = (
        f"Émotion détectée : {emotion}. Accélération : {accel}, Freinage : {freinage}. "
        f"Risque {niveau_risque}."
    )
    if alerte:
        message += " " + alerte

    # Encodage Base64 si image
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
                    mime = "image/jpeg"
                image_base64 = f"data:{mime};base64,{b64}"
                image_name = os.path.basename(image_path)
        except Exception as e:
            print("⚠️ Impossible d’encoder l’image en Base64 :", e)

    journal_entry = {
        "timestamp": now,
        "emotion": emotion,
        "acceleration": accel,
        "freinage": freinage,
        "niveau_risque": niveau_risque,
        "motifs_risque": motifs,
        "conseils": conseils,
        "alerte_vocale": alerte,       # ex: “Attention, somnolence détectée…”
        "message": message,            # phrase synthèse
        "image_base64": image_base64,  # data URL pour <img src="..."/>
        "image_name": image_name,
        "image_size_bytes": image_size_bytes
    }

    # Lire l'existant ou démarrer
    try:
        with open("journal_emotionnel.json", "r", encoding="utf-8") as f:
            data = json.load(f)
            if not isinstance(data, list):
                data = []
    except (json.JSONDecodeError, FileNotFoundError):
        data = []

    # Ajouter + tronquer
    data.append(journal_entry)
    N_MAX = 500
    if len(data) > N_MAX:
        data = data[-N_MAX:]

    with open("journal_emotionnel.json", "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=4)

    return message, conseils, alerte


# =========================
# 5) LECTURE VOCALE (robuste)
# =========================
import atexit
import subprocess

_ENGINE = None

def _init_tts():
    global _ENGINE
    if _ENGINE is None:
        try:
            # Sur Linux : espeak / espeak-ng
            _ENGINE = pyttsx3.init(driverName='espeak')
            # Réglages doux et intelligibles
            _ENGINE.setProperty('rate', 165)   # vitesse
            _ENGINE.setProperty('volume', 1.0) # volume 0.0–1.0
            # Optionnel : choisir une voix FR si dispo
            # for v in _ENGINE.getProperty('voices'):
            #     if 'fr' in v.languages or 'fr_' in getattr(v, 'id', ''):
            #         _ENGINE.setProperty('voice', v.id); break
        except Exception as e:
            print("⚠️ pyttsx3 init a échoué :", e)
            _ENGINE = None

def _shutdown_tts():
    global _ENGINE
    try:
        if _ENGINE is not None:
            _ENGINE.stop()
    except Exception:
        pass
    _ENGINE = None

atexit.register(_shutdown_tts)

def parler_queue(texts):
    """
    Parle plusieurs phrases en une seule session runAndWait().
    Fallback : espeak CLI si pyttsx3 échoue.
    """
    if isinstance(texts, str):
        texts = [texts]

    _init_tts()
    if _ENGINE is not None:
        try:
            for t in texts:
                if t:
                    _ENGINE.say(t)
            _ENGINE.runAndWait()
            return
        except Exception as e:
            print("⚠️ pyttsx3 a échoué en cours de lecture :", e)

    # --- Fallback CLI (espeak) pour éviter tout segfault pyttsx3 ---
    for t in texts:
        try:
            # -s = vitesse ; ajuste au besoin (165 ≈ medium)
            subprocess.run(["espeak", "-s", "165", t], check=False)
        except Exception as e2:
            print("⚠️ Fallback espeak a échoué :", e2)


# =========================
# 6) CAPTURE WEBCAM
# =========================
def capturer_image() -> Optional[str]:
    """
    Capture une image via la webcam (index 0).
    """
    cam = cv2.VideoCapture(0)
    if not cam.isOpened():
        print("❌ Webcam non accessible")
        return None

    print("📸 Capture en cours... Regardez la caméra...")
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


# =========================
# 7) EXPORT CSV + PDF
# =========================
def exporter_csv_et_pdf():
    """
    Exporte le journal en CSV (sans image Base64) et en PDF (texte).
    """
    try:
        with open("journal_emotionnel.json", "r", encoding="utf-8") as f:
            data = json.load(f)  # liste d'objets

        # Aplatir pour CSV
        flat = []
        for d in data:
            dd = {k: v for k, v in d.items() if k not in {"image_base64", "conseils"}}
            dd["conseils"] = " | ".join(d.get("conseils", []))
            flat.append(dd)

        df = pd.DataFrame(flat)
        df.to_csv("journal_emotionnel.csv", index=False)
        print("✅ Export CSV : journal_emotionnel.csv")

        # PDF
        pdf = FPDF()
        pdf.add_page()
        pdf.set_font("Arial", size=10)
        pdf.cell(200, 10, txt="Journal Émotionnel", ln=True, align="C")
        pdf.ln(5)

        for entry in data:
            conseils_txt = " ; ".join(entry.get("conseils", [])) or "—"
            ligne = (
                f"{entry.get('timestamp','')}  |  Émotion: {entry.get('emotion','')}  |  "
                f"Accél.: {entry.get('acceleration','')}  |  Freinage: {entry.get('freinage','')}  |  "
                f"Risque: {entry.get('niveau_risque','')}\n"
                f">> {entry.get('message','')}\n"
                f"Conseils: {conseils_txt}"
            )
            pdf.multi_cell(0, 6, txt=ligne, border=0)
            pdf.ln(1)

        pdf.output("journal_emotionnel.pdf")
        print("✅ Export PDF : journal_emotionnel.pdf")

    except Exception as e:
        print("❌ Erreur export CSV/PDF :", e)


# =========================
# 8) COPIE AUTO VERS REACT
# =========================
def copier_json_vers_react() -> bool:
    """
    Copie journal_emotionnel.json vers ../../front-end/public/journal_emotionnel.json
    (adaptable selon ton arborescence).
    """
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


# =========================
# 9) MAIN
# =========================
if __name__ == "__main__":
    image_path = capturer_image()
    if image_path:
        emotion = analyser_emotion(image_path)
        accel, freinage = simuler_conduite()

        message, conseils, alerte = generer_journal(emotion, accel, freinage, image_path=image_path)

        # Console
        print("📝", message)
        if conseils:
            print("🧭 Conseils :")
            for c in conseils:
                print("   -", c)

        to_say = [message]
        if alerte:
            to_say.append(alerte)
        if conseils:
            to_say.append("Conseil : " + conseils[0])
        parler_queue(to_say)

        # Exports + copie front
        exporter_csv_et_pdf()
        copier_json_vers_react()
    else:
        print("ℹ️ Aucune image capturée : analyse annulée.")
