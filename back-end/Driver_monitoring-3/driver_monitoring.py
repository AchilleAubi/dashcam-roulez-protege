import sys, math
import numpy as np
import cv2
import mediapipe as mp
import pyttsx3

def euclid(p1, p2):
    return np.linalg.norm(np.array(p1) - np.array(p2))

def eye_EAR(landmarks, eye_idx):
    p1, p2, p3, p4, p5, p6 = [landmarks[i] for i in eye_idx]
    num = euclid(p2, p6) + euclid(p3, p5)
    den = 2.0 * euclid(p1, p4)
    return num / den if den > 0 else 0.0

def head_yaw_pitch(landmarks):
    left_outer, left_inner = landmarks[33], landmarks[133]
    right_outer, right_inner = landmarks[362], landmarks[263]
    left_center = ((left_outer[0] + left_inner[0]) / 2, (left_outer[1] + left_inner[1]) / 2)
    right_center = ((right_outer[0] + right_inner[0]) / 2, (right_outer[1] + right_inner[1]) / 2)
    eye_vec = (right_center[0] - left_center[0], right_center[1] - left_center[1])

    yaw = math.degrees(math.atan2(eye_vec[1], eye_vec[0]))

    nose_tip = landmarks[1]
    eyes_mid = ((left_center[0] + right_center[0]) / 2, (left_center[1] + right_center[1]) / 2)
    inter_eye = euclid(left_center, right_center)
    nose_drop = eyes_mid[1] - nose_tip[1]
    pitch = math.degrees(math.atan2(nose_drop, inter_eye + 1e-6))
    return yaw, pitch

def rough_emotion(landmarks, left_ear, right_ear):
    ear = (left_ear + right_ear) / 2.0
    if ear < 0.19:
        return "fatigue"

    brow_left = landmarks[105]
    brow_right = landmarks[334]
    eye_left_center = ((landmarks[33][0]+landmarks[133][0])/2, (landmarks[33][1]+landmarks[133][1])/2)
    eye_right_center = ((landmarks[362][0]+landmarks[263][0])/2, (landmarks[362][1]+landmarks[263][1])/2)
    brow_to_eye = ( (eye_left_center[1]-brow_left[1]) + (eye_right_center[1]-brow_right[1]) ) / 2.0

    mouth_left, mouth_right = landmarks[61], landmarks[291]
    mouth_top, mouth_bottom = landmarks[13], landmarks[14]
    mouth_w = euclid(mouth_left, mouth_right)
    mouth_h = euclid(mouth_top, mouth_bottom)
    mouth_ratio = mouth_h / (mouth_w + 1e-6)

    if brow_to_eye < 6 and mouth_ratio < 0.10:
        return "colère"

    _, pitch = head_yaw_pitch(landmarks)
    if mouth_ratio > 0.15 and pitch > 3:
        return "stress"

    return "neutre"

def detect_from_image(image_bgr):
    mp_face = mp.solutions.face_mesh
    with mp_face.FaceMesh(static_image_mode=True, max_num_faces=1, refine_landmarks=True) as face_mesh:
        rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        res = face_mesh.process(rgb)

        out = {
            "face_found": False,
            "somnolence": False,
            "distraction": False,
            "regard_hors_route": False,
            "telephone": "inconnu (à implémenter via détection d'objets)",
            "emotion": "inconnue",
            "scores": {}
        }

        if not res.multi_face_landmarks:
            return out

        h, w = image_bgr.shape[:2]
        lms = res.multi_face_landmarks[0].landmark
        pts = [(lm.x * w, lm.y * h) for lm in lms]
        out["face_found"] = True

        left_idx  = [33, 160, 158, 133, 153, 144]
        right_idx = [362, 385, 387, 263, 373, 380]

        left_ear = eye_EAR(pts, left_idx)
        right_ear = eye_EAR(pts, right_idx)
        ear = (left_ear + right_ear) / 2.0
        out["scores"]["EAR"] = round(ear, 3)

        SLEEP_EAR_THR = 0.19
        out["somnolence"] = ear < SLEEP_EAR_THR

        yaw, pitch = head_yaw_pitch(pts)
        out["scores"]["yaw_deg"] = round(yaw, 1)
        out["scores"]["pitch_deg"] = round(pitch, 1)

        out["regard_hors_route"] = (abs(yaw) > 15) or (pitch > 10)
        out["distraction"] = out["regard_hors_route"]

        out["emotion"] = rough_emotion(pts, left_ear, right_ear)
        return out

def voice_alert(text):
    try:
        engine = pyttsx3.init()
        engine.say(text)
        engine.runAndWait()
    except Exception as e:
        print(f"[Alerte vocale] Impossible de parler: {e}")

# ------------------ Main ------------------
def main():
    if len(sys.argv) < 2:
        print("Usage: python driver_monitor_image.py <chemin_image>")
        sys.exit(1)

    img_path = sys.argv[1]
    img = cv2.imread(img_path)
    if img is None:
        print(f"Impossible de lire l'image: {img_path}")
        sys.exit(1)

    result = detect_from_image(img)

    print("=== Surveillance du conducteur (image) ===")
    print(f"- Visage détecté : {result['face_found']}")
    print(f"- Somnolence     : {result['somnolence']} (EAR={result['scores'].get('EAR','-')})")
    print(f"- Distraction    : {result['distraction']}")
    print(f"- Regard hors route (yaw/pitch): {result['regard_hors_route']} "
          f"(yaw={result['scores'].get('yaw_deg','-')}°, pitch={result['scores'].get('pitch_deg','-')}°)")
    print(f"- Téléphone      : {result['telephone']}")
    print(f"- Émotion        : {result['emotion']}")

    if result["somnolence"]:
        voice_alert("Attention, somnolence détectée")

if __name__ == "__main__":
    main()
