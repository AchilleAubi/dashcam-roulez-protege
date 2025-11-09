"""
Commande:
    python server_back.py                      # Mode webcam (défaut)
    python server_back.py --tracker deepsort   # Avec DeepSORT
    python server_back.py --video test.mp4     # Mode vidéo
"""

import threading
import time
import os
import cv2
import shutil
import signal
import argparse
from flask import Flask, Response, jsonify, request
from datetime import datetime
from pydrive.auth import GoogleAuth
from pydrive.drive import GoogleDrive
from flask_cors import CORS

import detection_behavior as dashcam

app = Flask(__name__)
CORS(app, origins=["http://localhost:3000", "http://127.0.0.1:3000"])
recording = True
drive = None

latest_frame = None
alerts_buffer = []

frame_buffer = []
BUFFER_DURATION = 30  
MAX_BUFFER_SIZE = 900  
CREDENTIALS_FILE = "mycreds.txt"

# def init_drive_old():
#     try:
#         gauth = GoogleAuth()
#         gauth.LocalWebserverAuth()
#         return GoogleDrive(gauth)
#     except Exception as e:
#         print(f"Impossible d'initialiser Google Drive : {e}")
#         return None
    

def init_drive():
    try:
        gauth = GoogleAuth()

        if os.path.exists(CREDENTIALS_FILE):
            gauth.LoadCredentialsFile(CREDENTIALS_FILE)
            print("Credentials chargés depuis mycreds.txt")

        if gauth.credentials is None:
            print("Première connexion requise — ouverture du navigateur...")
            gauth.LocalWebserverAuth()
        elif gauth.access_token_expired:
            print("♻️ Token expiré, rafraîchissement...")
            gauth.Refresh()
        else:
            gauth.Authorize()

        gauth.SaveCredentialsFile(CREDENTIALS_FILE)
        print("Credentials sauvegardés, plus besoin de se reconnecter la prochaine fois")

        return GoogleDrive(gauth)

    except Exception as e:
        print(f"Erreur initialisation Google Drive : {e}")
        import traceback; traceback.print_exc()
        return None

def backup_to_drive():
    global drive
    try:
        if not os.path.exists('output') or not os.listdir('output'):
            print("⚠️ Aucun fichier à sauvegarder")
            return

        filename = f"backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}.zip"
        shutil.make_archive("backup_temp", "zip", "output")
        print(f"Archive locale créée : backup_temp.zip")

        if not drive:
            drive = init_drive()

        if drive:
            f = drive.CreateFile({'title': filename})
            f.SetContentFile("backup_temp.zip")
            f.Upload()
            print(f"Sauvegarde Drive : {filename}")
        else:
            print("Google Drive non initialisé, upload ignoré")

    except Exception as e:
        print(f"Erreur sauvegarde : {e}")



def detection_loop(tracker_type='simple', use_webcam=True, webcam_id=0, video_path=None):

    global latest_frame, alerts_buffer, recording
    
    print(f"\n{'='*60}")
    print(f" Démarrage de la détection")
    print(f"{'='*60}")
    print(f"Tracker: {tracker_type.upper()}")
    print(f"Source: {'WEBCAM #' + str(webcam_id) if use_webcam else video_path}")
    print(f"{'='*60}\n")
    
    if use_webcam:
        cap = cv2.VideoCapture(webcam_id)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        cap.set(cv2.CAP_PROP_FPS, 30)
    else:
        cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        print("Impossible d'ouvrir la source vidéo")
        recording = False
        return
    
    try:
        yolo = dashcam.YOLO(dashcam.YOLO_MODEL)
        # depth_model = dashcam.MiDaSDepth(device=dashcam.DEVICE)
        depth_model = dashcam.SimplifiedDepth()
        tracker = dashcam.create_tracker(tracker_type)
        print("Modèles chargés avec succès\n")
    except Exception as e:
        print(f"Erreur chargement modèles : {e}")
        recording = False
        cap.release()
        return
    
    prev_gray = None
    lx, rx, lane_type, M, Minv = 0, 0, 'unknown', None, None
    frame_count = 0
    start_time = time.time()
    
    print("Début de la détection en temps réel...\n")
    
    while recording:
        ret, frame = cap.read()
        if not ret:
            if use_webcam:
                print("Perte du flux caméra")
                time.sleep(0.1)
                continue
            else:
                print("Fin de la vidéo")
                break
        
        try:
            if prev_gray is None:
                prev_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                lx, rx, lane_type, M, Minv = dashcam.detect_lane_lines_and_type(frame)
                continue
            
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            results = yolo(frame, verbose=False)[0]
            detections = []
            
            for r in results.boxes:
                cls = int(r.cls[0])
                conf = float(r.conf[0])
                
                if conf < 0.3:
                    continue
                
                x1, y1, x2, y2 = map(int, r.xyxy[0])
                detections.append({
                    'box': [x1, y1, x2, y2],
                    'class': cls,
                    'conf': conf
                })
            
            if tracker.tracker_type in ['deepsort', 'bytetrack']:
                tracks = tracker.update(detections, frame=frame)
            else:
                tracks = tracker.update(detections)
            
            depth_map = depth_model.predict(frame)
            
            flow_vec = dashcam.estimate_global_optical_flow(prev_gray, gray)
            prev_gray = gray
            
            alerts = dashcam.analyze_tracks_and_log(
                tracks, depth_map, frame.shape, M, lane_type,
                flow_vec, time.time(), frame, use_logging=True, depth_model=None
            )
            
            vis_frame = frame.copy()
            h, w = vis_frame.shape[:2]
            
            for tid, info in tracks.items():
                if len(info['boxes']) == 0:
                    continue
                
                box = info['boxes'][-1]
                x1, y1, x2, y2 = map(int, box)
                cls = info['classes'][-1] if info['classes'] else -1
                
                if cls == dashcam.PERSON_CLASS:
                    color = (0, 165, 255)  
                elif cls in dashcam.VEHICLE_CLASSES:
                    color = (0, 255, 0)  
                else:
                    color = (128, 128, 128)  
                
                cv2.rectangle(vis_frame, (x1, y1), (x2, y2), color, 2)
                
                label = f'ID{tid}'
                if cls == dashcam.PERSON_CLASS:
                    label += ' (Piéton)'
                elif cls in dashcam.VEHICLE_CLASSES:
                    label += f' (Véhicule)'
                
                cv2.putText(vis_frame, label, (x1, y1 - 6),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            
            cv2.line(vis_frame, (int(lx), int(h*0.6)), (int(lx), h), (255, 0, 0), 2)
            cv2.line(vis_frame, (int(rx), int(h*0.6)), (int(rx), h), (255, 0, 0), 2)
            
            info_y = 30
            cv2.putText(vis_frame, f'Tracker: {tracker.tracker_type.upper()}',
                       (10, info_y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
            info_y += 30
            cv2.putText(vis_frame, f'Lane: {lane_type}',
                       (10, info_y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
            info_y += 30
            cv2.putText(vis_frame, f'Tracks: {len(tracks)}',
                       (10, info_y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
            
            for i, (tid, evtype, msg, box) in enumerate(alerts):
                x1, y1, x2, y2 = map(int, box)
                cv2.rectangle(vis_frame, (x1, y1), (x2, y2), (0, 0, 255), 3)
                cv2.putText(vis_frame, msg, 
                           (max(10, x1), max(30, y1 - 10)),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
            
            _, buffer = cv2.imencode('.jpg', vis_frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
            latest_frame = buffer.tobytes()
            
            for tid, evtype, msg, box in alerts:
                alerts_buffer.append({
                    "type": evtype,
                    "message": msg,
                    "time": datetime.now().strftime("%H:%M:%S"),
                    "track_id": tid
                })
            
            if len(alerts_buffer) > 50:
                alerts_buffer = alerts_buffer[-50:]
            
            frame_count += 1
            
            if frame_count % 100 == 0:
                elapsed = time.time() - start_time
                fps = frame_count / elapsed if elapsed > 0 else 0
                print(f"Frames: {frame_count} | FPS: {fps:.1f} | "
                      f"Tracks: {len(tracks)} | Alertes: {len(alerts_buffer)}")
            
        except Exception as e:
            print(f"Erreur traitement frame : {e}")
            import traceback
            traceback.print_exc()
            continue
        
        time.sleep(0.001)  
    
    cap.release()
    backup_to_drive()
    print("\nDétection arrêtée proprement")
    print(f"Total frames traitées : {frame_count}")


@app.route('/')
def index():
    return """
    <html>
        <head><title>Dashcam IA</title></head>
        <body style="background: #1a1a1a; color: white; font-family: Arial;">
            <h1>🚗 Dashcam Intelligence Artificielle</h1>
            <h2>Flux vidéo en direct :</h2>
            <img src="/video" width="1280">
            <h2>API Endpoints :</h2>
            <ul>
                <li><a href="/video" style="color: #4CAF50;">/video</a> - Flux vidéo MJPEG</li>
                <li><a href="/alerts" style="color: #4CAF50;">/alerts</a> - Alertes JSON</li>
                <li><a href="/stats" style="color: #4CAF50;">/stats</a> - Statistiques</li>
                <li>POST /stop - Arrêter le système</li>
            </ul>
        </body>
    </html>
    """

@app.route('/video')
def video_feed():
    def generate():
        global latest_frame
        while recording:
            if latest_frame is not None:
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + 
                       latest_frame + b'\r\n')
            time.sleep(0.03)
    
    return Response(generate(), 
                   mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/alerts')
def alerts():
    return jsonify(alerts_buffer[-15:] if alerts_buffer else [])

@app.route('/stats')
def stats():
    return jsonify({
        "recording": recording,
        "alerts_count": len(alerts_buffer),
        "has_frame": latest_frame is not None,
        "timestamp": datetime.now().isoformat()
    })

@app.route('/stop', methods=['POST'])
def stop_system():
    global recording
    recording = False
    print("Arrêt demandé via API")
    
    backup_to_drive()
    
    def shutdown():
        time.sleep(1)
        os.kill(os.getpid(), signal.SIGTERM)
    
    threading.Thread(target=shutdown, daemon=True).start()
    
    return jsonify({
        "status": "stopped",
        "message": "Système arrêté, sauvegarde effectuée"
    })

def main():
    parser = argparse.ArgumentParser(
        description='Système de dashcam intelligent avec détection IA',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument('--tracker', default='simple',
                       choices=['simple', 'deepsort', 'bytetrack'],
                       help='Type de tracker (défaut: simple)')
    
    parser.add_argument('--webcam', action='store_true', default=True,
                       help='Utiliser la webcam (défaut)')
    
    parser.add_argument('--webcam-id', type=int, default=0,
                       help='ID de la webcam (défaut: 0)')
    
    parser.add_argument('--video', type=str,
                       help='Chemin vers un fichier vidéo (désactive webcam)')
    
    parser.add_argument('--port', type=int, default=8003,
                       help='Port du serveur Flask (défaut: 8003)')
    
    parser.add_argument('--host', default='0.0.0.0',
                       help='Host du serveur Flask (défaut: 0.0.0.0)')
    
    args = parser.parse_args()
    
    use_webcam = args.webcam
    if args.video:
        use_webcam = False
        if not os.path.exists(args.video):
            print(f"Vidéo introuvable : {args.video}")
            return
    
    print(f"\n{'='*60}")
    print(f"DASHCAM INTELLIGENCE ARTIFICIELLE")
    print(f"{'='*60}")
    print(f"Tracker: {args.tracker.upper()}")
    print(f"Source: {'Webcam #' + str(args.webcam_id) if use_webcam else args.video}")
    print(f"Serveur: http://{args.host}:{args.port}")
    print(f"{'='*60}\n")
    
    detection_thread = threading.Thread(
        target=detection_loop,
        args=(args.tracker, use_webcam, args.webcam_id, args.video),
        daemon=True
    )
    detection_thread.start()
    
    time.sleep(2)
    
    print(f"\n Serveur Flask démarré sur http://{args.host}:{args.port}")
    print(f" Flux vidéo : http://{args.host}:{args.port}/video")
    print(f"Alertes : http://{args.host}:{args.port}/alerts")
    print(f"\n Appuyez sur Ctrl+C pour arrêter\n")
    
    try:
        app.run(host=args.host, port=args.port, threaded=True, debug=False)
    except KeyboardInterrupt:
        print("\n\n Interruption clavier")
        recording = False
    except Exception as e:
        print(f"\n Erreur serveur : {e}")
        recording = False
    finally:
        backup_to_drive()
        print("\n Système arrêté proprement")

if __name__ == '__main__':
    main()