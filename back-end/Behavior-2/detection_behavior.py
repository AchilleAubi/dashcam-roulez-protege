"""
 python script.py                           # SimpleTracker (baseline)
 python script.py --tracker deepsort        # Avec DeepSORT
 python script.py --tracker bytetrack       # Avec ByteTrack
 python script.py --run-tests               # Lancer les tests
 python script.py --cpu-only                # Optimisations CPU

"""

import os
import cv2
import numpy as np
import time
import argparse
from ultralytics import YOLO
import torch
from scipy.optimize import linear_sum_assignment
import csv
from collections import defaultdict
import threading
import base64
import io
from audio_alerts import AudioAlertManager

try:
    from deep_sort_realtime.deepsort_tracker import DeepSort
    DEEPSORT_AVAILABLE = True
except ImportError:
    DEEPSORT_AVAILABLE = False
    print("⚠️  DeepSORT non disponible. Install: pip install deep-sort-realtime")

try:
    from byte_tracker import BYTETracker
    BYTETRACK_AVAILABLE = True
except ImportError:
    BYTETRACK_AVAILABLE = False
    print("⚠️  ByteTrack non disponible. Install: pip install byte-tracker")

VIDEO_PATH = 'video_test_3.mp4'
YOLO_MODEL = 'yolov8n.pt'
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

IOU_MATCH_THRESHOLD = 0.3
DISTANCE_ALERT_THRESHOLD = 0.45
TTC_ALERT_SECONDS = 2.0
LATERAL_CLOSE_X_METRIC = 0.25
REAR_APPROACH_SPEED_THRESHOLD = 0.03
ZIGZAG_STD_THRESHOLD = 0.04
BRAKE_DECEL_THRESHOLD = 0.06
MIN_TRACK_AGE = 3

PERSON_CLASS = 0
VEHICLE_CLASSES = {2, 3, 5, 7}  # car, motorcycle, bus, truck

OUTPUT_DIR = 'output'
THUMBS_DIR = os.path.join(OUTPUT_DIR, 'thumbs')
LOG_CSV = os.path.join(OUTPUT_DIR, 'events_log.csv')
os.makedirs(THUMBS_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

audio_manager = AudioAlertManager(
    global_cooldown=1.5,   
    alert_cooldown=5.0     
)

def iou(boxA, boxB):
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    interW = max(0, xB - xA)
    interH = max(0, yB - yA)
    interArea = interW * interH
    boxAArea = max(0, (boxA[2]-boxA[0])) * max(0, (boxA[3]-boxA[1]))
    boxBArea = max(0, (boxB[2]-boxB[0])) * max(0, (boxB[3]-boxB[1]))
    union = boxAArea + boxBArea - interArea
    return interArea / union if union > 0 else 0

def box_center(box):
    return ((box[0]+box[2])/2, (box[1]+box[3])/2)

class TrackerWrapper:
    def __init__(self, tracker_type='simple'):
        self.tracker_type = tracker_type
        self.tracks_history = defaultdict(lambda: {
            'boxes': [], 'classes': [], 'depths': [], 
            'lane_history': [], 'timestamps': [], 'alerts': set()
        })
        
    def update(self, detections, frame=None):
        raise NotImplementedError
    
    def _update_history(self, track_id, box, cls, timestamp=None):
        self.tracks_history[track_id]['boxes'].append(box)
        self.tracks_history[track_id]['classes'].append(cls)
        if timestamp:
            self.tracks_history[track_id]['timestamps'].append(timestamp)
        
        for key in ['boxes', 'classes', 'depths', 'timestamps']:
            if len(self.tracks_history[track_id][key]) > 150:
                self.tracks_history[track_id][key] = \
                    self.tracks_history[track_id][key][-100:]


class SimpleTracker(TrackerWrapper):
    def __init__(self):
        super().__init__(tracker_type='simple')
        self.next_id = 1
        
    def update(self, detections, frame=None):
        prev_ids = [tid for tid in self.tracks_history.keys() 
                    if len(self.tracks_history[tid]['boxes']) > 0]
        prev_boxes = [self.tracks_history[k]['boxes'][-1] for k in prev_ids]
        
        det_boxes = [d['box'] for d in detections]
        
        if len(prev_boxes) == 0:
            for d in detections:
                tid = self.next_id
                self.next_id += 1
                self._update_history(tid, d['box'], d['class'])
            return self.tracks_history
        
        cost = np.ones((len(prev_boxes), len(det_boxes)), dtype=np.float32)
        for i, pb in enumerate(prev_boxes):
            for j, db in enumerate(det_boxes):
                cost[i, j] = 1.0 - iou(pb, db)
        
        row_ind, col_ind = linear_sum_assignment(cost)
        
        assigned_det = set()
        assigned_prev = set()
        
        for r, c in zip(row_ind, col_ind):
            if 1.0 - cost[r, c] >= IOU_MATCH_THRESHOLD:
                tid = prev_ids[r]
                det = detections[c]
                self._update_history(tid, det['box'], det['class'])
                assigned_det.add(c)
                assigned_prev.add(r)
        
        for j, d in enumerate(detections):
            if j not in assigned_det:
                tid = self.next_id
                self.next_id += 1
                self._update_history(tid, d['box'], d['class'])
        
        
        return self.tracks_history


class DeepSORTTracker(TrackerWrapper):
    def __init__(self, max_age=5):
        super().__init__(tracker_type='deepsort')
        if not DEEPSORT_AVAILABLE:
            raise RuntimeError("DeepSORT non installé. pip install deep-sort-realtime")
        
        self.deepsort = DeepSort(
            max_age=max_age,
            n_init=3,
            max_iou_distance=0.7,
            max_cosine_distance=0.3,
            nn_budget=100
        )
        
    def update(self, detections, frame=None):
        if frame is None:
            raise ValueError("DeepSORT nécessite le frame pour extraction de features")
        
        dets_for_ds = []
        for d in detections:
            x1, y1, x2, y2 = d['box']
            w, h = x2 - x1, y2 - y1
            dets_for_ds.append(([x1, y1, w, h], d.get('conf', 0.5), d.get('class', -1)))
        
        tracks = self.deepsort.update_tracks(dets_for_ds, frame=frame)
        
        for tr in tracks:
            if not tr.is_confirmed():
                continue
            
            tid = tr.track_id
            ltwh = tr.to_ltwh()
            x, y, w, h = ltwh
            box = [int(x), int(y), int(x+w), int(y+h)]
            cls = tr.det_class if hasattr(tr, 'det_class') else -1
            
            self._update_history(tid, box, cls)
        
        return self.tracks_history


class ByteTrackWrapper(TrackerWrapper):
    def __init__(self, track_thresh=0.5, track_buffer=30, match_thresh=0.8):
        super().__init__(tracker_type='bytetrack')
        if not BYTETRACK_AVAILABLE:
            raise RuntimeError("ByteTrack non installé. pip install byte-tracker")
        
        class Args:
            track_thresh = track_thresh
            track_buffer = track_buffer
            match_thresh = match_thresh
            mot20 = False
        
        self.byte_tracker = BYTETracker(Args())
        self.frame_id = 0
        
    def update(self, detections, frame=None):
        self.frame_id += 1
        
        if len(detections) == 0:
            online_targets = self.byte_tracker.update(
                np.empty((0, 5)), [frame.shape[0], frame.shape[1]], [frame.shape[0], frame.shape[1]]
            )
        else:
            dets_array = []
            for d in detections:
                x1, y1, x2, y2 = d['box']
                score = d.get('conf', 0.5)
                dets_array.append([x1, y1, x2, y2, score])
            
            dets_array = np.array(dets_array)
            online_targets = self.byte_tracker.update(
                dets_array, 
                [frame.shape[0], frame.shape[1]] if frame is not None else [480, 640],
                [frame.shape[0], frame.shape[1]] if frame is not None else [480, 640]
            )
        
        for t in online_targets:
            tid = t.track_id
            tlwh = t.tlwh
            box = [int(tlwh[0]), int(tlwh[1]), 
                   int(tlwh[0] + tlwh[2]), int(tlwh[1] + tlwh[3])]
            cls = -1  
            
            self._update_history(tid, box, cls)
        
        return self.tracks_history


def create_tracker(tracker_type='simple', **kwargs):
    """Factory pattern pour instancier le tracker souhaité"""
    tracker_type = tracker_type.lower()
    
    if tracker_type == 'simple':
        return SimpleTracker()
    
    elif tracker_type == 'deepsort':
        if not DEEPSORT_AVAILABLE:
            print("DeepSORT non disponible, fallback vers SimpleTracker")
            return SimpleTracker()
        return DeepSORTTracker(**kwargs)
    
    elif tracker_type == 'bytetrack':
        if not BYTETRACK_AVAILABLE:
            print("ByteTrack non disponible, fallback vers SimpleTracker")
            return SimpleTracker()
        return ByteTrackWrapper(**kwargs)
    
    else:
        raise ValueError(f"Tracker inconnu: {tracker_type}")


def detect_lane_lines_and_type(frame):
    h, w = frame.shape[:2]
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blur, 50, 150)
    
    mask = np.zeros_like(edges)
    polygon = np.array([[(0, h), (w, h), (w, int(h*0.6)), (0, int(h*0.6))]], dtype=np.int32)
    cv2.fillPoly(mask, polygon, 255)
    roi = cv2.bitwise_and(edges, mask)
    
    lines = cv2.HoughLinesP(roi, 1, np.pi/180, threshold=40, 
                            minLineLength=30, maxLineGap=60)
    
    left_lines, right_lines = [], []
    
    if lines is None:
        return int(w*0.2), int(w*0.8), 'unknown', None, None
    
    for l in lines:
        x1, y1, x2, y2 = l[0]
        if x2 == x1:
            continue
        slope = (y2 - y1) / (x2 - x1)
        if abs(slope) < 0.3:
            continue
        
        if slope < 0:
            left_lines.append(l[0])
        else:
            right_lines.append(l[0])
    
    def avg_x_and_gap(lines):
        if not lines:
            return None, 0
        xs = []
        gaps = []
        for x1, y1, x2, y2 in lines:
            xs.extend([x1, x2])
            gaps.append(abs(x2 - x1))
        return int(np.mean(xs)), np.mean(gaps)
    
    lx, lgap = avg_x_and_gap(left_lines)
    rx, rgap = avg_x_and_gap(right_lines)
    
    lx = lx if lx else int(w*0.2)
    rx = rx if rx else int(w*0.8)
    
    gap_mean = np.mean([g for g in [lgap, rgap] if g > 0])
    lane_type = 'dashed' if gap_mean > 80 else 'solid' if gap_mean > 0 else 'unknown'
    
    src = np.float32([[lx, int(h*0.6)], [rx, int(h*0.6)], [w, h], [0, h]])
    dst = np.float32([[int(w*0.2), 0], [int(w*0.8), 0], 
                      [int(w*0.8), h], [int(w*0.2), h]])
    try:
        M = cv2.getPerspectiveTransform(src, dst)
        Minv = cv2.getPerspectiveTransform(dst, src)
    except:
        M, Minv = None, None
    
    return lx, rx, lane_type, M, Minv


class MiDaSDepth:
    def __init__(self, device='cpu', small=True):
        self.device = device
        self.small = small
        model_name = 'MiDaS_small' if small else 'MiDaS'
        
        print(f"Chargement du modèle {model_name}...")
        self.midas = torch.hub.load("intel-isl/MiDaS", model_name).to(device)
        self.midas.eval()
        
        self.transforms = torch.hub.load("intel-isl/MiDaS", "transforms")
        try:
            self.transform = self.transforms.small_transform if small else \
                           self.transforms.default_transform
        except:
            self.transform = self.transforms.default_transform
        
        print(f"MiDaS chargé sur {device}")
    
    def predict(self, frame):
        img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        inp = self.transform(img).to(self.device)
        
        with torch.no_grad():
            prediction = self.midas(inp)
            prediction = torch.nn.functional.interpolate(
                prediction.unsqueeze(1), 
                size=frame.shape[:2], 
                mode='bicubic', 
                align_corners=False
            ).squeeze().cpu().numpy()
        
        minv, maxv = np.min(prediction), np.max(prediction)
        if maxv - minv > 1e-6:
            norm = (prediction - minv) / (maxv - minv)
        else:
            norm = np.zeros_like(prediction)
        
        return norm


def estimate_global_optical_flow(prev_gray, gray):
    """Calcule le flux optique global (avg_x, avg_y)"""
    flow = cv2.calcOpticalFlowFarneback(
        prev_gray, gray, None, 0.5, 3, 15, 3, 5, 1.2, 0
    )
    mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
    
    if np.sum(mag) == 0:
        return 0.0, 0.0
    
    avg_x = np.sum(flow[..., 0] * mag) / np.sum(mag)
    avg_y = np.sum(flow[..., 1] * mag) / np.sum(mag)
    
    return avg_x, avg_y


def log_event(event_type, tid, msg, box, frame, timestamp):
    """Log un événement dans le CSV et sauvegarde une miniature"""
    x1, y1, x2, y2 = map(int, box)
    h, w = frame.shape[:2]
    
    x1 = max(0, x1)
    y1 = max(0, y1)
    x2 = min(w-1, x2)
    y2 = min(h-1, y2)
    
    crop = frame[y1:y2, x1:x2]
    thumb_name = f'{int(timestamp)}_ID{tid}_{event_type}.jpg'
    thumb_path = os.path.join(THUMBS_DIR, thumb_name)
    
    try:
        cv2.imwrite(thumb_path, crop)
    except:
        thumb_path = ''
    
    row = [
        time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(timestamp)),
        tid, event_type, msg, x1, y1, x2, y2, thumb_path, ''
    ]
    
    write_header = not os.path.exists(LOG_CSV)
    with open(LOG_CSV, 'a', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        if write_header:
            writer.writerow([
                'timestamp', 'track_id', 'event_type', 'message',
                'x1', 'y1', 'x2', 'y2', 'thumb', 'validated'
            ])
        writer.writerow(row)


def analyze_tracks_and_log(tracker_history, depth_map, frame_shape, 
                           perspective_M, lane_type, flow_vec, 
                           frame_time, frame, use_logging=True):
    alerts = []
    alert_types_this_frame = []
    h, w = frame_shape[:2]
    own_x = w / 2
    
    for tid, info in list(tracker_history.items()):
        boxes = info.get('boxes', [])
        classes = info.get('classes', [])
        
        if len(boxes) < 1:
            continue
        
        box = boxes[-1]
        cls = classes[-1] if classes else None
        x1, y1, x2, y2 = box
        cx = (x1 + x2) / 2
        cy = (y1 + y2) / 2
        
        lane_pos = 'unknown'
        if perspective_M is not None:
            try:
                bottom_center = np.array([[(x1+x2)/2, y2]], dtype='float32')
                warped = cv2.perspectiveTransform(
                    np.array([bottom_center]), perspective_M
                )[0][0]
                warped_x = warped[0]
                norm_x = warped_x / w
                
                if norm_x < 0.4:
                    lane_pos = 'left'
                elif norm_x > 0.6:
                    lane_pos = 'right'
                else:
                    lane_pos = 'center'
            except:
                lane_pos = 'unknown'
        
        info.setdefault('lane_history', []).append(lane_pos)
        
        cx1, cy1, cx2, cy2 = map(int, [x1, y1, x2, y2])
        cx1 = max(0, cx1)
        cy1 = max(0, cy1)
        cx2 = min(w-1, cx2)
        cy2 = min(h-1, cy2)
        
        crop = depth_map[cy1:cy2, cx1:cx2]
        depth_val = float(np.median(crop)) if crop.size > 0 else 1.0
        
        info.setdefault('depths', []).append(depth_val)
        info.setdefault('timestamps', []).append(frame_time)
        
        vel = 0.0
        if len(info['depths']) >= 2:
            dt = info['timestamps'][-1] - info['timestamps'][-2]
            if dt > 0:
                vel = (info['depths'][-1] - info['depths'][-2]) / dt
                
        if cls == PERSON_CLASS:
            if lane_pos == 'center':
                if 'pedestrian_on_road' not in info.get('alerts', set()):
                    info.setdefault('alerts', set()).add('pedestrian_on_road')
                    alerts.append((tid, 'pedestrian_on_road', 
                                 ' Piéton sur la voie!', box))
                    alert_types_this_frame.append('pedestrian_on_road')
                    if use_logging:
                        log_event('pedestrian_on_road', tid, 
                                'Piéton sur la voie!', box, frame, frame_time)
            continue
        
        if lane_pos == 'center' and depth_val < DISTANCE_ALERT_THRESHOLD:
            if vel < 0:
                ttc = depth_val / (-vel) if -vel > 1e-6 else None
                if ttc is not None and ttc < TTC_ALERT_SECONDS:
                    key = f'frontal_{int(frame_time)}'
                    if key not in info.get('alerts', set()):
                        info.setdefault('alerts', set()).add(key)
                        alerts.append((tid, 'frontal_collision_risk',
                                     f' Collision frontale! TTC={ttc:.1f}s', box))
                        alert_types_this_frame.append('frontal_collision_risk')
                        if use_logging:
                            log_event('frontal_collision_risk', tid,
                                    f'TTC={ttc:.1f}s', box, frame, frame_time)
        
        hor_dist = abs(cx - own_x) / w
        if (hor_dist < LATERAL_CLOSE_X_METRIC and 
            lane_pos != 'center' and 
            depth_val < 0.6):
            
            if not (lane_type == 'dashed' and abs(flow_vec[0]) < 1.0):
                key = f'lateral_{int(frame_time)}'
                if key not in info.get('alerts', set()):
                    info.setdefault('alerts', set()).add(key)
                    alerts.append((tid, 'lateral_close',
                                 ' Dépassement latéral trop proche', box))
                    alert_types_this_frame.append('lateral_close')
                    if use_logging:
                        log_event('lateral_close', tid,
                                'Dépassement latéral trop proche', 
                                box, frame, frame_time)
        
        if cy > h * 0.75:
            if vel < -REAR_APPROACH_SPEED_THRESHOLD:
                key = f'rear_{int(frame_time)}'
                if key not in info.get('alerts', set()):
                    info.setdefault('alerts', set()).add(key)
                    alerts.append((tid, 'rear_approach',
                                 " Véhicule approche vite par l'arrière", box))
                    alert_types_this_frame.append('rear_approach')
                    if use_logging:
                        log_event('rear_approach', tid,
                                "Approche rapide par l'arrière",
                                box, frame, frame_time)
        
        lateral_hist = [
            0 if v == 'center' else (-1 if v == 'left' else 1)
            for v in info.get('lane_history', [])[-8:]
        ]
        if len(lateral_hist) >= 4:
            std = np.std(lateral_hist)
            if std > ZIGZAG_STD_THRESHOLD:
                if 'zigzag' not in info.get('alerts', set()):
                    info.setdefault('alerts', set()).add('zigzag')
                    alerts.append((tid, 'zigzag',
                                 ' Comportement agressif: zigzag', box))
                    alert_types_this_frame.append('zigzag')
                    if use_logging:
                        log_event('zigzag', tid,
                                'Comportement agressif: zigzag',
                                box, frame, frame_time)
        
        if len(info.get('depths', [])) >= 3:
            a1 = info['depths'][-1] - info['depths'][-2]
            a0 = info['depths'][-2] - info['depths'][-3]
            decel = a1 - a0
            
            if decel < -BRAKE_DECEL_THRESHOLD:
                key = f'brake_{int(frame_time)}'
                if key not in info.get('alerts', set()):
                    info.setdefault('alerts', set()).add(key)
                    alerts.append((tid, 'brake_hard',
                                 ' Freinage brutal détecté', box))
                    alert_types_this_frame.append('brake_hard')
                    if use_logging:
                        log_event('brake_hard', tid,
                                'Freinage brutal détecté',
                                box, frame, frame_time)
    
    if alert_types_this_frame:
        audio_manager.play_multiple_alerts(alert_types_this_frame)
    return alerts


def process_video(video_path, tracker_type='simple', cpu_only=False, 
                 run_display=True, skip_frames=0, use_webcam=False, webcam_id=0):
    
    if cpu_only:
        torch.set_num_threads(max(1, os.cpu_count() // 2))
        device = 'cpu'
        midas_small = True
    else:
        device = DEVICE
        midas_small = (device == 'cpu')
    
    print(f"\n{'='*60}")
    print(f"Détection de Conduite Dangereuse")
    print(f"{'='*60}")
    
    if use_webcam:
        print(f"Source: WEBCAM #{webcam_id}")
        cap = cv2.VideoCapture(webcam_id)
        if not cap.isOpened():
            raise RuntimeError(f"Impossible d'ouvrir la webcam #{webcam_id}")
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        cap.set(cv2.CAP_PROP_FPS, 30)
    else:
        print(f"Source: {video_path}")
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise RuntimeError(f"Impossible d'ouvrir la vidéo: {video_path}")
    
    print(f"Tracker: {tracker_type.upper()}")
    print(f"Device: {device.upper()}")
    print(f"{'='*60}\n")
    
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    if use_webcam:
        print(f"FPS: {fps:.1f} | Mode: TEMPS RÉEL")
        total_frames = float('inf') 
    else:
        print(f"FPS: {fps:.1f} | Frames: {total_frames}")
    
    print("\nChargement des modèles...")
    yolo = YOLO(YOLO_MODEL)
    print(f"YOLO chargé: {YOLO_MODEL}")
    
    depth_model = MiDaSDepth(device=device, small=midas_small)
    
    tracker = create_tracker(tracker_type)
    print(f"Tracker initialisé: {tracker.tracker_type.upper()}\n")
    
    ret, prev_frame = cap.read()
    if not ret:
        raise RuntimeError("Vidéo vide ou illisible")
    
    lx, rx, lane_type, M, Minv = detect_lane_lines_and_type(prev_frame)
    prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
    print(f"🛣️  Lanes détectées: type={lane_type}, left_x={lx}, right_x={rx}\n")
    
    frame_idx = 0
    start_time = time.time()
    alert_count = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        if skip_frames > 0 and frame_idx % (skip_frames + 1) != 0:
            frame_idx += 1
            continue
        
        tnow = time.time()
        h, w = frame.shape[:2]
        
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
        
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        flow_vec = estimate_global_optical_flow(prev_gray, gray)
        prev_gray = gray
        
        alerts = analyze_tracks_and_log(
            tracks, depth_map, frame.shape, M, lane_type,
            flow_vec, tnow, frame, use_logging=True
        )
        
        alert_count += len(alerts)
        
        for tid, info in tracks.items():
            if len(info['boxes']) == 0:
                continue
            
            box = info['boxes'][-1]
            x1, y1, x2, y2 = map(int, box)
            cls = info['classes'][-1] if info['classes'] else -1
            
            if cls == PERSON_CLASS:
                color = (0, 165, 255)  
            elif cls in VEHICLE_CLASSES:
                color = (0, 255, 0) 
            else:
                color = (128, 128, 128) 
            
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            
            label = f'ID{tid}'
            if cls == PERSON_CLASS:
                label += ' (Piéton)'
            elif cls in VEHICLE_CLASSES:
                label += f' (V{cls})'
            
            cv2.putText(frame, label, (x1, y1 - 6),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        cv2.line(frame, (int(lx), int(h*0.6)), (int(lx), h), (255, 0, 0), 2)
        cv2.line(frame, (int(rx), int(h*0.6)), (int(rx), h), (255, 0, 0), 2)
        
        cv2.putText(frame, f'Tracker: {tracker.tracker_type.upper()}',
                   (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
        cv2.putText(frame, f'Lane: {lane_type}',
                   (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
        cv2.putText(frame, f'Tracks: {len(tracks)}',
                   (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
        
        if use_webcam:
            cv2.putText(frame, f'Mode: WEBCAM (temps réel)',
                       (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        else:
            cv2.putText(frame, f'Frame: {frame_idx}/{total_frames}',
                       (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
        
        for i, (tid, evtype, msg, box) in enumerate(alerts):
            x1, y1, x2, y2 = map(int, box)
            cv2.putText(frame, msg, 
                       (max(10, x1 - 10), max(30, y1 - 10)),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        
        if run_display:
            cv2.imshow('Détection de Conduite Dangereuse', frame)
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                print("\nArrêt demandé par l'utilisateur")
                break
            elif key == ord('p'):
                cv2.waitKey(0)  
        
        frame_idx += 1
        
        if frame_idx % 30 == 0 and not use_webcam:
            elapsed = time.time() - start_time
            fps_actual = frame_idx / elapsed if elapsed > 0 else 0
            print(f"Frame {frame_idx}/{total_frames} | "
                  f"FPS: {fps_actual:.1f} | Alertes: {alert_count}")
        elif frame_idx % 100 == 0 and use_webcam:
            elapsed = time.time() - start_time
            fps_actual = frame_idx / elapsed if elapsed > 0 else 0
            print(f"Frames traités: {frame_idx} | "
                  f"FPS: {fps_actual:.1f} | Alertes: {alert_count}")
    
    cap.release()
    if run_display:
        cv2.destroyAllWindows()
    
    elapsed = time.time() - start_time
    print(f"\n{'='*60}")
    print(f"Traitement terminé!")
    print(f"{'='*60}")
    print(f"Temps total: {elapsed:.1f}s")
    print(f"Frames traitées: {frame_idx}")
    print(f"Alertes générées: {alert_count}")
    print(f"Log CSV: {LOG_CSV}")
    print(f" Miniatures: {THUMBS_DIR}/")
    print(f"{'='*60}\n")


def run_sanity_tests():
    print("\n" + "="*60)
    print("LANCEMENT DES TESTS UNITAIRES")
    print("="*60 + "\n")
    
    tests_passed = 0
    tests_total = 0
    
    print("Test 1: Calcul IoU...")
    tests_total += 1
    b1 = [10, 10, 50, 50]
    b2 = [12, 12, 48, 48]
    iou_val = iou(b1, b2)
    assert iou_val > 0.5, f"IoU devrait être > 0.5, obtenu {iou_val}"
    print(f"IoU correct: {iou_val:.3f}\n")
    tests_passed += 1
    
    print("Test 2: Détection de lanes...")
    tests_total += 1
    blank = np.zeros((480, 640, 3), dtype=np.uint8)
    for y in range(300, 480, 20):
        cv2.line(blank, (320, y), (320, y + 8), (255, 255, 255), 2)
    
    lx, rx, lane_type, M, Minv = detect_lane_lines_and_type(blank)
    assert lane_type in ('dashed', 'solid', 'unknown'), \
        f"Type de lane invalide: {lane_type}"
    print(f"Lane détectée: type={lane_type}, lx={lx}, rx={rx}\n")
    tests_passed += 1
    
    print("Test 3: Flux optique...")
    tests_total += 1
    f1 = np.zeros((200, 200), dtype=np.uint8)
    f2 = np.zeros_like(f1)
    cv2.circle(f1, (120, 100), 10, 255, -1)
    cv2.circle(f2, (110, 100), 10, 255, -1)
    
    ax, ay = estimate_global_optical_flow(f1, f2)
    assert ax < 0, f"Flow devrait être négatif (gauche), obtenu {ax}"
    print(f"Flux optique correct: avg_x={ax:.3f}, avg_y={ay:.3f}\n")
    tests_passed += 1
    
    print("Test 4: SimpleTracker...")
    tests_total += 1
    tracker_simple = SimpleTracker()
    det1 = [{'box': [10, 10, 50, 50], 'class': 2, 'conf': 0.9}]
    det2 = [{'box': [15, 15, 55, 55], 'class': 2, 'conf': 0.9}]
    
    tracks1 = tracker_simple.update(det1)
    tracks2 = tracker_simple.update(det2)
    
    assert len(tracks1) == 1, "Devrait avoir 1 track"
    assert len(tracks2) == 1, "Devrait maintenir 1 track"
    print(f"SimpleTracker fonctionne: {len(tracks2)} track(s)\n")
    tests_passed += 1
    
    print("Test 5: Logging d'événements...")
    tests_total += 1
    fake_frame = np.zeros((240, 320, 3), dtype=np.uint8)
    log_event('test_event', 999, 'Test unitaire', 
             [10, 10, 60, 60], fake_frame, time.time())
    
    assert os.path.exists(LOG_CSV), "Le fichier CSV devrait exister"
    print(f"Logging fonctionne: {LOG_CSV}\n")
    tests_passed += 1
    
    print("Test 6: Factory de trackers...")
    tests_total += 1
    
    tracker_s = create_tracker('simple')
    assert tracker_s.tracker_type == 'simple'
    print(f"SimpleTracker créé")
    
    if DEEPSORT_AVAILABLE:
        tracker_d = create_tracker('deepsort')
        assert tracker_d.tracker_type == 'deepsort'
        print(f"DeepSORT créé")
    else:
        print(f"DeepSORT non disponible (normal)")
    
    if BYTETRACK_AVAILABLE:
        tracker_b = create_tracker('bytetrack')
        assert tracker_b.tracker_type == 'bytetrack'
        print(f"ByteTrack créé")
    else:
        print(f"ByteTrack non disponible (normal)")
    
    print(f"Factory fonctionne correctement\n")
    tests_passed += 1
    
    # Résumé
    print("="*60)
    print(f"RÉSULTATS: {tests_passed}/{tests_total} tests réussis")
    
    if tests_passed == tests_total:
        print(" TOUS LES TESTS SONT PASSÉS!")
    else:
        print(f"  {tests_total - tests_passed} test(s) échoué(s)")
    
    print("="*60 + "\n")
    
    return tests_passed == tests_total


latest_frame = None
latest_alerts = []
running = False

def start_realtime_detection(tracker_type='simple', use_webcam=True, webcam_id=0):
    global latest_frame, latest_alerts, running
    if running:
        print("Détection déjà en cours")
        return

    running = True

    def loop():
        try:
            for frame, alerts in generate_realtime_frames(tracker_type, use_webcam, webcam_id):
                latest_frame = frame
                latest_alerts = alerts
        except Exception as e:
            print("Erreur thread:", e)
        finally:
            running = False

    t = threading.Thread(target=loop, daemon=True)
    t.start()
    print("Détection temps réel lancée.")

def generate_realtime_frames(tracker_type='simple', use_webcam=True, webcam_id=0):
    cap = cv2.VideoCapture(webcam_id if use_webcam else VIDEO_PATH)
    yolo = YOLO(YOLO_MODEL)
    depth_model = MiDaSDepth(device=DEVICE)
    tracker = create_tracker(tracker_type)

    lx, rx, lane_type, M, Minv = 0, 0, 'unknown', None, None
    prev_gray = None

    frame_count = 0
    frame_skip = 5  
    prev_gray = None

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1
        if frame_count % frame_skip != 0:
            continue  

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        if prev_gray is None:
            prev_gray = gray
            lx, rx, lane_type, M, Minv = detect_lane_lines_and_type(frame)
            continue

        lx, rx, lane_type, M, Minv = detect_lane_lines_and_type(frame)
        if lane_type is None or lx is None or rx is None:
            continue  

        results = yolo(frame, verbose=False)[0]
        detections = [{'box': list(map(int, r.xyxy[0])),
                    'class': int(r.cls[0]),
                    'conf': float(r.conf[0])} for r in results.boxes if float(r.conf[0]) > 0.3]

        detections_on_lane = []
        for det in detections:
            x1, y1, x2, y2 = det['box']
            obj_x_center = (x1 + x2) // 2
            obj_y_center = (y1 + y2) // 2
            if lx[obj_y_center] < obj_x_center < rx[obj_y_center]:
                detections_on_lane.append(det)

        if not detections_on_lane:
            prev_gray = gray
            continue  

        tracks = tracker.update(detections_on_lane, frame=frame)
        depth_map = depth_model.predict(frame)
        flow_vec = estimate_global_optical_flow(prev_gray, gray)
        alerts = analyze_tracks_and_log(tracks, depth_map, frame.shape, M, lane_type, flow_vec, time.time(), frame, use_logging=False)

        _, buffer = cv2.imencode('.jpg', frame)
        yield buffer.tobytes(), alerts

        prev_gray = gray

    cap.release()
    cv2.destroyAllWindows()




if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Détection de conduite dangereuse avec YOLOv8 + tracking avancé',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Commandes exemples:
  python script.py                                 # SimpleTracker (baseline)
  python script.py --tracker deepsort              # Avec DeepSORT
  python script.py --tracker bytetrack             # Avec ByteTrack
  python script.py --tracker deepsort --cpu-only   # DeepSORT en mode CPU
  python script.py --run-tests                     # Tests unitaires
  python script.py --video ma_video.mp4 --skip 2   # Skip 2 frames sur 3
        """
    )
    
    parser.add_argument('--video', default=VIDEO_PATH,
                       help=f'Chemin vers la vidéo (défaut: {VIDEO_PATH})')
    
    parser.add_argument('--tracker', default='simple',
                       choices=['simple', 'deepsort', 'bytetrack'],
                       help='Type de tracker à utiliser (défaut: simple)')
    
    parser.add_argument('--run-tests', action='store_true',
                       help='Lancer les tests unitaires au lieu du traitement')
    
    parser.add_argument('--cpu-only', action='store_true',
                       help='Forcer CPU avec optimisations (MiDaS_small, threads limités)')
    
    parser.add_argument('--no-display', action='store_true',
                       help='Désactiver l\'affichage vidéo (traitement en arrière-plan)')
    
    parser.add_argument('--skip', type=int, default=0,
                       help='Nombre de frames à sauter (0=toutes, 1=1 sur 2, etc.)')
    
    parser.add_argument('--webcam', action='store_true',
                       help='Utiliser la webcam au lieu d\'un fichier vidéo')
    
    parser.add_argument('--webcam-id', type=int, default=0,
                       help='ID de la webcam à utiliser (défaut: 0)')
    
    args = parser.parse_args()
    
    if args.run_tests:
        success = run_sanity_tests()
        exit(0 if success else 1)
    
    try:
        if not args.webcam and not os.path.exists(args.video):
            print(f"❌ Erreur: Vidéo introuvable: {args.video}")
            exit(1)
        
        if args.tracker == 'deepsort' and not DEEPSORT_AVAILABLE:
            print(" DeepSORT non disponible, utilisation de SimpleTracker")
            args.tracker = 'simple'
        
        if args.tracker == 'bytetrack' and not BYTETRACK_AVAILABLE:
            print("ByteTrack non disponible, utilisation de SimpleTracker")
            args.tracker = 'simple'
        
        process_video(
            video_path=args.video,
            tracker_type=args.tracker,
            cpu_only=args.cpu_only,
            run_display=not args.no_display,
            skip_frames=args.skip,
            use_webcam=args.webcam,
            webcam_id=args.webcam_id
        )
        
    except KeyboardInterrupt:
        print("\n\n Interruption clavier (Ctrl+C)")
    except Exception as e:
        print(f"\n ERREUR: {e}")
        import traceback
        traceback.print_exc()
        exit(1)