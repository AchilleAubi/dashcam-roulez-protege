import numpy as np
import threading
import time

try:
    import simpleaudio as sa
    SIMPLEAUDIO_AVAILABLE = True
except Exception:
    SIMPLEAUDIO_AVAILABLE = False
    print("simpleaudio non disponible. Install: pip install simpleaudio")


class AudioAlertManager:
    
    def __init__(self, global_cooldown=1.5, alert_cooldown=5.0):
 
        self.global_cooldown = global_cooldown
        self.alert_cooldown = alert_cooldown
        self.last_global_beep = 0
        self.last_alert_times = {}
        self.lock = threading.Lock()
        
        self.alert_config = {
            'frontal_collision_risk': {
                'priority': 1,      
                'frequency': 1500, 
                'duration': 0.4,
                'volume': 0.4
            },
            'pedestrian_on_road': {
                'priority': 2,
                'frequency': 1200,
                'duration': 0.35,
                'volume': 0.35
            },
            'rear_approach': {
                'priority': 3,
                'frequency': 1000,
                'duration': 0.3,
                'volume': 0.3
            },
            'lateral_close': {
                'priority': 4,
                'frequency': 800,
                'duration': 0.25,
                'volume': 0.25
            },
            'brake_hard': {
                'priority': 5,
                'frequency': 700,
                'duration': 0.25,
                'volume': 0.25
            },
            'zigzag': {
                'priority': 6,
                'frequency': 600,
                'duration': 0.2,
                'volume': 0.2
            }
        }
    
    def should_play_sound(self, alert_type):
        with self.lock:
            current_time = time.time()
            
            if current_time - self.last_global_beep < self.global_cooldown:
                return False
            
            if alert_type in self.last_alert_times:
                if current_time - self.last_alert_times[alert_type] < self.alert_cooldown:
                    return False
            
            self.last_global_beep = current_time
            self.last_alert_times[alert_type] = current_time
            return True
    
    def play_alert(self, alert_type):
        if not self.should_play_sound(alert_type):
            return
        
        config = self.alert_config.get(alert_type)
        if not config:
            return  
        
        def _play():
            if not SIMPLEAUDIO_AVAILABLE:
                print(f" {alert_type}")
                return
            
            try:
                fs = 44100
                duration = config['duration']
                frequency = config['frequency']
                volume = config['volume']
                
                t = np.linspace(0, duration, int(fs * duration), False)
                
                attack = int(fs * 0.05)
                release = int(fs * 0.1)
                envelope = np.ones(len(t))
                envelope[:attack] = np.linspace(0, 1, attack)
                envelope[-release:] = np.linspace(1, 0, release)
                
                tone = np.sin(frequency * t * 2 * np.pi) * envelope
                audio = (tone * volume * (2**15 - 1)).astype(np.int16)
                
                sa.play_buffer(audio, 1, 2, fs)
                
            except Exception as e:
                print(f" Erreur audio: {e}")
        
        threading.Thread(target=_play, daemon=True).start()
    
    def play_multiple_alerts(self, alert_types):
        if not alert_types:
            return
        
        sorted_alerts = sorted(
            alert_types,
            key=lambda x: self.alert_config.get(x, {}).get('priority', 999)
        )
        
        self.play_alert(sorted_alerts[0])

