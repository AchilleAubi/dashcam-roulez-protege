# README - Système de Détection de Conduite Dangereuse

## Description

Système de détection intelligent basé sur YOLOv8 pour identifier les comportements de conduite dangereux en temps réel. Le système utilise la vision par ordinateur pour détecter les collisions potentielles, les piétons sur la route, les freinages brusques et autres comportements à risque.

## Fonctionnalités

- Détection d'objets (véhicules, piétons) avec YOLOv8
- Tracking multi-objets (SimpleTracker, DeepSORT, ByteTrack)
- Estimation de profondeur avec MiDaS
- Détection de voies de circulation
- Analyse comportementale (zigzag, freinage brutal, approche arrière)
- Alertes sonores différenciées par type de danger
- Interface web Flask avec flux vidéo en temps réel
- Sauvegarde automatique sur Google Drive
- Export des événements en CSV avec miniatures

## Prérequis

### Matériel

**Actuellement en développement sur PC** (performances optimales avec GPU)

Matériel prévu pour le déploiement final :
- **Raspberry Pi 4** (8GB RAM)
- **Google Coral USB Accelerator** (pour accélération EdgeTPU)
- Webcam USB ou caméra Raspberry Pi
- Carte SD 32GB minimum

### Système

**Configuration PC actuelle (développement) :**
- Windows/Linux/macOS
- Python 3.8+
- GPU CUDA recommandé (pour MiDaS et YOLOv8)
- 8GB RAM minimum

**Configuration Raspberry Pi (déploiement futur) :**
- Raspberry Pi OS (Bullseye ou supérieur)
- Python 3.8+
- Coral TPU (optionnel)

## Installation

### Installation sur PC (Développement)

#### 1. Créer un environnement virtuel (recommandé)
```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
venv\Scripts\activate     # Windows
```

#### 2. Installer les dépendances
```bash
pip install -r requirements.txt
```

#### 3. Vérifier l'installation CUDA (optionnel, pour GPU)
```bash
python -c "import torch; print(torch.cuda.is_available())"
```

### Installation sur Raspberry Pi (Déploiement futur)

#### 1. Dépendances système
```bash
sudo apt update
sudo apt upgrade -y
sudo apt install -y python3-pip python3-opencv
sudo apt install -y libatlas-base-dev libopenblas-dev libjpeg-dev
sudo apt install -y libportaudio2 portaudio19-dev
```

#### 2. Installation de Google Coral (optionnel)
```bash
echo "deb https://packages.cloud.google.com/apt coral-edgetpu-stable main" | sudo tee /etc/apt/sources.list.d/coral-edgetpu.list
curl https://packages.cloud.google.com/apt/doc/apt-key.gpg | sudo apt-key add -
sudo apt update
sudo apt install -y libedgetpu1-std python3-pycoral
```

#### 3. Packages Python
```bash
pip install -r pip-requirements.txt
```

## Utilisation

Voir le fichier `commande.txt` pour la liste complète des commandes disponibles.

### Démarrage Rapide

**Sur PC (développement) :**
```bash
# Serveur web avec webcam
python server.py

# Détection simple avec webcam
python detection_behavior.py --webcam
```

**Sur Raspberry Pi (futur) :**
```bash
# Mode optimisé pour Raspberry Pi
python server.py --tracker simple --cpu-only

# Détection avec économie de ressources
python detection_behavior.py --webcam --cpu-only --skip 2 --no-display
```

### Endpoints API

Une fois le serveur lancé sur `http://localhost:8003` :

- **Interface web** : `http://localhost:8003/`
- **Flux vidéo** : `http://localhost:8003/video`
- **Alertes JSON** : `http://localhost:8003/alerts`
- **Statistiques** : `http://localhost:8003/stats`
- **Arrêter** : `curl -X POST http://localhost:8003/stop`

### Tests Unitaires
```bash
python detection_behavior.py --run-tests
```

## Architecture et Performances

### Configuration Actuelle (PC avec GPU)

**Utilisé pour le développement et les tests :**
- MiDaS complet pour estimation de profondeur (haute qualité)
- YOLOv8 avec GPU CUDA
- Performances : 30-60 FPS selon GPU
- Mémoire : 4-8 GB RAM

### Configuration Future (Raspberry Pi + Coral)

**Optimisations prévues pour le déploiement :**

1. **Remplacer MiDaS par SimplifiedDepth**
```python
# Au lieu de :
depth_model = MiDaSDepth(device=device, small=midas_small)

# Utiliser :
depth_model = SimplifiedDepth()  # 100x plus rapide
```

2. **Activer EdgeTPU avec Coral**
```python
# Dans tflite_utils.py
interpreter = load_tflite_interpreter(model_path, use_edgetpu=True)
```

3. **Réduire la résolution**
```python
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
```

4. **Skip frames agressif**
```bash
python detection_behavior.py --cpu-only --webcam --skip 3 --no-display
```

## Configuration Google Drive

Le système sauvegarde automatiquement les logs et miniatures sur Google Drive.

1. Première exécution : un navigateur s'ouvrira pour autoriser l'accès
2. Les identifiants sont sauvegardés dans `mycreds.txt`
3. Les sauvegardes futures seront automatiques

## Structure des Fichiers
```
output/
├── events_log.csv          # Log de tous les événements
└── thumbs/                 # Miniatures des détections
    └── timestamp_IDxx_event.jpg

mycreds.txt                 # Credentials Google Drive
detection_behavior.py       # Script principal de détection
server.py                   # Serveur Flask
audio_alerts.py            # Gestion des alertes sonores
tflite_utils.py            # Utilitaires TensorFlow Lite (pour Coral)
commande.txt               # Liste complète des commandes
```

## Types d'Alertes

1. **frontal_collision_risk** - Risque de collision frontale (priorité haute)
2. **pedestrian_on_road** - Piéton sur la voie
3. **rear_approach** - Véhicule approchant rapidement par l'arrière
4. **lateral_close** - Dépassement latéral trop proche
5. **brake_hard** - Freinage brutal détecté
6. **zigzag** - Comportement de zigzag agressif

## Performances Mesurées

### PC avec GPU CUDA (Configuration actuelle)
- CPU uniquement : 8-12 FPS

### Raspberry Pi 4 (Estimations pour déploiement)
- Sans Coral (SimplifiedDepth) : 5-10 FPS
- Avec Coral TPU : 15-20 FPS
- Avec optimisations complètes : 20-25 FPS

## Dépannage

### Webcam non détectée (PC)
```bash
# Windows : vérifier dans Gestionnaire de périphériques
# Linux : lister les webcams
ls /dev/video*

# Tester avec un ID différent
python detection_behavior.py --webcam --webcam-id 1
```

### Erreur CUDA sur PC
```bash
# Vérifier installation PyTorch avec CUDA
pip uninstall torch torchvision
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

### Performances faibles sur PC
- Vérifier que le GPU est utilisé : `nvidia-smi`
- Utiliser un tracker plus léger : `--tracker simple`
- Réduire la résolution de la webcam


## Roadmap

- [x] Développement sur PC avec MiDaS
- [x] Interface Flask et API REST
- [x] Système d'alertes sonores
- [ ] Optimisation pour Raspberry Pi
- [ ] Intégration EdgeTPU Coral
- [ ] Remplacement MiDaS par SimplifiedDepth
- [ ] Tests sur matériel embarqué
- [ ] Modèles TFLite quantifiés
