# Dashcam – Roulez Protégé
## Module **Safety-5** (Sécurité & Urgence) + **Driver-3** (Vidéos) + Front (React)

> ✅ Guide de démarrage **copier–coller** pour installer, configurer et lancer le front + les deux back-ends (Driver-3 et Safety-5).

---

## 🧰 Prérequis

- **Node.js** LTS (v18 ou v20) + **npm**
- **Python** 3.10+ (idéal 3.11) + **pip**
- (Option) **Git**, **curl** ou **Postman**

Vérifier rapidement :
```bash
node -v
npm -v
python --version  # (macOS/Linux)  | sous Windows selon config: py --version
```

---

## 🗂️ Arborescence (résumé)

```
project-root/
├─ back-end/
│  ├─ Driver-3/           # API vidéos (FastAPI, port 8000)
│  └─ Safety-5/           # API sécurité & urgence (FastAPI, port 8005)
│     ├─ safety.py
│     ├─ data/            # contacts.json, incidents.json, privacy.json (créés auto)
│     └─ evidence/        # pièces jointes (images/vidéos)
└─ front-end/             # React (port 3000)
   └─ src/
      ├─ App.js
      └─ Safety.jsx       # UI pour Safety-5 (SOS, contacts, incidents…)
```

> ℹ️ Si `Safety-5` n’existe pas encore, crée le dossier avec `safety.py` selon le code de référence du module (voir doc interne).

---

## ⚙️ Configuration (Front)

Dans le code (ou via `.env`), assure-toi que les URLs pointent vers les bons ports en local :
```js
// Exemple dans du code :
const API_BASE   = "http://localhost:8000"; // Driver-3 (vidéos)
const API_SAFETY = "http://localhost:8005"; // Safety-5 (sécurité)
```

Exemples `.env` possibles :

**Vite**
```
VITE_API_BASE=http://localhost:8000
VITE_API_SAFETY=http://localhost:8005
```

**Create React App**
```
REACT_APP_API_BASE=http://localhost:8000
REACT_APP_API_SAFETY=http://localhost:8005
```

> 📌 Vérifie `package.json` du front : si c’est **Vite**, la commande sera `npm run dev`. Si c’est **CRA**, ce sera `npm start`.

---

## 🚀 Lancer les services (3 terminaux)

### 1) Back — Driver-3 (port 8000)

**Windows (PowerShell)**
```powershell
cd back-end\Driver-3
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install --upgrade pip
pip install fastapi uvicorn pydantic==1.* python-multipart
# Remplace 'main' par le nom du fichier contenant 'app = FastAPI(...)'
uvicorn main:app --reload --port 8000
```

**macOS / Linux**
```bash
cd back-end/Driver-3
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install fastapi uvicorn pydantic==1.* python-multipart
# Remplace 'main' par le nom du fichier contenant 'app = FastAPI(...)'
uvicorn main:app --reload --port 8000
```

**Tester :** ouvrir `http://localhost:8000/docs` (OpenAPI)  
> Selon l’implémentation, les vidéos peuvent être servies sous `/files/<name>`.

---

### 2) Back — Safety-5 (port 8005)

**Windows (PowerShell)**
```powershell
cd back-end\Safety-5
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install --upgrade pip
pip install fastapi uvicorn pydantic==1.* python-multipart
uvicorn safety:app --reload --port 8005
```

**macOS / Linux**
```bash
cd back-end/Safety-5
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install fastapi uvicorn pydantic==1.* python-multipart
uvicorn safety:app --reload --port 8005
```

**Tester :** ouvrir `http://localhost:8005/docs` (OpenAPI)  
**Endpoints principaux :**

- **Contacts d’urgence**
  - `GET  /api/emergency/contacts`
  - `POST /api/emergency/contacts`  → `{ name, phone?, email?, channels: ["sms","email","call"] }`
  - `DELETE /api/emergency/contacts/{id}`
- **Incidents & SOS**
  - `POST /api/emergency/sos` → `{ location?, note?, attachRecordingName? }`
  - `POST /api/emergency/crash-event` → `{ gForce, speedKmh?, location? }` (auto-SOS si `gForce ≥ 2.5`)
  - `GET  /api/emergency/incidents`
  - `GET  /api/emergency/incidents/{id}`
- **Pièces jointes**
  - `POST /api/emergency/evidence` (multipart file) → `{ url }`
- **Confidentialité**
  - `POST /api/privacy/lock` → `{ enabled: true|false }`
  - `GET  /api/privacy/status`

---

### 3) Front — React (port 3000)

```bash
cd front-end
npm install
# Démarrer selon le bundler :
npm start     # Create React App
# ou
npm run dev   # Vite
```

**Accès UI :** `http://localhost:3000`  
Le composant **Safety.jsx** expose : bouton **SOS**, **contacts**, **simulateur d’accident**, **historique**, **mode confidentialité**.

---

## 🧪 Tests rapides (cURL)

```bash
# Ajouter un contact
curl -X POST http://localhost:8005/api/emergency/contacts \
  -H "Content-Type: application/json" \
  -d '{"name":"Julien","email":"julien@ex.com","channels":["email"]}'

# Lister contacts
curl http://localhost:8005/api/emergency/contacts

# Déclencher un SOS manuel
curl -X POST http://localhost:8005/api/emergency/sos \
  -H "Content-Type: application/json" \
  -d '{"note":"Test SOS"}'

# Simuler un crash
curl -X POST http://localhost:8005/api/emergency/crash-event \
  -H "Content-Type: application/json" \
  -d '{"gForce":2.8,"speedKmh":45}'
  
# Historique des incidents
curl http://localhost:8005/api/emergency/incidents
```

---

## 🔧 Dépannage

- **Port déjà utilisé**  
  Change le port : `--port 8006`, ou ferme le process sur 8000/8005.

- **CORS (erreurs en front)**  
  Safety-5 active déjà :
  ```python
  CORSMiddleware(
    allow_origins=["http://localhost:3000","http://127.0.0.1:3000"],
    allow_methods=["*"], allow_headers=["*"]
  )
  ```
  Si ton front a une autre origine (port/host), ajoute-la dans `allow_origins`.

- **Uvicorn ne trouve pas `app`**  
  Utilise `uvicorn <fichier_sans_.py>:app` (le fichier où tu as `app = FastAPI(...)`).  
  Aide :  
  - Windows : `findstr /s /i "FastAPI(" *.py`  
  - macOS/Linux : `grep -R "FastAPI(" -n .`

- **Problèmes Node (npm)**  
  Utilise **Node LTS** (18/20). Re-installe le front : `rm -rf node_modules && npm install` (Linux/mac) ou suppression dossier manuelle (Windows).

- **Windows PowerShell – ExecutionPolicy**  
  Si l’activation du venv échoue, lance PowerShell en admin :  
  ```powershell
  Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
  ```

---

## 📦 Scripts utilitaires (optionnels)

**Windows – `dev.ps1`**
```powershell
Start-Process powershell -ArgumentList 'cd back-end\Driver-3; .\.venv\Scripts\Activate.ps1; uvicorn main:app --reload --port 8000'
Start-Process powershell -ArgumentList 'cd back-end\Safety-5; .\.venv\Scripts\Activate.ps1; uvicorn safety:app --reload --port 8005'
Start-Process powershell -ArgumentList 'cd front-end; npm start'
```

**macOS/Linux – `dev.sh`**
```bash
#!/usr/bin/env bash
( cd back-end/Driver-3 && source .venv/bin/activate && uvicorn main:app --reload --port 8000 ) &
( cd back-end/Safety-5 && source .venv/bin/activate && uvicorn safety:app --reload --port 8005 ) &
( cd front-end && npm run dev ) &
wait
```

> Donne les droits d’exécution : `chmod +x dev.sh`

---

## ✅ Checklist livraison (MVP)

- [ ] Driver-3 lancé sur **8000** (`/docs` OK)
- [ ] Safety-5 lancé sur **8005** (`/docs` OK)
- [ ] Front lancé sur **3000**
- [ ] 1–2 contacts d’urgence créés
- [ ] 1 SOS manuel + 1 crash simulé visibles dans **Historique**
- [ ] (Option) `attachRecordingName` pointe vers un clip Driver-3 (`http://localhost:8000/files/<name>`)

---

## 📄 Licence / Crédit

Projet éducatif de groupe – module **Safety-5** par <votre_nom>.  
Respectez les politiques d’utilisation des API/SDK externes (email/SMS) si vous intégrez de vraies notifications.
