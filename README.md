# DashCam IA - Système Intelligent de Sécurité Routière

## Vue d'ensemble

**DashCam IA** est un système complet de surveillance et d'assistance à la conduite basé sur l'intelligence artificielle. Conçu pour fonctionner sur Raspberry Pi 4 avec accélération matérielle, ce système combine analyse vidéo en temps réel, détection comportementale et assistance vocale pour améliorer la sécurité routière.

### Équipe de Développement

- **Rakotoarimanana Nampina Fanamperana – N26** : Enregistrement & Surveillance Vidéo
- **Andrianirimanjaka Onja Mbola – N7** : Analyse du Comportement Routier
- **Andriantsitohaina Lahatra Mamy Hajaina – N11** : Surveillance du Conducteur
- **Fanomezantsoa Achille Aubin – N14** : MoodCam & Journal Émotionnel
- **Rafamatanantsoa Princy Rolly – N20** : Sécurité & Urgences

---

## Fonctionnalités Complètes

### 1. Enregistrement & Surveillance Vidéo

**Responsable : Nampina (N26)**

- Enregistrement vidéo en boucle (caméra avant & conducteur)
- Sauvegarde automatique en cas d'incident détecté
- Vision nocturne avec support infrarouge
- Interface web intuitive pour accès aux vidéos
- Stockage local et sauvegarde cloud automatique
- Gestion intelligente de l'espace disque

### 2. Analyse du Comportement Routier

**Responsable : Onja (N7)**

**Détection de comportements dangereux autour du véhicule :**
- Détection de dépassements latéraux trop proches
- Alerte si véhicule approche rapidement par l'arrière
- Analyse de changements de voie dangereux
- Reconnaissance de comportements agressifs (zigzag, freinage brutal)
- Détection de piétons sur la chaussée
- Calcul du temps avant collision potentielle
- Système d'alertes sonores hiérarchisées par niveau de danger

### 3. Surveillance du Conducteur

**Responsable : Lahatra (N11)**

**Monitoring de l'état du conducteur :**
- Détection de somnolence (fermeture des yeux, bâillements)
- Détection de distraction (regards hors route, utilisation téléphone)
- Analyse des émotions en temps réel : stress, colère, fatigue
- Suivi de la position de la tête et direction du regard
- Score de vigilance en temps réel
- Alertes vocales de sécurité personnalisées :
  - "Attention, somnolence détectée"
  - "Gardez les yeux sur la route"
  - "Prenez une pause, vous semblez fatigué"

### 4. MoodCam – Journal Émotionnel de Conduite

**Responsable : Achille (N14)**

**Analyse comportementale et émotionnelle :**
- Analyse du style de conduite (accélérations, freinages, virages)
- Détection et analyse des émotions faciales
- Corrélation entre état émotionnel et comportement de conduite
- Génération automatique d'un journal de conduite (texte ou vocal)
- Statistiques hebdomadaires et mensuelles
- Conseils personnalisés :
  - Suggestions de pauses
  - Recommandations de musique relaxante
  - Détection de conduite stressante
  - Conseils d'amélioration du comportement

### 5. Sécurité & Urgences

**Responsable : Princy (N20)**

**Assistant vocal de sécurité :**
- Commandes vocales hors ligne :
  - "Signaler accident"
  - "Où suis-je ?" (géolocalisation)
  - "Appeler urgence"
  - "Sauvegarder vidéo"
- Alertes automatiques vocales contextuelles
- Interface mains-libres complète

**Module d'urgence intelligent :**
- Détection automatique d'accident (capteurs + analyse IA)
- Envoi automatique de la position GPS aux contacts d'urgence
- Sauvegarde prioritaire de la séquence vidéo avant/après impact
- Appel ou SMS automatique aux contacts prédéfinis
- Bouton SOS physique pour déclenchement manuel
- Notification aux services d'urgence avec données contextuelles

---

## Configuration Matérielle

**Configuration de développement (PC) :**
- Ordinateur avec carte graphique
- Webcam USB
- Mémoire : 8GB RAM minimum

**Configuration finale (embarquée) :**
- Raspberry Pi 4 (8GB RAM)
- Accélérateur Google Coral (EdgeTPU)
- 2 caméras : avant (1080p) + conducteur (720p)
- Module GPS
- Capteur de mouvement (accéléromètre/gyroscope)
- Microphone + Haut-parleur
- Bouton d'urgence physique
- Carte SD 64GB
- Alimentation 5V 3A

---

## Installation

Suivre chaque commande et readme dans back-end 
voici le nom de chaque repertoires :
- **Enregistrement & Surveillance Vidéo** : back-end/Driver-3/
- **Analyse du Comportement Routier** : back-end/Behavior-2/
- **Surveillance du Conducteur** : back-end/Driver_monitoring-3/
- **MoodCam & Journal Émotionnel** : back-end/MoodCam-4/
- **Sécurité & Urgences** : back-end/Safety-5/
