import React, { useEffect, useRef, useState } from "react";
import "./App.css";
import Safety from "./Safety";

const API_BASE = "http://localhost:8000"; // <-- adapte si besoin
const API_SAFETY = "http://localhost:8005";

function App() {
  const [journal, setJournal] = useState([]);
  const [filtre, setFiltre] = useState("");

  // Empêche l'upload double (StrictMode)
  const uploadingRef = useRef(false);
  const recordSessionRef = useRef(null);
  const processedSessionRef = useRef(null);

  useEffect(() => {
    fetch("/journal_emotionnel.json")
      .then((res) => res.json())
      .then((data) => setJournal(data))
      .catch((err) => console.error("Erreur chargement JSON", err));
  }, []);

  // ==================== DASHCAM Enregistrement video ====================
  const liveVideoRef = useRef(null);
  const [mediaStream, setMediaStream] = useState(null);
  const [recorder, setRecorder] = useState(null);
  const [chunks, setChunks] = useState([]);
  const [isRecording, setIsRecording] = useState(false);

  const [clipsLocal, setClipsLocal] = useState([]);
  const [clipsServer, setClipsServer] = useState([]);

  const loadServerClips = async () => {
    try {
      const res = await fetch(`${API_BASE}/api/recordings`);
      const data = await res.json();
      const mapped = data.map(d => ({
        id: d.name,
        url: `${API_BASE}${d.url}`,
        size: d.size,
        createdAt: d.createdAt,
        persisted: true,
      }));
      setClipsServer(mapped);
    } catch (e) {
      console.error("Erreur chargement recordings:", e);
    }
  };

  useEffect(() => {
    loadServerClips();
  }, []);

  const startCamera = async () => {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({
        video: { width: { ideal: 1280 }, height: { ideal: 720 } },
        audio: true,
      });
      setMediaStream(stream);
      if (liveVideoRef.current) {
        liveVideoRef.current.srcObject = stream;
        await liveVideoRef.current.play();
      }
    } catch (e) {
      console.error("Impossible d'accéder à la webcam:", e);
      alert("Autorise la caméra et le micro dans ton navigateur.");
    }
  };

  const stopCamera = () => {
    if (mediaStream) {
      mediaStream.getTracks().forEach(t => t.stop());
      setMediaStream(null);
    }
    if (liveVideoRef.current) {
      liveVideoRef.current.pause();
      liveVideoRef.current.srcObject = null;
    }
  };

  // ---- Enregistrement ----
  const startRecording = () => {
    if (!mediaStream) {
      alert("Démarre d'abord la caméra.");
      return;
    }
    recordSessionRef.current = `rec_${Date.now()}`;

    const mime = MediaRecorder.isTypeSupported("video/webm;codecs=vp9,opus")
      ? "video/webm;codecs=vp9,opus"
      : MediaRecorder.isTypeSupported("video/webm;codecs=vp8,opus")
        ? "video/webm;codecs=vp8,opus"
        : "video/webm";

    try {
      const mr = new MediaRecorder(mediaStream, { mimeType: mime, videoBitsPerSecond: 4_000_000 });
      setChunks([]);
      mr.ondataavailable = (e) => {
        if (e.data && e.data.size > 0) setChunks(prev => [...prev, e.data]);
      };
      mr.onstart = () => setIsRecording(true);
      mr.onstop = () => setIsRecording(false);

      mr.start(1000);
      setRecorder(mr);
    } catch (e) {
      console.error("Échec MediaRecorder:", e);
      alert("L’enregistrement n’est pas supporté par ce navigateur.");
    }
  };

  const stopRecording = () => {
    if (recorder && recorder.state !== "inactive") {
      recorder.stop();
      setRecorder(null);
    }
  };

  // ---- Après arrêt: créer le Blob local, puis uploader AUTO ----
  useEffect(() => {
    const autoUpload = async () => {
      if (recorder || chunks.length === 0) return;

      const sessionId = recordSessionRef.current;
      if (!sessionId) return;

      if (processedSessionRef.current === sessionId) return;

      if (uploadingRef.current) return;
      uploadingRef.current = true;

      try {
        processedSessionRef.current = sessionId;

        const blob = new Blob(chunks, { type: chunks[0]?.type || "video/webm" });
        const localUrl = URL.createObjectURL(blob);
        const tempId = `${sessionId}`;

        setClipsLocal(prev => [
          { id: tempId, url: localUrl, size: blob.size, createdAt: new Date().toISOString(), persisted: false },
          ...prev
        ]);

        const form = new FormData();
        form.append("file", blob, `${tempId}.webm`);
        const up = await fetch(`${API_BASE}/api/upload`, { method: "POST", body: form });
        const res = await up.json();
        if (!up.ok || !res.ok) throw new Error(res.error || "Upload échoué");

        setClipsLocal(prev => {
          const toDel = prev.find(c => c.id === tempId);
          if (toDel) URL.revokeObjectURL(toDel.url);
          return prev.filter(c => c.id !== tempId);
        });

        setClipsServer(prev => [
          {
            id: res.name,
            url: `${API_BASE}${res.url}`,
            size: blob.size,
            createdAt: new Date().toISOString(),
            persisted: true
          },
          ...prev
        ]);

      } catch (e) {
        console.error("Upload auto échoué:", e);
        alert("Upload échoué — le clip local reste visible mais ne sera pas persistant.");
      } finally {
        setChunks([]);
        uploadingRef.current = false;
      }
    };

    autoUpload();
  }, [recorder, chunks]);

  // ---- Téléchargement / Suppression (sur serveur) ----
  const downloadServerClip = (clip) => {
    const a = document.createElement("a");
    a.href = clip.url;
    a.download = clip.id;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
  };

  const deleteServerClip = async (clipId) => {
    try {
      const res = await fetch(`${API_BASE}/api/recordings/${encodeURIComponent(clipId)}`, { method: "DELETE" });
      if (!res.ok) throw new Error("Suppression échouée");
      setClipsServer(prev => prev.filter(c => c.id !== clipId));
    } catch (e) {
      console.error(e);
      alert("Échec suppression");
    }
  };
  // ==================== Fin enregistrement video ====================

  // ==================== UI ====================
  const getEmotionColor = (emotion) => {
    switch (emotion) {
      case "happy": return "#d4edda";
      case "angry": return "#f8d7da";
      case "neutral": return "#fdfdfe";
      case "sad": return "#fff3cd";
      case "fear": return "#d1ecf1";
      default: return "#f0f0f0";
    }
  };

  const journalFiltré = journal.filter((entry) =>
    Object.values(entry).some((val) =>
      String(val).toLowerCase().includes(filtre)
    )
  );

  const exportCSV = () => {
    if (journal.length === 0) return;
    const header = "Date;Émotion;Accélération;Freinage;Message\n";
    const rows = journal.map(j =>
      `${j.timestamp};${j.emotion};${j.acceleration};${j.freinage};"${j.message.replace(/"/g, '""')}"`
    );
    const blob = new Blob([header + rows.join("\n")], { type: 'text/csv;charset=utf-8;' });
    const link = document.createElement("a");
    link.href = URL.createObjectURL(blob);
    link.setAttribute("download", "journal_emotionnel.csv");
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
  };

  return (
    <div className="App">
      <h1>Journal Émotionnel - MoodCam 📊</h1>

      {/* Recherche + export */}
      <div style={{ display: 'flex', gap: 10, flexWrap: 'wrap', alignItems: 'center' }}>
        <input
          type="text"
          placeholder="🔎 Rechercher (mot, émotion, date...)"
          value={filtre}
          onChange={(e) => setFiltre(e.target.value.toLowerCase())}
          style={{
            padding: "10px",
            borderRadius: "5px",
            border: "1px solid #ccc",
            width: "100%",
            maxWidth: "400px",
          }}
        />
        <button onClick={exportCSV} className="btn">📁 Exporter en CSV</button>
      </div>

      {/* TABLEAU JOURNAL (existant) */}
      <div className="table-container" style={{ marginTop: 32 }}>
        <table>
          <thead>
            <tr>
              <th>Date</th>
              <th>Émotion</th>
              <th>Accélération</th>
              <th>Freinage</th>
              <th>Message</th>
            </tr>
          </thead>
          <tbody>
            {journalFiltré.map((entry, index) => (
              <tr key={index} style={{ backgroundColor: getEmotionColor(entry.emotion) }}>
                <td>{entry.timestamp}</td>
                <td>{entry.emotion}</td>
                <td>{entry.acceleration}</td>
                <td>{entry.freinage}</td>
                <td>{entry.message}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>

      {/* DASHCAM enregistrement video */}
      <section style={{ marginTop: 24 }}>
        <h2>🎥 DashCam — Enregistrement Webcam (persistant)</h2>

        <div style={{ display: 'grid', gridTemplateColumns: '1fr', gap: 12, maxWidth: 720 }}>
          <video
            ref={liveVideoRef}
            muted
            playsInline
            style={{ width: "100%", background: "#000", borderRadius: 8, border: "1px solid #ddd" }}
          />
        </div>

        <div style={{ display: 'flex', gap: 8, marginTop: 10, flexWrap: 'wrap' }}>
          <button onClick={startCamera} disabled={!!mediaStream} className="btn">▶️ Démarrer la caméra</button>
          <button onClick={stopCamera} disabled={!mediaStream} className="btn secondary">⏹️ Arrêter la caméra</button>
          <button onClick={startRecording} disabled={!mediaStream || isRecording} className="btn green">⭕ Démarrer l’enregistrement</button>
          <button onClick={stopRecording} disabled={!isRecording} className="btn danger">⏺️ Stopper & uploader</button>
          <button onClick={loadServerClips} className="btn secondary">🔄 Rafraîchir la liste</button>
        </div>

        <div style={{ marginTop: 8, fontSize: 14 }}>
          Caméra: {mediaStream ? "ON" : "OFF"} • Enregistrement: {isRecording ? "EN COURS..." : "—"}
        </div>

        {/* (Optionnel) Clips temporaires (non persistés) */}
        {clipsLocal.length > 0 && (
          <>
            <h3 style={{ marginTop: 20 }}>⏳ En cours (non persistés)</h3>
            <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fill, minmax(280px, 1fr))", gap: 16 }}>
              {clipsLocal.map((c) => (
                <div key={c.id} className="card">
                  <video src={c.url} controls style={{ width: "100%", borderRadius: 8, background: "#000" }} />
                  <div className="card-meta">
                    <div>ID: {c.id}</div>
                    <div>Taille: {(c.size / (1024 * 1024)).toFixed(2)} Mo</div>
                    <div>Créé: {new Date(c.createdAt).toLocaleString()}</div>
                    <div>Statut: pas encore sur le serveur</div>
                  </div>
                </div>
              ))}
            </div>
          </>
        )}

        {/* 🎞️ Clips enregistrés (persistants, backend) */}
        <h3 style={{ marginTop: 20 }}>🎞️ Clips enregistrés (persistants)</h3>
        {clipsServer.length === 0 && <div>Aucun clip persistant pour le moment.</div>}
        <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fill, minmax(280px, 1fr))", gap: 16 }}>
          {clipsServer.map((c) => (
            <div key={c.id} className="card">
              <video src={c.url} controls style={{ width: "100%", borderRadius: 8, background: "#000" }} />
              <div className="card-meta">
                <div>Nom: {c.id}</div>
                <div>Taille: {c.size ? (c.size / (1024 * 1024)).toFixed(2) + " Mo" : "—"}</div>
                <div>Créé: {new Date(c.createdAt).toLocaleString()}</div>
              </div>
              <div className="card-actions">
                <button onClick={() => downloadServerClip(c)} className="btn small">⬇️ Télécharger</button>
                <button onClick={() => deleteServerClip(c.id)} className="btn small danger">🗑️ Supprimer</button>
              </div>
            </div>
          ))}
        </div>
      </section>

      <hr />
      <Safety apiBase={API_SAFETY} />
    </div>
  );
}

export default App;
