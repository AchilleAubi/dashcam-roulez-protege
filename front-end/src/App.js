import React, { useEffect, useRef, useState } from "react";
import "./App.css";
import Safety from "./Safety";
import DriverMonitoring from "./Driver-monitoring";
import DashcamUI from "./DashcamUI";

const API_BASE = "http://localhost:8000";
const API_SAFETY = "http://localhost:8005";
const API_DASHCAM = "http://127.0.0.1:8003";

function App() {
  const [journal, setJournal] = useState([]);
  const [filtre, setFiltre] = useState("");

  const uploadingRef = useRef(false);
  const recordSessionRef = useRef(null);
  const processedSessionRef = useRef(null);

  useEffect(() => {
    fetch("/journal_emotionnel.json")
      .then((res) => res.json())
      .then((data) => setJournal(data))
      .catch((err) => console.error("Erreur chargement JSON", err));
  }, []);

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
      alert("L'enregistrement n'est pas supporté par ce navigateur.");
    }
  };

  const stopRecording = () => {
    if (recorder && recorder.state !== "inactive") {
      recorder.stop();
      setRecorder(null);
    }
  };

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

  // --- Aperçu image ---
  const [previewOpen, setPreviewOpen] = useState(false);
  const [previewSrc, setPreviewSrc] = useState(null);

  const openPreview = (src) => {
    setPreviewSrc(src);
    setPreviewOpen(true);
  };

  const closePreview = () => {
    setPreviewOpen(false);
    setPreviewSrc(null);
  };

  // Fermer avec la touche Échap
  useEffect(() => {
    if (!previewOpen) return;
    const onKey = (e) => {
      if (e.key === "Escape") closePreview();
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [previewOpen]);


  return (
    <div className="min-h-screen bg-gray-100 p-6">
      <h1 className="text-4xl font-bold text-center mb-8 text-gray-800">Journal Émotionnel - MoodCam 📊</h1>

      <div className="max-w-7xl mx-auto space-y-6">
        <div className="bg-white rounded-xl p-6 shadow-md">
          <div className="flex gap-4 flex-wrap items-center">
            <input
              type="text"
              placeholder="🔎 Rechercher (mot, émotion, date...)"
              value={filtre}
              onChange={(e) => setFiltre(e.target.value.toLowerCase())}
              className="flex-1 min-w-64 px-4 py-3 rounded-lg border border-gray-300 focus:outline-none focus:ring-2 focus:ring-blue-500"
            />
            <button
              onClick={exportCSV}
              className="px-6 py-3 bg-blue-600 text-white rounded-lg hover:bg-blue-700 font-semibold transition-colors"
            >
              Exporter en CSV
            </button>
          </div>
        </div>

        <div className="bg-white rounded-xl shadow-md overflow-hidden">
          <div className="overflow-x-auto">
            <table className="w-full">
              <thead className="bg-gray-200 text-black">
                <tr>
                  <th className="px-6 py-3 text-left font-semibold">Date</th>
                  <th className="px-6 py-3 text-left font-semibold">Émotion</th>
                  <th className="px-6 py-3 text-left font-semibold">Image</th>
                  <th className="px-6 py-3 text-left font-semibold">Accélération</th>
                  <th className="px-6 py-3 text-left font-semibold">Freinage</th>
                  <th className="px-6 py-3 text-left font-semibold">Message</th>
                  {/* -- NOUVEAU -- */}
                  <th className="px-6 py-3 text-left font-semibold">Conseils</th>
                </tr>
              </thead>
              <tbody>
                {journalFiltré.map((entry, index) => (
                  <tr key={index} style={{ backgroundColor: getEmotionColor(entry.emotion) }} className="border-b border-gray-200">
                    <td className="px-6 py-4">{entry.timestamp}</td>
                    <td className="px-6 py-4">{entry.emotion}</td>
                    <td className="px-6 py-4">
                      {entry.image_base64 ? (
                        <img
                          src={entry.image_base64}
                          alt="Capture"
                          className="w-20 h-auto rounded-md border border-gray-300 cursor-zoom-in hover:opacity-90 transition"
                          onClick={() => openPreview(entry.image_base64)}
                        />
                      ) : (
                        <span className="text-gray-500 italic">Aucune image</span>
                      )}
                    </td>
                    <td className="px-6 py-4">{entry.acceleration}</td>
                    <td className="px-6 py-4">{entry.freinage}</td>
                    <td className="px-6 py-4">{entry.message}</td>
                    <td className="px-6 py-4">
                      {Array.isArray(entry.conseils) && entry.conseils.length > 0 ? (
                        <div className="flex flex-wrap gap-2">
                          {entry.conseils.slice(0, 4).map((c, i) => (
                            <span
                              key={i}
                              className="px-2 py-0.5 text-xs rounded-full bg-indigo-50 text-indigo-700 ring-1 ring-indigo-200"
                              title={c}
                            >
                              {c.length > 42 ? c.slice(0, 39) + "…" : c}
                            </span>
                          ))}
                        </div>
                      ) : (
                        <span className="text-gray-500 italic">Aucun conseil</span>
                      )}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>

        <div className="bg-white rounded-xl p-6 shadow-md">
          <h2 className="text-2xl font-bold mb-6 text-gray-800">Flux Vidéo</h2>

          <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
            <div className="space-y-4">
              <div className="bg-gray-800 rounded-xl overflow-hidden">
                <div className="bg-gray-700 px-4 py-2 font-semibold text-white">
                  Webcam Locale
                </div>
                <video
                  ref={liveVideoRef}
                  muted
                  playsInline
                  className="w-full bg-black"
                  style={{ aspectRatio: '16/9' }}
                />
              </div>

              <div className="flex gap-2 flex-wrap">
                <button onClick={startCamera} disabled={!!mediaStream} className="px-4 py-2 bg-green-600 text-white rounded-lg hover:bg-green-700 disabled:bg-gray-400 disabled:cursor-not-allowed font-semibold transition-colors">
                  Démarrer caméra
                </button>
                <button onClick={stopCamera} disabled={!mediaStream} className="px-4 py-2 bg-gray-600 text-white rounded-lg hover:bg-gray-700 disabled:bg-gray-400 disabled:cursor-not-allowed font-semibold transition-colors">
                  Arrêter caméra
                </button>
                <button onClick={startRecording} disabled={!mediaStream || isRecording} className="px-4 py-2 bg-red-600 text-white rounded-lg hover:bg-red-700 disabled:bg-gray-400 disabled:cursor-not-allowed font-semibold transition-colors">
                  Enregistrer
                </button>
                <button onClick={stopRecording} disabled={!isRecording} className="px-4 py-2 bg-orange-600 text-white rounded-lg hover:bg-orange-700 disabled:bg-gray-400 disabled:cursor-not-allowed font-semibold transition-colors">
                  Stopper
                </button>
              </div>

              <div className="text-sm text-gray-600">
                Caméra: <span className="font-semibold">{mediaStream ? "ON" : "OFF"}</span> •
                Enregistrement: <span className="font-semibold">{isRecording ? "EN COURS..." : "—"}</span>
              </div>
            </div>

            <div className="bg-gray-800 rounded-xl overflow-hidden">
              <div className="bg-gray-700 px-4 py-2 font-semibold text-white">
                Dashcam Distante (Raspberry Pi)
              </div>
              <img
                src={`${API_DASHCAM}/video`}
                alt="Flux dashcam"
                className="w-full bg-black"
                style={{ aspectRatio: '16/9' }}
                onError={(e) => {
                  e.target.src = "data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' width='640' height='360'%3E%3Crect fill='%23111' width='640' height='360'/%3E%3Ctext x='50%25' y='50%25' font-size='18' fill='%23666' text-anchor='middle' dominant-baseline='middle'%3EFlux vidéo non disponible%3C/text%3E%3C/svg%3E";
                }}
              />
            </div>
          </div>
        </div>

        {clipsLocal.length > 0 && (
          <div className="bg-white rounded-xl p-6 shadow-md">
            <h3 className="text-2xl font-bold mb-6 text-gray-800">En cours (non persistés)</h3>
            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
              {clipsLocal.map((c) => (
                <div key={c.id} className="bg-gray-50 rounded-xl overflow-hidden shadow-md">
                  <video src={c.url} controls className="w-full bg-black" />
                  <div className="p-4 space-y-2">
                    <div className="text-sm text-gray-600">
                      <div className="font-semibold">ID: {c.id}</div>
                      <div>Taille: {(c.size / (1024 * 1024)).toFixed(2)} Mo</div>
                      <div>Créé: {new Date(c.createdAt).toLocaleString()}</div>
                      <div className="text-orange-600 font-semibold">Statut: En upload...</div>
                    </div>
                  </div>
                </div>
              ))}
            </div>
          </div>
        )}

        {clipsServer.length > 0 && (
          <div className="bg-white rounded-xl p-6 shadow-md">
            <div className="flex items-center justify-between mb-6">
              <h3 className="text-2xl font-bold text-gray-800">🎞️ Clips enregistrés</h3>
              <button onClick={loadServerClips} className="px-4 py-2 bg-blue-600 text-white rounded-lg hover:bg-blue-700 font-semibold transition-colors">
                Rafraîchir
              </button>
            </div>

            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
              {clipsServer.map((c) => (
                <div key={c.id} className="bg-gray-50 rounded-xl overflow-hidden shadow-md">
                  <video src={c.url} controls className="w-full bg-black" />
                  <div className="p-4 space-y-2">
                    <div className="text-sm text-gray-600">
                      <div className="font-semibold">Nom: {c.id}</div>
                      <div>Taille: {c.size ? (c.size / (1024 * 1024)).toFixed(2) + " Mo" : "—"}</div>
                      <div>Créé: {new Date(c.createdAt).toLocaleString()}</div>
                    </div>
                    <div className="flex gap-2">
                      <button onClick={() => downloadServerClip(c)} className="flex-1 px-3 py-2 bg-green-600 text-white rounded-lg hover:bg-green-700 text-sm font-semibold transition-colors">
                        Télécharger
                      </button>
                      <button onClick={() => deleteServerClip(c.id)} className="flex-1 px-3 py-2 bg-red-600 text-white rounded-lg hover:bg-red-700 text-sm font-semibold transition-colors">
                        Supprimer
                      </button>
                    </div>
                  </div>
                </div>
              ))}
            </div>
          </div>
        )}

        <DashcamUI apiBase={API_DASHCAM} />

        <Safety apiBase={API_SAFETY} />

        <DriverMonitoring apiBase={API_SAFETY} />

        {previewOpen && (
          <div
            className="fixed inset-0 z-[1000] bg-black/70 backdrop-blur-sm flex items-center justify-center p-4"
            onClick={closePreview}
            aria-modal="true"
            role="dialog"
          >
            {/* Conteneur pour empêcher la fermeture si on clique sur l’image */}
            <div className="relative" onClick={(e) => e.stopPropagation()}>
              <img
                src={previewSrc}
                alt="Aperçu"
                className="max-h-[90vh] max-w-[90vw] rounded-xl shadow-2xl"
              />

              {/* Bouton fermer */}
              <button
                onClick={closePreview}
                aria-label="Fermer l’aperçu"
                className="absolute -top-3 -right-3 bg-white/90 hover:bg-white text-gray-800 rounded-full w-9 h-9 shadow-md flex items-center justify-center font-bold"
                title="Fermer"
              >
                ✕
              </button>
            </div>
          </div>
        )}
      </div>
    </div>
  );
}

export default App;