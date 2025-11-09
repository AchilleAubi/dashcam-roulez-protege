import React, { useEffect, useMemo, useRef, useState } from "react";

export default function Safety({ apiBase = "http://localhost:8005" }) {
  const [journal, setJournal] = useState([]);
  const [filtre, setFiltre] = useState("");
  const [soundEnabled, setSoundEnabled] = useState(false);

  // DEMO: injecter une alerte 15s après le chargement
  const DEMO_AFTER_MS = 15000; // 15 secondes
  const DEMO_ENABLED = true; // passe à false pour désactiver la démo

  const lastNotifiedTsRef = useRef(null);
  const audioCtxRef = useRef(null);

  // --- CHARGEMENT DU JOURNAL ---
  useEffect(() => {
    fetch("/journal_driver.json")
      .then((res) => res.json())
      .then((data) => setJournal(Array.isArray(data) ? data : []))
      .catch((err) => console.error("Erreur chargement JSON", err));
  }, []);

  // --- MODE DÉMO: ajoute une entrée critique après 15s ---
  useEffect(() => {
    if (!DEMO_ENABLED) return;
    const t = setTimeout(() => {
      const now = new Date();
      const pad = (n) => `${n}`.padStart(2, "0");
      const ts = `${now.getFullYear()}-${pad(now.getMonth() + 1)}-${pad(
        now.getDate()
      )} ${pad(now.getHours())}:${pad(now.getMinutes())}:${pad(
        now.getSeconds()
      )}`;

      const demoEntry = {
        timestamp: ts,
        name: "Détection de téléphone",
        message: "Téléphone détecté en conduite",
        image_base64: null, // tu peux mettre une base64 si tu veux afficher une image
        image_size_bytes: 0,
        alert_type: "Sécurité",
        alert_vocale:
          "Attention, téléphone détecté. Gardez les mains sur le volant.",
        niveau_risque: "Élevé",
        source: "Caméra faciale",
      };

      // On insère en tête pour qu’elle apparaisse en haut
      setJournal((prev) => [demoEntry, ...prev]);
    }, DEMO_AFTER_MS);

    return () => clearTimeout(t);
  }, [DEMO_ENABLED]);

  // --- UTILS ---
  const normalize = (s) => (s || "").toString().toLowerCase();

  const isCritical = (entry) => {
    const n = normalize(entry.name);
    const m = normalize(entry.message);
    return (
      n.includes("téléphone") ||
      n.includes("telephone") ||
      n.includes("somnolence") ||
      m.includes("téléphone") ||
      m.includes("telephone") ||
      m.includes("somnolence") ||
      m.includes("endormissement") ||
      m.includes("fatigue")
    );
  };

  const rowBg = (niveau) => {
    const v = normalize(niveau);
    if (v.includes("élev") || v.includes("elev")) return "#fdecea";
    if (v.includes("mod")) return "#fff4e5";
    if (v.includes("faib")) return "#edf7ed";
    return "#f7f7f7";
  };

  const journalFiltre = useMemo(() => {
    const f = normalize(filtre);
    if (!f) return journal;
    return journal.filter((e) =>
      Object.values(e).some((val) => normalize(val).includes(f))
    );
  }, [journal, filtre]);

  // --- SON ---
  // Remplace ENTIEREMENT ta fonction playBeep par celle-ci
  const playBeep = () => {
    try {
      if (!audioCtxRef.current) {
        audioCtxRef.current = new (window.AudioContext ||
          window.webkitAudioContext)();
      }
      const ctx = audioCtxRef.current;

      // Oscillateur principal (glissando descendant)
      const osc = ctx.createOscillator();
      const gain = ctx.createGain();

      // LFO pour un léger trémolo (effet sirène)
      const lfo = ctx.createOscillator();
      const lfoGain = ctx.createGain();

      // Forme et courbe de fréquence (sirène courte)
      osc.type = "triangle";
      osc.frequency.setValueAtTime(1100, ctx.currentTime); // démarre assez aigu
      osc.frequency.exponentialRampToValueAtTime(520, ctx.currentTime + 0.65); // descend rapidement

      // Trémolo ~8–9 Hz
      lfo.type = "sine";
      lfo.frequency.setValueAtTime(9, ctx.currentTime);
      lfoGain.gain.setValueAtTime(0.25, ctx.currentTime); // profondeur du trémolo

      // Enveloppe d’amplitude (attaque courte, release rapide)
      gain.gain.setValueAtTime(0.0001, ctx.currentTime);
      gain.gain.exponentialRampToValueAtTime(0.32, ctx.currentTime + 0.05);
      gain.gain.exponentialRampToValueAtTime(0.0001, ctx.currentTime + 0.72);

      // Routing audio
      lfo.connect(lfoGain);
      lfoGain.connect(gain.gain);
      osc.connect(gain);
      gain.connect(ctx.destination);

      // Lecture
      osc.start();
      lfo.start();

      // Stop propre
      const stopAt = ctx.currentTime + 0.75; // ~750 ms
      osc.stop(stopAt);
      lfo.stop(stopAt);
    } catch (e) {
      console.warn("Impossible de jouer le son d'alerte :", e);
    }
  };

  useEffect(() => {
    if (!soundEnabled || !journal.length) return;

    const critical = [...journal].filter(isCritical);
    if (!critical.length) return;

    const parseTs = (e) => {
      const d = new Date(e.timestamp?.replace(" ", "T") || "");
      return isNaN(d.getTime()) ? null : d.getTime();
    };

    critical.sort((a, b) => {
      const ta = parseTs(a);
      const tb = parseTs(b);
      if (ta && tb) return tb - ta;
      return 0;
    });

    const latest = critical[0];
    const latestKey =
      parseTs(latest)?.toString() ||
      `${latest.name}|${latest.message}|${latest.source}`;

    if (latestKey && lastNotifiedTsRef.current !== latestKey) {
      lastNotifiedTsRef.current = latestKey;
      playBeep();
    }
  }, [journal, soundEnabled]);

  // --- RENDU ---
  return (
    <div className="bg-white rounded-xl p-6 shadow-md">
      <div className="flex items-center justify-between mb-4">
        <h3 className="text-lg font-semibold">Historique du conducteur</h3>

        <div className="flex items-center gap-3">
          <input
            type="text"
            placeholder="Rechercher…"
            value={filtre}
            onChange={(e) => setFiltre(e.target.value)}
            className="border border-gray-300 rounded-md px-3 py-1 text-sm"
          />
          <button
            type="button"
            onClick={() => {
              if (!soundEnabled && !audioCtxRef.current) {
                audioCtxRef.current = new (window.AudioContext ||
                  window.webkitAudioContext)();
              }
              setSoundEnabled((v) => !v);
            }}
            className={`px-3 py-1 rounded-md text-sm ${
              soundEnabled
                ? "bg-green-600 text-white"
                : "bg-gray-200 text-gray-800"
            }`}
            title="Active le bip pour téléphone/somnolence"
          >
            {soundEnabled ? "Son activé" : "Activer le son"}
          </button>
        </div>
      </div>

      <div className="bg-white rounded-xl shadow-md overflow-hidden">
        <div className="overflow-x-auto">
          <table className="w-full">
            <thead className="bg-gray-200 text-black">
              <tr>
                <th className="px-6 py-3 text-left font-semibold">Date</th>
                <th className="px-6 py-3 text-left font-semibold">Nom</th>
                <th className="px-6 py-3 text-left font-semibold">Image</th>
                <th className="px-6 py-3 text-left font-semibold">Message</th>
                <th className="px-6 py-3 text-left font-semibold">Risque</th>
                <th className="px-6 py-3 text-left font-semibold">Source</th>
              </tr>
            </thead>
            <tbody>
              {journalFiltre.map((entry, index) => {
                const critical = isCritical(entry);
                return (
                  <tr
                    key={index}
                    style={{ backgroundColor: rowBg(entry.niveau_risque) }}
                    className={`border-b border-gray-200 ${
                      critical ? "outline outline-1 outline-red-300" : ""
                    }`}
                  >
                    <td className="px-6 py-4">{entry.timestamp}</td>
                    <td className="px-6 py-4">
                      <div className="flex items-center gap-2">
                        {critical && (
                          <span
                            className="inline-block w-2 h-2 rounded-full bg-red-500"
                            title="Alerte critique (bip)"
                          />
                        )}
                        {entry.name}
                      </div>
                    </td>
                    <td className="px-6 py-4">
                      {entry.image_base64 ? (
                        <img
                          src={entry.image_base64}
                          alt="Capture"
                          className="w-20 h-auto rounded-md border border-gray-300"
                        />
                      ) : (
                        <span className="text-gray-500 italic">
                          Aucune image
                        </span>
                      )}
                    </td>
                    <td className="px-6 py-4">{entry.message}</td>
                    <td className="px-6 py-4">
                      <span
                        className={`px-2 py-1 rounded text-xs font-medium ${
                          normalize(entry.niveau_risque).includes("élev") ||
                          normalize(entry.niveau_risque).includes("elev")
                            ? "bg-red-100 text-red-700"
                            : normalize(entry.niveau_risque).includes("mod")
                            ? "bg-orange-100 text-orange-700"
                            : "bg-green-100 text-green-700"
                        }`}
                      >
                        {entry.niveau_risque || "—"}
                      </span>
                    </td>
                    <td className="px-6 py-4">{entry.source || "—"}</td>
                  </tr>
                );
              })}
              {!journalFiltre.length && (
                <tr>
                  <td
                    className="px-6 py-6 text-center text-gray-500"
                    colSpan={6}
                  >
                    Aucun résultat
                  </td>
                </tr>
              )}
            </tbody>
          </table>
        </div>

        <div className="p-3 text-xs text-gray-500">
          Démo : une alerte “Détection de téléphone” sera injectée 15s après le
          chargement. Active le son pour entendre le bip.
        </div>
      </div>
    </div>
  );
}
