import React, { useEffect, useState } from "react";

export default function DashcamUI({ apiBase = "http://127.0.0.1:8003" }) {
  const [alerts, setAlerts] = useState([]);
  const [time, setTime] = useState(new Date());
  const [isConnected, setIsConnected] = useState(false);
  const [stats, setStats] = useState({ 
    recording: false, 
    alerts_count: 0, 
    has_frame: false 
  });

  useEffect(() => {
    const timer = setInterval(() => setTime(new Date()), 1000);
    return () => clearInterval(timer);
  }, []);

  useEffect(() => {
    const fetchAlerts = async () => {
      try {
        const res = await fetch(`${apiBase}/alerts`);
        const data = await res.json();
        setAlerts(data);
        setIsConnected(true);
      } catch (err) {
        console.error("Erreur récupération alertes:", err);
        setIsConnected(false);
      }
    };

    const interval = setInterval(fetchAlerts, 1500);
    fetchAlerts();
    return () => clearInterval(interval);
  }, [apiBase]);

  useEffect(() => {
    const fetchStats = async () => {
      try {
        const res = await fetch(`${apiBase}/stats`);
        const data = await res.json();
        setStats(data || { 
          recording: false, 
          alerts_count: 0, 
          has_frame: false 
        });
      } catch (err) {
        console.error("⚠️ Erreur stats:", err);
      }
    };

    const interval = setInterval(fetchStats, 3000);
    fetchStats();
    return () => clearInterval(interval);
  }, [apiBase]);

  const handleStop = async () => {
    if (!window.confirm("⚠️ Voulez-vous vraiment arrêter le système ?")) {
      return;
    }
    
    try {
      const res = await fetch(`${apiBase}/stop`, { method: "POST" });
      const data = await res.json();
      alert("✅ " + data.message);
    } catch (err) {
      alert("Erreur lors de l'arrêt: " + err.message);
    }
  };

  const connectionDot = isConnected ? "bg-green-500 animate-pulse" : "bg-red-500";

  return (
    <div className="bg-gray-900 rounded-xl p-6 text-white">
      <h2 className="text-2xl font-bold mb-6">Dashcam Intelligente - Système de Sécurité</h2>
      
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        <div className="flex flex-col items-center justify-center p-6 bg-gray-800 rounded-xl">
          <div className="flex items-center gap-2 mb-4">
            <div className={`w-3 h-3 rounded-full ${connectionDot}`}></div>
            <span className="text-sm text-gray-400">
              {isConnected ? "Connecté" : "Déconnecté"}
            </span>
          </div>

          <div className="text-4xl font-bold mb-2">
            {time.toLocaleTimeString("fr-FR")}
          </div>
          <div className="text-sm text-gray-400 mb-6">
            {time.toLocaleDateString("fr-FR", { 
              weekday: "long", 
              day: "numeric", 
              month: "long" 
            })}
          </div>

          <div className="w-full bg-gray-700 rounded-xl p-4 space-y-3">
            <h3 className="text-sm font-semibold text-gray-300 mb-3">Statistiques</h3>
            
            <div className="flex justify-between">
              <span className="text-gray-400">Status:</span>
              <span className={`font-bold ${stats.recording ? 'text-green-400' : 'text-red-400'}`}>
                {stats.recording ? 'REC' : '⏸️ Pause'}
              </span>
            </div>
            
            <div className="flex justify-between">
              <span className="text-gray-400">Vidéo:</span>
              <span className={`font-bold ${stats.has_frame ? 'text-green-400' : 'text-gray-500'}`}>
                {stats.has_frame ? '✓ OK' : '✗ N/A'}
              </span>
            </div>
            
            <div className="flex justify-between">
              <span className="text-gray-400">Alertes:</span>
              <span className="font-bold text-red-400">{stats.alerts_count}</span>
            </div>
          </div>

          {stats.timestamp && (
            <div className="mt-4 text-xs text-gray-500 text-center">
              Dernière mise à jour:<br/>
              {new Date(stats.timestamp).toLocaleString("fr-FR")}
            </div>
          )}

          <button
            onClick={handleStop}
            className="mt-6 w-full bg-red-600 hover:bg-red-700 py-3 rounded-xl font-bold shadow-lg transition-all"
          >
            Arrêter et sauvegarder
          </button>
        </div>

        <div className="lg:col-span-1">
          <div className="bg-gray-800 rounded-xl overflow-hidden">
            <div className="bg-gray-700 px-4 py-2 font-semibold">
              📹 Flux Dashcam en Direct
            </div>
            <div className="relative">
              <img
                src={`${apiBase}/video`}
                alt="Flux dashcam"
                className="w-full h-auto"
                onError={(e) => {
                  e.target.src = "data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' width='640' height='480'%3E%3Crect fill='%23111' width='640' height='480'/%3E%3Ctext x='50%25' y='50%25' font-size='20' fill='%23666' text-anchor='middle' dominant-baseline='middle'%3EFlux vidéo non disponible%3C/text%3E%3C/svg%3E";
                }}
              />
              
              {isConnected && stats.recording && (
                <div className="absolute top-2 right-2 bg-red-600 px-3 py-1 rounded-full flex items-center gap-2 animate-pulse text-sm font-bold">
                  <div className="w-2 h-2 bg-white rounded-full"></div>
                  LIVE
                </div>
              )}

              {isConnected && !stats.recording && (
                <div className="absolute top-2 right-2 bg-yellow-600 px-3 py-1 rounded-full flex items-center gap-2 text-sm font-bold">
                  ⏸️ PAUSE
                </div>
              )}
            </div>
          </div>
        </div>

        <div className="flex flex-col bg-gray-800 rounded-xl p-4">
          <div className="flex items-center justify-between mb-4">
            <h3 className="text-xl font-bold">Alertes</h3>
            <div className="bg-red-600 text-white px-3 py-1 rounded-full text-sm font-bold">
              {alerts.length}
            </div>
          </div>

          <div className="flex-1 overflow-y-auto space-y-3 max-h-96">
            {alerts.length === 0 && (
              <div className="flex flex-col items-center justify-center h-full text-gray-500 py-8">
                <svg className="w-16 h-16 mb-3 opacity-50" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 12l2 2 4-4m6 2a9 9 0 11-18 0 9 9 0 0118 0z" />
                </svg>
                <p className="text-center">Aucune alerte détectée</p>
              </div>
            )}

            {alerts.map((alert, index) => (
              <div
                key={index}
                className={`p-3 rounded-lg shadow-lg transition-all hover:scale-105 ${
                  alert.type === "frontal_collision_risk" || alert.type === "pedestrian_on_road"
                    ? "bg-red-700"
                    : alert.type === "lateral_close"
                    ? "bg-orange-700"
                    : "bg-blue-700"
                }`}
              >
                <div className="flex items-start gap-3">
                  <div className="text-xl">
                    {alert.type === "frontal_collision_risk" && "🚨"}
                    {alert.type === "pedestrian_on_road" && "🚶"}
                    {alert.type === "lateral_close" && "⚠️"}
                    {alert.type === "rear_approach" && "🔴"}
                    {alert.type === "zigzag" && "〰️"}
                    {alert.type === "brake_hard" && "⛔"}
                    {!["frontal_collision_risk", "pedestrian_on_road", "lateral_close", "rear_approach", "zigzag", "brake_hard"].includes(alert.type) && "⚡"}
                  </div>
                  
                  <div className="flex-1">
                    <div className="font-bold text-sm mb-1">{alert.message}</div>
                    <div className="text-xs text-gray-200 opacity-90">
                      {alert.time}
                    </div>
                    {alert.track_id && (
                      <div className="text-xs text-gray-300 mt-1">
                        Track ID: {alert.track_id}
                      </div>
                    )}
                  </div>
                </div>
              </div>
            ))}
          </div>
        </div>
      </div>
    </div>
  );
}