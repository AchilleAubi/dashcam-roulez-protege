import React, { useEffect, useState } from "react";

export default function Safety({ apiBase = "http://localhost:8005" }) {
  const [contacts, setContacts] = useState([]);
  const [form, setForm] = useState({ name: "", email: "", phone: "", channels: ["email"] });
  const [incidents, setIncidents] = useState([]);
  const [privacy, setPrivacy] = useState({ enabled: false, locked_at: null });
  const [crash, setCrash] = useState({ gForce: 1.0, speedKmh: 0 });
  const [error, setError] = useState("");

  const loadAll = async () => {
    try {
      setError("");
      const [c, i, p] = await Promise.all([
        fetch(`${apiBase}/api/emergency/contacts`).then(r=>r.json()),
        fetch(`${apiBase}/api/emergency/incidents`).then(r=>r.json()),
        fetch(`${apiBase}/api/privacy/status`).then(r=>r.json()),
      ]);
      setContacts(c); setIncidents(i); setPrivacy(p);
    } catch (e) {
      console.error("Safety loadAll error:", e);
      setError("Impossible de joindre le service Safety.");
    }
  };

  useEffect(() => { loadAll(); }, []);

  const addContact = async (e) => {
    e.preventDefault();
    await fetch(`${apiBase}/api/emergency/contacts`, {
      method: "POST", headers: { "Content-Type": "application/json" },
      body: JSON.stringify(form)
    });
    setForm({ name: "", email: "", phone: "", channels: ["email"] });
    loadAll();
  };

  const delContact = async (id) => {
    await fetch(`${apiBase}/api/emergency/contacts/${id}`, { method: "DELETE" });
    loadAll();
  };

  const triggerSOS = async () => {
    await fetch(`${apiBase}/api/emergency/sos`, {
      method: "POST", headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ note: "SOS manuel depuis UI" })
    });
    loadAll();
  };

  const simulateCrash = async () => {
    await fetch(`${apiBase}/api/emergency/crash-event`, {
      method: "POST", headers: { "Content-Type": "application/json" },
      body: JSON.stringify(crash)
    });
    loadAll();
  };

  const togglePrivacy = async () => {
    await fetch(`${apiBase}/api/privacy/lock`, {
      method: "POST", headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ enabled: !privacy.enabled })
    });
    loadAll();
  };

  return (
    <div className="space-y-6">
      <h2 className="text-2xl font-bold text-gray-800">Sécurité & Urgence</h2>

      {error && <div className="bg-red-100 border border-red-400 text-red-700 px-4 py-3 rounded">{error}</div>}

      <div className="bg-white rounded-xl p-6 shadow-md">
        <h3 className="text-lg font-semibold mb-4">Bouton SOS</h3>
        <button onClick={triggerSOS} className="w-full px-6 py-4 bg-red-600 text-white rounded-lg hover:bg-red-700 font-bold text-lg shadow-lg transition-all">
          Déclencher SOS
        </button>
      </div>

      <div className="bg-white rounded-xl p-6 shadow-md">
        <h3 className="text-lg font-semibold mb-4">Confidentialité</h3>
        <button onClick={togglePrivacy} className={`w-full px-6 py-4 rounded-lg font-semibold transition-all ${
          privacy.enabled ? "bg-green-600 text-white hover:bg-green-700" : "bg-gray-200 text-gray-700 hover:bg-gray-300"
        }`}>
          {privacy.enabled ? "🔒 Désactiver le mode confidentialité" : "🔓 Activer le mode confidentialité"}
        </button>
        {privacy.enabled && <div className="text-sm text-gray-600 mt-2">Activé depuis: {privacy.locked_at}</div>}
      </div>

      <div className="bg-white rounded-xl p-6 shadow-md">
        <h3 className="text-lg font-semibold mb-4">Contacts d'urgence</h3>
        <form onSubmit={addContact} className="space-y-3 mb-4">
          <input required placeholder="Nom" value={form.name} onChange={e=>setForm({...form, name:e.target.value})}
                 className="w-full px-4 py-2 border border-gray-300 rounded-lg focus:outline-none focus:ring-2 focus:ring-blue-500"/>
          <input placeholder="Email" value={form.email} onChange={e=>setForm({...form, email:e.target.value})}
                 className="w-full px-4 py-2 border border-gray-300 rounded-lg focus:outline-none focus:ring-2 focus:ring-blue-500"/>
          <input placeholder="Téléphone" value={form.phone} onChange={e=>setForm({...form, phone:e.target.value})}
                 className="w-full px-4 py-2 border border-gray-300 rounded-lg focus:outline-none focus:ring-2 focus:ring-blue-500"/>
          <button className="w-full px-6 py-2 bg-blue-600 text-white rounded-lg hover:bg-blue-700 font-semibold">Ajouter</button>
        </form>
        <div className="space-y-2">
          {contacts.map(c => (
            <div key={c.id} className="flex items-center justify-between p-3 bg-gray-50 rounded-lg">
              <div>
                <div className="font-semibold">{c.name}</div>
                <div className="text-sm text-gray-600">{c.email || "—"} — {c.phone || "—"} — {c.channels?.join(",")}</div>
              </div>
              <button onClick={()=>delContact(c.id)} className="px-3 py-1 bg-red-500 text-white rounded hover:bg-red-600 text-sm font-semibold">
                Supprimer
              </button>
            </div>
          ))}
        </div>
      </div>

      <div className="bg-white rounded-xl p-6 shadow-md">
        <h3 className="text-lg font-semibold mb-4">Simulateur d'accident</h3>
        <div className="grid grid-cols-1 md:grid-cols-2 gap-4 mb-4">
          <div>
            <label className="block text-sm font-semibold mb-2">g-Force</label>
            <input type="number" step="0.1" value={crash.gForce}
                   onChange={e=>setCrash({...crash, gForce: parseFloat(e.target.value)})}
                   className="w-full px-4 py-2 border border-gray-300 rounded-lg focus:outline-none focus:ring-2 focus:ring-blue-500"/>
          </div>
          <div>
            <label className="block text-sm font-semibold mb-2">Vitesse (km/h)</label>
            <input type="number" step="1" value={crash.speedKmh}
                   onChange={e=>setCrash({...crash, speedKmh: parseFloat(e.target.value)})}
                   className="w-full px-4 py-2 border border-gray-300 rounded-lg focus:outline-none focus:ring-2 focus:ring-blue-500"/>
          </div>
        </div>
        <button onClick={simulateCrash} className="w-full px-6 py-3 bg-orange-600 text-white rounded-lg hover:bg-orange-700 font-bold transition-colors">
          Simuler
        </button>
      </div>

      <div className="bg-white rounded-xl p-6 shadow-md">
        <h3 className="text-lg font-semibold mb-4">Historique des incidents</h3>
        <div className="overflow-x-auto">
          <table className="w-full">
            <thead className="bg-gray-200 text-black">
              <tr>
                <th className="px-6 py-3 text-left font-semibold">Date</th>
                <th className="px-6 py-3 text-left font-semibold">Type</th>
                <th className="px-6 py-3 text-left font-semibold">Sévérité</th>
                <th className="px-6 py-3 text-left font-semibold">Statut</th>
                <th className="px-6 py-3 text-left font-semibold">Médias</th>
              </tr>
            </thead>
            <tbody>
              {incidents.map(it => (
                <tr key={it.id} className="border-b border-gray-200 hover:bg-gray-50">
                  <td className="px-6 py-4">{it.timestamp}</td>
                  <td className="px-6 py-4 font-semibold">{it.type}</td>
                  <td className="px-6 py-4">{it.severity}</td>
                  <td className="px-6 py-4">{it.status}</td>
                  <td className="px-6 py-4">
                    {(it.media||[]).map((m,i)=>(
                      <a key={i} href={m} target="_blank" rel="noreferrer" className="text-blue-600 hover:underline mr-2">
                        preuve {i+1}
                      </a>
                    ))}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>
    </div>
  );
}