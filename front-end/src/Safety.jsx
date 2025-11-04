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
    <section>
      <h2>Sécurité & Urgence</h2>

      {error && <div className="alert danger">{error}</div>}

      <div className="card">
        <div className="card-title">Bouton SOS</div>
        <div className="card-actions">
          <button className="btn danger" onClick={triggerSOS}>🚨 Déclencher SOS</button>
        </div>
      </div>

      <div className="card">
        <div className="card-title">Confidentialité</div>
        <div className="card-actions">
          <button className="btn" onClick={togglePrivacy}>
            {privacy.enabled ? "🔒 Désactiver le mode confidentialité" : "🔓 Activer le mode confidentialité"}
          </button>
          {privacy.enabled && <small>Activé depuis: {privacy.locked_at}</small>}
        </div>
      </div>

      <div className="card">
        <div className="card-title">Contacts d’urgence</div>
        <form onSubmit={addContact} className="stack">
          <input required placeholder="Nom" value={form.name} onChange={e=>setForm({...form, name:e.target.value})}/>
          <input placeholder="Email" value={form.email} onChange={e=>setForm({...form, email:e.target.value})}/>
          <input placeholder="Téléphone" value={form.phone} onChange={e=>setForm({...form, phone:e.target.value})}/>
          <button className="btn">Ajouter</button>
        </form>
        <ul>
          {contacts.map(c => (
            <li key={c.id}>
              <b>{c.name}</b> — {c.email || "—"} — {c.phone || "—"} — {c.channels?.join(",")}
              <button className="btn small danger" onClick={()=>delContact(c.id)} style={{ marginLeft: 8 }}>Supprimer</button>
            </li>
          ))}
        </ul>
      </div>

      <div className="card">
        <div className="card-title">Simulateur d’accident</div>
        <div className="grid2">
          <label>g-Force
            <input type="number" step="0.1" value={crash.gForce}
                   onChange={e=>setCrash({...crash, gForce: parseFloat(e.target.value)})}/>
          </label>
          <label>Vitesse (km/h)
            <input type="number" step="1" value={crash.speedKmh}
                   onChange={e=>setCrash({...crash, speedKmh: parseFloat(e.target.value)})}/>
          </label>
        </div>
        <button className="btn" onClick={simulateCrash}>Simuler</button>
      </div>

      <div className="card">
        <div className="card-title">Historique des incidents</div>
        <div className="table-container">
          <table>
            <thead><tr><th>Date</th><th>Type</th><th>Sévérité</th><th>Statut</th><th>Médias</th></tr></thead>
            <tbody>
              {incidents.map(it => (
                <tr key={it.id}>
                  <td>{it.timestamp}</td>
                  <td>{it.type}</td>
                  <td>{it.severity}</td>
                  <td>{it.status}</td>
                  <td>{(it.media||[]).map((m,i)=><a key={i} href={m} target="_blank" rel="noreferrer">preuve {i+1}</a>)}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>
    </section>
  );
}
