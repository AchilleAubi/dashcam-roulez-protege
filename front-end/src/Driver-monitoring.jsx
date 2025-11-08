import React, { useEffect, useRef, useState } from "react";

export default function Safety({ apiBase = "http://localhost:8005" }) {
  const [journal, setJournal] = useState([]);
  const [filtre, setFiltre] = useState("");

  useEffect(() => {
    fetch("/journal_driver.json")
      .then((res) => res.json())
      .then((data) => setJournal(data))
      .catch((err) => console.error("Erreur chargement JSON", err));
  }, []);

  const getEmotionColor = (emotion) => {
    switch (emotion) {
      case "happy":
        return "#d4edda";
      case "angry":
        return "#f8d7da";
      case "neutral":
        return "#fdfdfe";
      case "sad":
        return "#fff3cd";
      case "fear":
        return "#d1ecf1";
      default:
        return "#f0f0f0";
    }
  };

  const journalFiltré = journal.filter((entry) =>
    Object.values(entry).some((val) =>
      String(val).toLowerCase().includes(filtre)
    )
  );

  return (
    <div className="bg-white rounded-xl p-6 shadow-md">
      <h3 className="text-lg font-semibold mb-4">Historique du conducteur</h3>
      <div className="bg-white rounded-xl shadow-md overflow-hidden">
        <div className="overflow-x-auto">
          <table className="w-full">
            <thead className="bg-gray-200 text-black">
              <tr>
                <th className="px-6 py-3 text-left font-semibold">Date</th>
                <th className="px-6 py-3 text-left font-semibold">
                  Nom
                </th>
                <th className="px-6 py-3 text-left font-semibold">Image</th>
                <th className="px-6 py-3 text-left font-semibold">Message</th>
              </tr>
            </thead>
            <tbody>
              {journalFiltré.map((entry, index) => (
                <tr
                  key={index}
                  style={{ backgroundColor: getEmotionColor(entry.emotion) }}
                  className="border-b border-gray-200"
                >
                  <td className="px-6 py-4">{entry.timestamp}</td>
                  <td className="px-6 py-4">{entry.name}</td>
                  <td className="px-6 py-4">
                    {entry.image_base64 ? (
                      <img
                        src={entry.image_base64}
                        alt="Capture"
                        className="w-20 h-auto rounded-md border border-gray-300"
                      />
                    ) : (
                      <span className="text-gray-500 italic">Aucune image</span>
                    )}
                  </td>
                  <td className="px-6 py-4">{entry.message}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>
    </div>
  );
}
