export default function VerifyModal({ result, onClose }) {
  if (!result) return null;

  return (
    <div className="fixed inset-0 bg-black bg-opacity-70 flex justify-end">
      <div className="w-96 bg-gray-900 text-white p-6">
        <button onClick={onClose} className="text-gray-400 mb-4">✕</button>

        <h2 className="text-xl font-bold mb-3">Verification Result</h2>

        <p className="mb-2">
          Verdict: <span className="text-green-400">{result.verdict}</span>
        </p>

        <p className="mb-2">Confidence: {result.confidence}%</p>

        <ul className="list-disc list-inside text-sm">
          {result.reasons.map((r, i) => (
            <li key={i}>{r}</li>
          ))}
        </ul>
      </div>
    </div>
  );
}
