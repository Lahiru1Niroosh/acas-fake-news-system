import { useState } from "react";
import TweetFeed from "./components/TweetFeed";
import VerifyModal from "./components/VerifyModal";

export default function App() {
  const [result, setResult] = useState(null);

  const handleVerify = async (tweet) => {
    const payload = {
      tweet: {
        id: tweet.id,
        text: tweet.text,
        image_url: tweet.image
      },
      user: tweet.user
    };

    const res = await fetch("http://localhost:8000/verify", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload)
    });

    const data = await res.json();

    setResult({
      verdict: data.credibility.verdict || "LIKELY TRUE",
      confidence: Math.round(data.credibility.score * 100),
      reasons: [data.credibility.reason]
    });
  };

  return (
    <div className="max-w-2xl mx-auto p-6">
      <h1 className="text-2xl font-bold mb-4">Health News Verification</h1>

      <TweetFeed onVerify={handleVerify} />

      <VerifyModal result={result} onClose={() => setResult(null)} />
    </div>
  );
}
