import { useState } from "react"
import Feed from "../components/Feed"
import VerifyModal from "../components/VerifyModal"

export default function Home() {
  const [result, setResult] = useState(null)

  const handleVerify = () => {
    // TEMP dummy result (backend later)
    setResult({
      verdict: "LIKELY TRUE",
      confidence: 78,
      reasons: [
        "Health-related wording",
        "No emotional manipulation",
        "Consistent image context"
      ]
    })
  }

  return (
    <div className="max-w-xl mx-auto py-6">
      <h1 className="text-2xl font-bold text-white mb-4">
        Health Tweet Verification
      </h1>

      <Feed onVerify={handleVerify} />
      <VerifyModal result={result} onClose={() => setResult(null)} />
    </div>
  )
}
