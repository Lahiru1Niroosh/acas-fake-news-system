export default function TweetCard({ tweet, onVerify }) {
  return (
    <div className="border rounded-lg p-4 bg-white shadow">
      <div className="flex items-center gap-3">
        <img
          src={tweet.user.profile_image}
          className="w-10 h-10 rounded-full"
        />
        <div>
          <p className="font-bold">{tweet.user.name}</p>
          <p className="text-gray-500">@{tweet.user.username}</p>
        </div>
      </div>

      {tweet.text && <p className="mt-3">{tweet.text}</p>}

      {tweet.image && (
        <img src={tweet.image} className="mt-3 rounded-lg" />
      )}

      <button
        onClick={() => onVerify(tweet)}
        className="mt-4 bg-blue-500 text-white px-4 py-2 rounded"
      >
        Verify
      </button>
    </div>
  );
}
