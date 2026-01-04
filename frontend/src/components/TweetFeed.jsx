import { useEffect, useState } from "react";
import TweetCard from "./TweetCard";
import { fakeTweetsBatch1, fakeTweetsBatch2 } from "../data/fakeTweets";

export default function TweetFeed({ onVerify }) {
  const [tweets, setTweets] = useState(fakeTweetsBatch1);
  const [toggle, setToggle] = useState(false);

  useEffect(() => {
    const interval = setInterval(() => {
      setTweets(toggle ? fakeTweetsBatch1 : fakeTweetsBatch2);
      setToggle(!toggle);
    }, 10000);

    return () => clearInterval(interval);
  }, [toggle]);

  return (
    <div className="space-y-4">
      {tweets.map((tweet) => (
        <TweetCard key={tweet.id} tweet={tweet} onVerify={onVerify} />
      ))}
    </div>
  );
}
