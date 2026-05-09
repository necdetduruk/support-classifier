"""
Load generator for the banking77 inference service.

Sends mixed in-distribution and out-of-distribution traffic to the live endpoint
to demonstrate the drift detection signal. In-distribution queries score ~0.30+
on top-1 confidence; OOD queries collapse to <0.10.

Usage:
    python scripts/generate_traffic.py --url $SERVICE_URL --rounds 10 --ood-ratio 0.5
"""
import argparse
import json
import random
import time
import urllib.error
import urllib.request

IN_DIST_QUERIES = [
    "my card is broken and I need a new one",
    "I lost my pin",
    "transfer fee charged on my last transaction",
    "what is my current balance",
    "cancel a pending transfer",
    "my account is frozen, why",
    "where is my refund",
    "international payment failed",
    "how do I activate my new card",
    "card payment was declined",
    "I want to close my account",
    "change my registered phone number",
    "exchange rate for euros today",
    "top up my account by direct deposit",
    "extra charge on my statement",
    "ATM withdrawal showing twice",
    "report card lost or stolen",
    "verify my identity to unlock",
    "block a contactless payment",
    "set up direct debit for utilities",
]

OOD_QUERIES = [
    "hello there",
    "what is the weather today",
    "tell me a joke",
    "the sky is blue",
    "i love pizza",
    "what time is it",
    "good morning sunshine",
    "random gibberish text",
    "lorem ipsum dolor sit amet",
    "the quick brown fox jumps over the lazy dog",
    "can you sing me a song",
    "who won the world cup",
    "explain photosynthesis",
    "recommend a good movie",
    "how do I bake bread",
]


def predict(url: str, text: str, top_k: int = 1) -> dict:
    body = json.dumps({"text": text, "top_k": top_k}).encode("utf-8")
    req = urllib.request.Request(
        f"{url.rstrip('/')}/predict",
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=10) as resp:
        return json.loads(resp.read().decode("utf-8"))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", required=True, help="Service base URL")
    parser.add_argument("--rounds", type=int, default=20)
    parser.add_argument("--per-round", type=int, default=10)
    parser.add_argument("--ood-ratio", type=float, default=0.5,
                        help="Fraction of OOD queries (0.0-1.0)")
    parser.add_argument("--sleep", type=float, default=2.0)
    args = parser.parse_args()

    if not 0.0 <= args.ood_ratio <= 1.0:
        parser.error("--ood-ratio must be between 0.0 and 1.0")

    in_dist_n = round(args.per_round * (1 - args.ood_ratio))
    ood_n = args.per_round - in_dist_n

    print(f"Target: {args.rounds} rounds x {args.per_round} requests "
          f"({in_dist_n} in-dist + {ood_n} OOD per round)")
    print(f"URL: {args.url}\n")

    total_in, total_ood = 0, 0
    in_dist_scores, ood_scores = [], []
    errors = 0

    for r in range(1, args.rounds + 1):
        batch = (random.sample(IN_DIST_QUERIES, k=min(in_dist_n, len(IN_DIST_QUERIES)))
                 + random.sample(OOD_QUERIES, k=min(ood_n, len(OOD_QUERIES))))
        random.shuffle(batch)

        for text in batch:
            is_ood = text in OOD_QUERIES
            try:
                resp = predict(args.url, text)
                score = resp["predictions"][0]["score"]
                if is_ood:
                    ood_scores.append(score)
                    total_ood += 1
                else:
                    in_dist_scores.append(score)
                    total_in += 1
            except (urllib.error.URLError, KeyError, json.JSONDecodeError) as e:
                errors += 1
                print(f"  error: {e}")

        in_avg = sum(in_dist_scores) / len(in_dist_scores) if in_dist_scores else 0.0
        ood_avg = sum(ood_scores) / len(ood_scores) if ood_scores else 0.0
        print(f"Round {r:>3}/{args.rounds}  "
              f"in-dist mean: {in_avg:.3f} (n={total_in})  "
              f"OOD mean: {ood_avg:.3f} (n={total_ood})  "
              f"errors: {errors}")

        time.sleep(args.sleep)

    print("\n=== Summary ===")
    print(f"Total requests:       {total_in + total_ood}")
    print(f"Errors:               {errors}")
    if in_dist_scores:
        print(f"In-distribution mean: {sum(in_dist_scores)/len(in_dist_scores):.3f}  "
              f"(n={len(in_dist_scores)})")
    if ood_scores:
        print(f"OOD mean:             {sum(ood_scores)/len(ood_scores):.3f}  "
              f"(n={len(ood_scores)})")


if __name__ == "__main__":
    main()
