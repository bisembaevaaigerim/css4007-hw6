import csv
import random
from pathlib import Path

from policies import fixed_policy, random_policy
from runner import MAX_STEPS, make_env, run_episode

BATCH_SEEDS = [11, 22, 33]
EPISODES_PER_BATCH = 50
SLIPPERY_OPTIONS = [False, True]
POLICIES = {"random": random_policy, "fixed": fixed_policy}
OUTPUT_DIR = Path(__file__).resolve().parent.parent / "outputs"


def episode_seeds(batch_seed):
    return [1000 * batch_seed + i for i in range(EPISODES_PER_BATCH)]


def trace_text(condition, batch_seed, episode_seed, result):
    header = (
        f"condition={condition} batch_seed={batch_seed} episode_seed={episode_seed}\n"
        f"outcome={result['outcome']} steps={result['steps']} "
        f"total_reward={result['total_reward']}\n\n"
    )
    return header + result["trace"]


def main():
    OUTPUT_DIR.mkdir(exist_ok=True)
    episode_rows = []
    summary_rows = []
    example = None
    success = None
    fail = None

    for slippery in SLIPPERY_OPTIONS:
        env = make_env(slippery)
        for policy_name, policy in POLICIES.items():
            condition = f"slippery={slippery}, policy={policy_name}"
            for batch_seed in BATCH_SEEDS:
                rng = random.Random(batch_seed)
                counts = {"goal": 0, "hole": 0, "time_limit": 0}
                total_steps = 0

                for episode_seed in episode_seeds(batch_seed):
                    result = run_episode(env, policy, episode_seed, rng, MAX_STEPS)
                    counts[result["outcome"]] += 1
                    total_steps += result["steps"]

                    episode_rows.append({
                        "condition": condition,
                        "slippery": slippery,
                        "policy": policy_name,
                        "batch_seed": batch_seed,
                        "episode_seed": episode_seed,
                        "steps": result["steps"],
                        "total_reward": result["total_reward"],
                        "outcome": result["outcome"],
                    })

                    text = trace_text(condition, batch_seed, episode_seed, result)
                    if example is None:
                        example = text
                    if success is None and result["outcome"] == "goal":
                        success = text
                    if fail is None and result["outcome"] != "goal":
                        fail = text

                summary_rows.append({
                    "slippery": "Yes" if slippery else "No",
                    "policy": policy_name,
                    "batch_seed": batch_seed,
                    "goals": counts["goal"],
                    "holes": counts["hole"],
                    "time_limits": counts["time_limit"],
                    "success_pct": round(100 * counts["goal"] / EPISODES_PER_BATCH, 1),
                    "mean_steps": round(total_steps / EPISODES_PER_BATCH, 2),
                })
        env.close()

    with open(OUTPUT_DIR / "partB_episodes.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(episode_rows[0].keys()))
        writer.writeheader()
        writer.writerows(episode_rows)

    with open(OUTPUT_DIR / "partB_summary.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(summary_rows[0].keys()))
        writer.writeheader()
        writer.writerows(summary_rows)

    lines = [
        "| Slippery? | Policy | Batch seed | Goals /50 | Holes /50 | Time limits /50 | Success % | Mean steps |",
        "| --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for r in summary_rows:
        lines.append(
            f"| {r['slippery']} | {r['policy']} | {r['batch_seed']} | {r['goals']} | "
            f"{r['holes']} | {r['time_limits']} | {r['success_pct']} | {r['mean_steps']} |"
        )
    table = "\n".join(lines)
    (OUTPUT_DIR / "partB_summary.md").write_text(table + "\n", encoding="utf-8")

    (OUTPUT_DIR / "partB_example_episode.txt").write_text(example, encoding="utf-8")
    (OUTPUT_DIR / "partB_trace_success.txt").write_text(
        success if success else "No successful episode was observed.\n", encoding="utf-8"
    )
    (OUTPUT_DIR / "partB_trace_fail.txt").write_text(
        fail if fail else "No failed episode was observed.\n", encoding="utf-8"
    )

    print(table)
    print(f"\nSaved {len(episode_rows)} episode rows to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()