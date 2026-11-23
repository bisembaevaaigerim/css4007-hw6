import csv
from pathlib import Path

from checks import format_checks, run_checks
from gridworld import load_config, load_world
from simulate import run_episode
from solver import TIE_TOLERANCE, extract_policy, q_value, value_iteration

OUTPUT_DIR = Path(__file__).resolve().parent.parent / "outputs"
ARROWS = {"up": "↑", "right": "→", "down": "↓", "left": "←"}
MAPS = {"paper": "paper_map", "extended": "extended_map"}

PAPER_V1 = {
    (1, 1): -1.0, (1, 2): -1.0, (1, 3): 10.0,
    (2, 1): -1.0, (2, 3): -1.0,
    (3, 1): -1.0, (3, 2): -1.0, (3, 3): -1.0, (3, 4): -1.0,
}
PAPER_V2 = {
    (1, 2): {"up": -3.9, "right": 8.0, "down": -3.9, "left": -1.9},
    (1, 3): {"up": 8.0, "right": 10.0, "down": -1.9, "left": -3.9},
    (2, 3): {"up": 5.21, "right": -7.39, "down": -2.71, "left": -0.91},
}
TIE_RULE = (
    f"Tie-breaking: actions whose Q-values differ by less than {TIE_TOLERANCE} are tied. "
    "All tied actions are listed; the chosen arrow is the first in the order up, right, down, left."
)


def pos(state):
    return f"({state[0] + 1},{state[1] + 1})"


def value_map(world, values, decimals=4):
    width = decimals + 6
    lines = []
    for r in range(world.n_rows):
        cells = []
        for c in range(world.n_cols):
            s = (r, c)
            if not world.is_traversable(s):
                cells.append(f"{'#':>{width}}")
            elif world.is_terminal(s):
                cells.append(f"{world.tile(s) + '=0':>{width}}")
            else:
                cells.append(f"{values[s]:{width}.{decimals}f}")
        lines.append("|" + "|".join(cells) + "|")
    return "\n".join(lines)


def policy_map(world, policy):
    lines = []
    for r in range(world.n_rows):
        cells = []
        for c in range(world.n_cols):
            s = (r, c)
            letter = world.tile(s)
            if s in policy:
                cells.append(f" {letter}{ARROWS[policy[s]]} ")
            else:
                cells.append(f" {letter}  ")
        lines.append("|" + "|".join(cells) + "|")
    return "\n".join(lines)


def compare_with_paper(world, result, gamma, positions):
    v1 = result["history"][1]
    v2 = result["history"][2]
    lines = ["V1 for every nonterminal tile (code vs paper)", ""]
    lines.append(value_map(world, v1, 2))
    lines.append("")
    for state in world.states:
        if world.is_terminal(state):
            continue
        key = (state[0] + 1, state[1] + 1)
        paper = PAPER_V1[key]
        match = "OK" if abs(v1[state] - paper) < 1e-9 else "DIFFERENT"
        lines.append(f"{pos(state)} tile {world.tile(state)}: code={v1[state]:.4f} paper={paper:.4f} {match}")

    lines += ["", "V2 at the requested positions (Q-values use V1)", ""]
    for row, col in positions:
        state = (row - 1, col - 1)
        lines.append(f"{pos(state)} tile {world.tile(state)}")
        for action in world.actions:
            code = q_value(world, v1, state, action, gamma)
            paper = PAPER_V2[(row, col)][action]
            match = "OK" if abs(code - paper) < 1e-9 else "DIFFERENT"
            lines.append(f"   {action:>5}: code={code:.4f} paper={paper:.4f} {match}")
        paper_v2 = max(PAPER_V2[(row, col)].values())
        match = "OK" if abs(v2[state] - paper_v2) < 1e-9 else "DIFFERENT"
        lines.append(f"   V2 = max = code={v2[state]:.4f} paper={paper_v2:.4f} {match}")
        lines.append("")
    return "\n".join(lines)


def solve_report(name, world, result, policy, ties, q_table):
    status = "converged" if result["converged"] else "DID NOT CONVERGE (safeguard reached)"
    lines = [
        f"Map: {name}",
        "\n".join("".join(row) for row in world.grid),
        "",
        f"Status: {status}",
        f"Iterations: {result['iterations']}",
        f"Final max change: {result['final_delta']:.3e}",
        "",
        "Final values:",
        value_map(world, result["values"]),
        "",
        "Policy (tile letter + arrow):",
        policy_map(world, policy),
        "",
        TIE_RULE,
    ]
    tied_states = [s for s in ties if len(ties[s]) > 1]
    if tied_states:
        for s in tied_states:
            lines.append(f"  Tie at {pos(s)}: {', '.join(ties[s])}")
    else:
        lines.append("  No ties found.")
    lines += ["", "Q-values with final values:"]
    for s, q in q_table.items():
        parts = "  ".join(f"{a}={q[a]:8.4f}" for a in world.actions)
        lines.append(f"  {pos(s)} {world.tile(s)}: {parts}  best={policy[s]}")
    return "\n".join(lines)


def main():
    OUTPUT_DIR.mkdir(exist_ok=True)
    config = load_config()
    gamma = config["discount"]
    tolerance = config["solver"]["tolerance"]
    max_iterations = config["solver"]["max_iterations"]
    sim = config["simulation"]

    checks_text = format_checks(run_checks(config))
    (OUTPUT_DIR / "partC_rule_checks.txt").write_text(checks_text + "\n", encoding="utf-8")
    print("=== Rule checks ===")
    print(checks_text)

    solved = {}
    for name, key in MAPS.items():
        world = load_world(config, key)
        result = value_iteration(
            world, gamma, tolerance, max_iterations,
            config["initial_value"], config["terminal_value"],
        )
        policy, ties, q_table = extract_policy(world, result["values"], gamma)
        solved[name] = (world, result, policy)

        if name == "paper":
            compare_text = compare_with_paper(world, result, gamma, config["paper_v2_positions"])
            (OUTPUT_DIR / "partC_paper_compare.txt").write_text(compare_text + "\n", encoding="utf-8")
            print("\n=== Compare code with paper ===")
            print(compare_text)

        report = solve_report(name, world, result, policy, ties, q_table)
        (OUTPUT_DIR / f"partC_{name}_map.txt").write_text(report + "\n", encoding="utf-8")
        print(f"\n=== Solution: {name} map ===")
        print(report)

    episode_rows = []
    summary_rows = []
    for name, (world, result, policy) in solved.items():
        for batch_seed in sim["batch_seeds"]:
            counts = {"goal": 0, "hole": 0, "time_limit": 0}
            total_steps = 0
            total_return = 0.0
            for i in range(sim["episodes_per_batch"]):
                seed = 1000 * batch_seed + i
                ep = run_episode(world, policy, seed, gamma, sim["max_steps"])
                counts[ep["outcome"]] += 1
                total_steps += ep["steps"]
                total_return += ep["discounted_return"]
                episode_rows.append({
                    "map": name,
                    "batch_seed": batch_seed,
                    "episode_seed": seed,
                    "steps": ep["steps"],
                    "total_reward": ep["total_reward"],
                    "discounted_return": round(ep["discounted_return"], 6),
                    "outcome": ep["outcome"],
                    "path": " ".join(pos(s) for s in ep["path"]),
                })
            n = sim["episodes_per_batch"]
            summary_rows.append({
                "map": name,
                "batch_seed": batch_seed,
                "goals": counts["goal"],
                "holes": counts["hole"],
                "time_limits": counts["time_limit"],
                "mean_steps": round(total_steps / n, 2),
                "mean_discounted_return": round(total_return / n, 4),
            })

    with open(OUTPUT_DIR / "partC_episodes.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(episode_rows[0].keys()))
        writer.writeheader()
        writer.writerows(episode_rows)

    with open(OUTPUT_DIR / "partC_summary.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(summary_rows[0].keys()))
        writer.writeheader()
        writer.writerows(summary_rows)

    lines = [
        "| Map | Batch seed | Goals /50 | Holes /50 | Time limits /50 | Mean steps | Mean discounted return |",
        "| --- | --- | --- | --- | --- | --- | --- |",
    ]
    for r in summary_rows:
        lines.append(
            f"| {r['map']} | {r['batch_seed']} | {r['goals']} | {r['holes']} | "
            f"{r['time_limits']} | {r['mean_steps']} | {r['mean_discounted_return']} |"
        )
    lines.append("")
    for name, (world, result, policy) in solved.items():
        lines.append(f"V(S) on the {name} map = {result['values'][world.start]:.4f}")
    table = "\n".join(lines)
    (OUTPUT_DIR / "partC_summary.md").write_text(table + "\n", encoding="utf-8")

    print("\n=== Simulation summary ===")
    print(table)
    print(f"\nSaved results to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()