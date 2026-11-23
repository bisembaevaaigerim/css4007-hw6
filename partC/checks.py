from gridworld import load_world
from simulate import run_episode
from solver import value_iteration


def pos(state):
    return f"({state[0] + 1},{state[1] + 1})"


def describe(outcomes):
    return ", ".join(f"p={p:.2f} -> {pos(s)} r={r}" for p, s, r, _ in outcomes)


def check(results, name, passed, detail):
    results.append(("PASS" if passed else "FAIL", name, detail))


def run_checks(config):
    results = []
    paper = load_world(config, "paper_map")
    extended = load_world(config, "extended_map")
    gamma = config["discount"]

    out = paper.transitions((2, 0), "left")
    check(results, "Boundary: S (3,1) left stays in place with reward -1",
          out == [(1.0, (2, 0), -1, False)], describe(out))

    out = paper.transitions((0, 0), "up")
    check(results, "Boundary: (1,1) up stays in place with reward -1",
          out == [(1.0, (0, 0), -1, False)], describe(out))

    out = paper.transitions((0, 1), "down")
    check(results, "Wall: Mud (1,2) down hits wall, stays in mud with reward -3",
          out == [(1.0, (0, 1), -3, False)], describe(out))

    out = paper.transitions((1, 0), "right")
    check(results, "Wall: (2,1) right hits wall, stays with reward -1",
          out == [(1.0, (1, 0), -1, False)], describe(out))

    check(results, "Wall: (2,2) is not a state",
          (1, 1) not in paper.states, f"states={len(paper.states)}")

    for action in paper.actions:
        out = paper.transitions((1, 2), action)
        total = sum(p for p, _, _, _ in out)
        check(results, f"Ice (2,3) {action}: probabilities sum to 1",
              abs(total - 1.0) < 1e-12, f"sum={total:.6f}; {describe(out)}")

    out = paper.transitions((1, 2), "left")
    expected = {(1, 2): 0.8, (0, 2): 0.1, (2, 2): 0.1}
    got = {s: p for p, s, _, _ in out}
    check(results, "Ice (2,3) left: 0.8 stay (wall), 0.1 up, 0.1 down",
          all(abs(got.get(s, 0) - p) < 1e-12 for s, p in expected.items()), describe(out))

    out = paper.transitions((0, 0), "right")
    check(results, "Reward: entering Mud gives -3",
          out == [(1.0, (0, 1), -3, False)], describe(out))

    out = extended.transitions((2, 0), "right")
    check(results, "Reward: entering Sand (extended (3,2)) gives -2",
          out == [(1.0, (2, 1), -2, False)], describe(out))

    out = extended.transitions((2, 2), "right")
    check(results, "Reward: entering Water (extended (3,4)) gives -5",
          out == [(1.0, (2, 3), -5, False)], describe(out))

    out = extended.transitions((2, 1), "up")
    check(results, "Sand is deterministic: up from Sand hits wall and stays (-2)",
          out == [(1.0, (2, 1), -2, False)], describe(out))

    out = extended.transitions((2, 3), "right")
    check(results, "Water is deterministic: right from Water hits edge and stays (-5)",
          out == [(1.0, (2, 3), -5, False)], describe(out))

    out = paper.transitions((0, 2), "right")
    check(results, "Reward: entering Goal gives +10 and ends the episode",
          out == [(1.0, (0, 3), 10, True)], describe(out))

    out = paper.transitions((2, 3), "up")
    check(results, "Reward: entering Hole gives -10 and ends the episode",
          out == [(1.0, (1, 3), -10, True)], describe(out))

    no_moves = all(paper.transitions(s, a) == [] for s in [(0, 3), (1, 3)] for a in paper.actions)
    check(results, "Terminal: no transitions or rewards from G or H", no_moves,
          "transitions(G or H, any action) == []")

    solved = value_iteration(paper, gamma, 1e-6, 10000)
    terminal_values = [solved["values"][s] for s in [(0, 3), (1, 3)]]
    check(results, "Terminal: V(G) and V(H) stay 0 after value iteration",
          terminal_values == [0.0, 0.0], f"V(G)={terminal_values[0]}, V(H)={terminal_values[1]}")

    episode = run_episode(paper, {(0, 2): "right"}, 0, gamma, 100, start=(0, 2))
    check(results, "Terminal: simulation stops at G, total reward is exactly +10",
          episode["steps"] == 1 and episode["total_reward"] == 10 and episode["outcome"] == "goal",
          f"steps={episode['steps']}, total_reward={episode['total_reward']}, outcome={episode['outcome']}")

    return results


def format_checks(results):
    lines = [f"[{status}] {name}\n       {detail}" for status, name, detail in results]
    passed = sum(1 for status, _, _ in results if status == "PASS")
    lines.append(f"\n{passed}/{len(results)} checks passed")
    return "\n".join(lines)