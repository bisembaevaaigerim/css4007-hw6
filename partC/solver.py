TIE_TOLERANCE = 1e-9


def q_value(world, values, state, action, gamma):
    total = 0.0
    for prob, next_state, reward, terminal in world.transitions(state, action):
        future = 0.0 if terminal else values[next_state]
        total += prob * (reward + gamma * future)
    return total


def bellman_update(world, values, gamma, terminal_value):
    new_values = {}
    for state in world.states:
        if world.is_terminal(state):
            new_values[state] = terminal_value
        else:
            new_values[state] = max(
                q_value(world, values, state, action, gamma) for action in world.actions
            )
    return new_values


def value_iteration(world, gamma, tolerance, max_iterations, initial_value=0.0,
                    terminal_value=0.0, keep_rounds=3):
    values = {
        s: terminal_value if world.is_terminal(s) else initial_value for s in world.states
    }
    history = [dict(values)]
    iterations = 0
    delta = None
    converged = False

    while iterations < max_iterations:
        new_values = bellman_update(world, values, gamma, terminal_value)
        delta = max(abs(new_values[s] - values[s]) for s in world.states)
        values = new_values
        iterations += 1
        if len(history) < keep_rounds:
            history.append(dict(values))
        if delta < tolerance:
            converged = True
            break

    return {
        "values": values,
        "iterations": iterations,
        "final_delta": delta,
        "converged": converged,
        "history": history,
    }


def extract_policy(world, values, gamma):
    policy = {}
    ties = {}
    q_table = {}
    for state in world.states:
        if world.is_terminal(state):
            continue
        q = {a: q_value(world, values, state, a, gamma) for a in world.actions}
        best = max(q.values())
        tied = [a for a in world.actions if best - q[a] < TIE_TOLERANCE]
        policy[state] = tied[0]
        ties[state] = tied
        q_table[state] = q
    return policy, ties, q_table