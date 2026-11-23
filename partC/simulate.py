import random


def sample_transition(world, state, action, rng):
    outcomes = world.transitions(state, action)
    x = rng.random()
    cumulative = 0.0
    for prob, next_state, reward, terminal in outcomes:
        cumulative += prob
        if x < cumulative:
            return next_state, reward, terminal
    _, next_state, reward, terminal = outcomes[-1]
    return next_state, reward, terminal


def run_episode(world, policy, seed, gamma, max_steps, start=None):
    rng = random.Random(seed)
    state = world.start if start is None else start
    steps = 0
    total_reward = 0.0
    discounted_return = 0.0
    outcome = "time_limit"
    path = [state]

    while steps < max_steps:
        action = policy[state]
        next_state, reward, terminal = sample_transition(world, state, action, rng)
        discounted_return += (gamma ** steps) * reward
        total_reward += reward
        steps += 1
        state = next_state
        path.append(state)
        if terminal:
            outcome = world.rule(state)["meaning"]
            break

    return {
        "steps": steps,
        "total_reward": total_reward,
        "discounted_return": discounted_return,
        "outcome": outcome,
        "path": path,
    }