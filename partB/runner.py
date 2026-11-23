import gymnasium as gym

from policies import ACTION_NAMES

MAX_STEPS = 100


def make_env(is_slippery):
    return gym.make(
        "FrozenLake-v1",
        map_name="4x4",
        is_slippery=is_slippery,
        render_mode="ansi",
    )


def tile_of(env, state):
    desc = env.unwrapped.desc
    ncol = desc.shape[1]
    return desc[state // ncol][state % ncol].decode()


def run_episode(env, policy, seed, rng, max_steps=MAX_STEPS):
    state, _ = env.reset(seed=seed)
    total_reward = 0.0
    steps = 0
    terminated = False
    frames = [f"reset seed={seed} state={state}\n{env.render()}"]

    while steps < max_steps:
        action = policy(state, rng)
        next_state, reward, terminated, truncated, _ = env.step(action)
        steps += 1
        total_reward += reward
        frames.append(
            f"step {steps}: state={state} action={ACTION_NAMES[action]} "
            f"-> state={next_state} reward={reward}\n{env.render()}"
        )
        state = next_state
        if terminated or truncated:
            break

    if terminated:
        outcome = "goal" if tile_of(env, state) == "G" else "hole"
    else:
        outcome = "time_limit"

    return {
        "steps": steps,
        "total_reward": total_reward,
        "outcome": outcome,
        "final_state": state,
        "trace": "\n".join(frames),
    }