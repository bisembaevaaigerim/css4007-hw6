LEFT, DOWN, RIGHT, UP = 0, 1, 2, 3
ACTION_NAMES = {LEFT: "left", DOWN: "down", RIGHT: "right", UP: "up"}

FIXED_POLICY = {
    0: DOWN,
    1: LEFT,
    2: DOWN,
    3: LEFT,
    4: DOWN,
    6: DOWN,
    8: RIGHT,
    9: DOWN,
    10: DOWN,
    13: RIGHT,
    14: RIGHT,
}


def fixed_policy(state, rng):
    return FIXED_POLICY[state]


def random_policy(state, rng):
    return rng.randrange(4)