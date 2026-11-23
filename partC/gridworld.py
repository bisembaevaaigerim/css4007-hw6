import json
from pathlib import Path

DATA_PATH = Path(__file__).resolve().parent.parent / "data" / "maps.json"

MOVES = {"up": (-1, 0), "right": (0, 1), "down": (1, 0), "left": (0, -1)}
PERPENDICULAR = {
    "up": ("left", "right"),
    "down": ("left", "right"),
    "left": ("up", "down"),
    "right": ("up", "down"),
}


def load_config(path=DATA_PATH):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


class GridWorld:
    def __init__(self, rows, tiles, actions):
        self.grid = [list(row) for row in rows]
        self.n_rows = len(self.grid)
        self.n_cols = len(self.grid[0])
        self.tiles = tiles
        self.actions = list(actions)
        for row in self.grid:
            for letter in row:
                if letter not in tiles:
                    raise ValueError(f"Unknown tile '{letter}'. Add it to the tile rules.")
        self.states = [
            (r, c)
            for r in range(self.n_rows)
            for c in range(self.n_cols)
            if self.is_traversable((r, c))
        ]
        self.start = next(s for s in self.states if self.rule(s)["meaning"] == "start")

    def tile(self, state):
        return self.grid[state[0]][state[1]]

    def rule(self, state):
        return self.tiles[self.tile(state)]

    def is_traversable(self, state):
        return self.rule(state).get("traversable", True)

    def is_terminal(self, state):
        return self.rule(state).get("terminal", False)

    def reward(self, state):
        return self.rule(state)["reward"]

    def move(self, state, direction):
        dr, dc = MOVES[direction]
        r, c = state[0] + dr, state[1] + dc
        if not (0 <= r < self.n_rows and 0 <= c < self.n_cols):
            return state
        if not self.is_traversable((r, c)):
            return state
        return (r, c)

    def direction_probs(self, state, action):
        rule = self.rule(state)
        if rule.get("movement") == "slippery":
            side_a, side_b = PERPENDICULAR[action]
            side = rule["perpendicular_each"]
            return [(action, rule["intended"]), (side_a, side), (side_b, side)]
        return [(action, 1.0)]

    def transitions(self, state, action):
        if self.is_terminal(state):
            return []
        outcomes = {}
        for direction, prob in self.direction_probs(state, action):
            next_state = self.move(state, direction)
            outcomes[next_state] = outcomes.get(next_state, 0.0) + prob
        return [
            (prob, next_state, self.reward(next_state), self.is_terminal(next_state))
            for next_state, prob in outcomes.items()
        ]


def load_world(config, map_key):
    return GridWorld(config[map_key], config["tiles"], config["actions"])