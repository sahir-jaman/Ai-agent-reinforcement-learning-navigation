
import argparse
import ast
import math
import sys
from dataclasses import dataclass
from typing import Tuple, List, Dict, Optional

import numpy as np
import matplotlib.pyplot as plt


# -----------------------------
# Environment: GridWorld
# -----------------------------

@dataclass(frozen=True)
class Action:
    """Action mapping consistent with brief: North=1, South=2, East=3, West=4."""
    NORTH: int = 1
    SOUTH: int = 2
    EAST: int = 3
    WEST: int = 4

ACTIONS = (Action.NORTH, Action.SOUTH, Action.EAST, Action.WEST)
ACTION_TO_DELTA = {
    Action.NORTH: (-1, 0),
    Action.SOUTH: (1, 0),
    Action.EAST: (0, 1),
    Action.WEST: (0, -1),
}


class GridWorld:
    """
    5x5 GridWorld with:
      - Start at [2,1] (1-indexed in spec)
      - Goal at  [5,5] with +10 reward
      - Special jump from [2,4] -> [4,4] with +5 reward (applied when ENTERING [2,4])
      - Obstacles (blocked cells): configurable list of [r,c]
      - All other actions: -1 step penalty
      - Bordered world; invalid moves keep the agent in place and incur -1.

    Internal representation uses 0-indexing for arrays, but public cell coords in comments
    use 1-indexing to stay aligned with the assignment brief.
    """

    def __init__(
        self,
        n_rows: int = 5,
        n_cols: int = 5,
        start_cell: Tuple[int, int] = (2, 1),  # 1-indexed
        goal_cell: Tuple[int, int] = (5, 5),   # 1-indexed
        special_jump_src: Tuple[int, int] = (2, 4),  # 1-indexed
        special_jump_dst: Tuple[int, int] = (4, 4),  # 1-indexed
        special_jump_reward: float = 5.0,
        step_penalty: float = -1.0,
        goal_reward: float = 10.0,
        obstacles_1idx: Optional[List[Tuple[int, int]]] = None,
        end_on_jump_end: bool = False,
        max_steps_per_episode: int = 1000,  # safety cap
    ):
        self.n_rows = n_rows
        self.n_cols = n_cols
        self.start = self._to0(start_cell)
        self.goal = self._to0(goal_cell)
        self.special_src = self._to0(special_jump_src)
        self.special_dst = self._to0(special_jump_dst)
        self.special_reward = float(special_jump_reward)
        self.step_penalty = float(step_penalty)
        self.goal_reward = float(goal_reward)
        self.end_on_jump_end = bool(end_on_jump_end)
        self.max_steps_per_episode = int(max_steps_per_episode)

        if obstacles_1idx is None:
            obstacles_1idx = []  # default: no obstacles; set via CLI to match your figure
        self.obstacles = {self._to0(tuple(cell)) for cell in obstacles_1idx}

        self.reset()

    def _to0(self, cell_1idx: Tuple[int, int]) -> Tuple[int, int]:
        r, c = cell_1idx
        return (r - 1, c - 1)

    def _to1(self, cell_0idx: Tuple[int, int]) -> Tuple[int, int]:
        r, c = cell_0idx
        return (r + 1, c + 1)

    @property
    def n_states(self) -> int:
        return self.n_rows * self.n_cols

    @property
    def n_actions(self) -> int:
        return len(ACTIONS)

    def state_to_index(self, cell_0idx: Tuple[int, int]) -> int:
        r, c = cell_0idx
        return r * self.n_cols + c

    def index_to_state(self, idx: int) -> Tuple[int, int]:
        r = idx // self.n_cols
        c = idx % self.n_cols
        return (r, c)

    def in_bounds(self, r: int, c: int) -> bool:
        return 0 <= r < self.n_rows and 0 <= c < self.n_cols

    def is_obstacle(self, cell_0: Tuple[int, int]) -> bool:
        return cell_0 in self.obstacles

    def reset(self) -> Tuple[int, int]:
        """Reset to start state; returns 0-indexed (r,c)."""
        self.s = tuple(self.start)
        self.t = 0
        self.done = False
        return self.s

    def step(self, action: int) -> Tuple[Tuple[int, int], float, bool, Dict]:
        """
        Take an action (1..4). Returns (next_state, reward, done, info).
        Rules:
          - borders block movement (stay+penalty)
          - obstacles block movement (stay+penalty)
          - entering special_src teleports to special_dst and yields +special_reward
          - entering goal yields +goal_reward and done=True
          - otherwise, step_penalty
        """
        if self.done:
            return self.s, 0.0, True, {"info": "already_terminal"}

        dr, dc = ACTION_TO_DELTA[action]
        r, c = self.s
        nr, nc = r + dr, c + dc

        # Border or obstacle => blocked, stay put
        if not self.in_bounds(nr, nc) or self.is_obstacle((nr, nc)):
            nr, nc = r, c  # stay
            reward = self.step_penalty
        else:
            reward = self.step_penalty

        # Apply move
        next_state = (nr, nc)

        # Special jump (triggered when ENTERING the src cell)
        if next_state == self.special_src:
            next_state = self.special_dst
            reward = self.special_reward
            if self.end_on_jump_end:
                self.s = next_state
                self.t += 1
                self.done = True
                return next_state, reward, True, {"event": "special_jump_terminal"}

        # Goal check
        if next_state == self.goal:
            reward = self.goal_reward
            self.s = next_state
            self.t += 1
            self.done = True
            return next_state, reward, True, {"event": "goal_reached"}

        # Regular transition
        self.s = next_state
        self.t += 1
        if self.t >= self.max_steps_per_episode:
            # safety cap
            self.done = True
        return next_state, reward, self.done, {}

    def valid_actions(self, state: Optional[Tuple[int, int]] = None) -> List[int]:
        """Optional helper: list actions that don't immediately hit a border/obstacle."""
        if state is None:
            state = self.s
        r, c = state
        acts = []
        for a, (dr, dc) in ACTION_TO_DELTA.items():
            nr, nc = r + dr, c + dc
            if self.in_bounds(nr, nc) and not self.is_obstacle((nr, nc)):
                acts.append(a)
        return acts


# -----------------------------
# Agent: Q-Learning
# -----------------------------

class QLearningAgent:
    def __init__(
        self,
        n_states: int,
        n_actions: int,
        alpha: float = 1.0,   # learning rate
        gamma: float = 0.99,  # discount
        eps_start: float = 1.0,
        eps_end: float = 0.05,
        eps_decay: str = "linear",  # "linear" or "exp"
        seed: Optional[int] = 42,
    ):
        self.n_states = n_states
        self.n_actions = n_actions
        self.alpha = float(alpha)
        self.gamma = float(gamma)
        self.eps_start = float(eps_start)
        self.eps_end = float(eps_end)
        assert eps_decay in ("linear", "exp")
        self.eps_decay = eps_decay

        self.rng = np.random.default_rng(seed)
        self.Q = np.zeros((n_states, n_actions), dtype=np.float64)

    def epsilon_at(self, episode: int, total_episodes: int) -> float:
        if self.eps_decay == "linear":
            frac = max(0.0, 1.0 - episode / max(1, total_episodes - 1))
            return self.eps_end + (self.eps_start - self.eps_end) * frac
        else:  # exp
            # decay so that eps drops from start to end across episodes
            k = math.log((self.eps_end + 1e-12) / (self.eps_start + 1e-12)) / max(1, total_episodes - 1)
            return max(self.eps_end, self.eps_start * math.exp(k * episode))

    def select_action(self, state_idx: int, epsilon: float) -> int:
        if self.rng.random() < epsilon:
            return int(self.rng.integers(0, self.n_actions))  # random a_idx in [0,n_actions)
        # Greedy
        q = self.Q[state_idx]
        # tie-break randomly among max actions
        max_val = q.max()
        max_acts = np.flatnonzero(q == max_val)
        return int(self.rng.choice(max_acts))

    def train(
        self,
        env: GridWorld,
        episodes: int = 100,
        early_stop_avg_gt: float = 10.0,  # per brief (last 30 avg return > 10)
        early_stop_window: int = 30,
        end_on_jump_end: Optional[bool] = None,
        log_every: int = 10,
    ) -> Dict:
        returns = []
        steps_list = []
        reached_goal_flags = []

        for ep in range(episodes):
            if end_on_jump_end is not None:
                env.end_on_jump_end = end_on_jump_end

            s = env.reset()
            s_idx = env.state_to_index(s)
            epsilon = self.epsilon_at(ep, episodes)
            done = False
            G = 0.0
            steps = 0
            reached_goal = False

            while not done:
                a_idx = self.select_action(s_idx, epsilon)
                # map 0..3 -> 1..4 (spec actions)
                action = a_idx + 1

                s_next, reward, done, info = env.step(action)
                s_next_idx = env.state_to_index(s_next)

                # Q-learning update
                best_next = self.Q[s_next_idx].max() if not done else 0.0
                td_target = reward + self.gamma * best_next
                td_error = td_target - self.Q[s_idx, a_idx]
                self.Q[s_idx, a_idx] += self.alpha * td_error

                # bookkeeping
                G += reward
                steps += 1
                s_idx = s_next_idx
                if info.get("event") == "goal_reached":
                    reached_goal = True

            returns.append(G)
            steps_list.append(steps)
            reached_goal_flags.append(1 if reached_goal else 0)

            if (ep + 1) % log_every == 0 or ep == 0:
                avg30 = np.mean(returns[-early_stop_window:]) if len(returns) >= early_stop_window else np.nan
                print(f"Episode {ep+1}/{episodes} | Return={G:.1f} | Steps={steps} | Eps={epsilon:.3f} | "
                      f"Avg{early_stop_window}={avg30:.2f}")

            # Early stopping per brief
            if len(returns) >= early_stop_window:
                if np.mean(returns[-early_stop_window:]) > early_stop_avg_gt:
                    print(f"[EARLY STOP] Avg return over last {early_stop_window} episodes "
                          f"> {early_stop_avg_gt}. Stopping at episode {ep+1}.")
                    break

        result = {
            "episodes_ran": len(returns),
            "returns": np.array(returns, dtype=np.float64),
            "steps": np.array(steps_list, dtype=np.int32),
            "reached_goal_flags": np.array(reached_goal_flags, dtype=np.int8),
        }
        return result

    def state_values(self) -> np.ndarray:
        """V(s) = max_a Q(s,a)"""
        return self.Q.max(axis=1)


# -----------------------------
# Utilities
# -----------------------------

def parse_obstacles(obstacles_str: Optional[str]) -> List[Tuple[int, int]]:
    """
    Parse obstacles from a JSON-like string, e.g.: "[[2,2],[3,3]]".
    Returns a list of (row, col) in 1-indexed coordinates.
    """
    if not obstacles_str:
        return []
    try:
        arr = ast.literal_eval(obstacles_str)
        out = []
        for pair in arr:
            if not (isinstance(pair, (list, tuple)) and len(pair) == 2):
                raise ValueError
            r, c = int(pair[0]), int(pair[1])
            out.append((r, c))
        return out
    except Exception as e:
        raise argparse.ArgumentTypeError(
            f"Could not parse obstacles. Use a list of [r,c], e.g. '[[2,2],[3,3]]'. Error: {e}"
        )


def save_qtable(Q: np.ndarray, path_npy: str = "qtable.npy", path_csv: str = "qtable.csv"):
    np.save(path_npy, Q)
    # CSV with header
    header = ",".join([f"a{a}" for a in range(1, Q.shape[1] + 1)])
    np.savetxt(path_csv, Q, delimiter=",", header=header, comments="")


def save_returns(returns: np.ndarray, steps: np.ndarray, path_csv: str = "training_returns.csv"):
    import csv
    with open(path_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["episode", "return", "steps"])
        for i, (g, s) in enumerate(zip(returns, steps), start=1):
            w.writerow([i, float(g), int(s)])


def plot_state_values_heatmap(env: GridWorld, V: np.ndarray, path_png: str = "state_values_heatmap.png"):
    """
    Plot V(s) as a 2D heatmap matching the grid layout (row-major).
    Obstacles are masked.
    """
    grid = V.reshape(env.n_rows, env.n_cols).copy()
    # Mask obstacles
    for (r, c) in env.obstacles:
        grid[r, c] = np.nan

    plt.figure(figsize=(6, 6))
    im = plt.imshow(grid, origin="upper")  # default colormap; do NOT set colors explicitly
    plt.title("State Values Heatmap (V[s] = max_a Q[s,a])")
    plt.xlabel("Column (1-indexed)")
    plt.ylabel("Row (1-indexed)")
    plt.colorbar(im, fraction=0.046, pad=0.04)
    # Annotate values
    for r in range(env.n_rows):
        for c in range(env.n_cols):
            val = grid[r, c]
            txt = "X" if (r, c) in env.obstacles else (f"{val:.1f}" if not np.isnan(val) else "")
            plt.text(c, r, txt, ha="center", va="center")
    # Mark start, goal, and jump
    rs, cs = env.start
    rg, cg = env.goal
    rj, cj = env.special_src
    rjd, cjd = env.special_dst
    plt.text(cs, rs, "S", ha="center", va="center", fontweight="bold")
    plt.text(cg, rg, "G", ha="center", va="center", fontweight="bold")
    plt.text(cj, rj, "J", ha="center", va="center", fontweight="bold")
    plt.text(cjd, rjd, "E", ha="center", va="center", fontweight="bold")
    plt.tight_layout()
    plt.savefig(path_png, dpi=200)
    plt.close()


def write_summary(text: str, path: str = "training_summary.txt"):
    with open(path, "w") as f:
        f.write(text)


def run_training_once(args, alpha: float) -> Dict:
    obstacles = parse_obstacles(args.obstacles)
    env = GridWorld(
        n_rows=5,
        n_cols=5,
        start_cell=(2, 1),
        goal_cell=(5, 5),
        special_jump_src=(2, 4),
        special_jump_dst=(4, 4),
        special_jump_reward=5.0,
        step_penalty=-1.0,
        goal_reward=10.0,
        obstacles_1idx=obstacles,
        end_on_jump_end=args.end_on_jump_end,
        max_steps_per_episode=args.max_steps,
    )
    agent = QLearningAgent(
        n_states=env.n_states,
        n_actions=env.n_actions,
        alpha=alpha,
        gamma=args.gamma,
        eps_start=args.eps_start,
        eps_end=args.eps_end,
        eps_decay=args.eps_decay,
        seed=args.seed,
    )
    result = agent.train(
        env,
        episodes=args.episodes,
        early_stop_avg_gt=args.early_stop_avg_gt,
        early_stop_window=args.early_stop_window,
        end_on_jump_end=args.end_on_jump_end,
        log_every=args.log_every,
    )
    # Save artifacts (with alpha suffix if sweeping)
    suffix = f"_alpha{alpha}".replace(".", "p") if args.alpha_sweep else ""
    save_returns(result["returns"], result["steps"], f"training_returns{suffix}.csv")
    save_qtable(agent.Q, f"qtable{suffix}.npy", f"qtable{suffix}.csv")
    V = agent.state_values()
    plot_state_values_heatmap(env, V, f"state_values_heatmap{suffix}.png")

    success_rate = result["reached_goal_flags"].mean() if len(result["reached_goal_flags"]) > 0 else 0.0
    summary = (
        f"Episodes run: {result['episodes_ran']}\n"
        f"Alpha: {alpha}\n"
        f"Gamma: {args.gamma}\n"
        f"Epsilon schedule: {args.eps_decay} from {args.eps_start} to {args.eps_end}\n"
        f"End on jump end: {args.end_on_jump_end}\n"
        f"Obstacles (1-indexed): {obstacles}\n"
        f"Mean return: {result['returns'].mean():.3f}\n"
        f"Mean steps: {result['steps'].mean():.1f}\n"
        f"Goal success rate: {success_rate:.2%}\n"
    )
    write_summary(summary, f"training_summary{suffix}.txt")
    return {"env": env, "agent": agent, "result": result, "alpha": alpha}


def main(argv=None):
    parser = argparse.ArgumentParser(description="COM762 CW2: Q-Learning in 5x5 GridWorld")
    parser.add_argument("--episodes", type=int, default=100, help="Training episodes (default 100)")
    parser.add_argument("--gamma", type=float, default=0.99, help="Discount factor")
    parser.add_argument("--alpha", type=float, default=None, help="Learning rate (if not sweeping)")
    parser.add_argument("--alpha-sweep", type=str, default="", help="Comma-separated alphas, e.g. '1.0,0.7,0.5,0.3,0.1'")
    parser.add_argument("--eps-start", type=float, default=1.0, help="Epsilon start")
    parser.add_argument("--eps-end", type=float, default=0.05, help="Epsilon end")
    parser.add_argument("--eps-decay", type=str, choices=["linear", "exp"], default="linear", help="Epsilon decay schedule")
    parser.add_argument("--obstacles", type=str, default="", help="JSON-like list of [r,c] 1-indexed, e.g. '[[2,2],[3,3]]'")
    parser.add_argument("--end-on-jump-end", type=lambda x: str(x).lower() in ("true","1","yes","y"), default=False,
                        help="If True, reaching the jump destination [4,4] ends the episode")
    parser.add_argument("--max-steps", type=int, default=1000, help="Safety cap per episode (unlimited in principle)")
    parser.add_argument("--early-stop-avg-gt", type=float, default=10.0, help="Early stop if last window avg return > this")
    parser.add_argument("--early-stop-window", type=int, default=30, help="Window size for early stopping criterion")
    parser.add_argument("--log-every", type=int, default=10, help="Logging cadence")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")

    args = parser.parse_args(argv)

    # Determine alpha list
    alpha_list: List[float] = []
    if args.alpha_sweep.strip():
        alpha_list = [float(x.strip()) for x in args.alpha_sweep.split(",") if x.strip()]
    else:
        if args.alpha is None:
            alpha_list = [1.0]  # default per brief (also run more values later)
        else:
            alpha_list = [float(args.alpha)]

    all_runs = []
    for alpha in alpha_list:
        print(f"\n=== Training with alpha={alpha} ===")
        run = run_training_once(args, alpha)
        all_runs.append(run)

    # If a single run, print a concise summary to stdout
    if len(all_runs) == 1:
        run = all_runs[0]
        R = run["result"]["returns"]
        steps = run["result"]["steps"]
        print("\n--- SUMMARY ---")
        print(f"Episodes run: {run['result']['episodes_ran']}")
        print(f"Mean return: {R.mean():.3f} (min={R.min():.1f}, max={R.max():.1f})")
        print(f"Mean steps: {steps.mean():.1f}")
        print(f"State-values heatmap saved to 'state_values_heatmap.png'")
        print(f"Q-table saved to 'qtable.npy' and 'qtable.csv'")
        print(f"Training log saved to 'training_returns.csv' and 'training_summary.txt'")
    else:
        print("\nMultiple runs completed (alpha sweep). Check the generated files for each alpha.")

if __name__ == "__main__":
    main()
