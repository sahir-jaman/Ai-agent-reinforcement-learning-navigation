AI Agent Learns to Navigate a Grid World (Reinforcement Learning)

An AI agent that teaches itself to navigate a grid by trial and error, using Q-learning. While analysing the results, I found that it had learned to exploit a loophole in its reward system instead of solving the task. This repository shows the problem, explains why it happened, and fixes it.

Built in Python (NumPy, Matplotlib) as part of the Deep Learning module (COM762) at Ulster University.

The problem

The agent lives in a 5×5 grid and can move North, South, East or West.

Cell	Position (row, col)	Reward
S — Start	(2, 1)	–
G — Goal	(5, 5)	+10, episode ends
J — Jump square	(2, 4)	+5, teleports agent to E (4, 4)
Every other move	–	−1 (including bumping into a wall)

The best possible route is East → East → East (jump) → South → East, which scores +12 in 5 moves.

How it works
Q-learning: the agent keeps a table of values Q(state, action) and updates it after every move: Q(s,a) ← Q(s,a) + α · [ r + γ · max Q(s′, ·) − Q(s,a) ]
ε-greedy exploration: the agent starts by moving completely at random (ε = 1.0). It then gradually trusts what it has learned, until only 5% of its moves are random (ε = 0.05).
Settings: learning rate α = 1.0, discount factor γ = 0.99, up to 100 episodes. Training stops early once the average return over the last 30 episodes is above 10.
Key finding: the agent learned to cheat

<img width="1200" height="1200" alt="image" src="https://github.com/user-attachments/assets/8c8afc12-392b-4c00-ae31-477b5460d0e0" />


On the open grid, the learned state values came out at around 200, even though the best honest route is worth only about 12.

Show Image

Why: the jump can be used again and again. After landing on E, the agent can step North twice and land back on J, for +5 − 1 = +4 every two moves, forever. Solving the Bellman equations for this loop gives:

V(3,4) = (5 − γ) / (1 − γ²) = 4.01 / 0.0199 ≈ 201.5

This matches the 200.8 the agent learned for that cell. Its training log shows the same thing: late episodes scored +77 and +60, but took 240 and 347 steps, because the agent was circling the jump to collect reward before finishing.

This is a small example of reward hacking. The agent did exactly what the reward signal told it to do, but not what the designer intended.

The fix

I added walls at (3,3), (3,4), (3,5) and (4,3), which block the shortcut back to the jump square. With the loop gone:

The value of the start square drops to 11.5. This is the correct discounted value of the optimal route.
The agent converges to the optimal 5-move, +12 route.

Show Image

Setup	Episodes to converge	Start-state value	Final episodes
Open grid (loophole)	61	186.7 (inflated)	+77 in 240 steps, +60 in 347 steps
With walls (fixed)	93	11.5 (correct)	+12 in 5 steps
Effect of the learning rate (with walls)
α	1.0	0.7	0.5	0.3	0.1
Episodes to converge	93	95	95	99	94
Mean return during training	1.42	0.33	0.30	−0.39	−0.92

Because this environment is deterministic, a learning rate of α = 1.0 works best: there is no noise to average out, so the agent can trust each update fully.

Run it yourself
bash
pip install -r requirements.txt

# Open grid (shows the reward loop)
python gridworld_qlearning.py

# With walls (fixed environment)
python gridworld_qlearning.py --obstacles "[[3,3],[3,4],[3,5],[4,3]]"

# Compare learning rates
python gridworld_qlearning.py --obstacles "[[3,3],[3,4],[3,5],[4,3]]" --alpha-sweep "1.0,0.7,0.5,0.3,0.1"

Each run saves the Q-table, a training log and a state-value heatmap.

What I learned
Reward design matters as much as the algorithm. A single reusable bonus was enough to change what the agent learned.
Check the learned values, not just the success rate. The agent reached the goal in 100% of episodes, so the bug only showed up in the value table.
Verify learned behaviour against theory. Comparing the Q-values with a hand-calculated Bellman solution confirmed what was going on.
Files
File	Description
gridworld_qlearning.py	Environment, Q-learning agent, training and plotting
state_values_heatmap.png	Learned values on the open grid
heatmap_with_obstacles.png	Learned values after the fix
qtable.csv / qtable.npy	Learned Q-table (25 states × 4 actions)
training_returns.csv	Return and step count for each episode
training_summary.txt	Summary of the training run
