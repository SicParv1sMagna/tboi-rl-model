# The Binding Of Isaac RL

## Reinforcement learning agent for The Binding of Isaac: Afterbirth+

A custom reinforcement learning environment that connects TBOI: Afterbirth+ with PPO via Lua mod and Gymnasium

## About

This project explores wheter an RL agent can learn to play TBOI: Afterbirth+ using a custom game-to-Python bridge.

The game itself is controlled through a Lua mod. Game State is serialized to JSON and
streamed over TCP to a Python environment implementing the Gymnasium API.

The Python side uses Stable-Baselines3 PPO to train policies which send movement and shooting actions back to game.

The project currently focuses on structured observations rather than raw pixels.

The long-term goal is to progressively move from simple controlled encounters toward increasingly realistic Isaac gameplay.

---

## Current Status

The project has already progressed through several curriculum stages:

- Kill one enemy
- Preserve health
- Random enemy positions
- Room navigation
- Door approach + exit
- Multi-room behaviour
- Obstacle/hazard awareness

The current generation is:

```
structured_v2
```

It extends the original structured observation with local information about:

- Walls and obstacles
- Blocked movement directions
- Spikes
- Fireplaces and other local hazards

The previous `structured_v1` generation successfully demonstrated:

- Combat against a controlled enemy
- Randomized enemy positioning
- HP-aware behaviour
- Switching between combat and navigation
- Door selection
- Actual room transitions
- Repeated multi-room behaviour

The strongest single-room `structured_v1` evaluation reached:

```
Episodes:              10
Wins:                   9
Deaths:                 0
Timeouts:               1

Win rate:              90.0%
Mean winning length:   43.4
Mean HP lost:           0.80
```
