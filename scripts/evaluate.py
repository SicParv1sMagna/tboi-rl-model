from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from stable_baselines3 import PPO

from isaac_rl import IsaacEnv


PROJECT_ROOT = Path(__file__).resolve().parent.parent


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate a trained PPO agent "
            "in The Binding of Isaac."
        )
    )

    parser.add_argument(
        "--model",
        type=Path,
        required=True,
        help="Path to the trained PPO .zip model.",
    )

    parser.add_argument(
        "--episodes",
        type=int,
        default=20,
        help="Number of evaluation episodes.",
    )

    parser.add_argument(
        "--max-episode-steps",
        type=int,
        default=600,
        help="Maximum number of steps per episode.",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    model_path = args.model

    if not model_path.is_absolute():
        model_path = (
            PROJECT_ROOT
            / model_path
        )

    if not model_path.exists():
        raise FileNotFoundError(
            f"Model not found: {model_path}"
        )

    print("=" * 60)
    print("ISAAC RL - PPO EVALUATION")
    print("=" * 60)

    print(
        f"Model: {model_path}"
    )

    print(
        f"Episodes: {args.episodes}"
    )

    print()

    print(
        "Start Isaac with --luadebug "
        "and start a new run."
    )

    print()

    env = IsaacEnv(
        max_episode_steps=args.max_episode_steps,
    )

    model = PPO.load(
        model_path,
        env=env,
    )

    episode_rewards: list[float] = []
    episode_lengths: list[int] = []
    episode_hp_lost: list[float] = []
    remaining_hp_values: list[float] = []

    wins = 0
    deaths = 0
    timeouts = 0

    no_hit_episodes = 0
    no_hit_wins = 0

    try:
        for episode in range(
            1,
            args.episodes + 1,
        ):
            observation, info = env.reset()

            total_reward = 0.0
            steps = 0

            terminated = False
            truncated = False

            while (
                not terminated
                and not truncated
            ):
                action, _state = model.predict(
                    observation,
                    deterministic=True,
                )

                (
                    observation,
                    reward,
                    terminated,
                    truncated,
                    info,
                ) = env.step(
                    action
                )

                total_reward += reward
                steps += 1

            hp_lost = float(
                info.get(
                    "episode_hp_lost",
                    0.0,
                )
            )

            remaining_hp = float(
                info.get(
                    "remaining_hp",
                    info.get(
                        "player_hp",
                        0.0,
                    ),
                )
            )

            no_hit = bool(
                info.get(
                    "no_hit",
                    hp_lost == 0.0,
                )
            )

            episode_rewards.append(
                total_reward
            )

            episode_lengths.append(
                steps
            )

            episode_hp_lost.append(
                hp_lost
            )

            remaining_hp_values.append(
                remaining_hp
            )

            # ---------------------------------------------
            # Episode result
            # ---------------------------------------------

            if info.get(
                "enemy_defeated",
                False,
            ):
                result = "WIN"
                wins += 1

                if no_hit:
                    no_hit_wins += 1

            elif info.get(
                "dead",
                False,
            ):
                result = "DEATH"
                deaths += 1

            elif truncated:
                result = "TIMEOUT"
                timeouts += 1

            else:
                result = "UNKNOWN"

            if no_hit:
                no_hit_episodes += 1

            # ---------------------------------------------
            # Per-episode output
            # ---------------------------------------------

            print(
                f"episode={episode:03d} "
                f"result={result:<7} "
                f"reward={total_reward:+8.3f} "
                f"steps={steps:4d} "
                f"hp={remaining_hp:4.1f} "
                f"hp_lost={hp_lost:4.1f} "
                f"no_hit={'YES' if no_hit else 'NO'}"
            )

        # =================================================
        # Aggregate metrics
        # =================================================

        mean_reward = float(
            np.mean(
                episode_rewards
            )
        )

        mean_length = float(
            np.mean(
                episode_lengths
            )
        )

        mean_hp_lost = float(
            np.mean(
                episode_hp_lost
            )
        )

        mean_remaining_hp = float(
            np.mean(
                remaining_hp_values
            )
        )

        win_rate = (
            wins
            / args.episodes
        )

        no_hit_episode_rate = (
            no_hit_episodes
            / args.episodes
        )

        no_hit_win_rate = (
            no_hit_wins
            / args.episodes
        )

        no_hit_rate_among_wins = (
            no_hit_wins / wins
            if wins > 0
            else 0.0
        )

        # =================================================
        # Results
        # =================================================

        print()
        print("=" * 60)
        print("EVALUATION RESULTS")
        print("=" * 60)

        print(
            f"Episodes:              "
            f"{args.episodes}"
        )

        print(
            f"Wins:                  "
            f"{wins}"
        )

        print(
            f"Deaths:                "
            f"{deaths}"
        )

        print(
            f"Timeouts:              "
            f"{timeouts}"
        )

        print(
            f"Win rate:              "
            f"{win_rate:.1%}"
        )

        print()

        print(
            f"Mean reward:           "
            f"{mean_reward:+.3f}"
        )

        print(
            f"Mean episode length:   "
            f"{mean_length:.1f}"
        )

        print()

        print(
            f"Mean HP lost:          "
            f"{mean_hp_lost:.2f}"
        )

        print(
            f"Mean remaining HP:     "
            f"{mean_remaining_hp:.2f}"
        )

        print()

        print(
            f"No-hit episodes:       "
            f"{no_hit_episodes}"
        )

        print(
            f"No-hit episode rate:   "
            f"{no_hit_episode_rate:.1%}"
        )

        print(
            f"No-hit wins:           "
            f"{no_hit_wins}"
        )

        print(
            f"No-hit win rate:       "
            f"{no_hit_win_rate:.1%}"
        )

        print(
            f"No-hit among wins:     "
            f"{no_hit_rate_among_wins:.1%}"
        )

        print("=" * 60)

    finally:
        env.close()


if __name__ == "__main__":
    main()