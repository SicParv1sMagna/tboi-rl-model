from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from stable_baselines3 import PPO

from isaac_rl import IsaacEnv


PROJECT_ROOT = (
    Path(__file__)
    .resolve()
    .parents[1]
)


# =========================================================
# ARGUMENTS
# =========================================================


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
        help="Path to PPO model.",
    )

    parser.add_argument(
        "--episodes",
        type=int,
        default=10,
        help="Number of episodes.",
    )

    parser.add_argument(
        "--max-episode-steps",
        type=int,
        default=600,
        help="Maximum steps per episode.",
    )

    parser.add_argument(
        "--stochastic",
        action="store_true",
        help=(
            "Use stochastic policy actions "
            "instead of deterministic ones."
        ),
    )

    return parser.parse_args()


# =========================================================
# MODEL PATH
# =========================================================


def resolve_model_path(
    model_path: Path,
) -> Path:
    if not model_path.is_absolute():
        model_path = (
            PROJECT_ROOT
            / model_path
        )

    model_path = (
        model_path.resolve()
    )

    if model_path.exists():
        return model_path

    if model_path.suffix != ".zip":
        zip_path = (
            model_path.with_suffix(
                ".zip"
            )
        )

        if zip_path.exists():
            return zip_path

    raise FileNotFoundError(
        f"Model not found: {model_path}"
    )


# =========================================================
# SAFE MEAN
# =========================================================


def safe_mean(
    values: list[float],
) -> float:
    if not values:
        return 0.0

    return float(
        np.mean(
            values
        )
    )


# =========================================================
# MAIN
# =========================================================


def main() -> None:
    args = parse_args()

    model_path = resolve_model_path(
        args.model
    )

    deterministic = (
        not args.stochastic
    )

    print(
        "=" * 60
    )

    print(
        "ISAAC RL EVALUATION"
    )

    print(
        "=" * 60
    )

    print(
        f"Model:         {model_path}"
    )

    print(
        f"Episodes:      {args.episodes}"
    )

    print(
        f"Deterministic: {deterministic}"
    )

    print(
        "=" * 60
    )

    model = PPO.load(
        str(
            model_path
        )
    )

    env = IsaacEnv(
        max_episode_steps=(
            args.max_episode_steps
        )
    )

    # -----------------------------------------------------
    # Aggregates
    # -----------------------------------------------------

    episode_rewards: list[float] = []

    episode_lengths: list[float] = []

    episode_hp_lost: list[float] = []

    remaining_hp_values: list[float] = []

    winning_lengths: list[float] = []

    rooms_exited_values: list[float] = []


    wins = 0

    deaths = 0

    timeouts = 0

    unknowns = 0


    no_hit_episodes = 0

    no_hit_wins = 0


    try:
        for episode_index in range(
            1,
            args.episodes + 1,
        ):
            observation, reset_info = (
                env.reset()
            )

            total_reward = 0.0

            steps = 0

            terminated = False

            truncated = False

            info = reset_info


            while (
                not terminated
                and not truncated
            ):
                action, _state = (
                    model.predict(
                        observation,
                        deterministic=deterministic,
                    )
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

                total_reward += float(
                    reward
                )

                steps += 1


            # =================================================
            # FINAL STATE
            # =================================================

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

            rooms_exited = int(
                info.get(
                    "rooms_exited",
                    0,
                )
            )

            target_room_exits = int(
                info.get(
                    "target_room_exits",
                    3,
                )
            )

            training_complete = bool(
                info.get(
                    "training_complete",
                    False,
                )
            )

            dead = bool(
                info.get(
                    "dead",
                    False,
                )
            )

            timeout = bool(
                info.get(
                    "timeout",
                    False,
                )
                or (
                    truncated
                    and not terminated
                )
            )


            # =================================================
            # RESULT
            # =================================================

            if training_complete:
                result = "WIN"

                wins += 1

                winning_lengths.append(
                    float(
                        steps
                    )
                )

                if no_hit:
                    no_hit_wins += 1


            elif dead:
                result = "DEATH"

                deaths += 1


            elif timeout:
                result = "TIMEOUT"

                timeouts += 1


            else:
                result = "UNKNOWN"

                unknowns += 1


            if no_hit:
                no_hit_episodes += 1


            episode_rewards.append(
                total_reward
            )

            episode_lengths.append(
                float(
                    steps
                )
            )

            episode_hp_lost.append(
                hp_lost
            )

            remaining_hp_values.append(
                remaining_hp
            )

            rooms_exited_values.append(
                float(
                    rooms_exited
                )
            )


            no_hit_text = (
                "YES"
                if no_hit
                else "NO"
            )


            print(
                f"episode={episode_index:03d} "
                f"result={result:<7} "
                f"rooms="
                f"{rooms_exited}/"
                f"{target_room_exits} "
                f"reward={total_reward:+8.3f} "
                f"steps={steps:4d} "
                f"hp={remaining_hp:4.1f} "
                f"hp_lost={hp_lost:4.1f} "
                f"no_hit={no_hit_text}"
            )


    finally:
        env.close()


    # =====================================================
    # AGGREGATES
    # =====================================================

    episode_count = len(
        episode_rewards
    )


    if episode_count > 0:
        win_rate = (
            wins
            / episode_count
            * 100.0
        )

        timeout_rate = (
            timeouts
            / episode_count
            * 100.0
        )

        no_hit_episode_rate = (
            no_hit_episodes
            / episode_count
            * 100.0
        )

        no_hit_win_rate = (
            no_hit_wins
            / episode_count
            * 100.0
        )

    else:
        win_rate = 0.0

        timeout_rate = 0.0

        no_hit_episode_rate = 0.0

        no_hit_win_rate = 0.0


    if wins > 0:
        no_hit_among_wins = (
            no_hit_wins
            / wins
            * 100.0
        )

    else:
        no_hit_among_wins = 0.0


    mean_reward = safe_mean(
        episode_rewards
    )

    mean_episode_length = safe_mean(
        episode_lengths
    )

    mean_winning_length = safe_mean(
        winning_lengths
    )

    mean_hp_lost = safe_mean(
        episode_hp_lost
    )

    mean_remaining_hp = safe_mean(
        remaining_hp_values
    )

    mean_rooms_exited = safe_mean(
        rooms_exited_values
    )


    # =====================================================
    # OUTPUT
    # =====================================================

    print()

    print(
        "=" * 60
    )

    print(
        "EVALUATION RESULTS"
    )

    print(
        "=" * 60
    )

    print(
        f"Episodes:              "
        f"{episode_count}"
    )

    print(
        f"Wins (3-room runs):    "
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
        f"Unknown:               "
        f"{unknowns}"
    )

    print(
        f"Win rate:              "
        f"{win_rate:.1f}%"
    )

    print(
        f"Timeout rate:          "
        f"{timeout_rate:.1f}%"
    )

    print()

    print(
        f"Mean rooms exited:     "
        f"{mean_rooms_exited:.2f}/3"
    )

    print(
        f"Mean reward:           "
        f"{mean_reward:+.3f}"
    )

    print(
        f"Mean episode length:   "
        f"{mean_episode_length:.1f}"
    )

    if wins > 0:
        print(
            f"Mean winning length:   "
            f"{mean_winning_length:.1f}"
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
        f"{no_hit_episode_rate:.1f}%"
    )

    print(
        f"No-hit wins:           "
        f"{no_hit_wins}"
    )

    print(
        f"No-hit win rate:       "
        f"{no_hit_win_rate:.1f}%"
    )

    print(
        f"No-hit among wins:     "
        f"{no_hit_among_wins:.1f}%"
    )

    print(
        "=" * 60
    )


if __name__ == "__main__":
    main()