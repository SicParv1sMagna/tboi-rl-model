from __future__ import annotations

import argparse
import os
from pathlib import Path

import numpy as np
from stable_baselines3 import PPO

from isaac_rl import IsaacEnv


PROJECT_ROOT = (
    Path(__file__)
    .resolve()
    .parents[1]
)

DIRECTIONS = (
    "NONE",
    "LEFT",
    "RIGHT",
    "UP",
    "DOWN",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--model",
        type=Path,
        required=True,
    )

    parser.add_argument(
        "--episodes",
        type=int,
        default=10,
    )

    parser.add_argument(
        "--stage",
        type=int,
        choices=(1, 2, 3, 4, 5),
        default=None,
    )

    parser.add_argument(
        "--max-episode-steps",
        type=int,
        default=None,
    )

    parser.add_argument(
        "--stochastic",
        action="store_true",
    )

    return parser.parse_args()


def resolve_stage(
    stage: int | None,
) -> int:
    if stage is not None:
        return stage

    return int(
        os.environ.get(
            "ISAAC_RL_STAGE",
            "1",
        )
    )


def resolve_model_path(
    path: Path,
) -> Path:
    if not path.is_absolute():
        path = (
            PROJECT_ROOT
            / path
        )

    path = path.resolve()

    if path.exists():
        return path

    if path.suffix != ".zip":
        candidate = (
            path.with_suffix(
                ".zip"
            )
        )

        if candidate.exists():
            return candidate

    raise FileNotFoundError(
        f"Model not found: {path}"
    )


def mean(
    values: list[float],
) -> float:
    if not values:
        return 0.0

    return float(
        np.mean(values)
    )


def print_distribution(
    counts: dict[str, int],
) -> None:
    total = sum(
        counts.values()
    )

    for direction in DIRECTIONS:
        count = counts[
            direction
        ]

        percent = (
            count
            / total
            * 100.0
            if total > 0
            else 0.0
        )

        print(
            f"{direction:<6} "
            f"{count:6d} "
            f"{percent:6.1f}%"
        )


def main() -> None:
    args = parse_args()

    stage = resolve_stage(
        args.stage
    )

    model_path = (
        resolve_model_path(
            args.model
        )
    )

    env = IsaacEnv(
        curriculum_stage=stage,
        max_episode_steps=(
            args.max_episode_steps
        ),
    )

    model = PPO.load(
        str(model_path)
    )

    deterministic = (
        not args.stochastic
    )

    rewards: list[float] = []
    lengths: list[float] = []

    hp_lost_values: list[float] = []
    remaining_hp_values: list[float] = []

    rooms_values: list[float] = []

    encounter_enemy_values: list[float] = []
    target_switch_values: list[float] = []

    nav_steps_values: list[float] = []
    nav_missing_values: list[float] = []

    blocked_values: list[float] = []
    stuck_values: list[float] = []

    tears_fired_values: list[float] = []
    tears_hit_values: list[float] = []
    tears_missed_values: list[float] = []

    combat_move_counts = {
        direction: 0
        for direction in DIRECTIONS
    }

    combat_shoot_counts = {
        direction: 0
        for direction in DIRECTIONS
    }

    wins = 0
    deaths = 0
    timeouts = 0
    unknown = 0

    no_hit_episodes = 0
    no_hit_wins = 0

    target_room_exits = 0

    print("=" * 60)
    print("ISAAC RL EVALUATION")
    print("=" * 60)
    print(f"Stage:          {stage}")
    print(
        f"Curriculum:     "
        f"{env.stage_config['name']}"
    )
    print(f"Episodes:       {args.episodes}")
    print(f"Deterministic:  {deterministic}")
    print("=" * 60)

    try:
        for episode_number in range(
            1,
            args.episodes + 1,
        ):
            observation, info = (
                env.reset()
            )

            initial_enemy_target = int(
                info.get(
                    "encounter_enemy_target_count",
                    info.get(
                        "enemy_count",
                        0,
                    ),
                )
            )

            encounter_enemy_values.append(
                float(
                    initial_enemy_target
                )
            )

            total_reward = 0.0
            steps = 0

            navigation_steps = 0
            navigation_missing = 0

            blocked_moves = 0
            stuck_moves = 0

            target_switches = 0

            previous_target_seed = (
                info.get(
                    "combat_target_seed"
                )
            )

            terminated = False
            truncated = False

            while (
                not terminated
                and not truncated
            ):
                action, _ = (
                    model.predict(
                        observation,
                        deterministic=(
                            deterministic
                        ),
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

                navigation_mode = bool(
                    info.get(
                        "navigation_mode",
                        False,
                    )
                )

                if navigation_mode:
                    navigation_steps += 1
                else:
                    move = str(
                        info.get(
                            "applied_move",
                            "NONE",
                        )
                    )

                    shoot = str(
                        info.get(
                            "applied_shoot",
                            "NONE",
                        )
                    )

                    if move in combat_move_counts:
                        combat_move_counts[
                            move
                        ] += 1

                    if shoot in combat_shoot_counts:
                        combat_shoot_counts[
                            shoot
                        ] += 1

                if bool(
                    info.get(
                        "navigation_target_missing",
                        False,
                    )
                ):
                    navigation_missing += 1

                if bool(
                    info.get(
                        "blocked_move_attempted",
                        False,
                    )
                ):
                    blocked_moves += 1

                if bool(
                    info.get(
                        "stuck_move_attempted",
                        False,
                    )
                ):
                    stuck_moves += 1

                current_target_seed = (
                    info.get(
                        "combat_target_seed"
                    )
                )

                if (
                    previous_target_seed is not None
                    and current_target_seed is not None
                    and current_target_seed
                    != previous_target_seed
                ):
                    target_switches += 1

                if current_target_seed is not None:
                    previous_target_seed = (
                        current_target_seed
                    )

            complete = bool(
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
            )

            rooms = int(
                info.get(
                    "rooms_exited",
                    0,
                )
            )

            target_room_exits = int(
                info.get(
                    "target_room_exits",
                    0,
                )
            )

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

            tears_fired = int(
                info.get(
                    "tears_fired",
                    0,
                )
            )

            tears_hit = int(
                info.get(
                    "tears_hit",
                    0,
                )
            )

            tears_missed = int(
                info.get(
                    "tears_missed",
                    0,
                )
            )

            confirmed = (
                tears_hit
                + tears_missed
            )

            unresolved = max(
                0,
                tears_fired
                - confirmed,
            )

            accuracy = (
                tears_hit
                / confirmed
                * 100.0
                if confirmed > 0
                else 0.0
            )

            if complete:
                result = "WIN"
                wins += 1

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
                unknown += 1

            if no_hit:
                no_hit_episodes += 1

            rewards.append(
                total_reward
            )

            lengths.append(
                float(steps)
            )

            hp_lost_values.append(
                hp_lost
            )

            remaining_hp_values.append(
                remaining_hp
            )

            rooms_values.append(
                float(rooms)
            )

            target_switch_values.append(
                float(
                    target_switches
                )
            )

            nav_steps_values.append(
                float(
                    navigation_steps
                )
            )

            nav_missing_values.append(
                float(
                    navigation_missing
                )
            )

            blocked_values.append(
                float(
                    blocked_moves
                )
            )

            stuck_values.append(
                float(
                    stuck_moves
                )
            )

            tears_fired_values.append(
                float(
                    tears_fired
                )
            )

            tears_hit_values.append(
                float(
                    tears_hit
                )
            )

            tears_missed_values.append(
                float(
                    tears_missed
                )
            )

            print(
                f"episode={episode_number:03d} "
                f"result={result:<7} "
                f"enemies={initial_enemy_target} "
                f"reward={total_reward:+8.3f} "
                f"steps={steps:4d} "
                f"hp={remaining_hp:4.1f} "
                f"lost={hp_lost:4.1f} "
                f"tears="
                f"{tears_hit}H/"
                f"{tears_missed}M "
                f"acc={accuracy:5.1f}% "
                f"switches={target_switches} "
                f"in_flight={unresolved}"
            )

    finally:
        env.close()

    count = len(
        rewards
    )

    win_rate = (
        wins / count * 100.0
        if count
        else 0.0
    )

    timeout_rate = (
        timeouts / count * 100.0
        if count
        else 0.0
    )

    no_hit_rate = (
        no_hit_episodes
        / count
        * 100.0
        if count
        else 0.0
    )

    no_hit_among_wins = (
        no_hit_wins
        / wins
        * 100.0
        if wins
        else 0.0
    )

    total_hit = int(
        sum(
            tears_hit_values
        )
    )

    total_missed = int(
        sum(
            tears_missed_values
        )
    )

    total_confirmed = (
        total_hit
        + total_missed
    )

    overall_accuracy = (
        total_hit
        / total_confirmed
        * 100.0
        if total_confirmed > 0
        else 0.0
    )

    print()
    print("=" * 60)
    print("EVALUATION RESULTS")
    print("=" * 60)

    print(
        f"Curriculum stage:      "
        f"{stage}"
    )

    print(
        f"Curriculum:            "
        f"{env.stage_config['name']}"
    )

    print()
    print(
        f"Episodes:              {count}"
    )
    print(
        f"Wins:                  {wins}"
    )
    print(
        f"Deaths:                {deaths}"
    )
    print(
        f"Timeouts:              {timeouts}"
    )
    print(
        f"Unknown:               {unknown}"
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
        f"Mean encounter enemies:"
        f" {mean(encounter_enemy_values):.2f}"
    )

    if target_room_exits > 0:
        print(
            f"Mean rooms exited:     "
            f"{mean(rooms_values):.2f}/"
            f"{target_room_exits}"
        )

    print(
        f"Mean reward:           "
        f"{mean(rewards):+.3f}"
    )

    print(
        f"Mean episode length:   "
        f"{mean(lengths):.1f}"
    )

    print(
        f"Mean target switches:  "
        f"{mean(target_switch_values):.1f}"
    )

    print()

    print(
        f"Mean HP lost:          "
        f"{mean(hp_lost_values):.2f}"
    )

    print(
        f"Mean remaining HP:     "
        f"{mean(remaining_hp_values):.2f}"
    )

    print(
        f"No-hit episodes:       "
        f"{no_hit_episodes}"
    )

    print(
        f"No-hit episode rate:   "
        f"{no_hit_rate:.1f}%"
    )

    print(
        f"No-hit wins:           "
        f"{no_hit_wins}"
    )

    print(
        f"No-hit among wins:     "
        f"{no_hit_among_wins:.1f}%"
    )

    print()
    print("-" * 60)
    print("COMBAT ACCURACY")
    print("-" * 60)

    print(
        f"Mean tears fired:      "
        f"{mean(tears_fired_values):.1f}"
    )

    print(
        f"Mean tears hit:        "
        f"{mean(tears_hit_values):.1f}"
    )

    print(
        f"Mean tears missed:     "
        f"{mean(tears_missed_values):.1f}"
    )

    print(
        f"Overall tear accuracy: "
        f"{overall_accuracy:.1f}%"
    )

    print()
    print("-" * 60)
    print("COMBAT SHOOT ACTION DISTRIBUTION")
    print("-" * 60)

    print_distribution(
        combat_shoot_counts
    )

    print()
    print("-" * 60)
    print("COMBAT MOVE ACTION DISTRIBUTION")
    print("-" * 60)

    print_distribution(
        combat_move_counts
    )

    print()
    print("-" * 60)
    print("NAVIGATION / COLLISION DIAGNOSTICS")
    print("-" * 60)

    print(
        f"Mean navigation steps: "
        f"{mean(nav_steps_values):.1f}"
    )

    print(
        f"Mean missing targets:  "
        f"{mean(nav_missing_values):.1f}"
    )

    print(
        f"Mean blocked moves:    "
        f"{mean(blocked_values):.1f}"
    )

    print(
        f"Mean stuck moves:      "
        f"{mean(stuck_values):.1f}"
    )

    print("=" * 60)


if __name__ == "__main__":
    main()