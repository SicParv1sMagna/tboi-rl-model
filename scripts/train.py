from __future__ import annotations

import argparse
import os
from datetime import datetime
from pathlib import Path

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import (
    CheckpointCallback,
)

from isaac_rl import IsaacEnv


PROJECT_ROOT = (
    Path(__file__)
    .resolve()
    .parents[1]
)

DEFAULT_MODELS_DIR = (
    PROJECT_ROOT
    / "models"
    / "structured_v2"
)

DEFAULT_RUNS_DIR = (
    PROJECT_ROOT
    / "runs"
    / "structured_v2"
)


STAGE_NAMES = {
    1: "combat_basics",
    2: "combat_health_accuracy",
    3: "combat_multi_enemy",
    4: "single_room_navigation",
    5: "three_room_run",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--timesteps",
        type=int,
        default=50_000,
    )

    parser.add_argument(
        "--resume",
        type=Path,
        default=None,
    )

    parser.add_argument(
        "--stage",
        type=int,
        choices=(1, 2, 3, 4, 5),
        default=None,
    )

    parser.add_argument(
        "--checkpoint-freq",
        type=int,
        default=10_000,
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=None,
    )

    parser.add_argument(
        "--models-dir",
        type=Path,
        default=DEFAULT_MODELS_DIR,
    )

    parser.add_argument(
        "--runs-dir",
        type=Path,
        default=DEFAULT_RUNS_DIR,
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


def main() -> None:
    args = parse_args()

    stage = resolve_stage(
        args.stage
    )

    if stage not in STAGE_NAMES:
        raise ValueError(
            "Stage must be 1..5"
        )

    stage_name = (
        STAGE_NAMES[stage]
    )

    timestamp = (
        datetime.now()
        .strftime(
            "%Y-%m-%d_%H-%M-%S"
        )
    )

    models_dir = (
        args.models_dir
        / f"stage_{stage}_{stage_name}"
    )

    runs_dir = (
        args.runs_dir
        / f"stage_{stage}_{stage_name}"
    )

    models_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    runs_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    print("=" * 60)
    print("ISAAC RL TRAINING")
    print("=" * 60)
    print(f"Stage:      {stage}")
    print(f"Curriculum: {stage_name}")
    print(f"Timesteps:  {args.timesteps}")
    print(f"Models:     {models_dir}")
    print("=" * 60)

    env = IsaacEnv(
        curriculum_stage=stage
    )

    callback = None

    if args.checkpoint_freq > 0:
        callback = CheckpointCallback(
            save_freq=(
                args.checkpoint_freq
            ),
            save_path=str(
                models_dir
            ),
            name_prefix=(
                f"ppo_stage{stage}"
                f"_{timestamp}"
                "_checkpoint"
            ),
        )

    try:
        if args.resume is not None:
            resume_path = (
                resolve_model_path(
                    args.resume
                )
            )

            print(
                f"[train] Resume: "
                f"{resume_path}"
            )

            model = PPO.load(
                str(resume_path),
                env=env,
                tensorboard_log=str(
                    runs_dir
                ),
            )

            reset_num_timesteps = False

        else:
            model = PPO(
                "MlpPolicy",
                env,
                verbose=1,
                tensorboard_log=str(
                    runs_dir
                ),
                seed=args.seed,
            )

            reset_num_timesteps = True

        model.learn(
            total_timesteps=(
                args.timesteps
            ),
            callback=callback,
            reset_num_timesteps=(
                reset_num_timesteps
            ),
        )

        final_path = (
            models_dir
            / (
                f"ppo_stage{stage}"
                f"_{timestamp}"
                "_final"
            )
        )

        model.save(
            str(final_path)
        )

        print()
        print(
            f"Saved: "
            f"{final_path}.zip"
        )

    finally:
        env.close()


if __name__ == "__main__":
    main()