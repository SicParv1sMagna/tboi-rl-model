from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.monitor import Monitor

from isaac_rl import IsaacEnv


PROJECT_ROOT = Path(__file__).resolve().parent.parent

RUNS_DIR = PROJECT_ROOT / "runs"
MODELS_DIR = PROJECT_ROOT / "models"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Train a PPO agent "
            "in The Binding of Isaac."
        )
    )

    parser.add_argument(
        "--timesteps",
        type=int,
        default=10_000,
        help=(
            "Number of additional environment "
            "steps to train."
        ),
    )

    parser.add_argument(
        "--max-episode-steps",
        type=int,
        default=600,
        help=(
            "Maximum number of steps "
            "per episode."
        ),
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed.",
    )

    parser.add_argument(
        "--resume",
        type=Path,
        default=None,
        help=(
            "Optional path to an existing PPO "
            ".zip model to continue training."
        ),
    )

    parser.add_argument(
        "--name",
        type=str,
        default=None,
        help=(
            "Optional human-readable run name, "
            "for example stage2_health."
        ),
    )

    return parser.parse_args()


def resolve_model_path(
    path: Path,
) -> Path:
    if path.is_absolute():
        return path

    return PROJECT_ROOT / path


def create_run_name(
    requested_name: str | None,
    is_resume: bool,
) -> str:
    timestamp = datetime.now().strftime(
        "%Y-%m-%d_%H-%M-%S"
    )

    if requested_name:
        return (
            f"{requested_name}_{timestamp}"
        )

    if is_resume:
        prefix = "ppo_resume"
    else:
        prefix = "ppo"

    return (
        f"{prefix}_{timestamp}"
    )


def save_run_config(
    *,
    run_dir: Path,
    args: argparse.Namespace,
    run_name: str,
    resume_path: Path | None,
) -> None:
    config = {
        "run_name": run_name,
        "timesteps": args.timesteps,
        "max_episode_steps":
            args.max_episode_steps,
        "seed": args.seed,
        "resume_model": (
            str(resume_path)
            if resume_path is not None
            else None
        ),
    }

    config_path = (
        run_dir
        / "config.json"
    )

    with config_path.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            config,
            file,
            indent=4,
        )


def main() -> None:
    args = parse_args()

    # =====================================================
    # RESUME MODEL
    # =====================================================

    resume_path: Path | None = None

    if args.resume is not None:
        resume_path = resolve_model_path(
            args.resume
        )

        if not resume_path.exists():
            raise FileNotFoundError(
                f"Resume model not found: "
                f"{resume_path}"
            )

    # =====================================================
    # RUN DIRECTORIES
    # =====================================================

    run_name = create_run_name(
        requested_name=args.name,
        is_resume=resume_path is not None,
    )

    run_dir = (
        RUNS_DIR
        / run_name
    )

    checkpoint_dir = (
        run_dir
        / "checkpoints"
    )

    tensorboard_dir = (
        run_dir
        / "tensorboard"
    )

    run_dir.mkdir(
        parents=True,
        exist_ok=False,
    )

    checkpoint_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    tensorboard_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    MODELS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    save_run_config(
        run_dir=run_dir,
        args=args,
        run_name=run_name,
        resume_path=resume_path,
    )

    # =====================================================
    # OUTPUT
    # =====================================================

    print("=" * 60)
    print("ISAAC RL - PPO TRAINING")
    print("=" * 60)

    print(
        f"Run: {run_name}"
    )

    print(
        f"Additional timesteps: "
        f"{args.timesteps}"
    )

    print(
        f"Max episode steps: "
        f"{args.max_episode_steps}"
    )

    print(
        f"Seed: {args.seed}"
    )

    if resume_path is None:
        print(
            "Mode: fresh training"
        )

    else:
        print(
            "Mode: resume training"
        )

        print(
            f"Resume model: "
            f"{resume_path}"
        )

    print()

    print(
        "Start Isaac with --luadebug "
        "and start a new run."
    )

    print()

    # =====================================================
    # ENVIRONMENT
    # =====================================================

    base_env = IsaacEnv(
        max_episode_steps=
            args.max_episode_steps,
    )

    env = Monitor(
        base_env,
        filename=str(
            run_dir
            / "monitor.csv"
        ),
    )

    # =====================================================
    # CHECKPOINTS
    # =====================================================

    checkpoint_callback = (
        CheckpointCallback(
            save_freq=5_000,
            save_path=str(
                checkpoint_dir
            ),
            name_prefix="ppo_isaac",
        )
    )

    # =====================================================
    # MODEL
    # =====================================================

    if resume_path is None:
        # ---------------------------------------------
        # Fresh PPO model
        # ---------------------------------------------

        model = PPO(
            policy="MlpPolicy",
            env=env,

            learning_rate=3e-4,

            n_steps=256,
            batch_size=64,
            n_epochs=10,

            gamma=0.99,
            gae_lambda=0.95,

            clip_range=0.2,

            verbose=1,

            seed=args.seed,

            tensorboard_log=str(
                tensorboard_dir
            ),

            device="auto",
        )

        reset_num_timesteps = True

    else:
        # ---------------------------------------------
        # Continue training existing PPO model
        # ---------------------------------------------

        model = PPO.load(
            resume_path,
            env=env,

            tensorboard_log=str(
                tensorboard_dir
            ),
        )

        reset_num_timesteps = False

        print(
            f"Loaded model with "
            f"{model.num_timesteps} "
            f"existing timesteps."
        )

        print()

    # =====================================================
    # TRAIN
    # =====================================================

    try:
        model.learn(
            total_timesteps=
                args.timesteps,

            callback=
                checkpoint_callback,

            tb_log_name=
                "ppo",

            reset_num_timesteps=
                reset_num_timesteps,
        )

        # =================================================
        # SAVE FINAL MODEL
        # =================================================

        final_model_path = (
            MODELS_DIR
            / f"{run_name}_final"
        )

        model.save(
            final_model_path
        )

        print()
        print("=" * 60)
        print("TRAINING FINISHED")
        print("=" * 60)

        print(
            f"Total model timesteps: "
            f"{model.num_timesteps}"
        )

        print(
            f"Model saved to: "
            f"{final_model_path}.zip"
        )

        print(
            f"Run data saved to: "
            f"{run_dir}"
        )

    finally:
        env.close()


if __name__ == "__main__":
    main()