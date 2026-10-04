from stable_baselines3.common.env_checker import check_env

from isaac_rl import IsaacEnv

def main() -> None:
    print("=" * 60)
    print("ISAAC RL - ENVIRONMENT CHECK")
    print("=" * 60)

    env = IsaacEnv(
        max_episode_steps=100,
    )

    try:
        print("Environment created.")

        print(
            "Start Isaac with --luadebug "
            "and start a new run."
        )

        print()

        check_env(env, warn=True)

        print()

        print("=" * 60)
        print("ENVIRONMENT CHECK PASSED")
        print("=" * 60)
    finally:
        env.close()


if __name__ == "__main__":
    main()
