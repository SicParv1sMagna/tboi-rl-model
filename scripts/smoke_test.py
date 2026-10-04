from isaac_rl import IsaacEnv


def main() -> None:
    print("=" * 60)
    print("ISAAC RL - SMOKE TEST")
    print("=" * 60)

    print(
        "Start this script first, "
        "then start Isaac with --luadebug."
    )

    print()

    env = IsaacEnv(
        max_episode_steps=100,
        render_mode="human",
    )

    try:
        observation, info = env.reset()

        print(
            "Initial observation:",
            observation,
        )

        print(
            "Initial info:",
            info,
        )

        total_reward = 0.0

        for step_number in range(100):
            action = env.action_space.sample()

            (
                observation,
                reward,
                terminated,
                truncated,
                info,
            ) = env.step(action)

            total_reward += reward

            print(
                f"step={step_number + 1:03d} "
                f"action={action.tolist()} "
                f"reward={reward:+.3f} "
                f"total={total_reward:+.3f} "
                f"enemies={info['enemy_count']} "
                f"hp={info['player_hp']:.1f}"
            )

            if terminated or truncated:
                print(
                    "Episode finished:",
                    info,
                )

                break

    finally:
        env.close()


if __name__ == "__main__":
    main()