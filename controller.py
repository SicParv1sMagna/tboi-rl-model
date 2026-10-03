from isaac_env import IsaacEnv


def main() -> None:
    print("=" * 60)
    print("ISAAC RL CONTROLLER - RANDOM SMOKE TEST")
    print("=" * 60)
    print()
    print("Waiting for Isaac...")
    print()

    env = IsaacEnv(
        max_episode_steps=300,
        render_mode="human",
    )

    try:
        observation, info = env.reset()

        print()
        print("Environment ready!")
        print("Observation:", observation)
        print("Info:", info)
        print()

        total_reward = 0.0

        for step_number in range(300):
            # Пока никакой нейросети.
            # Просто случайное действие:
            #
            # [move, shoot]
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
                f"step={step_number + 1:03d} | "
                f"action={action.tolist()} | "
                f"reward={reward:+.3f} | "
                f"total={total_reward:+.3f} | "
                f"hp={info['player_hp']:.1f} | "
                f"enemies={info['enemy_count']}"
            )

            if terminated:
                print()
                print("EPISODE TERMINATED")

                if info.get("enemy_defeated"):
                    print("Enemy defeated!")

                if info.get("dead"):
                    print("Isaac died.")

                break

            if truncated:
                print()
                print("EPISODE TRUNCATED")
                print("Step limit reached.")
                break

    except KeyboardInterrupt:
        print()
        print("Stopped by user.")

    finally:
        env.close()


if __name__ == "__main__":
    main()