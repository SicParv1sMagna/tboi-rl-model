# TBOI RL Model

Проект по обучению RL-агента играть в The Binding Of Isaac: Afterbirth +

Цель - построить среду, в которой агент получает состояние игры, выбирает действие,
а Lua-мод передает это действие в игру. Агент обучается через reinforcement learning
с помощью PPO.

# Требования

Проект тестируется с:

- MacOS
- TBOI: Afterbirth+
- Python 3.10+;
- Gymnasium
- NumPy

Для Lua socket API TBOI должен запускаться с:

```
--luadebug
```

Флаг ослабляет ограничения Lua-среды Isaac. Использовать его рекомендуется только
с доверенным кодом.

# Training Env

Пока задача намеренно очень простая. Агент должен научиться ориентироваться относительно врага,
стрелять в правильном направлении, двигаться, избегать урона, уничтожать врага. Только после успешного
решения этой задачи среда будет усложняться.

# Roadmap

Следующие этапы проекта:

- [x] Read Isaac State From Lua
- [x] Control Isaac From Lua
- [x] Python-Lua TCP Bridge
- [x] Gymnasium env
- [x] Observation space
- [x] Action space
- [x] Basic reward
- [x] Automatic episode reset
- [x] Random smoke test
- [ ] Stable-Baselines3
- [ ] PPO training
- [ ] TensorBoard metrics
- [ ] Model checkpoints
- [ ] Better reward shaping
- [ ] Projectile observations
- [ ] Multiple enemies
- [ ] Multiple enemy types
- [ ] Boss training
- [ ] Item observations
- [ ] Item selection
- [ ] Room navigation
- [ ] Full floor
- [ ] Full run

# Long-term goal

Конечная цель проекта - постепенный переход от простого боя с одним противником к полноценному прохождению
TBOI.