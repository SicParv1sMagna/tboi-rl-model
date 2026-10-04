from __future__ import annotations

import json
import socket
import time
from typing import Any, Callable

import gymnasium as gym
import numpy as np
from gymnasium import spaces


# =========================================================
# NETWORK CONFIG
# =========================================================

HOST = "127.0.0.1"
PORT = 5000


# =========================================================
# ACTION NAMES
# =========================================================

MOVE_NAMES = (
    "NONE",
    "LEFT",
    "RIGHT",
    "UP",
    "DOWN",
)

SHOOT_NAMES = (
    "NONE",
    "LEFT",
    "RIGHT",
    "UP",
    "DOWN",
)


# =========================================================
# REWARD CONFIG
# =========================================================

STEP_PENALTY = -0.01

DAMAGE_DEALT_REWARD = 0.05

KILL_REWARD = 3.0

VICTORY_REWARD = 20.0

HP_LOST_PENALTY = 5.0

VICTORY_HP_BONUS_MULTIPLIER = 1.5

DEATH_PENALTY = -25.0

TIMEOUT_PENALTY = -2.0


class IsaacEnv(gym.Env):
    """
    Gymnasium environment for The Binding of Isaac: Afterbirth+.

    Action:
        [move, shoot]

        0 = NONE
        1 = LEFT
        2 = RIGHT
        3 = UP
        4 = DOWN

    Observation contains 13 normalized float values:

        0  player x relative to room center
        1  player y relative to room center

        2  player velocity x
        3  player velocity y

        4  red hearts
        5  soul hearts

        6  nearest enemy dx
        7  nearest enemy dy

        8  nearest enemy velocity x
        9  nearest enemy velocity y

        10 nearest enemy hp fraction

        11 enemy count

        12 room clear flag
    """

    metadata = {
        "render_modes": ["human"],
        "render_fps": 10,
    }

    def __init__(
        self,
        host: str = HOST,
        port: int = PORT,
        *,
        max_episode_steps: int = 600,
        socket_timeout: float = 5.0,
        render_mode: str | None = None,
    ) -> None:
        super().__init__()

        self.host = host
        self.port = port

        self.max_episode_steps = max_episode_steps
        self.socket_timeout = socket_timeout

        self.render_mode = render_mode

        # -------------------------------------------------
        # Gymnasium spaces
        # -------------------------------------------------

        # action = [move, shoot]
        self.action_space = spaces.MultiDiscrete(
            [5, 5]
        )

        self.observation_space = spaces.Box(
            low=-2.0,
            high=2.0,
            shape=(13,),
            dtype=np.float32,
        )

        # -------------------------------------------------
        # TCP
        # -------------------------------------------------

        self._server: socket.socket | None = None
        self._conn: socket.socket | None = None

        self._rx_buffer = b""

        # -------------------------------------------------
        # Environment state
        # -------------------------------------------------

        self._last_state: dict[str, Any] | None = None

        self._episode_steps = 0
        self._action_id = 0

        # Total damage taken during the current episode.
        self._episode_hp_lost = 0.0

        self._open_server()

    # =====================================================
    # TCP SERVER
    # =====================================================

    def _open_server(self) -> None:
        server = socket.socket(
            socket.AF_INET,
            socket.SOCK_STREAM,
        )

        server.setsockopt(
            socket.SOL_SOCKET,
            socket.SO_REUSEADDR,
            1,
        )

        server.bind(
            (
                self.host,
                self.port,
            )
        )

        server.listen(1)

        self._server = server

        print(
            f"[IsaacEnv] Listening on "
            f"{self.host}:{self.port}"
        )

    def _ensure_connection(self) -> None:
        if self._conn is not None:
            return

        if self._server is None:
            self._open_server()

        assert self._server is not None

        print(
            "[IsaacEnv] Waiting for Isaac..."
        )

        conn, addr = self._server.accept()

        conn.setsockopt(
            socket.IPPROTO_TCP,
            socket.TCP_NODELAY,
            1,
        )

        conn.settimeout(
            self.socket_timeout
        )

        self._conn = conn
        self._rx_buffer = b""

        print(
            f"[IsaacEnv] Isaac connected: {addr}"
        )

    def _drop_connection(self) -> None:
        if self._conn is not None:
            try:
                self._conn.close()

            except OSError:
                pass

        self._conn = None
        self._rx_buffer = b""

    # =====================================================
    # SEND
    # =====================================================

    def _send_json(
        self,
        payload: dict[str, Any],
    ) -> None:
        self._ensure_connection()

        assert self._conn is not None

        data = (
            json.dumps(
                payload,
                separators=(",", ":"),
            )
            + "\n"
        ).encode("utf-8")

        try:
            self._conn.sendall(data)

        except OSError:
            self._drop_connection()
            raise

    # =====================================================
    # RECEIVE
    # =====================================================

    def _recv_line(self) -> bytes:
        self._ensure_connection()

        assert self._conn is not None

        while b"\n" not in self._rx_buffer:
            try:
                chunk = self._conn.recv(
                    65536
                )

            except socket.timeout as exc:
                raise TimeoutError(
                    "Timed out waiting for Isaac state."
                ) from exc

            except OSError:
                self._drop_connection()
                raise

            if not chunk:
                self._drop_connection()

                raise ConnectionError(
                    "Isaac disconnected."
                )

            self._rx_buffer += chunk

            if len(self._rx_buffer) > 1_000_000:
                self._rx_buffer = b""

                raise RuntimeError(
                    "TCP receive buffer overflow."
                )

        line, self._rx_buffer = (
            self._rx_buffer.split(
                b"\n",
                1,
            )
        )

        return line

    def _recv_state(
        self,
    ) -> dict[str, Any]:
        while True:
            line = self._recv_line()

            if not line:
                continue

            try:
                message = json.loads(
                    line.decode("utf-8")
                )

            except json.JSONDecodeError:
                continue

            if (
                isinstance(message, dict)
                and message.get("type") == "state"
            ):
                return message

    def _wait_for(
        self,
        predicate: Callable[
            [dict[str, Any]],
            bool,
        ],
        *,
        timeout: float = 15.0,
    ) -> dict[str, Any]:
        deadline = (
            time.monotonic()
            + timeout
        )

        while True:
            if time.monotonic() > deadline:
                raise TimeoutError(
                    "Timed out waiting for "
                    "the expected Isaac state."
                )

            state = self._recv_state()

            if predicate(state):
                return state

    # =====================================================
    # OBSERVATION
    # =====================================================

    @staticmethod
    def _clip(
        value: float,
        low: float = -2.0,
        high: float = 2.0,
    ) -> float:
        return float(
            np.clip(
                value,
                low,
                high,
            )
        )

    def _state_to_obs(
        self,
        state: dict[str, Any],
    ) -> np.ndarray:
        player = state.get(
            "player",
            {},
        )

        room = state.get(
            "room",
            {},
        )

        enemies = state.get(
            "enemies",
            [],
        )

        # -------------------------------------------------
        # Player
        # -------------------------------------------------

        px = float(
            player.get(
                "x",
                0.0,
            )
        )

        py = float(
            player.get(
                "y",
                0.0,
            )
        )

        pvx = float(
            player.get(
                "velocity_x",
                0.0,
            )
        )

        pvy = float(
            player.get(
                "velocity_y",
                0.0,
            )
        )

        # -------------------------------------------------
        # Room
        # -------------------------------------------------

        center_x = float(
            room.get(
                "center_x",
                320.0,
            )
        )

        center_y = float(
            room.get(
                "center_y",
                280.0,
            )
        )

        # -------------------------------------------------
        # Health
        # -------------------------------------------------

        hearts = float(
            player.get(
                "hearts",
                0.0,
            )
        )

        soul_hearts = float(
            player.get(
                "soul_hearts",
                0.0,
            )
        )

        # -------------------------------------------------
        # Nearest enemy
        # -------------------------------------------------

        nearest = None
        nearest_dist_sq = None

        for enemy in enemies:
            ex = float(
                enemy.get(
                    "x",
                    0.0,
                )
            )

            ey = float(
                enemy.get(
                    "y",
                    0.0,
                )
            )

            dx = ex - px
            dy = ey - py

            dist_sq = (
                dx * dx
                + dy * dy
            )

            if (
                nearest_dist_sq is None
                or dist_sq < nearest_dist_sq
            ):
                nearest_dist_sq = dist_sq
                nearest = enemy

        if nearest is None:
            enemy_dx = 0.0
            enemy_dy = 0.0

            enemy_vx = 0.0
            enemy_vy = 0.0

            enemy_hp_fraction = 0.0

        else:
            enemy_dx = (
                float(
                    nearest.get(
                        "x",
                        0.0,
                    )
                )
                - px
            )

            enemy_dy = (
                float(
                    nearest.get(
                        "y",
                        0.0,
                    )
                )
                - py
            )

            enemy_vx = float(
                nearest.get(
                    "velocity_x",
                    0.0,
                )
            )

            enemy_vy = float(
                nearest.get(
                    "velocity_y",
                    0.0,
                )
            )

            enemy_hp = float(
                nearest.get(
                    "hp",
                    0.0,
                )
            )

            enemy_max_hp = max(
                1.0,
                float(
                    nearest.get(
                        "max_hp",
                        enemy_hp,
                    )
                ),
            )

            enemy_hp_fraction = (
                enemy_hp
                / enemy_max_hp
            )

        # -------------------------------------------------
        # Room state
        # -------------------------------------------------

        enemy_count = float(
            state.get(
                "enemy_count",
                len(enemies),
            )
        )

        room_clear = (
            1.0
            if state.get(
                "room_clear",
                False,
            )
            else 0.0
        )

        # -------------------------------------------------
        # Build observation
        # -------------------------------------------------

        observation = np.array(
            [
                self._clip(
                    (px - center_x)
                    / 400.0
                ),

                self._clip(
                    (py - center_y)
                    / 300.0
                ),

                self._clip(
                    pvx / 10.0
                ),

                self._clip(
                    pvy / 10.0
                ),

                self._clip(
                    hearts / 24.0,
                    0.0,
                    2.0,
                ),

                self._clip(
                    soul_hearts / 24.0,
                    0.0,
                    2.0,
                ),

                self._clip(
                    enemy_dx / 400.0
                ),

                self._clip(
                    enemy_dy / 300.0
                ),

                self._clip(
                    enemy_vx / 10.0
                ),

                self._clip(
                    enemy_vy / 10.0
                ),

                self._clip(
                    enemy_hp_fraction,
                    0.0,
                    2.0,
                ),

                self._clip(
                    enemy_count / 10.0,
                    0.0,
                    2.0,
                ),

                room_clear,
            ],
            dtype=np.float32,
        )

        return observation

    # =====================================================
    # HEALTH
    # =====================================================

    @staticmethod
    def _player_hp(
        state: dict[str, Any],
    ) -> float:
        player = state.get(
            "player",
            {},
        )

        return (
            float(
                player.get(
                    "hearts",
                    0.0,
                )
            )
            + float(
                player.get(
                    "soul_hearts",
                    0.0,
                )
            )
        )

    # =====================================================
    # REWARD
    # =====================================================

    def _calculate_reward(
        self,
        previous: dict[str, Any],
        current: dict[str, Any],
    ) -> tuple[
        float,
        bool,
        dict[str, Any],
    ]:
        # -------------------------------------------------
        # Small time penalty
        # -------------------------------------------------

        reward = STEP_PENALTY

        # -------------------------------------------------
        # Enemy damage
        # -------------------------------------------------

        previous_enemy_hp = float(
            previous.get(
                "total_enemy_hp",
                0.0,
            )
        )

        current_enemy_hp = float(
            current.get(
                "total_enemy_hp",
                0.0,
            )
        )

        damage_dealt = max(
            0.0,
            previous_enemy_hp
            - current_enemy_hp,
        )

        damage_reward = (
            damage_dealt
            * DAMAGE_DEALT_REWARD
        )

        reward += damage_reward

        # -------------------------------------------------
        # Enemy kills
        # -------------------------------------------------

        previous_enemy_count = int(
            previous.get(
                "enemy_count",
                0,
            )
        )

        current_enemy_count = int(
            current.get(
                "enemy_count",
                0,
            )
        )

        killed_enemies = max(
            0,
            previous_enemy_count
            - current_enemy_count,
        )

        kill_reward = (
            killed_enemies
            * KILL_REWARD
        )

        reward += kill_reward

        # -------------------------------------------------
        # Player damage
        # -------------------------------------------------

        previous_hp = self._player_hp(
            previous
        )

        current_hp = self._player_hp(
            current
        )

        hp_lost = max(
            0.0,
            previous_hp
            - current_hp,
        )

        damage_taken_penalty = (
            hp_lost
            * HP_LOST_PENALTY
        )

        reward -= damage_taken_penalty

        # -------------------------------------------------
        # Episode result
        # -------------------------------------------------

        enemy_defeated = (
            previous_enemy_count > 0
            and current_enemy_count == 0
        )

        dead = bool(
            current.get(
                "dead",
                False,
            )
            or current.get(
                "game_over",
                False,
            )
        )

        # -------------------------------------------------
        # Victory
        # -------------------------------------------------

        victory_health_bonus = 0.0

        if enemy_defeated:
            reward += VICTORY_REWARD

            # Stage 2:
            # winning with more HP is more valuable.
            victory_health_bonus = (
                current_hp
                * VICTORY_HP_BONUS_MULTIPLIER
            )

            reward += victory_health_bonus

        # -------------------------------------------------
        # Death
        # -------------------------------------------------

        if dead:
            reward += DEATH_PENALTY

        terminated = (
            enemy_defeated
            or dead
        )

        # -------------------------------------------------
        # Debug / evaluation information
        # -------------------------------------------------

        info = {
            "damage_dealt": damage_dealt,
            "damage_reward": damage_reward,

            "killed_enemies": killed_enemies,
            "kill_reward": kill_reward,

            "hp_lost": hp_lost,
            "damage_taken_penalty":
                damage_taken_penalty,

            "remaining_hp": current_hp,

            "victory_health_bonus":
                victory_health_bonus,

            "enemy_defeated":
                enemy_defeated,

            "dead":
                dead,
        }

        return (
            reward,
            terminated,
            info,
        )

    # =====================================================
    # RESET
    # =====================================================

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[
        np.ndarray,
        dict[str, Any],
    ]:
        super().reset(
            seed=seed
        )

        self._ensure_connection()

        # On the very first reset we first receive
        # the state of the manually started run.
        if self._last_state is None:
            baseline = self._wait_for(
                lambda state: bool(
                    state.get(
                        "ready",
                        False,
                    )
                ),
                timeout=20.0,
            )

        else:
            baseline = self._last_state

        old_episode_id = int(
            baseline.get(
                "episode_id",
                -1,
            )
        )

        print(
            f"[IsaacEnv] Resetting episode "
            f"{old_episode_id}..."
        )

        self._send_json(
            {
                "type": "reset",
            }
        )

        new_state = self._wait_for(
            lambda state: (
                bool(
                    state.get(
                        "ready",
                        False,
                    )
                )
                and int(
                    state.get(
                        "episode_id",
                        -1,
                    )
                )
                > old_episode_id
                and int(
                    state.get(
                        "enemy_count",
                        0,
                    )
                )
                > 0
            ),
            timeout=20.0,
        )

        self._last_state = new_state

        self._episode_steps = 0
        self._action_id = 0

        self._episode_hp_lost = 0.0

        observation = self._state_to_obs(
            new_state
        )

        info = self._make_info(
            new_state
        )

        info["episode_steps"] = 0
        info["episode_hp_lost"] = 0.0
        info["no_hit"] = True

        print(
            f"[IsaacEnv] Episode "
            f"{info['episode_id']} ready."
        )

        return (
            observation,
            info,
        )

    # =====================================================
    # STEP
    # =====================================================

    def step(
        self,
        action: np.ndarray,
    ) -> tuple[
        np.ndarray,
        float,
        bool,
        bool,
        dict[str, Any],
    ]:
        if self._last_state is None:
            raise RuntimeError(
                "Call env.reset() before env.step()."
            )

        # -------------------------------------------------
        # Validate action
        # -------------------------------------------------

        action_array = np.asarray(
            action,
            dtype=np.int64,
        ).reshape(-1)

        if action_array.shape != (2,):
            raise ValueError(
                "Action must be exactly "
                "[move, shoot]."
            )

        move_index = int(
            action_array[0]
        )

        shoot_index = int(
            action_array[1]
        )

        if not (
            0
            <= move_index
            < len(MOVE_NAMES)
        ):
            raise ValueError(
                f"Invalid move index: "
                f"{move_index}"
            )

        if not (
            0
            <= shoot_index
            < len(SHOOT_NAMES)
        ):
            raise ValueError(
                f"Invalid shoot index: "
                f"{shoot_index}"
            )

        # -------------------------------------------------
        # Send action
        # -------------------------------------------------

        self._action_id += 1

        action_id = self._action_id

        episode_id = int(
            self._last_state.get(
                "episode_id",
                -1,
            )
        )

        self._send_json(
            {
                "type": "action",

                "action_id":
                    action_id,

                "enabled":
                    True,

                "move":
                    MOVE_NAMES[
                        move_index
                    ],

                "shoot":
                    SHOOT_NAMES[
                        shoot_index
                    ],
            }
        )

        # -------------------------------------------------
        # Wait for state after action
        # -------------------------------------------------

        new_state = self._wait_for(
            lambda state: (
                int(
                    state.get(
                        "episode_id",
                        -1,
                    )
                )
                == episode_id
                and int(
                    state.get(
                        "action_id",
                        -1,
                    )
                )
                >= action_id
            ),
            timeout=10.0,
        )

        # -------------------------------------------------
        # Reward
        # -------------------------------------------------

        (
            reward,
            terminated,
            reward_info,
        ) = self._calculate_reward(
            self._last_state,
            new_state,
        )

        # -------------------------------------------------
        # Episode statistics
        # -------------------------------------------------

        self._episode_steps += 1

        self._episode_hp_lost += float(
            reward_info["hp_lost"]
        )

        # -------------------------------------------------
        # Timeout
        # -------------------------------------------------

        truncated = (
            self._episode_steps
            >= self.max_episode_steps
        )

        if (
            truncated
            and not terminated
        ):
            reward += TIMEOUT_PENALTY

        # -------------------------------------------------
        # Store current state
        # -------------------------------------------------

        self._last_state = new_state

        observation = self._state_to_obs(
            new_state
        )

        # -------------------------------------------------
        # Info
        # -------------------------------------------------

        info = self._make_info(
            new_state
        )

        info.update(
            reward_info
        )

        info["episode_steps"] = (
            self._episode_steps
        )

        info["episode_hp_lost"] = (
            self._episode_hp_lost
        )

        info["no_hit"] = (
            self._episode_hp_lost
            == 0.0
        )

        info["timeout"] = bool(
            truncated
            and not terminated
        )

        return (
            observation,
            float(reward),
            bool(terminated),
            bool(truncated),
            info,
        )

    # =====================================================
    # INFO
    # =====================================================

    def _make_info(
        self,
        state: dict[str, Any],
    ) -> dict[str, Any]:
        return {
            "episode_id": int(
                state.get(
                    "episode_id",
                    -1,
                )
            ),

            "frame": int(
                state.get(
                    "frame",
                    -1,
                )
            ),

            "enemy_count": int(
                state.get(
                    "enemy_count",
                    0,
                )
            ),

            "total_enemy_hp": float(
                state.get(
                    "total_enemy_hp",
                    0.0,
                )
            ),

            "player_hp":
                self._player_hp(
                    state
                ),
        }

    # =====================================================
    # RENDER
    # =====================================================

    def render(self) -> None:
        if self._last_state is None:
            print(
                "[IsaacEnv] No state yet."
            )

            return

        info = self._make_info(
            self._last_state
        )

        control = self._last_state.get(
            "control",
            {},
        )

        print(
            "[IsaacEnv] "
            f"episode={info['episode_id']} "
            f"frame={info['frame']} "
            f"hp={info['player_hp']:.1f} "
            f"enemies={info['enemy_count']} "
            f"move={control.get('move')} "
            f"shoot={control.get('shoot')}"
        )

    # =====================================================
    # CLOSE
    # =====================================================

    def close(self) -> None:
        if self._conn is not None:
            try:
                self._send_json(
                    {
                        "type":
                            "action",

                        "action_id":
                            self._action_id
                            + 1,

                        "enabled":
                            False,

                        "move":
                            "NONE",

                        "shoot":
                            "NONE",
                    }
                )

            except Exception:
                pass

        self._drop_connection()

        if self._server is not None:
            try:
                self._server.close()

            except OSError:
                pass

        self._server = None