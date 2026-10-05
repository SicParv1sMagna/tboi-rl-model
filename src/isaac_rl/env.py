from __future__ import annotations

import json
import math
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
# STEP / TIME REWARD
# =========================================================

# Combat is deliberately a little more expensive than
# navigation.
#
# This discourages the old strategy:
#
#   stand near a door
#   shoot in one direction
#   wait for the enemy to walk into tears
#
COMBAT_STEP_PENALTY = -0.03

NAVIGATION_STEP_PENALTY = -0.01


# =========================================================
# ROOM OVERTIME
# =========================================================

# For the first 100 agent actions in each room, there is no
# additional overtime penalty.
ROOM_OVERTIME_START = 100

# After that, the extra cost grows smoothly:
#
# step 101 -> -0.001 extra
# step 120 -> -0.020 extra
# step 150 -> -0.050 extra
#
ROOM_OVERTIME_PENALTY_GROWTH = 0.001

# Never exceed this additional per-step room penalty.
MAX_ROOM_OVERTIME_PENALTY = 0.05


# =========================================================
# COMBAT REWARD
# =========================================================

DAMAGE_DEALT_REWARD = 0.05

KILL_REWARD = 3.0

ROOM_CLEAR_REWARD = 10.0

HP_LOST_PENALTY = 5.0


# =========================================================
# NAVIGATION / PROGRESS REWARD
# =========================================================

# Reward for completing a NEW unique combat room.
#
# Backtracking through an already completed room does not
# increase the Lua progress counter, therefore it receives
# no room completion reward.
ROOM_EXIT_REWARD = 10.0

TRAINING_COMPLETE_REWARD = 20.0

FINAL_HP_BONUS_MULTIPLIER = 1.5

NAVIGATION_PROGRESS_REWARD = 0.01

MAX_NAVIGATION_PROGRESS_REWARD = 0.5


# =========================================================
# FAILURE
# =========================================================

DEATH_PENALTY = -25.0

TIMEOUT_PENALTY = -2.0


class IsaacEnv(gym.Env):
    """
    Multi-room RL environment for
    The Binding of Isaac: Afterbirth+.

    Action:
        [move, shoot]

        0 = NONE
        1 = LEFT
        2 = RIGHT
        3 = UP
        4 = DOWN

    Observation:
        13 float values.

        0  player relative x
        1  player relative y

        2  player velocity x
        3  player velocity y

        4  red hearts
        5  soul hearts

        6  target dx
        7  target dy

        8  target velocity x
        9  target velocity y

        10 target hp fraction

        11 enemy count

        12 room clear

    Target semantics:

        room_clear == 0:
            nearest enemy

        room_clear == 1:
            navigation waypoint / exit target

    Observation remains shape=(13,), so previous PPO
    checkpoints remain compatible.
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

        self.max_episode_steps = (
            max_episode_steps
        )

        self.socket_timeout = (
            socket_timeout
        )

        self.render_mode = render_mode

        # -------------------------------------------------
        # Gymnasium spaces
        # -------------------------------------------------

        self.action_space = (
            spaces.MultiDiscrete(
                [5, 5]
            )
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

        self._server: (
            socket.socket
            | None
        ) = None

        self._conn: (
            socket.socket
            | None
        ) = None

        self._rx_buffer = b""

        # -------------------------------------------------
        # Environment state
        # -------------------------------------------------

        self._last_state: (
            dict[str, Any]
            | None
        ) = None

        self._episode_steps = 0

        self._room_steps = 0

        self._action_id = 0

        self._episode_hp_lost = 0.0

        self._open_server()

    # =====================================================
    # TCP SERVER
    # =====================================================

    def _open_server(
        self,
    ) -> None:
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
            "[IsaacEnv] Listening on "
            f"{self.host}:{self.port}"
        )

    def _ensure_connection(
        self,
    ) -> None:
        if self._conn is not None:
            return

        if self._server is None:
            self._open_server()

        assert self._server is not None

        print(
            "[IsaacEnv] Waiting for Isaac..."
        )

        conn, addr = (
            self._server.accept()
        )

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
            "[IsaacEnv] Isaac connected: "
            f"{addr}"
        )

    def _drop_connection(
        self,
    ) -> None:
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
        ).encode(
            "utf-8"
        )

        try:
            self._conn.sendall(
                data
            )

        except OSError:
            self._drop_connection()

            raise

    # =====================================================
    # RECEIVE
    # =====================================================

    def _recv_line(
        self,
    ) -> bytes:
        self._ensure_connection()

        assert self._conn is not None

        while b"\n" not in self._rx_buffer:
            try:
                chunk = self._conn.recv(
                    65536
                )

            except socket.timeout as exc:
                raise TimeoutError(
                    "Timed out waiting for "
                    "Isaac state."
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

            if (
                len(self._rx_buffer)
                > 1_000_000
            ):
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
                    line.decode(
                        "utf-8"
                    )
                )

            except json.JSONDecodeError:
                continue

            if (
                isinstance(
                    message,
                    dict,
                )
                and message.get(
                    "type"
                )
                == "state"
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
            if (
                time.monotonic()
                > deadline
            ):
                raise TimeoutError(
                    "Timed out waiting for "
                    "the expected Isaac state."
                )

            state = self._recv_state()

            if predicate(
                state
            ):
                return state

    # =====================================================
    # GENERIC HELPERS
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

    @staticmethod
    def _player_position(
        state: dict[str, Any],
    ) -> tuple[
        float,
        float,
    ]:
        player = state.get(
            "player",
            {},
        )

        return (
            float(
                player.get(
                    "x",
                    0.0,
                )
            ),
            float(
                player.get(
                    "y",
                    0.0,
                )
            ),
        )

    @staticmethod
    def _navigation_target(
        state: dict[str, Any],
    ) -> dict[str, Any] | None:
        target = state.get(
            "navigation_target"
        )

        if isinstance(
            target,
            dict,
        ):
            return target

        return None

    @staticmethod
    def _room_index(
        state: dict[str, Any],
    ) -> int:
        return int(
            state.get(
                "room_index",
                -1,
            )
        )

    @staticmethod
    def _rooms_completed(
        state: dict[str, Any],
    ) -> int:
        # New explicit name.
        if "rooms_completed" in state:
            return int(
                state.get(
                    "rooms_completed",
                    0,
                )
            )

        # Compatibility with previous main.lua.
        return int(
            state.get(
                "rooms_exited",
                0,
            )
        )

    @staticmethod
    def _target_room_count(
        state: dict[str, Any],
    ) -> int:
        if "target_training_rooms" in state:
            return int(
                state.get(
                    "target_training_rooms",
                    3,
                )
            )

        return int(
            state.get(
                "target_room_exits",
                3,
            )
        )

    @staticmethod
    def _training_complete(
        state: dict[str, Any],
    ) -> bool:
        return bool(
            state.get(
                "training_complete",
                False,
            )
        )

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
    # ROOM TRANSITION
    # =====================================================

    def _room_transitioned(
        self,
        previous: dict[str, Any],
        current: dict[str, Any],
    ) -> bool:
        previous_room = (
            self._room_index(
                previous
            )
        )

        current_room = (
            self._room_index(
                current
            )
        )

        return (
            previous_room >= 0
            and current_room >= 0
            and previous_room
            != current_room
        )

    # =====================================================
    # NAVIGATION
    # =====================================================

    def _navigation_distance(
        self,
        state: dict[str, Any],
    ) -> float | None:
        if self._training_complete(
            state
        ):
            return None

        if not bool(
            state.get(
                "room_clear",
                False,
            )
        ):
            return None

        target = (
            self._navigation_target(
                state
            )
        )

        if target is None:
            return None

        px, py = (
            self._player_position(
                state
            )
        )

        target_x = float(
            target.get(
                "x",
                px,
            )
        )

        target_y = float(
            target.get(
                "y",
                py,
            )
        )

        return math.hypot(
            target_x - px,
            target_y - py,
        )

    # =====================================================
    # OBSERVATION
    # =====================================================

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

        room_clear = bool(
            state.get(
                "room_clear",
                False,
            )
        )

        # -------------------------------------------------
        # Shared target channels
        # -------------------------------------------------

        target_dx = 0.0
        target_dy = 0.0

        target_vx = 0.0
        target_vy = 0.0

        target_hp_fraction = 0.0

        # -------------------------------------------------
        # NAVIGATION TARGET
        # -------------------------------------------------

        if room_clear:
            navigation_target = (
                self._navigation_target(
                    state
                )
            )

            if (
                navigation_target
                is not None
            ):
                target_dx = (
                    float(
                        navigation_target.get(
                            "x",
                            px,
                        )
                    )
                    - px
                )

                target_dy = (
                    float(
                        navigation_target.get(
                            "y",
                            py,
                        )
                    )
                    - py
                )

        # -------------------------------------------------
        # COMBAT TARGET
        # -------------------------------------------------

        else:
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
                    nearest_dist_sq
                    is None
                    or dist_sq
                    < nearest_dist_sq
                ):
                    nearest_dist_sq = (
                        dist_sq
                    )

                    nearest = enemy


            if nearest is not None:
                target_dx = (
                    float(
                        nearest.get(
                            "x",
                            0.0,
                        )
                    )
                    - px
                )

                target_dy = (
                    float(
                        nearest.get(
                            "y",
                            0.0,
                        )
                    )
                    - py
                )

                target_vx = float(
                    nearest.get(
                        "velocity_x",
                        0.0,
                    )
                )

                target_vy = float(
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

                target_hp_fraction = (
                    enemy_hp
                    / enemy_max_hp
                )

        # -------------------------------------------------
        # Enemy count
        # -------------------------------------------------

        enemy_count = float(
            state.get(
                "enemy_count",
                len(enemies),
            )
        )

        # -------------------------------------------------
        # Observation
        # -------------------------------------------------

        return np.array(
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
                    target_dx / 400.0
                ),

                self._clip(
                    target_dy / 300.0
                ),

                self._clip(
                    target_vx / 10.0
                ),

                self._clip(
                    target_vy / 10.0
                ),

                self._clip(
                    target_hp_fraction,
                    0.0,
                    2.0,
                ),

                self._clip(
                    enemy_count / 10.0,
                    0.0,
                    2.0,
                ),

                (
                    1.0
                    if room_clear
                    else 0.0
                ),
            ],
            dtype=np.float32,
        )

    # =====================================================
    # REWARD
    # =====================================================

    def _calculate_reward(
        self,
        previous: dict[str, Any],
        current: dict[str, Any],
        *,
        room_steps: int,
    ) -> tuple[
        float,
        bool,
        dict[str, Any],
    ]:
        # -------------------------------------------------
        # PHASE-DEPENDENT STEP PENALTY
        # -------------------------------------------------

        navigation_mode = bool(
            previous.get(
                "room_clear",
                False,
            )
        )

        if navigation_mode:
            step_penalty = (
                NAVIGATION_STEP_PENALTY
            )

        else:
            step_penalty = (
                COMBAT_STEP_PENALTY
            )


        reward = (
            step_penalty
        )

        # -------------------------------------------------
        # ROOM OVERTIME
        # -------------------------------------------------

        overtime_steps = max(
            0,
            room_steps
            - ROOM_OVERTIME_START,
        )

        room_overtime_penalty = min(
            MAX_ROOM_OVERTIME_PENALTY,
            overtime_steps
            * ROOM_OVERTIME_PENALTY_GROWTH,
        )

        reward -= (
            room_overtime_penalty
        )

        # -------------------------------------------------
        # ENEMY DAMAGE
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

        reward += (
            damage_reward
        )

        # -------------------------------------------------
        # KILLS
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

        reward += (
            kill_reward
        )

        # -------------------------------------------------
        # PLAYER DAMAGE
        # -------------------------------------------------

        previous_hp = (
            self._player_hp(
                previous
            )
        )

        current_hp = (
            self._player_hp(
                current
            )
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

        reward -= (
            damage_taken_penalty
        )

        # -------------------------------------------------
        # ROOM CLEAR
        # -------------------------------------------------

        enemy_defeated = (
            previous_enemy_count > 0
            and current_enemy_count == 0
        )

        room_clear_reward = 0.0


        if enemy_defeated:
            room_clear_reward = (
                ROOM_CLEAR_REWARD
            )

            reward += (
                room_clear_reward
            )

        # -------------------------------------------------
        # NAVIGATION PROGRESS
        # -------------------------------------------------

        previous_navigation_distance = (
            self._navigation_distance(
                previous
            )
        )

        current_navigation_distance = (
            self._navigation_distance(
                current
            )
        )

        navigation_progress = 0.0

        navigation_progress_reward = 0.0


        if (
            previous_navigation_distance
            is not None
            and current_navigation_distance
            is not None
        ):
            navigation_progress = (
                previous_navigation_distance
                - current_navigation_distance
            )

            navigation_progress_reward = (
                navigation_progress
                * NAVIGATION_PROGRESS_REWARD
            )

            navigation_progress_reward = (
                float(
                    np.clip(
                        navigation_progress_reward,
                        -MAX_NAVIGATION_PROGRESS_REWARD,
                        MAX_NAVIGATION_PROGRESS_REWARD,
                    )
                )
            )

            reward += (
                navigation_progress_reward
            )

        # -------------------------------------------------
        # ACTUAL ROOM TRANSITION
        # -------------------------------------------------

        room_transitioned = (
            self._room_transitioned(
                previous,
                current,
            )
        )

        # -------------------------------------------------
        # UNIQUE CURRICULUM PROGRESS
        # -------------------------------------------------

        previous_rooms_completed = (
            self._rooms_completed(
                previous
            )
        )

        current_rooms_completed = (
            self._rooms_completed(
                current
            )
        )

        rooms_completed_delta = max(
            0,
            current_rooms_completed
            - previous_rooms_completed,
        )

        room_exit_reward = (
            rooms_completed_delta
            * ROOM_EXIT_REWARD
        )

        reward += (
            room_exit_reward
        )

        # A transition with no progress means we entered
        # an already-completed room.
        backtracked = (
            room_transitioned
            and rooms_completed_delta == 0
        )

        # -------------------------------------------------
        # FULL TASK COMPLETE
        # -------------------------------------------------

        training_complete = (
            self._training_complete(
                current
            )
        )

        was_training_complete = (
            self._training_complete(
                previous
            )
        )

        completed_this_step = (
            training_complete
            and not was_training_complete
        )

        training_complete_reward = 0.0

        final_health_bonus = 0.0


        if completed_this_step:
            training_complete_reward = (
                TRAINING_COMPLETE_REWARD
            )

            reward += (
                training_complete_reward
            )

            final_health_bonus = (
                current_hp
                * FINAL_HP_BONUS_MULTIPLIER
            )

            reward += (
                final_health_bonus
            )

        # -------------------------------------------------
        # DEATH
        # -------------------------------------------------

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


        if dead:
            reward += (
                DEATH_PENALTY
            )

        # -------------------------------------------------
        # TERMINATION
        # -------------------------------------------------

        terminated = (
            dead
            or training_complete
        )

        # -------------------------------------------------
        # INFO
        # -------------------------------------------------

        info = {
            "navigation_mode":
                navigation_mode,

            "step_penalty":
                step_penalty,

            "room_steps_for_reward":
                room_steps,

            "overtime_steps":
                overtime_steps,

            "room_overtime_penalty":
                room_overtime_penalty,

            "damage_dealt":
                damage_dealt,

            "damage_reward":
                damage_reward,

            "killed_enemies":
                killed_enemies,

            "kill_reward":
                kill_reward,

            "hp_lost":
                hp_lost,

            "damage_taken_penalty":
                damage_taken_penalty,

            "remaining_hp":
                current_hp,

            "enemy_defeated":
                enemy_defeated,

            "room_clear_reward":
                room_clear_reward,

            "navigation_progress":
                navigation_progress,

            "navigation_progress_reward":
                navigation_progress_reward,

            "room_transitioned":
                room_transitioned,

            "backtracked":
                backtracked,

            "rooms_completed_delta":
                rooms_completed_delta,

            # Compatibility with previous evaluation code.
            "room_exits_delta":
                rooms_completed_delta,

            "exited_room_this_step":
                room_transitioned,

            "room_exit_reward":
                room_exit_reward,

            "rooms_completed":
                current_rooms_completed,

            # Compatibility alias.
            "rooms_exited":
                current_rooms_completed,

            "target_room_exits":
                self._target_room_count(
                    current
                ),

            "training_complete":
                training_complete,

            "completed_this_step":
                completed_this_step,

            "training_complete_reward":
                training_complete_reward,

            "final_health_bonus":
                final_health_bonus,

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
        options: (
            dict[str, Any]
            | None
        ) = None,
    ) -> tuple[
        np.ndarray,
        dict[str, Any],
    ]:
        super().reset(
            seed=seed
        )

        self._ensure_connection()

        # -------------------------------------------------
        # Baseline episode
        # -------------------------------------------------

        if self._last_state is None:
            baseline = (
                self._wait_for(
                    lambda state: bool(
                        state.get(
                            "ready",
                            False,
                        )
                    ),
                    timeout=20.0,
                )
            )

        else:
            baseline = (
                self._last_state
            )


        old_episode_id = int(
            baseline.get(
                "episode_id",
                -1,
            )
        )


        print(
            "[IsaacEnv] Resetting episode "
            f"{old_episode_id}..."
        )


        self._send_json(
            {
                "type":
                    "reset",
            }
        )


        # -------------------------------------------------
        # Wait for first controlled enemy
        # -------------------------------------------------

        new_state = (
            self._wait_for(
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
                    and self._rooms_completed(
                        state
                    )
                    == 0
                ),
                timeout=20.0,
            )
        )


        self._last_state = (
            new_state
        )

        self._episode_steps = 0

        self._room_steps = 0

        self._action_id = 0

        self._episode_hp_lost = 0.0


        observation = (
            self._state_to_obs(
                new_state
            )
        )


        info = (
            self._make_info(
                new_state
            )
        )

        info["episode_steps"] = 0

        info["room_steps"] = 0

        info["episode_hp_lost"] = 0.0

        info["no_hit"] = True

        info["timeout"] = False


        print(
            "[IsaacEnv] Episode "
            f"{info['episode_id']} ready. "
            f"Target rooms: "
            f"{info['target_room_exits']}"
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
                "Call env.reset() "
                "before env.step()."
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
                "Invalid move index: "
                f"{move_index}"
            )


        if not (
            0
            <= shoot_index
            < len(SHOOT_NAMES)
        ):
            raise ValueError(
                "Invalid shoot index: "
                f"{shoot_index}"
            )

        # -------------------------------------------------
        # Resolve action
        # -------------------------------------------------

        move_name = (
            MOVE_NAMES[
                move_index
            ]
        )

        shoot_name = (
            SHOOT_NAMES[
                shoot_index
            ]
        )


        # -------------------------------------------------
        # Navigation mode
        # -------------------------------------------------

        navigation_mode = bool(
            self._last_state.get(
                "room_clear",
                False,
            )
        )

        forced_shoot_none = False


        if navigation_mode:
            # Keep action space compatible with older PPO
            # checkpoints but physically ignore shooting
            # during navigation.
            shoot_name = "NONE"

            forced_shoot_none = True

        # -------------------------------------------------
        # Send action
        # -------------------------------------------------

        self._action_id += 1

        action_id = (
            self._action_id
        )

        episode_id = int(
            self._last_state.get(
                "episode_id",
                -1,
            )
        )


        self._send_json(
            {
                "type":
                    "action",

                "action_id":
                    action_id,

                "enabled":
                    True,

                "move":
                    move_name,

                "shoot":
                    shoot_name,
            }
        )

        # -------------------------------------------------
        # Wait for applied action
        # -------------------------------------------------

        new_state = (
            self._wait_for(
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
        )

        # -------------------------------------------------
        # Step counters
        # -------------------------------------------------

        self._episode_steps += 1

        self._room_steps += 1

        room_steps_for_reward = (
            self._room_steps
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
            room_steps=(
                room_steps_for_reward
            ),
        )

        # -------------------------------------------------
        # HP statistics
        # -------------------------------------------------

        self._episode_hp_lost += float(
            reward_info[
                "hp_lost"
            ]
        )

        # -------------------------------------------------
        # Room timer reset
        # -------------------------------------------------
        #
        # Reset on EVERY actual transition, including
        # backtracking, because overtime is defined per
        # physical room stay rather than per curriculum
        # progress unit.
        # -------------------------------------------------

        completed_room_steps = None


        if reward_info[
            "room_transitioned"
        ]:
            completed_room_steps = (
                self._room_steps
            )

            self._room_steps = 0

        # -------------------------------------------------
        # Episode timeout
        # -------------------------------------------------

        truncated = (
            self._episode_steps
            >= self.max_episode_steps
        )


        if (
            truncated
            and not terminated
        ):
            reward += (
                TIMEOUT_PENALTY
            )

        # -------------------------------------------------
        # Store state
        # -------------------------------------------------

        self._last_state = (
            new_state
        )


        observation = (
            self._state_to_obs(
                new_state
            )
        )

        # -------------------------------------------------
        # Info
        # -------------------------------------------------

        info = (
            self._make_info(
                new_state
            )
        )


        info.update(
            reward_info
        )


        info["episode_steps"] = (
            self._episode_steps
        )


        info["room_steps"] = (
            self._room_steps
        )


        info["completed_room_steps"] = (
            completed_room_steps
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


        info["forced_shoot_none"] = (
            forced_shoot_none
        )


        info["requested_move"] = (
            MOVE_NAMES[
                move_index
            ]
        )


        info["requested_shoot"] = (
            SHOOT_NAMES[
                shoot_index
            ]
        )


        info["applied_move"] = (
            move_name
        )


        info["applied_shoot"] = (
            shoot_name
        )


        return (
            observation,
            float(
                reward
            ),
            bool(
                terminated
            ),
            bool(
                truncated
            ),
            info,
        )

    # =====================================================
    # INFO
    # =====================================================

    def _make_info(
        self,
        state: dict[str, Any],
    ) -> dict[str, Any]:
        navigation_distance = (
            self._navigation_distance(
                state
            )
        )


        rooms_completed = (
            self._rooms_completed(
                state
            )
        )


        target_room_count = (
            self._target_room_count(
                state
            )
        )


        navigation_target = (
            self._navigation_target(
                state
            )
        )


        navigation_phase = None

        navigation_target_room = None

        navigation_target_visited = None


        if navigation_target is not None:
            navigation_phase = (
                navigation_target.get(
                    "phase"
                )
            )

            navigation_target_room = (
                navigation_target.get(
                    "target_room_index"
                )
            )

            navigation_target_visited = (
                navigation_target.get(
                    "target_room_visited"
                )
            )


        return {
            "episode_id":
                int(
                    state.get(
                        "episode_id",
                        -1,
                    )
                ),

            "frame":
                int(
                    state.get(
                        "frame",
                        -1,
                    )
                ),

            "room_index":
                self._room_index(
                    state
                ),

            "room_clear":
                bool(
                    state.get(
                        "room_clear",
                        False,
                    )
                ),

            "enemy_count":
                int(
                    state.get(
                        "enemy_count",
                        0,
                    )
                ),

            "total_enemy_hp":
                float(
                    state.get(
                        "total_enemy_hp",
                        0.0,
                    )
                ),

            "player_hp":
                self._player_hp(
                    state
                ),

            "navigation_distance":
                navigation_distance,

            "navigation_phase":
                navigation_phase,

            "navigation_target_room":
                navigation_target_room,

            "navigation_target_visited":
                navigation_target_visited,

            "rooms_completed":
                rooms_completed,

            # Compatibility with evaluate.py.
            "rooms_exited":
                rooms_completed,

            "target_room_exits":
                target_room_count,

            "rooms_remaining":
                max(
                    0,
                    target_room_count
                    - rooms_completed,
                ),

            "total_room_transitions":
                int(
                    state.get(
                        "total_room_transitions",
                        0,
                    )
                ),

            "training_complete":
                self._training_complete(
                    state
                ),
        }

    # =====================================================
    # RENDER
    # =====================================================

    def render(
        self,
    ) -> None:
        if self._last_state is None:
            print(
                "[IsaacEnv] No state yet."
            )

            return


        info = (
            self._make_info(
                self._last_state
            )
        )


        control = (
            self._last_state.get(
                "control",
                {},
            )
        )


        mode = (
            "NAVIGATION"
            if info[
                "room_clear"
            ]
            else "COMBAT"
        )


        print(
            "[IsaacEnv] "
            f"episode={info['episode_id']} "
            f"frame={info['frame']} "
            f"room={info['room_index']} "
            f"mode={mode} "
            f"progress="
            f"{info['rooms_completed']}/"
            f"{info['target_room_exits']} "
            f"room_steps={self._room_steps} "
            f"hp={info['player_hp']:.1f} "
            f"enemies={info['enemy_count']} "
            f"move={control.get('move')} "
            f"shoot={control.get('shoot')}"
        )

    # =====================================================
    # CLOSE
    # =====================================================

    def close(
        self,
    ) -> None:
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