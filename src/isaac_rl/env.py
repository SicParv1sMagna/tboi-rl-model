from __future__ import annotations

import json
import math
import os
import socket
import time
from typing import Any, Callable

import gymnasium as gym
import numpy as np
from gymnasium import spaces


HOST = "127.0.0.1"
PORT = 5000


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
# CURRICULUM
# =========================================================

STAGE_CONFIGS: dict[int, dict[str, Any]] = {
    1: {
        "name": "combat_basics",

        "target_room_exits": 0,
        "default_max_episode_steps": 250,

        "combat_step_penalty": -0.01,

        "hp_lost_penalty": 1.0,

        # Collect statistics only.
        "missed_tear_penalty": 0.0,

        "death_penalty": -12.0,
        "timeout_penalty": -15.0,

        "training_complete_reward": 10.0,

        "final_hp_bonus_multiplier": 0.0,

        "room_overtime_enabled": False,
    },

    2: {
        "name": "combat_health_accuracy",

        "target_room_exits": 0,
        "default_max_episode_steps": 250,

        "combat_step_penalty": -0.01,

        # Strong health preservation signal.
        "hp_lost_penalty": 5.0,

        # Mild accuracy shaping.
        # 20 confirmed misses = -1 reward.
        "missed_tear_penalty": 0.05,

        "death_penalty": -25.0,
        "timeout_penalty": -15.0,

        "training_complete_reward": 10.0,

        "final_hp_bonus_multiplier": 1.5,

        "room_overtime_enabled": False,
    },

    3: {
        "name": "single_room_navigation",

        "target_room_exits": 1,
        "default_max_episode_steps": 400,

        "combat_step_penalty": -0.025,

        "hp_lost_penalty": 5.0,
        "missed_tear_penalty": 0.05,

        "death_penalty": -25.0,
        "timeout_penalty": -5.0,

        "training_complete_reward": 20.0,

        "final_hp_bonus_multiplier": 1.5,

        "room_overtime_enabled": True,
    },

    4: {
        "name": "three_room_run",

        "target_room_exits": 3,
        "default_max_episode_steps": 600,

        "combat_step_penalty": -0.03,

        "hp_lost_penalty": 5.0,
        "missed_tear_penalty": 0.05,

        "death_penalty": -25.0,
        "timeout_penalty": -5.0,

        "training_complete_reward": 20.0,

        "final_hp_bonus_multiplier": 1.5,

        "room_overtime_enabled": True,
    },
}


# =========================================================
# COMMON REWARD
# =========================================================

NAVIGATION_STEP_PENALTY = -0.01

ROOM_OVERTIME_START = 100
ROOM_OVERTIME_PENALTY_GROWTH = 0.001
MAX_ROOM_OVERTIME_PENALTY = 0.05

DAMAGE_DEALT_REWARD = 0.05
KILL_REWARD = 3.0
ROOM_CLEAR_REWARD = 10.0

ROOM_EXIT_REWARD = 10.0

NAVIGATION_PROGRESS_REWARD = 0.01
MAX_NAVIGATION_PROGRESS_REWARD = 0.5


# =========================================================
# COLLISION SHAPING
# =========================================================

# Increased after Stage 1 evaluation showed that ~60%
# of movement actions could become stuck/wall actions.
BLOCKED_MOVE_PENALTY = 0.10
STUCK_MOVE_PENALTY = 0.05

STUCK_DISTANCE_THRESHOLD = 2.0


class IsaacEnv(gym.Env):
    """
    structured_v2 curriculum environment.

    Stage can be selected either through:

        IsaacEnv(curriculum_stage=2)

    or:

        ISAAC_RL_STAGE=2

    Observation ALWAYS stays (21,), which makes PPO
    checkpoints compatible between curriculum stages.

    Action:
        [move, shoot]

        0 NONE
        1 LEFT
        2 RIGHT
        3 UP
        4 DOWN

    Observation:

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
        10 target HP fraction

        11 enemy count
        12 room clear

        13 blocked left
        14 blocked right
        15 blocked up
        16 blocked down

        17 danger left
        18 danger right
        19 danger up
        20 danger down
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
        curriculum_stage: int | None = None,
        max_episode_steps: int | None = None,
        socket_timeout: float = 5.0,
        render_mode: str | None = None,
    ) -> None:
        super().__init__()

        if curriculum_stage is None:
            curriculum_stage = int(
                os.environ.get(
                    "ISAAC_RL_STAGE",
                    "1",
                )
            )

        if curriculum_stage not in STAGE_CONFIGS:
            raise ValueError(
                "curriculum_stage must be 1, 2, 3 or 4"
            )

        self.curriculum_stage = curriculum_stage

        self.stage_config = STAGE_CONFIGS[
            curriculum_stage
        ]

        self.host = host
        self.port = port

        if max_episode_steps is None:
            max_episode_steps = int(
                self.stage_config[
                    "default_max_episode_steps"
                ]
            )

        self.max_episode_steps = (
            max_episode_steps
        )

        self.socket_timeout = (
            socket_timeout
        )

        self.render_mode = render_mode

        self.action_space = (
            spaces.MultiDiscrete([5, 5])
        )

        self.observation_space = (
            spaces.Box(
                low=-2.0,
                high=2.0,
                shape=(21,),
                dtype=np.float32,
            )
        )

        self._server: socket.socket | None = None
        self._conn: socket.socket | None = None

        self._rx_buffer = b""

        self._last_state: dict[str, Any] | None = None

        self._episode_steps = 0
        self._room_steps = 0

        self._action_id = 0

        self._episode_hp_lost = 0.0

        self._open_server()

        print(
            "[IsaacEnv] Curriculum stage "
            f"{self.curriculum_stage}: "
            f"{self.stage_config['name']}"
        )

        print(
            "[IsaacEnv] Max episode steps: "
            f"{self.max_episode_steps}"
        )

    # =====================================================
    # TCP
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
            (self.host, self.port)
        )

        server.listen(1)

        self._server = server

        print(
            "[IsaacEnv] Listening on "
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
            "[IsaacEnv] Isaac connected: "
            f"{addr}"
        )

    def _drop_connection(self) -> None:
        if self._conn is not None:
            try:
                self._conn.close()
            except OSError:
                pass

        self._conn = None
        self._rx_buffer = b""

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
                and message.get("type")
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
        timeout: float = 20.0,
    ) -> dict[str, Any]:
        deadline = (
            time.monotonic() + timeout
        )

        while True:
            if time.monotonic() > deadline:
                raise TimeoutError(
                    "Timed out waiting for expected Isaac state."
                )

            state = self._recv_state()

            if predicate(state):
                return state

    # =====================================================
    # HELPERS
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
    def _bool_float(
        value: Any,
    ) -> float:
        return 1.0 if bool(value) else 0.0

    @staticmethod
    def _player_position(
        state: dict[str, Any],
    ) -> tuple[float, float]:
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

    @staticmethod
    def _navigation_target(
        state: dict[str, Any],
    ) -> dict[str, Any] | None:
        target = state.get(
            "navigation_target"
        )

        if isinstance(target, dict):
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
        return int(
            state.get(
                "rooms_completed",
                state.get(
                    "rooms_exited",
                    0,
                ),
            )
        )

    @staticmethod
    def _target_room_count(
        state: dict[str, Any],
    ) -> int:
        return int(
            state.get(
                "target_room_exits",
                0,
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

    def _room_transitioned(
        self,
        previous: dict[str, Any],
        current: dict[str, Any],
    ) -> bool:
        previous_room = self._room_index(
            previous
        )

        current_room = self._room_index(
            current
        )

        return (
            previous_room >= 0
            and current_room >= 0
            and previous_room != current_room
        )

    # =====================================================
    # COLLISION
    # =====================================================

    @staticmethod
    def _blocked_for_move(
        state: dict[str, Any],
        move_name: str,
    ) -> bool:
        mapping = {
            "LEFT": "left",
            "RIGHT": "right",
            "UP": "up",
            "DOWN": "down",
        }

        key = mapping.get(move_name)

        if key is None:
            return False

        sensors = state.get(
            "local_sensors",
            {},
        )

        blocked = sensors.get(
            "blocked",
            {},
        )

        return bool(
            blocked.get(
                key,
                False,
            )
        )

    def _movement_distance(
        self,
        previous: dict[str, Any],
        current: dict[str, Any],
    ) -> float:
        x1, y1 = self._player_position(
            previous
        )

        x2, y2 = self._player_position(
            current
        )

        return math.hypot(
            x2 - x1,
            y2 - y1,
        )

    # =====================================================
    # NAVIGATION
    # =====================================================

    def _navigation_distance(
        self,
        state: dict[str, Any],
    ) -> float | None:
        if self.curriculum_stage < 3:
            return None

        if self._training_complete(state):
            return None

        if not bool(
            state.get(
                "room_clear",
                False,
            )
        ):
            return None

        target = self._navigation_target(
            state
        )

        if target is None:
            return None

        px, py = self._player_position(
            state
        )

        tx = float(
            target.get(
                "x",
                px,
            )
        )

        ty = float(
            target.get(
                "y",
                py,
            )
        )

        return math.hypot(
            tx - px,
            ty - py,
        )

    def _navigation_identity(
        self,
        state: dict[str, Any],
    ) -> tuple[Any, Any] | None:
        target = self._navigation_target(
            state
        )

        if target is None:
            return None

        return (
            target.get("slot"),
            target.get("phase"),
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

        sensors = state.get(
            "local_sensors",
            {},
        )

        blocked = sensors.get(
            "blocked",
            {},
        )

        danger = sensors.get(
            "danger",
            {},
        )

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

        target_dx = 0.0
        target_dy = 0.0

        target_vx = 0.0
        target_vy = 0.0

        target_hp_fraction = 0.0

        # Navigation target.
        if (
            room_clear
            and self.curriculum_stage >= 3
        ):
            target = self._navigation_target(
                state
            )

            if target is not None:
                target_dx = (
                    float(
                        target.get(
                            "x",
                            px,
                        )
                    )
                    - px
                )

                target_dy = (
                    float(
                        target.get(
                            "y",
                            py,
                        )
                    )
                    - py
                )

        # Combat target.
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
                    nearest_dist_sq is None
                    or dist_sq
                    < nearest_dist_sq
                ):
                    nearest_dist_sq = dist_sq
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

                hp = float(
                    nearest.get(
                        "hp",
                        0.0,
                    )
                )

                max_hp = max(
                    1.0,
                    float(
                        nearest.get(
                            "max_hp",
                            hp,
                        )
                    ),
                )

                target_hp_fraction = (
                    hp / max_hp
                )

        enemy_count = float(
            state.get(
                "enemy_count",
                len(enemies),
            )
        )

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

                self._bool_float(
                    room_clear
                ),

                self._bool_float(
                    blocked.get(
                        "left",
                        False,
                    )
                ),

                self._bool_float(
                    blocked.get(
                        "right",
                        False,
                    )
                ),

                self._bool_float(
                    blocked.get(
                        "up",
                        False,
                    )
                ),

                self._bool_float(
                    blocked.get(
                        "down",
                        False,
                    )
                ),

                self._bool_float(
                    danger.get(
                        "left",
                        False,
                    )
                ),

                self._bool_float(
                    danger.get(
                        "right",
                        False,
                    )
                ),

                self._bool_float(
                    danger.get(
                        "up",
                        False,
                    )
                ),

                self._bool_float(
                    danger.get(
                        "down",
                        False,
                    )
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
        navigation_mode = (
            self.curriculum_stage >= 3
            and bool(
                previous.get(
                    "room_clear",
                    False,
                )
            )
        )

        if navigation_mode:
            step_penalty = (
                NAVIGATION_STEP_PENALTY
            )
        else:
            step_penalty = float(
                self.stage_config[
                    "combat_step_penalty"
                ]
            )

        reward = step_penalty

        # -------------------------------------------------
        # OVERTIME
        # -------------------------------------------------

        overtime_steps = 0
        room_overtime_penalty = 0.0

        if bool(
            self.stage_config[
                "room_overtime_enabled"
            ]
        ):
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

        reward += damage_reward

        # -------------------------------------------------
        # TEAR ACCURACY
        # -------------------------------------------------

        previous_tears_fired = int(
            previous.get(
                "tears_fired",
                0,
            )
        )

        current_tears_fired = int(
            current.get(
                "tears_fired",
                0,
            )
        )

        previous_tears_hit = int(
            previous.get(
                "tears_hit",
                0,
            )
        )

        current_tears_hit = int(
            current.get(
                "tears_hit",
                0,
            )
        )

        previous_tears_missed = int(
            previous.get(
                "tears_missed",
                0,
            )
        )

        current_tears_missed = int(
            current.get(
                "tears_missed",
                0,
            )
        )

        tears_fired_delta = max(
            0,
            current_tears_fired
            - previous_tears_fired,
        )

        tears_hit_delta = max(
            0,
            current_tears_hit
            - previous_tears_hit,
        )

        tears_missed_delta = max(
            0,
            current_tears_missed
            - previous_tears_missed,
        )

        missed_tear_penalty = (
            tears_missed_delta
            * float(
                self.stage_config[
                    "missed_tear_penalty"
                ]
            )
        )

        reward -= (
            missed_tear_penalty
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

        reward += kill_reward

        # -------------------------------------------------
        # PLAYER HP
        # -------------------------------------------------

        previous_hp = self._player_hp(
            previous
        )

        current_hp = self._player_hp(
            current
        )

        hp_lost = max(
            0.0,
            previous_hp - current_hp,
        )

        damage_taken_penalty = (
            hp_lost
            * float(
                self.stage_config[
                    "hp_lost_penalty"
                ]
            )
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

        navigation_progress = 0.0
        navigation_progress_reward = 0.0

        previous_distance = (
            self._navigation_distance(
                previous
            )
        )

        current_distance = (
            self._navigation_distance(
                current
            )
        )

        previous_identity = (
            self._navigation_identity(
                previous
            )
        )

        current_identity = (
            self._navigation_identity(
                current
            )
        )

        # Don't compare distance across door changes or
        # APPROACH -> EXIT transitions.
        if (
            previous_distance is not None
            and current_distance is not None
            and previous_identity is not None
            and previous_identity
            == current_identity
        ):
            navigation_progress = (
                previous_distance
                - current_distance
            )

            navigation_progress_reward = (
                navigation_progress
                * NAVIGATION_PROGRESS_REWARD
            )

            navigation_progress_reward = float(
                np.clip(
                    navigation_progress_reward,
                    -MAX_NAVIGATION_PROGRESS_REWARD,
                    MAX_NAVIGATION_PROGRESS_REWARD,
                )
            )

            reward += (
                navigation_progress_reward
            )

        # -------------------------------------------------
        # ROOM TRANSITION
        # -------------------------------------------------

        room_transitioned = (
            self._room_transitioned(
                previous,
                current,
            )
        )

        previous_rooms = (
            self._rooms_completed(
                previous
            )
        )

        current_rooms = (
            self._rooms_completed(
                current
            )
        )

        rooms_completed_delta = max(
            0,
            current_rooms
            - previous_rooms,
        )

        room_exit_reward = (
            rooms_completed_delta
            * ROOM_EXIT_REWARD
        )

        reward += room_exit_reward

        backtracked = (
            room_transitioned
            and rooms_completed_delta == 0
        )

        # -------------------------------------------------
        # CURRICULUM COMPLETE
        # -------------------------------------------------

        training_complete = (
            self._training_complete(
                current
            )
        )

        previously_complete = (
            self._training_complete(
                previous
            )
        )

        completed_this_step = (
            training_complete
            and not previously_complete
        )

        training_complete_reward = 0.0
        final_health_bonus = 0.0

        if completed_this_step:
            training_complete_reward = float(
                self.stage_config[
                    "training_complete_reward"
                ]
            )

            reward += (
                training_complete_reward
            )

            final_health_bonus = (
                current_hp
                * float(
                    self.stage_config[
                        "final_hp_bonus_multiplier"
                    ]
                )
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

        death_penalty = 0.0

        if dead:
            death_penalty = abs(
                float(
                    self.stage_config[
                        "death_penalty"
                    ]
                )
            )

            reward -= death_penalty

        terminated = (
            dead
            or training_complete
        )

        return (
            reward,
            terminated,
            {
                "curriculum_stage":
                    self.curriculum_stage,

                "curriculum_name":
                    self.stage_config[
                        "name"
                    ],

                "navigation_mode":
                    navigation_mode,

                "step_penalty":
                    step_penalty,

                "overtime_steps":
                    overtime_steps,

                "room_overtime_penalty":
                    room_overtime_penalty,

                "damage_dealt":
                    damage_dealt,

                "damage_reward":
                    damage_reward,

                "tears_fired_delta":
                    tears_fired_delta,

                "tears_hit_delta":
                    tears_hit_delta,

                "tears_missed_delta":
                    tears_missed_delta,

                "missed_tear_penalty":
                    missed_tear_penalty,

                "tears_fired":
                    current_tears_fired,

                "tears_hit":
                    current_tears_hit,

                "tears_missed":
                    current_tears_missed,

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

                "room_exits_delta":
                    rooms_completed_delta,

                "room_exit_reward":
                    room_exit_reward,

                "rooms_completed":
                    current_rooms,

                "rooms_exited":
                    current_rooms,

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

                "death_penalty":
                    death_penalty,
            },
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
            "[IsaacEnv] Resetting episode "
            f"{old_episode_id}..."
        )

        self._send_json(
            {
                "type": "reset",
                "curriculum_stage":
                    self.curriculum_stage,
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
                        "curriculum_stage",
                        -1,
                    )
                )
                == self.curriculum_stage
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

        info.update(
            {
                "episode_steps": 0,
                "room_steps": 0,

                "episode_hp_lost": 0.0,

                "no_hit": True,
                "timeout": False,
            }
        )

        print(
            "[IsaacEnv] Episode "
            f"{info['episode_id']} ready "
            f"| stage={self.curriculum_stage} "
            f"{self.stage_config['name']} "
            f"| obs={observation.shape}"
        )

        return observation, info

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

        previous = self._last_state

        action_array = np.asarray(
            action,
            dtype=np.int64,
        ).reshape(-1)

        if action_array.shape != (2,):
            raise ValueError(
                "Action must be exactly [move, shoot]."
            )

        move_index = int(
            action_array[0]
        )

        shoot_index = int(
            action_array[1]
        )

        if not (
            0 <= move_index < len(MOVE_NAMES)
        ):
            raise ValueError(
                f"Invalid move index: {move_index}"
            )

        if not (
            0 <= shoot_index < len(SHOOT_NAMES)
        ):
            raise ValueError(
                f"Invalid shoot index: {shoot_index}"
            )

        move_name = (
            MOVE_NAMES[move_index]
        )

        requested_shoot = (
            SHOOT_NAMES[shoot_index]
        )

        shoot_name = requested_shoot

        navigation_mode = (
            self.curriculum_stage >= 3
            and bool(
                previous.get(
                    "room_clear",
                    False,
                )
            )
        )

        previous_target = (
            self._navigation_target(
                previous
            )
        )

        navigation_phase = None

        if previous_target is not None:
            navigation_phase = (
                previous_target.get(
                    "phase"
                )
            )

        forced_shoot_none = False

        if navigation_mode:
            shoot_name = "NONE"
            forced_shoot_none = True

        blocked_sensor = (
            self._blocked_for_move(
                previous,
                move_name,
            )
        )

        self._action_id += 1

        action_id = self._action_id

        episode_id = int(
            previous.get(
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
                    move_name,

                "shoot":
                    shoot_name,
            }
        )

        current = self._wait_for(
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

        self._episode_steps += 1
        self._room_steps += 1

        (
            reward,
            terminated,
            reward_info,
        ) = self._calculate_reward(
            previous,
            current,
            room_steps=self._room_steps,
        )

        movement_distance = (
            self._movement_distance(
                previous,
                current,
            )
        )

        room_transitioned = bool(
            reward_info[
                "room_transitioned"
            ]
        )

        # -------------------------------------------------
        # BLOCKED / STUCK SHAPING
        # -------------------------------------------------

        collision_penalties_enabled = (
            navigation_phase != "EXIT"
        )

        stuck_move_attempted = (
            move_name != "NONE"
            and not room_transitioned
            and movement_distance
            < STUCK_DISTANCE_THRESHOLD
        )

        blocked_move_attempted = (
            blocked_sensor
            and stuck_move_attempted
            and collision_penalties_enabled
        )

        blocked_move_penalty = 0.0
        stuck_move_penalty = 0.0

        if (
            stuck_move_attempted
            and collision_penalties_enabled
        ):
            stuck_move_penalty = (
                STUCK_MOVE_PENALTY
            )

            reward -= (
                stuck_move_penalty
            )

        if blocked_move_attempted:
            blocked_move_penalty = (
                BLOCKED_MOVE_PENALTY
            )

            reward -= (
                blocked_move_penalty
            )

        self._episode_hp_lost += float(
            reward_info[
                "hp_lost"
            ]
        )

        completed_room_steps = None

        if room_transitioned:
            completed_room_steps = (
                self._room_steps
            )

            self._room_steps = 0

        truncated = (
            self._episode_steps
            >= self.max_episode_steps
        )

        timeout_penalty = 0.0

        if (
            truncated
            and not terminated
        ):
            timeout_penalty = abs(
                float(
                    self.stage_config[
                        "timeout_penalty"
                    ]
                )
            )

            reward -= timeout_penalty

        self._last_state = current

        observation = (
            self._state_to_obs(
                current
            )
        )

        info = (
            self._make_info(
                current
            )
        )

        info.update(
            reward_info
        )

        info.update(
            {
                "episode_steps":
                    self._episode_steps,

                "room_steps":
                    self._room_steps,

                "completed_room_steps":
                    completed_room_steps,

                "episode_hp_lost":
                    self._episode_hp_lost,

                "no_hit":
                    self._episode_hp_lost
                    == 0.0,

                "timeout":
                    bool(
                        truncated
                        and not terminated
                    ),

                "timeout_penalty":
                    timeout_penalty,

                "forced_shoot_none":
                    forced_shoot_none,

                "requested_move":
                    move_name,

                "requested_shoot":
                    requested_shoot,

                "applied_move":
                    move_name,

                "applied_shoot":
                    shoot_name,

                "movement_distance":
                    movement_distance,

                "blocked_move_sensor":
                    blocked_sensor,

                "blocked_move_attempted":
                    blocked_move_attempted,

                "blocked_move_penalty":
                    blocked_move_penalty,

                "stuck_move_attempted":
                    stuck_move_attempted,

                "stuck_move_penalty":
                    stuck_move_penalty,
            }
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
        target = (
            self._navigation_target(
                state
            )
        )

        room_clear = bool(
            state.get(
                "room_clear",
                False,
            )
        )

        complete = (
            self._training_complete(
                state
            )
        )

        target_missing = (
            self.curriculum_stage >= 3
            and room_clear
            and not complete
            and target is None
        )

        rooms_completed = (
            self._rooms_completed(
                state
            )
        )

        tears_fired = int(
            state.get(
                "tears_fired",
                0,
            )
        )

        tears_hit = int(
            state.get(
                "tears_hit",
                0,
            )
        )

        tears_missed = int(
            state.get(
                "tears_missed",
                0,
            )
        )

        confirmed_tears = (
            tears_hit
            + tears_missed
        )

        if confirmed_tears > 0:
            tear_accuracy = (
                tears_hit
                / confirmed_tears
            )
        else:
            tear_accuracy = 0.0

        return {
            "curriculum_stage":
                int(
                    state.get(
                        "curriculum_stage",
                        self.curriculum_stage,
                    )
                ),

            "curriculum_name":
                state.get(
                    "curriculum_name",
                    self.stage_config[
                        "name"
                    ],
                ),

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
                room_clear,

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

            "tears_fired":
                tears_fired,

            "tears_hit":
                tears_hit,

            "tears_missed":
                tears_missed,

            "tear_accuracy":
                tear_accuracy,

            "navigation_distance":
                self._navigation_distance(
                    state
                ),

            "navigation_phase":
                (
                    target.get("phase")
                    if target is not None
                    else None
                ),

            "navigation_target_room":
                (
                    target.get(
                        "target_room_index"
                    )
                    if target is not None
                    else None
                ),

            "navigation_target_missing":
                target_missing,

            "navigation_candidate_count":
                int(
                    state.get(
                        "navigation_candidate_count",
                        0,
                    )
                ),

            "rooms_completed":
                rooms_completed,

            "rooms_exited":
                rooms_completed,

            "target_room_exits":
                self._target_room_count(
                    state
                ),

            "rooms_remaining":
                int(
                    state.get(
                        "rooms_remaining",
                        0,
                    )
                ),

            "total_room_transitions":
                int(
                    state.get(
                        "total_room_transitions",
                        0,
                    )
                ),

            "training_complete":
                complete,

            "local_sensors":
                state.get(
                    "local_sensors",
                    {},
                ),
        }

    # =====================================================
    # RENDER
    # =====================================================

    def render(self) -> None:
        if self._last_state is None:
            return

        info = self._make_info(
            self._last_state
        )

        print(
            "[IsaacEnv] "
            f"stage={info['curriculum_stage']} "
            f"{info['curriculum_name']} "
            f"episode={info['episode_id']} "
            f"room={info['room_index']} "
            f"clear={info['room_clear']} "
            f"enemies={info['enemy_count']} "
            f"tears="
            f"{info['tears_hit']}H/"
            f"{info['tears_missed']}M "
            f"acc="
            f"{info['tear_accuracy'] * 100.0:.1f}% "
            f"progress="
            f"{info['rooms_exited']}/"
            f"{info['target_room_exits']} "
            f"nav_missing="
            f"{info['navigation_target_missing']}"
        )

    # =====================================================
    # CLOSE
    # =====================================================

    def close(self) -> None:
        if self._conn is not None:
            try:
                self._send_json(
                    {
                        "type": "action",

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