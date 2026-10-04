-- =========================================================
-- ISAAC RL BRIDGE
-- The Binding of Isaac: Afterbirth+
--
-- Requires Steam launch option:
--
-- --luadebug
--
-- Python server:
-- 127.0.0.1:5000
-- =========================================================


local mod = RegisterMod(
    "Isaac RL Bridge",
    1
)

local game = Game()

local json = require("json")
local socket = require("socket")


-- =========================================================
-- CONFIG
-- =========================================================

local HOST = "127.0.0.1"
local PORT = 5000

-- Isaac updates roughly 30 times/sec.
-- 3 = ~10 observations/sec.
local STATE_SEND_INTERVAL = 3

-- Log once per second.
local STATE_LOG_INTERVAL = 30

-- Retry connection roughly once/sec.
local RECONNECT_INTERVAL = 30

-- Wait several updates after a run starts.
local TRAINING_SPAWN_DELAY = 8


-- =========================================================
-- TRAINING ENEMY
-- =========================================================

local TRAINING_ENEMY_TYPE = 10
local TRAINING_ENEMY_VARIANT = 0
local TRAINING_ENEMY_SUBTYPE = 0


local TRAINING_ENEMY_OFFSETS = {
    Vector(160, 0),
    Vector(-160, 0),
    Vector(0, 120),
    Vector(0, -120)
}


-- =========================================================
-- NAVIGATION
-- =========================================================
--
-- Navigation now uses two implicit phases:
--
-- APPROACH:
--     Target is inside the room, in front of the doorway.
--
-- EXIT:
--     Once Isaac is aligned with the doorway and has
--     reached the approach zone, target moves beyond
--     the doorway.
--
-- No door is permanently selected. On every state update
-- the most convenient door is selected dynamically.
-- =========================================================


-- How far inside the room the approach point sits.
local DOOR_APPROACH_DISTANCE = 70

-- How far beyond the door the final target sits.
local DOOR_EXIT_DISTANCE = 90

-- How precisely Isaac must be aligned with the doorway
-- before target changes from APPROACH to EXIT.
local DOOR_ALIGNMENT_TOLERANCE = 32


-- =========================================================
-- RL CONTROL
-- =========================================================

RL_ENABLED = false

RL_MOVE = "NONE"
RL_SHOOT = "NONE"

-- Last action which has had at least one game update
-- in which it could affect Isaac.
RL_ACTION_ID = 0

local pendingActionId = nil
local pendingActionFrame = nil


-- =========================================================
-- EPISODE STATE
-- =========================================================

local episodeId = 0

local gameOver = false

local resetInProgress = false

local trainingReady = false

local trainingSpawnCountdown = -1

-- Room in which the training encounter started.
local trainingRoomIndex = nil


-- =========================================================
-- TCP STATE
-- =========================================================

local tcp = nil

local lastConnectAttempt = -99999

local rxBuffer = ""
local txBuffer = ""


-- =========================================================
-- ACTION MAP
-- =========================================================

local MOVE_ACTIONS = {
    LEFT = ButtonAction.ACTION_LEFT,
    RIGHT = ButtonAction.ACTION_RIGHT,
    UP = ButtonAction.ACTION_UP,
    DOWN = ButtonAction.ACTION_DOWN
}


local SHOOT_ACTIONS = {
    LEFT = ButtonAction.ACTION_SHOOTLEFT,
    RIGHT = ButtonAction.ACTION_SHOOTRIGHT,
    UP = ButtonAction.ACTION_SHOOTUP,
    DOWN = ButtonAction.ACTION_SHOOTDOWN
}


-- =========================================================
-- STARTUP LOG
-- =========================================================

Isaac.DebugString(
    "======================================"
)

Isaac.DebugString(
    "ISAAC RL BRIDGE LOADED"
)

Isaac.DebugString(
    "TCP: 127.0.0.1:5000"
)

Isaac.DebugString(
    "======================================"
)


-- =========================================================
-- HELPERS
-- =========================================================

local function IsValidDirection(value)

    return
        value == "NONE"
        or value == "LEFT"
        or value == "RIGHT"
        or value == "UP"
        or value == "DOWN"

end


local function GetCurrentRoomIndex()

    local level = game:GetLevel()

    if level == nil then
        return -1
    end

    return level:GetCurrentRoomIndex()

end


-- =========================================================
-- DOOR DIRECTION
-- =========================================================

local function GetDoorOutwardDirection(slot)

    if slot == DoorSlot.LEFT0
    or slot == DoorSlot.LEFT1 then

        return Vector(
            -1,
            0
        )

    elseif slot == DoorSlot.RIGHT0
    or slot == DoorSlot.RIGHT1 then

        return Vector(
            1,
            0
        )

    elseif slot == DoorSlot.UP0
    or slot == DoorSlot.UP1 then

        return Vector(
            0,
            -1
        )

    elseif slot == DoorSlot.DOWN0
    or slot == DoorSlot.DOWN1 then

        return Vector(
            0,
            1
        )

    end


    return nil

end


-- =========================================================
-- DOOR NAVIGATION DATA
-- =========================================================

local function GetDoorNavigationData(
    room,
    player,
    slot
)

    local door =
        room:GetDoor(
            slot
        )


    if door == nil then
        return nil
    end


    local outwardDirection =
        GetDoorOutwardDirection(
            slot
        )


    if outwardDirection == nil then
        return nil
    end


    local doorPosition =
        room:GetDoorSlotPosition(
            slot
        )


    -- -----------------------------------------------------
    -- APPROACH POINT
    --
    -- This point sits INSIDE the room.
    --
    -- Example for a right-hand door:
    --
    --        approach     door       exit
    --           X          |           X
    --                      |
    --     room             | outside
    --
    -- -----------------------------------------------------

    local approachPosition =
        Vector(
            doorPosition.X
                - outwardDirection.X
                * DOOR_APPROACH_DISTANCE,

            doorPosition.Y
                - outwardDirection.Y
                * DOOR_APPROACH_DISTANCE
        )


    -- -----------------------------------------------------
    -- EXIT POINT
    --
    -- This point sits BEYOND the doorway.
    -- -----------------------------------------------------

    local exitPosition =
        Vector(
            doorPosition.X
                + outwardDirection.X
                * DOOR_EXIT_DISTANCE,

            doorPosition.Y
                + outwardDirection.Y
                * DOOR_EXIT_DISTANCE
        )


    local approachDx =
        approachPosition.X
        - player.Position.X


    local approachDy =
        approachPosition.Y
        - player.Position.Y


    local approachDistanceSquared =
        approachDx * approachDx
        + approachDy * approachDy


    -- -----------------------------------------------------
    -- ALIGNMENT CHECK
    -- -----------------------------------------------------

    local aligned = false


    if outwardDirection.X ~= 0 then

        -- Left/right door.
        -- Y must line up with doorway.
        aligned =
            math.abs(
                player.Position.Y
                - doorPosition.Y
            )
            <= DOOR_ALIGNMENT_TOLERANCE

    else

        -- Up/down door.
        -- X must line up with doorway.
        aligned =
            math.abs(
                player.Position.X
                - doorPosition.X
            )
            <= DOOR_ALIGNMENT_TOLERANCE

    end


    -- -----------------------------------------------------
    -- HAS ISAAC REACHED THE APPROACH PLANE?
    --
    -- This is deliberately directional rather than
    -- distance-only.
    --
    -- That means once Isaac starts moving through the
    -- doorway, we do not suddenly switch the target back
    -- to the point inside the room.
    -- -----------------------------------------------------

    local reachedApproachPlane = false


    if outwardDirection.X > 0 then

        -- Right door
        reachedApproachPlane =
            player.Position.X
            >= approachPosition.X

    elseif outwardDirection.X < 0 then

        -- Left door
        reachedApproachPlane =
            player.Position.X
            <= approachPosition.X

    elseif outwardDirection.Y > 0 then

        -- Down door
        reachedApproachPlane =
            player.Position.Y
            >= approachPosition.Y

    elseif outwardDirection.Y < 0 then

        -- Up door
        reachedApproachPlane =
            player.Position.Y
            <= approachPosition.Y

    end


    local navigationPhase =
        "APPROACH"


    local targetPosition =
        approachPosition


    if aligned
    and reachedApproachPlane then

        navigationPhase =
            "EXIT"

        targetPosition =
            exitPosition

    end


    return {

        door =
            door,

        slot =
            slot,

        phase =
            navigationPhase,

        target_position =
            targetPosition,

        approach_position =
            approachPosition,

        exit_position =
            exitPosition,

        door_position =
            doorPosition,

        approach_distance_squared =
            approachDistanceSquared

    }

end


-- =========================================================
-- FIND BEST NAVIGATION TARGET
-- =========================================================

local function GetNearestDoorTarget(player)

    local room =
        game:GetRoom()


    if room == nil then
        return nil
    end


    local bestDoorData = nil


    -- -----------------------------------------------------
    -- IMPORTANT:
    --
    -- Doors are NOT fixed for the episode.
    --
    -- We calculate the approach point for every actual
    -- door and select the one whose approach point is
    -- currently closest to Isaac.
    -- -----------------------------------------------------

    for slot = 0,
        DoorSlot.NUM_DOOR_SLOTS - 1 do


        local doorData =
            GetDoorNavigationData(
                room,
                player,
                slot
            )


        if doorData ~= nil then


            if bestDoorData == nil
            or doorData.approach_distance_squared
                < bestDoorData.approach_distance_squared then


                bestDoorData =
                    doorData

            end

        end

    end


    if bestDoorData == nil then
        return nil
    end


    local targetPosition =
        bestDoorData.target_position


    local approachPosition =
        bestDoorData.approach_position


    local exitPosition =
        bestDoorData.exit_position


    local doorPosition =
        bestDoorData.door_position


    return {

        x =
            targetPosition.X,

        y =
            targetPosition.Y,

        phase =
            bestDoorData.phase,

        slot =
            bestDoorData.slot,

        door_x =
            doorPosition.X,

        door_y =
            doorPosition.Y,

        approach_x =
            approachPosition.X,

        approach_y =
            approachPosition.Y,

        exit_x =
            exitPosition.X,

        exit_y =
            exitPosition.Y,

        target_room_index =
            bestDoorData.door.TargetRoomIndex

    }

end


-- =========================================================
-- DISCONNECT
-- =========================================================

local function DisconnectPython(reason)

    if tcp ~= nil then

        tcp:close()

        tcp = nil

    end


    rxBuffer = ""
    txBuffer = ""


    RL_ENABLED = false

    RL_MOVE = "NONE"
    RL_SHOOT = "NONE"


    pendingActionId = nil
    pendingActionFrame = nil


    Isaac.DebugString(
        "RL_SOCKET DISCONNECTED: "
        .. tostring(
            reason
        )
    )

end


-- =========================================================
-- CONNECT
-- =========================================================

local function ConnectToPython()

    if tcp ~= nil then
        return
    end


    local frame =
        game:GetFrameCount()


    if frame - lastConnectAttempt
        < RECONNECT_INTERVAL then

        return

    end


    lastConnectAttempt =
        frame


    local client =
        socket.tcp()


    if client == nil then

        Isaac.DebugString(
            "RL_SOCKET CREATE FAILED"
        )

        return

    end


    client:settimeout(
        0.05
    )


    local success, err =
        client:connect(
            HOST,
            PORT
        )


    if success then

        tcp =
            client


        rxBuffer =
            ""

        txBuffer =
            ""


        tcp:settimeout(
            0
        )


        Isaac.DebugString(
            "RL_SOCKET CONNECTED"
        )

    else

        client:close()

    end

end


-- =========================================================
-- TRAINING ROOM DOORS
-- =========================================================

local function CloseTrainingRoomDoors()

    local room =
        game:GetRoom()


    if room == nil then
        return
    end


    for slot = 0,
        DoorSlot.NUM_DOOR_SLOTS - 1 do


        local door =
            room:GetDoor(
                slot
            )


        if door ~= nil then

            door:Close(
                true
            )

        end

    end

end


local function OpenTrainingRoomDoors()

    local room =
        game:GetRoom()


    if room == nil then
        return
    end


    for slot = 0,
        DoorSlot.NUM_DOOR_SLOTS - 1 do


        local door =
            room:GetDoor(
                slot
            )


        if door ~= nil then

            door:Open()

        end

    end

end


-- =========================================================
-- SPAWN TRAINING ENEMY
-- =========================================================

local function SpawnTrainingEnemy()

    local room =
        game:GetRoom()


    if room == nil then
        return false
    end


    local center =
        room:GetCenterPos()


    local offsetIndex =
        (
            Random()
            % #TRAINING_ENEMY_OFFSETS
        )
        + 1


    local spawnOffset =
        TRAINING_ENEMY_OFFSETS[
            offsetIndex
        ]


    local desiredPosition =
        center
        + spawnOffset


    local spawnPosition =
        room:GetClampedPosition(
            desiredPosition,
            40
        )


    room:SetClear(
        false
    )


    local enemy =
        Isaac.Spawn(
            TRAINING_ENEMY_TYPE,
            TRAINING_ENEMY_VARIANT,
            TRAINING_ENEMY_SUBTYPE,
            spawnPosition,
            Vector(
                0,
                0
            ),
            nil
        )


    if enemy == nil then

        Isaac.DebugString(
            "RL_TRAINING ENEMY SPAWN FAILED"
        )

        return false

    end


    trainingRoomIndex =
        GetCurrentRoomIndex()


    trainingReady =
        true


    CloseTrainingRoomDoors()


    Isaac.DebugString(
        "RL_TRAINING READY episode="
        .. tostring(
            episodeId
        )
        .. " room="
        .. tostring(
            trainingRoomIndex
        )
        .. " spawn_index="
        .. tostring(
            offsetIndex
        )
    )


    return true

end


-- =========================================================
-- RESET RUN
-- =========================================================

local function RequestReset()

    if resetInProgress then
        return
    end


    resetInProgress =
        true


    trainingReady =
        false


    trainingRoomIndex =
        nil


    gameOver =
        false


    RL_ENABLED =
        false


    RL_MOVE =
        "NONE"


    RL_SHOOT =
        "NONE"


    pendingActionId =
        nil


    pendingActionFrame =
        nil


    Isaac.DebugString(
        "RL_RESET REQUESTED"
    )


    Isaac.ExecuteCommand(
        "restart"
    )

end


-- =========================================================
-- COMMAND FROM PYTHON
-- =========================================================

local function ProcessCommand(line)

    local success, command =
        pcall(
            json.decode,
            line
        )


    if not success then

        Isaac.DebugString(
            "RL BAD JSON: "
            .. tostring(
                line
            )
        )

        return

    end


    if type(
        command
    ) ~= "table" then

        return

    end


    -- =====================================================
    -- RESET
    -- =====================================================

    if command.type
        == "reset" then


        RequestReset()

        return

    end


    -- =====================================================
    -- ACTION
    -- =====================================================

    if command.type
        ~= "action" then

        return

    end


    if command.action_id
        ~= nil then


        local newActionId =
            tonumber(
                command.action_id
            )


        if newActionId
            ~= nil then


            pendingActionId =
                newActionId


            pendingActionFrame =
                game:GetFrameCount()

        end

    end


    if command.enabled
        ~= nil then


        RL_ENABLED =
            command.enabled
            == true

    end


    -- =====================================================
    -- MOVEMENT
    -- =====================================================

    if command.move
        ~= nil then


        local move =
            string.upper(
                tostring(
                    command.move
                )
            )


        if IsValidDirection(
            move
        ) then

            RL_MOVE =
                move

        end

    end


    -- =====================================================
    -- SHOOTING
    -- =====================================================

    if command.shoot
        ~= nil then


        local shoot =
            string.upper(
                tostring(
                    command.shoot
                )
            )


        if IsValidDirection(
            shoot
        ) then

            RL_SHOOT =
                shoot

        end

    end

end


-- =========================================================
-- RECEIVE FROM PYTHON
-- =========================================================

local function ReadPythonCommands()

    if tcp == nil then

        ConnectToPython()

        return

    end


    local messagesRead =
        0


    while messagesRead
        < 20 do


        local line, err, partial =
            tcp:receive(
                "*l"
            )


        if line ~= nil then


            local completeLine =
                rxBuffer
                .. line


            rxBuffer =
                ""


            ProcessCommand(
                completeLine
            )


            messagesRead =
                messagesRead
                + 1


        else


            if partial ~= nil
            and #partial > 0 then


                rxBuffer =
                    rxBuffer
                    .. partial

            end


            if #rxBuffer
                > 65536 then


                rxBuffer =
                    ""


                Isaac.DebugString(
                    "RL RX BUFFER OVERFLOW"
                )

            end


            if err
                == "timeout" then


                break


            elseif err
                == "closed" then


                DisconnectPython(
                    "remote closed"
                )


                break


            else


                if err ~= nil then

                    DisconnectPython(
                        err
                    )

                end


                break

            end

        end

    end

end


-- =========================================================
-- SEND TCP BUFFER
-- =========================================================

local function FlushOutgoing()

    if tcp == nil then
        return
    end


    if txBuffer == "" then
        return
    end


    local sent, err, last =
        tcp:send(
            txBuffer
        )


    if sent ~= nil then

        txBuffer = ""

        return

    end


    if err
        == "timeout" then


        if last ~= nil
        and last > 0 then


            txBuffer =
                string.sub(
                    txBuffer,
                    last + 1
                )

        end


        return

    end


    if err
        == "closed" then


        DisconnectPython(
            "send closed"
        )


        return

    end


    if err ~= nil then

        DisconnectPython(
            err
        )

    end

end


-- =========================================================
-- BUILD STATE
-- =========================================================

local function GetGameState()

    local player =
        Isaac.GetPlayer(
            0
        )


    if player == nil then
        return nil
    end


    local room =
        game:GetRoom()


    if room == nil then
        return nil
    end


    local enemies =
        {}


    local totalEnemyHp =
        0


    -- =====================================================
    -- ENEMIES
    -- =====================================================

    for _, entity
    in ipairs(
        Isaac.GetRoomEntities()
    ) do


        if entity:IsActiveEnemy(
            false
        )
        and entity:IsVulnerableEnemy() then


            local hp =
                math.max(
                    0,
                    entity.HitPoints
                )


            totalEnemyHp =
                totalEnemyHp
                + hp


            table.insert(
                enemies,
                {

                    type =
                        entity.Type,

                    variant =
                        entity.Variant,

                    x =
                        entity.Position.X,

                    y =
                        entity.Position.Y,

                    velocity_x =
                        entity.Velocity.X,

                    velocity_y =
                        entity.Velocity.Y,

                    hp =
                        hp,

                    max_hp =
                        math.max(
                            0,
                            entity.MaxHitPoints
                        )

                }
            )

        end

    end


    local roomCenter =
        room:GetCenterPos()


    local currentRoomIndex =
        GetCurrentRoomIndex()


    -- =====================================================
    -- NAVIGATION TARGET
    -- =====================================================

    local navigationTarget =
        nil


    if room:IsClear()
    and trainingRoomIndex ~= nil
    and currentRoomIndex
        == trainingRoomIndex then


        navigationTarget =
            GetNearestDoorTarget(
                player
            )

    end


    -- =====================================================
    -- STATE
    -- =====================================================

    return {

        type =
            "state",


        episode_id =
            episodeId,


        action_id =
            RL_ACTION_ID,


        ready =
            trainingReady,


        frame =
            game:GetFrameCount(),


        room_frame =
            room:GetFrameCount(),


        room_index =
            currentRoomIndex,


        training_room_index =
            trainingRoomIndex,


        dead =
            player:IsDead(),


        game_over =
            gameOver,


        room_clear =
            room:IsClear(),


        room = {

            center_x =
                roomCenter.X,

            center_y =
                roomCenter.Y

        },


        player = {

            x =
                player.Position.X,

            y =
                player.Position.Y,

            velocity_x =
                player.Velocity.X,

            velocity_y =
                player.Velocity.Y,

            hearts =
                player:GetHearts(),

            soul_hearts =
                player:GetSoulHearts()

        },


        enemies =
            enemies,


        enemy_count =
            #enemies,


        total_enemy_hp =
            totalEnemyHp,


        navigation_target =
            navigationTarget,


        control = {

            enabled =
                RL_ENABLED,

            move =
                RL_MOVE,

            shoot =
                RL_SHOOT,

            connected =
                tcp ~= nil

        }

    }

end


-- =========================================================
-- QUEUE STATE
-- =========================================================

local function QueueState(state)

    if tcp == nil then
        return
    end


    -- Prefer the newest observation instead of building
    -- a queue of stale observations.
    if txBuffer ~= "" then
        return
    end


    local success, encoded =
        pcall(
            json.encode,
            state
        )


    if not success then

        Isaac.DebugString(
            "RL JSON ENCODE ERROR"
        )

        return

    end


    txBuffer =
        encoded
        .. "\n"

end


-- =========================================================
-- EPISODE START
-- =========================================================

function mod:OnGameStarted(
    isContinued
)

    episodeId =
        episodeId
        + 1


    gameOver =
        false


    resetInProgress =
        false


    trainingReady =
        false


    trainingRoomIndex =
        nil


    trainingSpawnCountdown =
        TRAINING_SPAWN_DELAY


    RL_ENABLED =
        false


    RL_MOVE =
        "NONE"


    RL_SHOOT =
        "NONE"


    RL_ACTION_ID =
        0


    pendingActionId =
        nil


    pendingActionFrame =
        nil


    Isaac.DebugString(
        "RL_EPISODE START id="
        .. tostring(
            episodeId
        )
    )

end


mod:AddCallback(
    ModCallbacks.MC_POST_GAME_STARTED,
    mod.OnGameStarted
)


-- =========================================================
-- EPISODE END
-- =========================================================

function mod:OnGameEnd(
    isGameOver
)

    gameOver =
        true


    RL_ENABLED =
        false


    RL_MOVE =
        "NONE"


    RL_SHOOT =
        "NONE"


    pendingActionId =
        nil


    pendingActionFrame =
        nil


    Isaac.DebugString(
        "RL_EPISODE END"
    )

end


mod:AddCallback(
    ModCallbacks.MC_POST_GAME_END,
    mod.OnGameEnd
)


-- =========================================================
-- UPDATE
-- =========================================================

function mod:OnUpdate()

    ReadPythonCommands()

    FlushOutgoing()


    -- =====================================================
    -- ACKNOWLEDGE APPLIED ACTION
    -- =====================================================

    local currentFrame =
        game:GetFrameCount()


    if pendingActionId ~= nil
    and pendingActionFrame ~= nil
    and currentFrame
        > pendingActionFrame then


        RL_ACTION_ID =
            pendingActionId


        pendingActionId =
            nil


        pendingActionFrame =
            nil

    end


    -- =====================================================
    -- SPAWN TRAINING ENEMY
    -- =====================================================

    if trainingSpawnCountdown
        > 0 then


        trainingSpawnCountdown =
            trainingSpawnCountdown
            - 1


        if trainingSpawnCountdown
            == 0 then


            SpawnTrainingEnemy()

        end

    end


    if resetInProgress
    or not trainingReady then

        return

    end


    -- =====================================================
    -- TRAINING ROOM DOORS
    -- =====================================================

    local currentRoomIndex =
        GetCurrentRoomIndex()


    if trainingRoomIndex ~= nil
    and currentRoomIndex
        == trainingRoomIndex then


        local room =
            game:GetRoom()


        if room:IsClear() then

            -- Combat finished.
            -- Navigation stage starts.
            OpenTrainingRoomDoors()

        else

            -- Combat still active.
            CloseTrainingRoomDoors()

        end

    end


    local frame =
        game:GetFrameCount()


    -- =====================================================
    -- SEND STATE TO PYTHON
    -- =====================================================

    if frame
        % STATE_SEND_INTERVAL
        == 0 then


        local state =
            GetGameState()


        if state ~= nil then

            QueueState(
                state
            )

            FlushOutgoing()

        end

    end


    -- =====================================================
    -- DEBUG LOG
    -- =====================================================

    if frame
        % STATE_LOG_INTERVAL
        == 0 then


        local state =
            GetGameState()


        if state ~= nil then


            local success, encoded =
                pcall(
                    json.encode,
                    state
                )


            if success then

                Isaac.DebugString(
                    "RL_STATE "
                    .. encoded
                )

            end

        end

    end

end


mod:AddCallback(
    ModCallbacks.MC_POST_UPDATE,
    mod.OnUpdate
)


-- =========================================================
-- INPUT HELPERS
-- =========================================================

local function IsMovementButton(button)

    return
        button
        == ButtonAction.ACTION_LEFT

        or button
        == ButtonAction.ACTION_RIGHT

        or button
        == ButtonAction.ACTION_UP

        or button
        == ButtonAction.ACTION_DOWN

end


local function IsShootingButton(button)

    return
        button
        == ButtonAction.ACTION_SHOOTLEFT

        or button
        == ButtonAction.ACTION_SHOOTRIGHT

        or button
        == ButtonAction.ACTION_SHOOTUP

        or button
        == ButtonAction.ACTION_SHOOTDOWN

end


-- =========================================================
-- INPUT OVERRIDE
-- =========================================================

function mod:OnInput(
    entity,
    inputHook,
    buttonAction
)

    if not RL_ENABLED then
        return nil
    end


    if entity == nil then
        return nil
    end


    local player =
        entity:ToPlayer()


    if player == nil then
        return nil
    end


    local mainPlayer =
        Isaac.GetPlayer(
            0
        )


    if mainPlayer ~= nil
    and player.InitSeed
        ~= mainPlayer.InitSeed then

        return nil

    end


    -- =====================================================
    -- MOVEMENT
    -- =====================================================

    if IsMovementButton(
        buttonAction
    ) then


        local wantedMove =
            MOVE_ACTIONS[
                RL_MOVE
            ]


        local pressed =
            wantedMove ~= nil
            and buttonAction
            == wantedMove


        if inputHook
            == InputHook.IS_ACTION_PRESSED then

            return pressed

        end


        if inputHook
            == InputHook.IS_ACTION_TRIGGERED then

            return pressed

        end


        if inputHook
            == InputHook.GET_ACTION_VALUE then


            if pressed then
                return 1.0
            else
                return 0.0
            end

        end

    end


    -- =====================================================
    -- SHOOTING
    -- =====================================================

    if IsShootingButton(
        buttonAction
    ) then


        local wantedShoot =
            SHOOT_ACTIONS[
                RL_SHOOT
            ]


        local pressed =
            wantedShoot ~= nil
            and buttonAction
            == wantedShoot


        if inputHook
            == InputHook.IS_ACTION_PRESSED then

            return pressed

        end


        if inputHook
            == InputHook.IS_ACTION_TRIGGERED then

            return pressed

        end


        if inputHook
            == InputHook.GET_ACTION_VALUE then


            if pressed then
                return 1.0
            else
                return 0.0
            end

        end

    end


    return nil

end


mod:AddCallback(
    ModCallbacks.MC_INPUT_ACTION,
    mod.OnInput
)