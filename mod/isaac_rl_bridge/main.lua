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

-- Isaac runs at roughly 30 updates/sec.
-- 3 = roughly 10 observations/sec.
local STATE_SEND_INTERVAL = 3

local STATE_LOG_INTERVAL = 30

local RECONNECT_INTERVAL = 30

-- Wait several updates before spawning the controlled
-- training enemy in a newly entered room.
local TRAINING_SPAWN_DELAY = 8

-- Number of UNIQUE combat rooms that must be cleared
-- and exited to complete one episode.
local TARGET_TRAINING_ROOMS = 3


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
-- NAVIGATION CONFIG
-- =========================================================

-- First target:
-- point INSIDE the room, in front of a doorway.
local DOOR_APPROACH_DISTANCE = 70

-- Second target:
-- point BEYOND the doorway.
local DOOR_EXIT_DISTANCE = 90

-- Alignment tolerance with the center of a doorway.
local DOOR_ALIGNMENT_TOLERANCE = 32

-- Door hysteresis.
--
-- A different door must be at least 25% closer than the
-- currently selected door before we switch to it.
--
-- 0.75 means:
--
-- current door = 100 px
-- alternative  = 90 px  -> keep current
-- alternative  = 70 px  -> switch
local DOOR_SWITCH_RATIO = 0.75


-- =========================================================
-- RL CONTROL
-- =========================================================

RL_ENABLED = false

RL_MOVE = "NONE"
RL_SHOOT = "NONE"

-- Last action that has had at least one game update
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

-- True after the first controlled encounter has spawned.
local trainingReady = false

-- True while the current room is controlled by the
-- curriculum.
--
-- This can mean either:
--   combat room
--   navigation-only revisited room
local encounterActive = false

-- True after TARGET_TRAINING_ROOMS unique combat rooms
-- have been successfully completed.
local trainingComplete = false

local trainingSpawnCountdown = -1

-- Current room being controlled.
local trainingRoomIndex = nil

-- Number of unique combat rooms completed.
--
-- We keep the old "rooms_exited" field in the JSON state
-- for Python compatibility, but semantically this now means
-- unique training rooms completed.
local roomsCompleted = 0

-- Number of actual room-to-room transitions.
--
-- Diagnostic only. Backtracking increments this number,
-- but DOES NOT increase roomsCompleted.
local totalRoomTransitions = 0

-- True after combat is finished and Isaac is currently
-- expected to navigate through a door.
local waitingForRoomExit = false


-- =========================================================
-- MAP MEMORY
-- =========================================================

-- Rooms Isaac has physically entered during this episode.
local visitedRoomIndices = {}

-- Rooms in which the training enemy was killed and Isaac
-- successfully exited afterward.
local completedRoomIndices = {}


-- =========================================================
-- NAVIGATION MEMORY
-- =========================================================

-- This is NOT a permanently fixed door.
--
-- It only provides hysteresis:
-- keep using the current door until another valid door is
-- substantially better.
local navigationDoorSlot = nil


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
    "TARGET TRAINING ROOMS: "
    .. tostring(
        TARGET_TRAINING_ROOMS
    )
)

Isaac.DebugString(
    "======================================"
)


-- =========================================================
-- GENERIC HELPERS
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

    local level =
        game:GetLevel()


    if level == nil then
        return -1
    end


    return level:GetCurrentRoomIndex()

end


local function MarkRoomVisited(roomIndex)

    if roomIndex == nil then
        return
    end


    if roomIndex < 0 then
        return
    end


    visitedRoomIndices[
        roomIndex
    ] = true

end


local function IsRoomVisited(roomIndex)

    if roomIndex == nil then
        return false
    end


    return visitedRoomIndices[
        roomIndex
    ] == true

end


local function IsRoomCompleted(roomIndex)

    if roomIndex == nil then
        return false
    end


    return completedRoomIndices[
        roomIndex
    ] == true

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
-- DOOR FILTERING
-- =========================================================

local function IsDirectDoorVariant(door)

    if door == nil then
        return false
    end


    local variant =
        door:GetVariant()


    -- We intentionally allow only ordinary/free doors.
    --
    -- Hidden secret-room doors and all locked variants are
    -- outside the current action space because the agent
    -- cannot use bombs/keys deliberately yet.
    if variant
        ~= DoorVariant.DOOR_UNSPECIFIED
    and variant
        ~= DoorVariant.DOOR_UNLOCKED then

        return false

    end


    return true

end


local function CanOpenDoorWithoutResource(door)

    if door == nil then
        return false
    end


    if not IsDirectDoorVariant(
        door
    ) then

        return false
    end


    if door:IsLocked() then
        return false
    end


    local targetRoomIndex =
        door.TargetRoomIndex


    if targetRoomIndex == nil
    or targetRoomIndex < 0 then

        return false

    end


    return true

end


local function IsDoorNavigationCandidate(door)

    if not CanOpenDoorWithoutResource(
        door
    ) then

        return false

    end


    -- At this point the agent should only target an actual
    -- currently passable opening.
    if not door:IsOpen() then
        return false
    end


    return true

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
-- DOOR CONTROL
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


local function ReleaseTraversableTrainingRoomDoors()

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


        -- IMPORTANT:
        --
        -- Do not blindly Open() every GridEntityDoor.
        --
        -- Hidden / locked / special resource-gated doors
        -- remain untouched.
        if CanOpenDoorWithoutResource(
            door
        ) then

            door:Open()

        end

    end

end


-- =========================================================
-- REMOVE UNCONTROLLED COMBAT
-- =========================================================

local function RemoveUncontrolledCombatEntities()

    for _, entity
    in ipairs(
        Isaac.GetRoomEntities()
    ) do


        if entity:IsActiveEnemy(
            false
        ) then

            entity:Remove()


        elseif entity.Type
            == EntityType.ENTITY_PROJECTILE then

            entity:Remove()

        end

    end

end


-- =========================================================
-- PREPARE COMBAT ROOM
-- =========================================================

local function PrepareTrainingRoom()

    local room =
        game:GetRoom()


    if room == nil then
        return
    end


    RemoveUncontrolledCombatEntities()


    room:SetClear(
        false
    )


    navigationDoorSlot =
        nil


    waitingForRoomExit =
        false


    CloseTrainingRoomDoors()

end


-- =========================================================
-- PREPARE REVISITED ROOM
-- =========================================================

local function PrepareNavigationOnlyRoom()

    local room =
        game:GetRoom()


    if room == nil then
        return
    end


    -- A room already completed earlier should not create
    -- another combat encounter when we backtrack through it.
    RemoveUncontrolledCombatEntities()


    room:SetClear(
        true
    )


    encounterActive =
        true


    waitingForRoomExit =
        true


    trainingSpawnCountdown =
        -1


    navigationDoorSlot =
        nil


    ReleaseTraversableTrainingRoomDoors()


    Isaac.DebugString(
        "RL_NAVIGATION ONLY room="
        .. tostring(
            GetCurrentRoomIndex()
        )
    )

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


    if not IsDoorNavigationCandidate(
        door
    ) then

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


    -- =====================================================
    -- APPROACH POINT
    -- =====================================================

    local approachPosition =
        Vector(

            doorPosition.X
                - outwardDirection.X
                * DOOR_APPROACH_DISTANCE,

            doorPosition.Y
                - outwardDirection.Y
                * DOOR_APPROACH_DISTANCE

        )


    -- =====================================================
    -- EXIT POINT
    -- =====================================================

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


    local approachDistance =
        math.sqrt(
            approachDistanceSquared
        )


    -- =====================================================
    -- ALIGNMENT CHECK
    -- =====================================================

    local aligned = false


    if outwardDirection.X ~= 0 then

        -- Left / right doorway.
        aligned =
            math.abs(
                player.Position.Y
                - doorPosition.Y
            )
            <= DOOR_ALIGNMENT_TOLERANCE

    else

        -- Up / down doorway.
        aligned =
            math.abs(
                player.Position.X
                - doorPosition.X
            )
            <= DOOR_ALIGNMENT_TOLERANCE

    end


    -- =====================================================
    -- APPROACH PLANE CHECK
    -- =====================================================

    local reachedApproachPlane =
        false


    if outwardDirection.X > 0 then

        reachedApproachPlane =
            player.Position.X
            >= approachPosition.X


    elseif outwardDirection.X < 0 then

        reachedApproachPlane =
            player.Position.X
            <= approachPosition.X


    elseif outwardDirection.Y > 0 then

        reachedApproachPlane =
            player.Position.Y
            >= approachPosition.Y


    elseif outwardDirection.Y < 0 then

        reachedApproachPlane =
            player.Position.Y
            <= approachPosition.Y

    end


    -- =====================================================
    -- APPROACH / EXIT PHASE
    -- =====================================================

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


    local targetRoomIndex =
        door.TargetRoomIndex


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

        approach_distance =
            approachDistance,

        approach_distance_squared =
            approachDistanceSquared,

        target_room_index =
            targetRoomIndex,

        target_room_visited =
            IsRoomVisited(
                targetRoomIndex
            )

    }

end


-- =========================================================
-- DOOR SELECTION WITH HYSTERESIS
-- =========================================================

local function SelectNavigationDoor(
    room,
    player
)

    local candidates =
        {}


    local bestUnvisited =
        nil


    local bestVisited =
        nil


    local currentCandidate =
        nil


    -- =====================================================
    -- BUILD VALID CANDIDATES
    -- =====================================================

    for slot = 0,
        DoorSlot.NUM_DOOR_SLOTS - 1 do


        local doorData =
            GetDoorNavigationData(
                room,
                player,
                slot
            )


        if doorData ~= nil then


            candidates[
                slot
            ] = doorData


            if slot
                == navigationDoorSlot then

                currentCandidate =
                    doorData

            end


            if doorData.target_room_visited then


                if bestVisited == nil
                or doorData.approach_distance
                    < bestVisited.approach_distance then


                    bestVisited =
                        doorData

                end


            else


                if bestUnvisited == nil
                or doorData.approach_distance
                    < bestUnvisited.approach_distance then


                    bestUnvisited =
                        doorData

                end

            end

        end

    end


    -- =====================================================
    -- PREFER UNVISITED ROOMS
    -- =====================================================
    --
    -- This is preference, not prohibition.
    --
    -- If no unvisited usable door exists, backtracking is
    -- still allowed.
    -- =====================================================

    local bestCandidate =
        bestUnvisited
        or bestVisited


    if bestCandidate == nil then

        navigationDoorSlot =
            nil

        return nil

    end


    -- =====================================================
    -- CURRENT DOOR INVALID
    -- =====================================================

    if currentCandidate == nil then

        navigationDoorSlot =
            bestCandidate.slot

        return bestCandidate

    end


    -- =====================================================
    -- CURRENT DOOR LEADS TO VISITED ROOM WHILE AN
    -- UNVISITED ROUTE EXISTS
    -- =====================================================

    if currentCandidate.target_room_visited
    and bestUnvisited ~= nil then

        navigationDoorSlot =
            bestUnvisited.slot

        return bestUnvisited

    end


    -- =====================================================
    -- DETERMINE BEST CANDIDATE IN SAME PRIORITY CLASS
    -- =====================================================

    local comparisonCandidate =
        nil


    if currentCandidate.target_room_visited then

        comparisonCandidate =
            bestVisited

    else

        comparisonCandidate =
            bestUnvisited

    end


    if comparisonCandidate == nil then

        return currentCandidate

    end


    if comparisonCandidate.slot
        == currentCandidate.slot then

        return currentCandidate

    end


    -- =====================================================
    -- HYSTERESIS
    -- =====================================================
    --
    -- Alternative must be significantly closer.
    -- =====================================================

    local switchThreshold =
        currentCandidate.approach_distance
        * DOOR_SWITCH_RATIO


    if comparisonCandidate.approach_distance
        < switchThreshold then


        navigationDoorSlot =
            comparisonCandidate.slot


        return comparisonCandidate

    end


    -- Difference is too small.
    -- Keep current door.
    return currentCandidate

end


-- =========================================================
-- GET NAVIGATION TARGET
-- =========================================================

local function GetNavigationTarget(player)

    local room =
        game:GetRoom()


    if room == nil then
        return nil
    end


    local selected =
        SelectNavigationDoor(
            room,
            player
        )


    if selected == nil then
        return nil
    end


    local targetPosition =
        selected.target_position


    local doorPosition =
        selected.door_position


    local approachPosition =
        selected.approach_position


    local exitPosition =
        selected.exit_position


    return {

        x =
            targetPosition.X,

        y =
            targetPosition.Y,

        phase =
            selected.phase,

        slot =
            selected.slot,

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
            selected.target_room_index,

        target_room_visited =
            selected.target_room_visited

    }

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


    if trainingComplete then
        return false
    end


    RemoveUncontrolledCombatEntities()


    room:SetClear(
        false
    )


    CloseTrainingRoomDoors()


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


    MarkRoomVisited(
        trainingRoomIndex
    )


    encounterActive =
        true


    waitingForRoomExit =
        false


    navigationDoorSlot =
        nil


    trainingReady =
        true


    Isaac.DebugString(
        "RL_ENCOUNTER READY"
        .. " episode="
        .. tostring(
            episodeId
        )
        .. " room="
        .. tostring(
            trainingRoomIndex
        )
        .. " completed="
        .. tostring(
            roomsCompleted
        )
        .. "/"
        .. tostring(
            TARGET_TRAINING_ROOMS
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


    encounterActive =
        false


    trainingComplete =
        false


    waitingForRoomExit =
        false


    trainingRoomIndex =
        nil


    roomsCompleted =
        0


    totalRoomTransitions =
        0


    trainingSpawnCountdown =
        -1


    visitedRoomIndices =
        {}


    completedRoomIndices =
        {}


    navigationDoorSlot =
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
    -- SHOOT
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

        txBuffer =
            ""

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


    if encounterActive
    and not trainingComplete
    and room:IsClear()
    and trainingRoomIndex ~= nil
    and currentRoomIndex
        == trainingRoomIndex then


        navigationTarget =
            GetNavigationTarget(
                player
            )

    end


    local roomsRemaining =
        math.max(
            0,
            TARGET_TRAINING_ROOMS
                - roomsCompleted
        )


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


        encounter_active =
            encounterActive,


        training_complete =
            trainingComplete,


        -- Compatibility with existing Python/evaluate.py.
        rooms_exited =
            roomsCompleted,


        rooms_completed =
            roomsCompleted,


        target_room_exits =
            TARGET_TRAINING_ROOMS,


        target_training_rooms =
            TARGET_TRAINING_ROOMS,


        rooms_remaining =
            roomsRemaining,


        total_room_transitions =
            totalRoomTransitions,


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


    -- Prefer latest state over a large stale queue.
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
-- NEW ROOM
-- =========================================================

function mod:OnNewRoom()

    if resetInProgress then
        return
    end


    -- Initial room callback can happen before the training
    -- run is fully initialized.
    if not trainingReady then
        return
    end


    if trainingComplete then
        return
    end


    -- We only consider the transition valid if the previous
    -- room was actually in navigation mode.
    if not waitingForRoomExit then
        return
    end


    local newRoomIndex =
        GetCurrentRoomIndex()


    local previousRoomIndex =
        trainingRoomIndex


    if previousRoomIndex == nil then
        return
    end


    if newRoomIndex
        == previousRoomIndex then

        return

    end


    -- =====================================================
    -- ROOM TRANSITION
    -- =====================================================

    totalRoomTransitions =
        totalRoomTransitions
        + 1


    MarkRoomVisited(
        newRoomIndex
    )


    -- =====================================================
    -- UNIQUE ROOM COMPLETION
    -- =====================================================
    --
    -- Backtracking out of an already completed room does
    -- NOT increase curriculum progress.
    -- =====================================================

    if not IsRoomCompleted(
        previousRoomIndex
    ) then


        completedRoomIndices[
            previousRoomIndex
        ] = true


        roomsCompleted =
            roomsCompleted
            + 1


        Isaac.DebugString(
            "RL_ROOM COMPLETE "
            .. tostring(
                roomsCompleted
            )
            .. "/"
            .. tostring(
                TARGET_TRAINING_ROOMS
            )
            .. " room="
            .. tostring(
                previousRoomIndex
            )
        )


    else


        Isaac.DebugString(
            "RL_BACKTRACK TRANSITION "
            .. tostring(
                previousRoomIndex
            )
            .. " -> "
            .. tostring(
                newRoomIndex
            )
        )

    end


    encounterActive =
        false


    waitingForRoomExit =
        false


    navigationDoorSlot =
        nil


    -- Stop old movement while the new room state settles.
    RL_MOVE =
        "NONE"


    RL_SHOOT =
        "NONE"


    -- =====================================================
    -- COMPLETE FULL CURRICULUM
    -- =====================================================

    if roomsCompleted
        >= TARGET_TRAINING_ROOMS then


        trainingComplete =
            true


        trainingSpawnCountdown =
            -1


        trainingRoomIndex =
            newRoomIndex


        RemoveUncontrolledCombatEntities()


        local room =
            game:GetRoom()


        if room ~= nil then

            room:SetClear(
                true
            )

        end


        ReleaseTraversableTrainingRoomDoors()


        Isaac.DebugString(
            "RL_TRAINING COMPLETE episode="
            .. tostring(
                episodeId
            )
        )


        return

    end


    -- =====================================================
    -- NEXT ROOM
    -- =====================================================

    trainingRoomIndex =
        newRoomIndex


    -- If the agent comes back to a room it already
    -- completed, do NOT spawn another training enemy.
    if IsRoomCompleted(
        newRoomIndex
    ) then


        PrepareNavigationOnlyRoom()


        return

    end


    -- First visit to a new room:
    -- prepare another controlled combat encounter.
    encounterActive =
        false


    PrepareTrainingRoom()


    trainingSpawnCountdown =
        TRAINING_SPAWN_DELAY


    Isaac.DebugString(
        "RL_PREPARING NEW COMBAT ROOM "
        .. tostring(
            newRoomIndex
        )
    )

end


mod:AddCallback(
    ModCallbacks.MC_POST_NEW_ROOM,
    mod.OnNewRoom
)


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


    encounterActive =
        false


    trainingComplete =
        false


    waitingForRoomExit =
        false


    roomsCompleted =
        0


    totalRoomTransitions =
        0


    visitedRoomIndices =
        {}


    completedRoomIndices =
        {}


    navigationDoorSlot =
        nil


    trainingRoomIndex =
        GetCurrentRoomIndex()


    MarkRoomVisited(
        trainingRoomIndex
    )


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


    PrepareTrainingRoom()


    Isaac.DebugString(
        "RL_EPISODE START id="
        .. tostring(
            episodeId
        )
        .. " room="
        .. tostring(
            trainingRoomIndex
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


    navigationDoorSlot =
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


    if resetInProgress then
        return
    end


    -- =====================================================
    -- WAITING TO SPAWN NEXT ENCOUNTER
    -- =====================================================

    if trainingSpawnCountdown
        > 0 then


        trainingSpawnCountdown =
            trainingSpawnCountdown
            - 1


        if not trainingComplete then

            CloseTrainingRoomDoors()

        end


        if trainingSpawnCountdown
            == 0 then


            SpawnTrainingEnemy()

        end

    end


    -- =====================================================
    -- TERMINAL STATE
    -- =====================================================

    if trainingComplete then


        local frame =
            game:GetFrameCount()


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


        return

    end


    -- While waiting for the next enemy spawn we suppress
    -- observations so Python sees the room only after it is
    -- ready.
    if not encounterActive then
        return
    end


    -- =====================================================
    -- COMBAT / NAVIGATION
    -- =====================================================

    local currentRoomIndex =
        GetCurrentRoomIndex()


    if trainingRoomIndex ~= nil
    and currentRoomIndex
        == trainingRoomIndex then


        local room =
            game:GetRoom()


        if room:IsClear() then


            waitingForRoomExit =
                true


            -- Only ordinary/free doors are released.
            ReleaseTraversableTrainingRoomDoors()


        else


            waitingForRoomExit =
                false


            navigationDoorSlot =
                nil


            CloseTrainingRoomDoors()

        end

    end


    local frame =
        game:GetFrameCount()


    -- =====================================================
    -- SEND STATE
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
    -- DEBUG STATE
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