-- =========================================================
-- ISAAC RL BRIDGE
-- The Binding of Isaac: Afterbirth+
--
-- Requires:
-- --luadebug
--
-- Python:
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

local STATE_SEND_INTERVAL = 3
local STATE_LOG_INTERVAL = 30
local RECONNECT_INTERVAL = 30

local TRAINING_SPAWN_DELAY = 8

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
-- NAVIGATION
-- =========================================================

local DOOR_APPROACH_DISTANCE = 70
local DOOR_EXIT_DISTANCE = 90

local DOOR_ALIGNMENT_TOLERANCE = 32

-- Alternative door must be at least 25% better
-- before target switches.
local DOOR_SWITCH_RATIO = 0.75


-- =========================================================
-- LOCAL SENSORS
-- =========================================================
--
-- The agent can move only cardinally, so for now we expose
-- four cardinal obstacle sensors and four hazard sensors.
-- =========================================================

local SENSOR_BLOCK_DISTANCE = 70

local SENSOR_DANGER_DISTANCE_1 = 35
local SENSOR_DANGER_DISTANCE_2 = 65

local FIRE_DANGER_RADIUS = 42


local SENSOR_DIRECTIONS = {
    left = Vector(-1, 0),
    right = Vector(1, 0),
    up = Vector(0, -1),
    down = Vector(0, 1)
}


-- =========================================================
-- RL CONTROL
-- =========================================================

RL_ENABLED = false

RL_MOVE = "NONE"
RL_SHOOT = "NONE"

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
local encounterActive = false
local trainingComplete = false

local trainingSpawnCountdown = -1

local trainingRoomIndex = nil

local roomsCompleted = 0
local totalRoomTransitions = 0

local waitingForRoomExit = false


-- =========================================================
-- MAP MEMORY
-- =========================================================

local visitedRoomIndices = {}
local completedRoomIndices = {}

-- Hysteresis only.
-- This does NOT permanently fix a door.
local navigationDoorSlot = nil


-- =========================================================
-- TCP
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
-- STARTUP
-- =========================================================

Isaac.DebugString(
    "======================================"
)

Isaac.DebugString(
    "ISAAC RL BRIDGE STRUCTURED V2"
)

Isaac.DebugString(
    "OBSERVATION: 21 FEATURES"
)

Isaac.DebugString(
    "TARGET TRAINING ROOMS: "
    .. tostring(TARGET_TRAINING_ROOMS)
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


local function MarkRoomVisited(roomIndex)

    if roomIndex == nil
    or roomIndex < 0 then

        return

    end

    visitedRoomIndices[roomIndex] = true

end


local function IsRoomVisited(roomIndex)

    if roomIndex == nil then
        return false
    end

    return visitedRoomIndices[roomIndex] == true

end


local function IsRoomCompleted(roomIndex)

    if roomIndex == nil then
        return false
    end

    return completedRoomIndices[roomIndex] == true

end


local function DistanceSquared(
    positionA,
    positionB
)

    local dx =
        positionA.X - positionB.X

    local dy =
        positionA.Y - positionB.Y

    return
        dx * dx
        + dy * dy

end


-- =========================================================
-- HAZARDS
-- =========================================================

local function IsGridHazardAtPosition(
    room,
    position
)

    local gridEntity =
        room:GetGridEntityFromPos(
            position
        )


    if gridEntity == nil then
        return false
    end


    local gridType =
        gridEntity:GetType()


    if gridType
        == GridEntityType.GRID_SPIKES then

        return true

    end


    if gridType
        == GridEntityType.GRID_SPIKES_ONOFF then

        -- Conservative for now:
        -- treat retractable spikes as dangerous regardless
        -- of their current animation state.
        return true

    end


    return false

end


local function IsFireHazardNearPosition(
    position
)

    local radiusSquared =
        FIRE_DANGER_RADIUS
        * FIRE_DANGER_RADIUS


    for _, entity
    in ipairs(
        Isaac.GetRoomEntities()
    ) do

        if entity.Type
            == EntityType.ENTITY_FIREPLACE then


            -- Extinguished fireplaces can remain as an
            -- entity, so only count living ones.
            if entity.HitPoints > 0 then


                if DistanceSquared(
                    entity.Position,
                    position
                ) <= radiusSquared then

                    return true

                end

            end

        end

    end


    return false

end


local function IsPositionDangerous(
    room,
    position
)

    if IsGridHazardAtPosition(
        room,
        position
    ) then

        return true

    end


    if IsFireHazardNearPosition(
        position
    ) then

        return true

    end


    return false

end


-- =========================================================
-- LOCAL OBSTACLE SENSORS
-- =========================================================

local function IsDirectionBlocked(
    room,
    player,
    direction
)

    local target =
        player.Position
        + direction
        * SENSOR_BLOCK_DISTANCE


    -- Mode 0 checks obstacles that impede ground movement.
    local clear =
        room:CheckLine(
            player.Position,
            target,
            0,
            0,
            false,
            false
        )


    return not clear

end


local function IsDirectionDangerous(
    room,
    player,
    direction
)

    local sample1 =
        player.Position
        + direction
        * SENSOR_DANGER_DISTANCE_1


    local sample2 =
        player.Position
        + direction
        * SENSOR_DANGER_DISTANCE_2


    return
        IsPositionDangerous(
            room,
            sample1
        )
        or IsPositionDangerous(
            room,
            sample2
        )

end


local function GetLocalSensors(player)

    local room = game:GetRoom()


    local blocked = {
        left = false,
        right = false,
        up = false,
        down = false
    }


    local danger = {
        left = false,
        right = false,
        up = false,
        down = false
    }


    if room == nil
    or player == nil then

        return {
            blocked = blocked,
            danger = danger
        }

    end


    for name, direction
    in pairs(
        SENSOR_DIRECTIONS
    ) do

        blocked[name] =
            IsDirectionBlocked(
                room,
                player,
                direction
            )


        danger[name] =
            IsDirectionDangerous(
                room,
                player,
                direction
            )

    end


    return {
        blocked = blocked,
        danger = danger
    }

end


-- =========================================================
-- SAFE TRAINING SPAWN
-- =========================================================

local function IsSafeTrainingSpawnPosition(
    room,
    player,
    position
)

    if not room:IsPositionInRoom(
        position,
        40
    ) then

        return false

    end


    local collision =
        room:GetGridCollisionAtPos(
            position
        )


    if collision
        ~= GridCollisionClass.COLLISION_NONE then

        return false

    end


    if IsPositionDangerous(
        room,
        position
    ) then

        return false

    end


    if player ~= nil then

        -- Do not spawn directly on top of Isaac.
        if DistanceSquared(
            player.Position,
            position
        ) < 120 * 120 then

            return false

        end

    end


    return true

end


local function FindSafeTrainingSpawnPosition(
    room,
    player
)

    local center =
        room:GetCenterPos()


    local startIndex =
        (
            Random()
            % #TRAINING_ENEMY_OFFSETS
        )
        + 1


    -- Try all four configured directions, starting from a
    -- random one.
    for offsetStep = 0,
        #TRAINING_ENEMY_OFFSETS - 1 do


        local offsetIndex =
            (
                (
                    startIndex
                    - 1
                    + offsetStep
                )
                % #TRAINING_ENEMY_OFFSETS
            )
            + 1


        local desiredPosition =
            center
            + TRAINING_ENEMY_OFFSETS[
                offsetIndex
            ]


        -- AB+ helper which avoids solid grid and pits.
        local freePosition =
            room:FindFreePickupSpawnPosition(
                desiredPosition,
                0,
                true
            )


        if IsSafeTrainingSpawnPosition(
            room,
            player,
            freePosition
        ) then

            return
                freePosition,
                offsetIndex

        end

    end


    -- Fallback:
    -- ask the game for a generic free tile near center.
    local fallback =
        room:FindFreeTilePosition(
            center,
            40
        )


    if IsSafeTrainingSpawnPosition(
        room,
        player,
        fallback
    ) then

        return fallback, 0

    end


    return nil, nil

end


-- =========================================================
-- DOOR DIRECTION
-- =========================================================

local function GetDoorOutwardDirection(slot)

    if slot == DoorSlot.LEFT0
    or slot == DoorSlot.LEFT1 then

        return Vector(-1, 0)

    elseif slot == DoorSlot.RIGHT0
    or slot == DoorSlot.RIGHT1 then

        return Vector(1, 0)

    elseif slot == DoorSlot.UP0
    or slot == DoorSlot.UP1 then

        return Vector(0, -1)

    elseif slot == DoorSlot.DOWN0
    or slot == DoorSlot.DOWN1 then

        return Vector(0, 1)

    end


    return nil

end


-- =========================================================
-- DOOR FILTER
-- =========================================================

local function IsDirectDoorVariant(door)

    if door == nil then
        return false
    end


    local variant =
        door:GetVariant()


    -- Current curriculum intentionally excludes secret,
    -- bomb-only and locked/special door variants.
    if variant
        ~= DoorVariant.DOOR_UNSPECIFIED
    and variant
        ~= DoorVariant.DOOR_UNLOCKED then

        return false

    end


    return true

end


local function IsDangerousRoomDoor(door)

    if door == nil then
        return true
    end


    -- Curse room doors cause entry/exit damage.
    if door:IsRoomType(
        RoomType.ROOM_CURSE
    ) then

        return true

    end


    -- Secret rooms currently require mechanics outside the
    -- agent's action space.
    if door:IsRoomType(
        RoomType.ROOM_SECRET
    ) then

        return true

    end


    if door:IsRoomType(
        RoomType.ROOM_SUPERSECRET
    ) then

        return true

    end


    return false

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


    if IsDangerousRoomDoor(
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


    -- Target only a doorway which is physically open now.
    if not door:IsOpen() then
        return false
    end


    return true

end


-- =========================================================
-- TCP DISCONNECT
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
        .. tostring(reason)
    )

end


-- =========================================================
-- TCP CONNECT
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


    lastConnectAttempt = frame


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

        tcp = client

        rxBuffer = ""
        txBuffer = ""

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


        -- Do not force curse/secret/locked doors open.
        if CanOpenDoorWithoutResource(
            door
        ) then

            door:Open()

        end

    end

end


-- =========================================================
-- REMOVE UNCONTROLLED ENEMIES
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
-- ROOM PREPARATION
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


    navigationDoorSlot = nil
    waitingForRoomExit = false


    CloseTrainingRoomDoors()

end


local function PrepareNavigationOnlyRoom()

    local room =
        game:GetRoom()


    if room == nil then
        return
    end


    RemoveUncontrolledCombatEntities()


    room:SetClear(
        true
    )


    encounterActive = true
    waitingForRoomExit = true

    trainingSpawnCountdown = -1

    navigationDoorSlot = nil


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


    local approachPosition =
        Vector(

            doorPosition.X
                - outwardDirection.X
                * DOOR_APPROACH_DISTANCE,

            doorPosition.Y
                - outwardDirection.Y
                * DOOR_APPROACH_DISTANCE

        )


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


    local aligned = false


    if outwardDirection.X ~= 0 then

        aligned =
            math.abs(
                player.Position.Y
                - doorPosition.Y
            )
            <= DOOR_ALIGNMENT_TOLERANCE

    else

        aligned =
            math.abs(
                player.Position.X
                - doorPosition.X
            )
            <= DOOR_ALIGNMENT_TOLERANCE

    end


    local reachedApproachPlane = false


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

        door = door,

        slot = slot,

        phase = navigationPhase,

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

        target_room_index =
            targetRoomIndex,

        target_room_visited =
            IsRoomVisited(
                targetRoomIndex
            )

    }

end


-- =========================================================
-- DOOR SELECTION
-- =========================================================

local function SelectNavigationDoor(
    room,
    player
)

    local bestUnvisited = nil
    local bestVisited = nil
    local currentCandidate = nil


    for slot = 0,
        DoorSlot.NUM_DOOR_SLOTS - 1 do


        local doorData =
            GetDoorNavigationData(
                room,
                player,
                slot
            )


        if doorData ~= nil then


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


    local bestCandidate =
        bestUnvisited
        or bestVisited


    if bestCandidate == nil then

        navigationDoorSlot = nil

        return nil

    end


    if currentCandidate == nil then

        navigationDoorSlot =
            bestCandidate.slot

        return bestCandidate

    end


    -- Always prefer a usable unvisited room over a
    -- visited-room target.
    if currentCandidate.target_room_visited
    and bestUnvisited ~= nil then

        navigationDoorSlot =
            bestUnvisited.slot

        return bestUnvisited

    end


    local comparisonCandidate = nil


    if currentCandidate.target_room_visited then

        comparisonCandidate =
            bestVisited

    else

        comparisonCandidate =
            bestUnvisited

    end


    if comparisonCandidate == nil
    or comparisonCandidate.slot
        == currentCandidate.slot then

        return currentCandidate

    end


    -- Hysteresis.
    local switchThreshold =
        currentCandidate.approach_distance
        * DOOR_SWITCH_RATIO


    if comparisonCandidate.approach_distance
        < switchThreshold then


        navigationDoorSlot =
            comparisonCandidate.slot


        return comparisonCandidate

    end


    return currentCandidate

end


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


    return {

        x =
            selected.target_position.X,

        y =
            selected.target_position.Y,

        phase =
            selected.phase,

        slot =
            selected.slot,

        door_x =
            selected.door_position.X,

        door_y =
            selected.door_position.Y,

        approach_x =
            selected.approach_position.X,

        approach_y =
            selected.approach_position.Y,

        exit_x =
            selected.exit_position.X,

        exit_y =
            selected.exit_position.Y,

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


    if room == nil
    or trainingComplete then

        return false

    end


    RemoveUncontrolledCombatEntities()


    room:SetClear(
        false
    )


    CloseTrainingRoomDoors()


    local player =
        Isaac.GetPlayer(
            0
        )


    local spawnPosition,
        offsetIndex =
            FindSafeTrainingSpawnPosition(
                room,
                player
            )


    if spawnPosition == nil then

        Isaac.DebugString(
            "RL_TRAINING NO SAFE SPAWN"
        )

        return false

    end


    local enemy =
        Isaac.Spawn(

            TRAINING_ENEMY_TYPE,

            TRAINING_ENEMY_VARIANT,

            TRAINING_ENEMY_SUBTYPE,

            spawnPosition,

            Vector(0, 0),

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


    encounterActive = true
    waitingForRoomExit = false

    navigationDoorSlot = nil

    trainingReady = true


    Isaac.DebugString(
        "RL_ENCOUNTER READY"
        .. " episode="
        .. tostring(episodeId)
        .. " room="
        .. tostring(trainingRoomIndex)
        .. " completed="
        .. tostring(roomsCompleted)
        .. "/"
        .. tostring(TARGET_TRAINING_ROOMS)
        .. " spawn_index="
        .. tostring(offsetIndex)
    )


    return true

end


-- =========================================================
-- RESET
-- =========================================================

local function RequestReset()

    if resetInProgress then
        return
    end


    resetInProgress = true

    trainingReady = false
    encounterActive = false
    trainingComplete = false

    waitingForRoomExit = false

    trainingRoomIndex = nil

    roomsCompleted = 0
    totalRoomTransitions = 0

    trainingSpawnCountdown = -1

    visitedRoomIndices = {}
    completedRoomIndices = {}

    navigationDoorSlot = nil

    gameOver = false


    RL_ENABLED = false
    RL_MOVE = "NONE"
    RL_SHOOT = "NONE"


    pendingActionId = nil
    pendingActionFrame = nil


    Isaac.DebugString(
        "RL_RESET REQUESTED"
    )


    Isaac.ExecuteCommand(
        "restart"
    )

end


-- =========================================================
-- PYTHON COMMANDS
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
            .. tostring(line)
        )

        return

    end


    if type(command)
        ~= "table" then

        return

    end


    if command.type
        == "reset" then


        RequestReset()

        return

    end


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

            RL_MOVE = move

        end

    end


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

            RL_SHOOT = shoot

        end

    end

end


-- =========================================================
-- RECEIVE
-- =========================================================

local function ReadPythonCommands()

    if tcp == nil then

        ConnectToPython()

        return

    end


    local messagesRead = 0


    while messagesRead < 20 do


        local line, err, partial =
            tcp:receive(
                "*l"
            )


        if line ~= nil then


            local completeLine =
                rxBuffer
                .. line


            rxBuffer = ""


            ProcessCommand(
                completeLine
            )


            messagesRead =
                messagesRead + 1


        else


            if partial ~= nil
            and #partial > 0 then


                rxBuffer =
                    rxBuffer
                    .. partial

            end


            if #rxBuffer > 65536 then

                rxBuffer = ""

                Isaac.DebugString(
                    "RL RX BUFFER OVERFLOW"
                )

            end


            if err == "timeout" then

                break


            elseif err == "closed" then


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
-- SEND
-- =========================================================

local function FlushOutgoing()

    if tcp == nil
    or txBuffer == "" then

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


    if err == "timeout" then


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


    if err == "closed" then

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
-- STATE
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


    local enemies = {}
    local totalEnemyHp = 0


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
                totalEnemyHp + hp


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


    local navigationTarget = nil


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


    local localSensors =
        GetLocalSensors(
            player
        )


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

        local_sensors =
            localSensors,

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

    if resetInProgress
    or not trainingReady
    or trainingComplete
    or not waitingForRoomExit then

        return

    end


    local newRoomIndex =
        GetCurrentRoomIndex()


    local previousRoomIndex =
        trainingRoomIndex


    if previousRoomIndex == nil
    or newRoomIndex
        == previousRoomIndex then

        return

    end


    totalRoomTransitions =
        totalRoomTransitions
        + 1


    MarkRoomVisited(
        newRoomIndex
    )


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
            .. tostring(roomsCompleted)
            .. "/"
            .. tostring(TARGET_TRAINING_ROOMS)
        )


    else


        Isaac.DebugString(
            "RL_BACKTRACK "
            .. tostring(previousRoomIndex)
            .. " -> "
            .. tostring(newRoomIndex)
        )

    end


    encounterActive = false
    waitingForRoomExit = false

    navigationDoorSlot = nil


    RL_MOVE = "NONE"
    RL_SHOOT = "NONE"


    if roomsCompleted
        >= TARGET_TRAINING_ROOMS then


        trainingComplete = true

        trainingSpawnCountdown = -1

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
            .. tostring(episodeId)
        )


        return

    end


    trainingRoomIndex =
        newRoomIndex


    if IsRoomCompleted(
        newRoomIndex
    ) then


        PrepareNavigationOnlyRoom()

        return

    end


    encounterActive = false


    PrepareTrainingRoom()


    trainingSpawnCountdown =
        TRAINING_SPAWN_DELAY

end


mod:AddCallback(
    ModCallbacks.MC_POST_NEW_ROOM,
    mod.OnNewRoom
)


-- =========================================================
-- GAME START
-- =========================================================

function mod:OnGameStarted(
    isContinued
)

    episodeId =
        episodeId + 1


    gameOver = false
    resetInProgress = false

    trainingReady = false
    encounterActive = false
    trainingComplete = false

    waitingForRoomExit = false

    roomsCompleted = 0
    totalRoomTransitions = 0

    visitedRoomIndices = {}
    completedRoomIndices = {}

    navigationDoorSlot = nil


    trainingRoomIndex =
        GetCurrentRoomIndex()


    MarkRoomVisited(
        trainingRoomIndex
    )


    trainingSpawnCountdown =
        TRAINING_SPAWN_DELAY


    RL_ENABLED = false
    RL_MOVE = "NONE"
    RL_SHOOT = "NONE"

    RL_ACTION_ID = 0

    pendingActionId = nil
    pendingActionFrame = nil


    PrepareTrainingRoom()


    Isaac.DebugString(
        "RL_EPISODE START id="
        .. tostring(episodeId)
    )

end


mod:AddCallback(
    ModCallbacks.MC_POST_GAME_STARTED,
    mod.OnGameStarted
)


-- =========================================================
-- GAME END
-- =========================================================

function mod:OnGameEnd(
    isGameOver
)

    gameOver = true

    RL_ENABLED = false
    RL_MOVE = "NONE"
    RL_SHOOT = "NONE"

    pendingActionId = nil
    pendingActionFrame = nil

    navigationDoorSlot = nil

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


    local currentFrame =
        game:GetFrameCount()


    -- =====================================================
    -- ACK ACTION
    -- =====================================================

    if pendingActionId ~= nil
    and pendingActionFrame ~= nil
    and currentFrame
        > pendingActionFrame then


        RL_ACTION_ID =
            pendingActionId


        pendingActionId = nil
        pendingActionFrame = nil

    end


    if resetInProgress then
        return
    end


    -- =====================================================
    -- SPAWN
    -- =====================================================

    if trainingSpawnCountdown > 0 then


        trainingSpawnCountdown =
            trainingSpawnCountdown - 1


        if not trainingComplete then

            CloseTrainingRoomDoors()

        end


        if trainingSpawnCountdown == 0 then


            local spawned =
                SpawnTrainingEnemy()


            -- If the room happened not to contain any safe
            -- spawn position, retry later rather than create
            -- a broken episode.
            if not spawned then

                trainingSpawnCountdown =
                    TRAINING_SPAWN_DELAY

            end

        end

    end


    -- =====================================================
    -- TERMINAL STATE
    -- =====================================================

    if trainingComplete then


        if currentFrame
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

            waitingForRoomExit = true

            ReleaseTraversableTrainingRoomDoors()


        else

            waitingForRoomExit = false

            navigationDoorSlot = nil

            CloseTrainingRoomDoors()

        end

    end


    -- =====================================================
    -- SEND STATE
    -- =====================================================

    if currentFrame
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
    -- DEBUG
    -- =====================================================

    if currentFrame
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

    if not RL_ENABLED
    or entity == nil then

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
    -- MOVE
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
            end


            return 0.0

        end

    end


    -- =====================================================
    -- SHOOT
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
            end


            return 0.0

        end

    end


    return nil

end


mod:AddCallback(
    ModCallbacks.MC_INPUT_ACTION,
    mod.OnInput
)