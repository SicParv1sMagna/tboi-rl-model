local mod = RegisterMod("Isaac RL Bridge", 1)

local game = Game()
local json = require("json")
local socket = require("socket")


-- =========================================================
-- NETWORK
-- =========================================================

local HOST = "127.0.0.1"
local PORT = 5000

local STATE_SEND_INTERVAL = 3
local STATE_LOG_INTERVAL = 30
local RECONNECT_INTERVAL = 30

local TRAINING_SPAWN_DELAY = 8


-- =========================================================
-- CURRICULUM
-- =========================================================
--
-- Stage 1:
--   1 enemy
--   combat basics
--
-- Stage 2:
--   1 enemy
--   8 spawn directions
--   preserve health
--   tear accuracy
--
-- Stage 3:
--   2 enemies
--   generic multi-enemy combat
--
-- Stage 4:
--   random 1..3 enemies
--   clear + exit one room
--
-- Stage 5:
--   random 1..3 enemies
--   three-room run
--
-- Observation stays fixed at 21 values.
-- =========================================================

local curriculumStage = 1


local function NormalizeCurriculumStage(value)
    local stage = tonumber(value)

    if stage == nil then
        return 1
    end

    stage = math.floor(stage)

    if stage < 1 then
        stage = 1
    elseif stage > 5 then
        stage = 5
    end

    return stage
end


local function GetCurriculumName()
    if curriculumStage == 1 then
        return "combat_basics"
    elseif curriculumStage == 2 then
        return "combat_health_accuracy"
    elseif curriculumStage == 3 then
        return "combat_multi_enemy"
    elseif curriculumStage == 4 then
        return "single_room_navigation"
    end

    return "three_room_run"
end


local function StageUsesNavigation()
    return curriculumStage >= 4
end


local function StageUsesEnemyDanger()
    -- Keep Stage 1/2 sensor semantics compatible with
    -- the already trained checkpoint.
    return curriculumStage >= 3
end


local function GetTargetRoomExits()
    if curriculumStage <= 3 then
        return 0
    elseif curriculumStage == 4 then
        return 1
    end

    return 3
end


local function GetDesiredEnemyCount()
    if curriculumStage <= 2 then
        return 1
    elseif curriculumStage == 3 then
        return 2
    end

    -- Stages 4/5:
    -- random 1..3 enemies per new combat room.
    return (Random() % 3) + 1
end


-- =========================================================
-- TRAINING ENEMY
-- =========================================================

local TRAINING_ENEMY_TYPE = 10
local TRAINING_ENEMY_VARIANT = 0
local TRAINING_ENEMY_SUBTYPE = 0


local TRAINING_ENEMY_OFFSETS_STAGE_1 = {
    Vector(160, 0),
    Vector(-160, 0),
    Vector(0, 120),
    Vector(0, -120)
}


local TRAINING_ENEMY_OFFSETS_GENERAL = {
    Vector(160, 0),
    Vector(-160, 0),
    Vector(0, 120),
    Vector(0, -120),

    Vector(130, 95),
    Vector(-130, 95),
    Vector(130, -95),
    Vector(-130, -95)
}


local MIN_ENEMY_SPAWN_DISTANCE = 110


local function GetTrainingEnemyOffsets()
    if curriculumStage == 1 then
        return TRAINING_ENEMY_OFFSETS_STAGE_1
    end

    return TRAINING_ENEMY_OFFSETS_GENERAL
end


-- =========================================================
-- NAVIGATION
-- =========================================================

local DOOR_APPROACH_DISTANCE = 70
local DOOR_EXIT_DISTANCE = 90
local DOOR_ALIGNMENT_TOLERANCE = 32

local DOOR_SWITCH_RATIO = 0.75


-- =========================================================
-- LOCAL SENSORS
-- =========================================================

local SENSOR_BLOCK_DISTANCE = 70

local SENSOR_DANGER_DISTANCE_1 = 35
local SENSOR_DANGER_DISTANCE_2 = 65

local FIRE_DANGER_RADIUS = 42

-- Radius around each directional sample point where
-- an enemy contributes to danger.
local ENEMY_DANGER_RADIUS = 60


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

-- Desired number for current combat encounter.
local currentEncounterEnemyTargetCount = 0


-- =========================================================
-- MAP MEMORY
-- =========================================================

local visitedRoomIndices = {}
local completedRoomIndices = {}

local navigationDoorSlot = nil


-- =========================================================
-- TEAR ACCURACY
-- =========================================================

local tearsFired = 0
local tearsHit = 0
local tearsMissed = 0

local trackedTears = {}


local function ResetTearStatistics()
    tearsFired = 0
    tearsHit = 0
    tearsMissed = 0

    trackedTears = {}
end


-- =========================================================
-- TCP
-- =========================================================

local tcp = nil

local lastConnectAttempt = -99999

local rxBuffer = ""
local txBuffer = ""


-- =========================================================
-- INPUT MAP
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


Isaac.DebugString("======================================")
Isaac.DebugString("ISAAC RL BRIDGE")
Isaac.DebugString("STRUCTURED V2 / GENERIC ENEMIES")
Isaac.DebugString("OBSERVATION SIZE: 21")
Isaac.DebugString("======================================")


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
    local level = game:GetLevel()

    if level == nil then
        return -1
    end

    return level:GetCurrentRoomIndex()
end


local function MarkRoomVisited(roomIndex)
    if roomIndex == nil or roomIndex < 0 then
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


local function DistanceSquared(a, b)
    local dx = a.X - b.X
    local dy = a.Y - b.Y

    return dx * dx + dy * dy
end


local function ShuffleIndices(count)
    local indices = {}

    for i = 1, count do
        indices[i] = i
    end

    for i = count, 2, -1 do
        local j = (Random() % i) + 1

        local tmp = indices[i]
        indices[i] = indices[j]
        indices[j] = tmp
    end

    return indices
end


-- =========================================================
-- TEAR TRACKING
-- =========================================================

local function IsMainPlayerTear(tear)
    if tear == nil then
        return false
    end

    local mainPlayer = Isaac.GetPlayer(0)

    if mainPlayer == nil then
        return false
    end

    local spawner = tear.SpawnerEntity

    if spawner ~= nil then
        local player = spawner:ToPlayer()

        if player ~= nil then
            return player.InitSeed == mainPlayer.InitSeed
        end
    end

    -- Fallback for INIT frames where the spawner reference
    -- is incomplete.
    return tear.SpawnerType == EntityType.ENTITY_PLAYER
end


local function TryTrackPlayerTear(tear)
    if tear == nil then
        return
    end

    if resetInProgress
        or trainingComplete
        or not encounterActive then

        return
    end

    local room = game:GetRoom()

    if room == nil or room:IsClear() then
        return
    end

    if not IsMainPlayerTear(tear) then
        return
    end

    local seed = tear.InitSeed

    if seed == nil then
        return
    end

    if trackedTears[seed] ~= nil then
        return
    end

    tearsFired = tearsFired + 1

    trackedTears[seed] = {
        hit = false,
        episode_id = episodeId,
        room_index = GetCurrentRoomIndex()
    }
end


function mod:OnTearInit(tear)
    TryTrackPlayerTear(tear)
end


mod:AddCallback(
    ModCallbacks.MC_POST_TEAR_INIT,
    mod.OnTearInit
)


function mod:OnTearUpdate(tear)
    -- Retry once tear data is fully populated.
    TryTrackPlayerTear(tear)
end


mod:AddCallback(
    ModCallbacks.MC_POST_TEAR_UPDATE,
    mod.OnTearUpdate
)


function mod:OnTearCollision(tear, collider, low)
    if tear == nil or collider == nil then
        return nil
    end

    local tracked =
        trackedTears[tear.InitSeed]

    if tracked == nil or tracked.hit then
        return nil
    end

    if collider:IsActiveEnemy(false)
        and collider:IsVulnerableEnemy() then

        tracked.hit = true
        tearsHit = tearsHit + 1
    end

    return nil
end


mod:AddCallback(
    ModCallbacks.MC_PRE_TEAR_COLLISION,
    mod.OnTearCollision
)


function mod:OnTearRemoved(entity)
    if entity == nil then
        return
    end

    local tracked =
        trackedTears[entity.InitSeed]

    if tracked == nil then
        return
    end

    trackedTears[entity.InitSeed] = nil

    if tracked.hit then
        return
    end

    if resetInProgress
        or trainingComplete
        or not encounterActive then

        return
    end

    if tracked.episode_id ~= episodeId then
        return
    end

    -- Entity removal during room unloading must not
    -- count as a miss.
    if tracked.room_index ~= GetCurrentRoomIndex() then
        return
    end

    local room = game:GetRoom()

    if room == nil or room:IsClear() then
        return
    end

    tearsMissed = tearsMissed + 1
end


mod:AddCallback(
    ModCallbacks.MC_POST_ENTITY_REMOVE,
    mod.OnTearRemoved,
    EntityType.ENTITY_TEAR
)


-- =========================================================
-- HAZARDS
-- =========================================================

local function IsGridHazardAtPosition(room, position)
    local gridEntity =
        room:GetGridEntityFromPos(position)

    if gridEntity == nil then
        return false
    end

    local gridType =
        gridEntity:GetType()

    if gridType == GridEntityType.GRID_SPIKES then
        return true
    end

    if GridEntityType.GRID_SPIKES_ONOFF ~= nil
        and gridType
        == GridEntityType.GRID_SPIKES_ONOFF then

        return true
    end

    return false
end


local function IsFireHazardNearPosition(position)
    local radiusSquared =
        FIRE_DANGER_RADIUS * FIRE_DANGER_RADIUS

    for _, entity in ipairs(Isaac.GetRoomEntities()) do
        if entity.Type == EntityType.ENTITY_FIREPLACE
            and entity.HitPoints > 0 then

            if DistanceSquared(
                entity.Position,
                position
            ) <= radiusSquared then

                return true
            end
        end
    end

    return false
end


local function IsStaticDangerAtPosition(room, position)
    return
        IsGridHazardAtPosition(room, position)
        or IsFireHazardNearPosition(position)
end


local function IsEnemyDangerNearPosition(position)
    if not StageUsesEnemyDanger() then
        return false
    end

    local radiusSquared =
        ENEMY_DANGER_RADIUS
        * ENEMY_DANGER_RADIUS

    for _, entity in ipairs(Isaac.GetRoomEntities()) do
        if entity:IsActiveEnemy(false)
            and entity:IsVulnerableEnemy() then

            if DistanceSquared(
                entity.Position,
                position
            ) <= radiusSquared then

                return true
            end
        end
    end

    return false
end


-- =========================================================
-- LOCAL SENSORS
-- =========================================================

local function IsDirectionBlocked(
    room,
    player,
    direction
)
    local target =
        player.Position
        + direction * SENSOR_BLOCK_DISTANCE

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


local function GetDirectionDanger(
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

    local staticDanger =
        IsStaticDangerAtPosition(
            room,
            sample1
        )
        or IsStaticDangerAtPosition(
            room,
            sample2
        )

    local enemyDanger =
        IsEnemyDangerNearPosition(sample1)
        or IsEnemyDangerNearPosition(sample2)

    return
        staticDanger or enemyDanger,
        staticDanger,
        enemyDanger
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

    local staticDanger = {
        left = false,
        right = false,
        up = false,
        down = false
    }

    local enemyDanger = {
        left = false,
        right = false,
        up = false,
        down = false
    }

    if room == nil or player == nil then
        return {
            blocked = blocked,
            danger = danger,
            static_danger = staticDanger,
            enemy_danger = enemyDanger
        }
    end

    for name, direction in pairs(SENSOR_DIRECTIONS) do
        blocked[name] =
            IsDirectionBlocked(
                room,
                player,
                direction
            )

        local combined, staticValue, enemyValue =
            GetDirectionDanger(
                room,
                player,
                direction
            )

        danger[name] = combined
        staticDanger[name] = staticValue
        enemyDanger[name] = enemyValue
    end

    return {
        blocked = blocked,
        danger = danger,
        static_danger = staticDanger,
        enemy_danger = enemyDanger
    }
end


-- =========================================================
-- GENERIC ENEMY SPAWN
-- =========================================================

local function IsSafeTrainingSpawnPosition(
    room,
    player,
    position
)
    if position == nil then
        return false
    end

    if not room:IsPositionInRoom(position, 40) then
        return false
    end

    local collision =
        room:GetGridCollisionAtPos(position)

    if collision
        ~= GridCollisionClass.COLLISION_NONE then

        return false
    end

    if IsStaticDangerAtPosition(
        room,
        position
    ) then
        return false
    end

    if player ~= nil
        and DistanceSquared(
            player.Position,
            position
        ) < 120 * 120 then

        return false
    end

    return true
end


local function IsFarEnoughFromChosen(
    position,
    chosenPositions
)
    local minimumSquared =
        MIN_ENEMY_SPAWN_DISTANCE
        * MIN_ENEMY_SPAWN_DISTANCE

    for _, existing in ipairs(chosenPositions) do
        if DistanceSquared(
            position,
            existing
        ) < minimumSquared then

            return false
        end
    end

    return true
end


local function FindTrainingSpawnPositions(
    room,
    player,
    requestedCount
)
    local offsets =
        GetTrainingEnemyOffsets()

    local order =
        ShuffleIndices(#offsets)

    local center =
        room:GetCenterPos()

    local chosen = {}

    for _, index in ipairs(order) do
        local desiredPosition =
            center + offsets[index]

        local freePosition =
            room:FindFreePickupSpawnPosition(
                desiredPosition,
                40,
                true
            )

        if IsSafeTrainingSpawnPosition(
            room,
            player,
            freePosition
        )
            and IsFarEnoughFromChosen(
                freePosition,
                chosen
            ) then

            table.insert(
                chosen,
                freePosition
            )

            if #chosen >= requestedCount then
                return chosen
            end
        end
    end

    return nil
end


-- =========================================================
-- DOORS
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


local function IsDangerousRoomDoor(door)
    if door == nil then
        return true
    end

    if door:IsRoomType(RoomType.ROOM_CURSE) then
        return true
    end

    if door:IsRoomType(RoomType.ROOM_SECRET) then
        return true
    end

    if door:IsRoomType(RoomType.ROOM_SUPERSECRET) then
        return true
    end

    return false
end


local function CanTraverseTrainingDoor(door)
    if door == nil then
        return false
    end

    if door:IsLocked() then
        return false
    end

    if IsDangerousRoomDoor(door) then
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


local function CountNavigationCandidates(room)
    if room == nil then
        return 0
    end

    local count = 0

    for slot = 0, DoorSlot.NUM_DOOR_SLOTS - 1 do
        local door =
            room:GetDoor(slot)

        if CanTraverseTrainingDoor(door) then
            count = count + 1
        end
    end

    return count
end


-- =========================================================
-- TCP CONNECTION
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
        return
    end

    client:settimeout(0.05)

    local success =
        client:connect(
            HOST,
            PORT
        )

    if success then
        tcp = client

        rxBuffer = ""
        txBuffer = ""

        tcp:settimeout(0)

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
    local room = game:GetRoom()

    if room == nil then
        return
    end

    for slot = 0, DoorSlot.NUM_DOOR_SLOTS - 1 do
        local door =
            room:GetDoor(slot)

        if door ~= nil then
            door:Close(true)
        end
    end
end


local function ReleaseTraversableTrainingRoomDoors()
    local room = game:GetRoom()

    if room == nil then
        return
    end

    for slot = 0, DoorSlot.NUM_DOOR_SLOTS - 1 do
        local door =
            room:GetDoor(slot)

        if CanTraverseTrainingDoor(door) then
            door:Open()
        end
    end
end


-- =========================================================
-- CONTROLLED ROOM
-- =========================================================

local function RemoveUncontrolledCombatEntities()
    for _, entity in ipairs(Isaac.GetRoomEntities()) do
        if entity:IsActiveEnemy(false) then
            entity:Remove()

        elseif entity.Type
            == EntityType.ENTITY_PROJECTILE then

            entity:Remove()
        end
    end
end


local function PrepareTrainingRoom()
    local room = game:GetRoom()

    if room == nil then
        return
    end

    RemoveUncontrolledCombatEntities()

    room:SetClear(false)

    navigationDoorSlot = nil
    waitingForRoomExit = false

    currentEncounterEnemyTargetCount = 0

    CloseTrainingRoomDoors()
end


local function PrepareNavigationOnlyRoom()
    local room = game:GetRoom()

    if room == nil then
        return
    end

    RemoveUncontrolledCombatEntities()

    room:SetClear(true)

    encounterActive = true
    waitingForRoomExit = true

    trainingSpawnCountdown = -1

    currentEncounterEnemyTargetCount = 0

    navigationDoorSlot = nil

    ReleaseTraversableTrainingRoomDoors()
end


-- =========================================================
-- NAVIGATION TARGET
-- =========================================================

local function GetDoorNavigationData(
    room,
    player,
    slot
)
    local door =
        room:GetDoor(slot)

    if not CanTraverseTrainingDoor(door) then
        return nil
    end

    local outwardDirection =
        GetDoorOutwardDirection(slot)

    if outwardDirection == nil then
        return nil
    end

    local doorPosition =
        room:GetDoorSlotPosition(slot)

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

    local dx =
        approachPosition.X
        - player.Position.X

    local dy =
        approachPosition.Y
        - player.Position.Y

    local approachDistance =
        math.sqrt(
            dx * dx + dy * dy
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

    local phase = "APPROACH"
    local targetPosition =
        approachPosition

    if aligned and reachedApproachPlane then
        phase = "EXIT"
        targetPosition = exitPosition
    end

    local targetRoomIndex =
        door.TargetRoomIndex

    return {
        slot = slot,
        phase = phase,

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
            ),

        door_open =
            door:IsOpen()
    }
end


local function SelectNavigationDoor(room, player)
    local bestUnvisited = nil
    local bestVisited = nil
    local currentCandidate = nil

    for slot = 0, DoorSlot.NUM_DOOR_SLOTS - 1 do
        local data =
            GetDoorNavigationData(
                room,
                player,
                slot
            )

        if data ~= nil then
            if slot == navigationDoorSlot then
                currentCandidate = data
            end

            if data.target_room_visited then
                if bestVisited == nil
                    or data.approach_distance
                    < bestVisited.approach_distance then

                    bestVisited = data
                end
            else
                if bestUnvisited == nil
                    or data.approach_distance
                    < bestUnvisited.approach_distance then

                    bestUnvisited = data
                end
            end
        end
    end

    local bestCandidate =
        bestUnvisited or bestVisited

    if bestCandidate == nil then
        navigationDoorSlot = nil
        return nil
    end

    if currentCandidate == nil then
        navigationDoorSlot =
            bestCandidate.slot

        return bestCandidate
    end

    if currentCandidate.target_room_visited
        and bestUnvisited ~= nil then

        navigationDoorSlot =
            bestUnvisited.slot

        return bestUnvisited
    end

    local alternative = nil

    if currentCandidate.target_room_visited then
        alternative = bestVisited
    else
        alternative = bestUnvisited
    end

    if alternative == nil
        or alternative.slot
        == currentCandidate.slot then

        return currentCandidate
    end

    if alternative.approach_distance
        < currentCandidate.approach_distance
        * DOOR_SWITCH_RATIO then

        navigationDoorSlot =
            alternative.slot

        return alternative
    end

    return currentCandidate
end


local function GetNavigationTarget(player)
    local room = game:GetRoom()

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
            selected.target_room_visited,

        door_open =
            selected.door_open
    }
end


-- =========================================================
-- GENERIC COMBAT ENCOUNTER
-- =========================================================

local function SpawnTrainingEncounter()
    local room = game:GetRoom()

    if room == nil or trainingComplete then
        return false
    end

    RemoveUncontrolledCombatEntities()

    room:SetClear(false)
    CloseTrainingRoomDoors()

    local player =
        Isaac.GetPlayer(0)

    if currentEncounterEnemyTargetCount <= 0 then
        currentEncounterEnemyTargetCount =
            GetDesiredEnemyCount()
    end

    local positions =
        FindTrainingSpawnPositions(
            room,
            player,
            currentEncounterEnemyTargetCount
        )

    if positions == nil then
        Isaac.DebugString(
            "RL_TRAINING NO SAFE MULTI SPAWN"
            .. " requested="
            .. tostring(
                currentEncounterEnemyTargetCount
            )
        )

        return false
    end

    local spawned = {}

    for _, position in ipairs(positions) do
        local enemy =
            Isaac.Spawn(
                TRAINING_ENEMY_TYPE,
                TRAINING_ENEMY_VARIANT,
                TRAINING_ENEMY_SUBTYPE,
                position,
                Vector(0, 0),
                nil
            )

        if enemy == nil then
            for _, existing in ipairs(spawned) do
                existing:Remove()
            end

            return false
        end

        table.insert(
            spawned,
            enemy
        )
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
        .. " stage="
        .. tostring(curriculumStage)
        .. " room="
        .. tostring(trainingRoomIndex)
        .. " enemies="
        .. tostring(#spawned)
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

    trainingSpawnCountdown = -1
    trainingRoomIndex = nil

    roomsCompleted = 0
    totalRoomTransitions = 0

    waitingForRoomExit = false

    currentEncounterEnemyTargetCount = 0

    visitedRoomIndices = {}
    completedRoomIndices = {}

    navigationDoorSlot = nil

    ResetTearStatistics()

    gameOver = false

    RL_ENABLED = false
    RL_MOVE = "NONE"
    RL_SHOOT = "NONE"

    pendingActionId = nil
    pendingActionFrame = nil

    Isaac.DebugString(
        "RL_RESET stage="
        .. tostring(curriculumStage)
        .. " "
        .. GetCurriculumName()
    )

    Isaac.ExecuteCommand("restart")
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

    if not success
        or type(command) ~= "table" then

        return
    end

    if command.type == "reset" then
        if command.curriculum_stage ~= nil then
            curriculumStage =
                NormalizeCurriculumStage(
                    command.curriculum_stage
                )
        end

        RequestReset()
        return
    end

    if command.type ~= "action" then
        return
    end

    if command.action_id ~= nil then
        local newActionId =
            tonumber(
                command.action_id
            )

        if newActionId ~= nil then
            pendingActionId =
                newActionId

            pendingActionFrame =
                game:GetFrameCount()
        end
    end

    if command.enabled ~= nil then
        RL_ENABLED =
            command.enabled == true
    end

    if command.move ~= nil then
        local value =
            string.upper(
                tostring(command.move)
            )

        if IsValidDirection(value) then
            RL_MOVE = value
        end
    end

    if command.shoot ~= nil then
        local value =
            string.upper(
                tostring(command.shoot)
            )

        if IsValidDirection(value) then
            RL_SHOOT = value
        end
    end
end


local function ReadPythonCommands()
    if tcp == nil then
        ConnectToPython()
        return
    end

    local messagesRead = 0

    while messagesRead < 20 do
        local line, err, partial =
            tcp:receive("*l")

        if line ~= nil then
            local completeLine =
                rxBuffer .. line

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
                    rxBuffer .. partial
            end

            if #rxBuffer > 65536 then
                rxBuffer = ""
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
                    DisconnectPython(err)
                end

                break
            end
        end
    end
end


local function FlushOutgoing()
    if tcp == nil or txBuffer == "" then
        return
    end

    local sent, err, last =
        tcp:send(txBuffer)

    if sent ~= nil then
        txBuffer = ""
        return
    end

    if err == "timeout" then
        if last ~= nil and last > 0 then
            txBuffer =
                string.sub(
                    txBuffer,
                    last + 1
                )
        end

        return
    end

    DisconnectPython(
        err or "send failed"
    )
end


-- =========================================================
-- STATE
-- =========================================================

local function GetGameState()
    local player =
        Isaac.GetPlayer(0)

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

    for _, entity in ipairs(Isaac.GetRoomEntities()) do
        if entity:IsActiveEnemy(false)
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
                    seed =
                        entity.InitSeed,

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

    local currentRoomIndex =
        GetCurrentRoomIndex()

    local navigationTarget = nil
    local navigationCandidateCount = 0

    if StageUsesNavigation()
        and encounterActive
        and not trainingComplete
        and room:IsClear()
        and trainingRoomIndex ~= nil
        and currentRoomIndex
        == trainingRoomIndex then

        navigationCandidateCount =
            CountNavigationCandidates(
                room
            )

        navigationTarget =
            GetNavigationTarget(
                player
            )
    end

    local targetRoomExits =
        GetTargetRoomExits()

    local roomsRemaining =
        math.max(
            0,
            targetRoomExits
            - roomsCompleted
        )

    local center =
        room:GetCenterPos()

    return {
        type = "state",

        curriculum_stage =
            curriculumStage,

        curriculum_name =
            GetCurriculumName(),

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

        encounter_enemy_target_count =
            currentEncounterEnemyTargetCount,

        enemy_danger_enabled =
            StageUsesEnemyDanger(),

        rooms_exited =
            roomsCompleted,

        rooms_completed =
            roomsCompleted,

        target_room_exits =
            targetRoomExits,

        target_training_rooms =
            targetRoomExits,

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

        navigation_candidate_count =
            navigationCandidateCount,

        room = {
            center_x =
                center.X,

            center_y =
                center.Y
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

        tears_fired =
            tearsFired,

        tears_hit =
            tearsHit,

        tears_missed =
            tearsMissed,

        navigation_target =
            navigationTarget,

        local_sensors =
            GetLocalSensors(
                player
            ),

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


local function QueueState(state)
    if tcp == nil
        or txBuffer ~= "" then

        return
    end

    local success, encoded =
        pcall(
            json.encode,
            state
        )

    if not success then
        return
    end

    txBuffer =
        encoded .. "\n"
end


-- =========================================================
-- ROOM TRANSITIONS
-- =========================================================

function mod:OnNewRoom()
    if not StageUsesNavigation() then
        return
    end

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
        totalRoomTransitions + 1

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
            roomsCompleted + 1
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

    currentEncounterEnemyTargetCount = 0

    navigationDoorSlot = nil

    RL_MOVE = "NONE"
    RL_SHOOT = "NONE"

    if roomsCompleted
        >= GetTargetRoomExits() then

        trainingComplete = true
        trainingSpawnCountdown = -1

        trainingRoomIndex =
            newRoomIndex

        RemoveUncontrolledCombatEntities()

        local room =
            game:GetRoom()

        if room ~= nil then
            room:SetClear(true)
        end

        ReleaseTraversableTrainingRoomDoors()

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

    PrepareTrainingRoom()

    encounterActive = false

    trainingSpawnCountdown =
        TRAINING_SPAWN_DELAY
end


mod:AddCallback(
    ModCallbacks.MC_POST_NEW_ROOM,
    mod.OnNewRoom
)


-- =========================================================
-- GAME START / END
-- =========================================================

function mod:OnGameStarted(isContinued)
    episodeId =
        episodeId + 1

    gameOver = false
    resetInProgress = false

    trainingReady = false
    encounterActive = false
    trainingComplete = false

    roomsCompleted = 0
    totalRoomTransitions = 0

    waitingForRoomExit = false

    currentEncounterEnemyTargetCount = 0

    visitedRoomIndices = {}
    completedRoomIndices = {}

    navigationDoorSlot = nil

    ResetTearStatistics()

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
        "RL_EPISODE START"
        .. " id="
        .. tostring(episodeId)
        .. " stage="
        .. tostring(curriculumStage)
        .. " "
        .. GetCurriculumName()
    )
end


mod:AddCallback(
    ModCallbacks.MC_POST_GAME_STARTED,
    mod.OnGameStarted
)


function mod:OnGameEnd(isGameOver)
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

    -- Spawn encounter.
    if trainingSpawnCountdown > 0 then
        trainingSpawnCountdown =
            trainingSpawnCountdown - 1

        CloseTrainingRoomDoors()

        if trainingSpawnCountdown == 0 then
            local spawned =
                SpawnTrainingEncounter()

            if not spawned then
                trainingSpawnCountdown =
                    TRAINING_SPAWN_DELAY
            end
        end
    end

    -- Terminal state still needs to be transmitted.
    if trainingComplete then
        if currentFrame
            % STATE_SEND_INTERVAL
            == 0 then

            local state =
                GetGameState()

            if state ~= nil then
                QueueState(state)
                FlushOutgoing()
            end
        end

        return
    end

    if not encounterActive then
        return
    end

    local currentRoomIndex =
        GetCurrentRoomIndex()

    if trainingRoomIndex ~= nil
        and currentRoomIndex
        == trainingRoomIndex then

        local room =
            game:GetRoom()

        if room:IsClear() then
            if StageUsesNavigation() then
                waitingForRoomExit = true

                ReleaseTraversableTrainingRoomDoors()
            else
                -- Stage 1/2/3:
                -- clearing the encounter = success.
                trainingComplete = true

                encounterActive = false
                waitingForRoomExit = false

                navigationDoorSlot = nil

                RL_MOVE = "NONE"
                RL_SHOOT = "NONE"
            end
        else
            waitingForRoomExit = false
            navigationDoorSlot = nil

            CloseTrainingRoomDoors()
        end
    end

    if currentFrame
        % STATE_SEND_INTERVAL
        == 0 then

        local state =
            GetGameState()

        if state ~= nil then
            QueueState(state)
            FlushOutgoing()
        end
    end

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
-- INPUT
-- =========================================================

local function IsMovementButton(button)
    return
        button == ButtonAction.ACTION_LEFT
        or button == ButtonAction.ACTION_RIGHT
        or button == ButtonAction.ACTION_UP
        or button == ButtonAction.ACTION_DOWN
end


local function IsShootingButton(button)
    return
        button == ButtonAction.ACTION_SHOOTLEFT
        or button == ButtonAction.ACTION_SHOOTRIGHT
        or button == ButtonAction.ACTION_SHOOTUP
        or button == ButtonAction.ACTION_SHOOTDOWN
end


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
        Isaac.GetPlayer(0)

    if mainPlayer ~= nil
        and player.InitSeed
        ~= mainPlayer.InitSeed then

        return nil
    end

    if IsMovementButton(buttonAction) then
        local wanted =
            MOVE_ACTIONS[RL_MOVE]

        local pressed =
            wanted ~= nil
            and buttonAction == wanted

        if inputHook
            == InputHook.IS_ACTION_PRESSED then

            return pressed

        elseif inputHook
            == InputHook.IS_ACTION_TRIGGERED then

            return pressed

        elseif inputHook
            == InputHook.GET_ACTION_VALUE then

            return pressed and 1.0 or 0.0
        end
    end

    if IsShootingButton(buttonAction) then
        local wanted =
            SHOOT_ACTIONS[RL_SHOOT]

        local pressed =
            wanted ~= nil
            and buttonAction == wanted

        if inputHook
            == InputHook.IS_ACTION_PRESSED then

            return pressed

        elseif inputHook
            == InputHook.IS_ACTION_TRIGGERED then

            return pressed

        elseif inputHook
            == InputHook.GET_ACTION_VALUE then

            return pressed and 1.0 or 0.0
        end
    end

    return nil
end


mod:AddCallback(
    ModCallbacks.MC_INPUT_ACTION,
    mod.OnInput
)