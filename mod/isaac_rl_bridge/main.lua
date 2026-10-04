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
--
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


-- Possible enemy spawn offsets relative to room center.
--
-- One position is selected randomly for each episode.
local TRAINING_ENEMY_OFFSETS = {

    -- Right
    Vector(
        160,
        0
    ),

    -- Left
    Vector(
        -160,
        0
    ),

    -- Down
    Vector(
        0,
        120
    ),

    -- Up
    Vector(
        0,
        -120
    )

}


-- =========================================================
-- RL CONTROL
--
-- Global intentionally so we can also change these
-- from the Isaac debug console.
-- =========================================================

RL_ENABLED = false

RL_MOVE = "NONE"
RL_SHOOT = "NONE"

-- Last action that has had at least one game update
-- to affect Isaac. Python treats this ID as applied.
RL_ACTION_ID = 0

-- Action received from Python but not yet acknowledged.
local pendingActionId = nil

-- Game frame on which the pending action was received.
local pendingActionFrame = nil


-- =========================================================
-- EPISODE STATE
-- =========================================================

local episodeId = 0

local gameOver = false

local resetInProgress = false

local trainingReady = false

local trainingSpawnCountdown = -1


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

    LEFT =
        ButtonAction.ACTION_LEFT,

    RIGHT =
        ButtonAction.ACTION_RIGHT,

    UP =
        ButtonAction.ACTION_UP,

    DOWN =
        ButtonAction.ACTION_DOWN

}


local SHOOT_ACTIONS = {

    LEFT =
        ButtonAction.ACTION_SHOOTLEFT,

    RIGHT =
        ButtonAction.ACTION_SHOOTRIGHT,

    UP =
        ButtonAction.ACTION_SHOOTUP,

    DOWN =
        ButtonAction.ACTION_SHOOTDOWN

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

local function IsValidDirection(
    value
)

    return
        value == "NONE"
        or value == "LEFT"
        or value == "RIGHT"
        or value == "UP"
        or value == "DOWN"

end


-- =========================================================
-- DISCONNECT
-- =========================================================

local function DisconnectPython(
    reason
)

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


    -- Small timeout only for connect().
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


        -- Non-blocking after connection.
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


    -- Pick one of the configured spawn positions.
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


    trainingReady =
        true


    Isaac.DebugString(
        "RL_TRAINING READY episode="
        .. tostring(
            episodeId
        )
        .. " spawn_index="
        .. tostring(
            offsetIndex
        )
    )


    return true

end


-- =========================================================
-- CLOSE TRAINING ROOM DOORS
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

local function ProcessCommand(
    line
)

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


        -- =================================================
        -- COMPLETE MESSAGE
        -- =================================================

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


            -- =============================================
            -- PARTIAL TCP MESSAGE
            -- =============================================

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


            -- =============================================
            -- NOTHING AVAILABLE
            -- =============================================

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


    if txBuffer
        == "" then

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
-- BUILD OBSERVATION
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
-- QUEUE OBSERVATION
-- =========================================================

local function QueueState(
    state
)

    if tcp == nil then
        return
    end


    -- Prefer latest observation instead of building
    -- a giant queue of old observations.
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
    -- KEEP AGENT INSIDE TRAINING ROOM
    -- =====================================================

    CloseTrainingRoomDoors()


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

local function IsMovementButton(
    button
)

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


local function IsShootingButton(
    button
)

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