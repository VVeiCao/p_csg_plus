#!/bin/bash
export DATA_ROOT=carla_data
export YAML_ROOT=data_collection_attack/yamls
export CARLA_ROOT=/home/slt/gitlab_file/transfuser/carla
export CARLA_SERVER=${CARLA_ROOT}/CarlaUE4.sh
export PYTHONPATH=$PYTHONPATH:${CARLA_ROOT}/PythonAPI
export PYTHONPATH=$PYTHONPATH:${CARLA_ROOT}/PythonAPI/carla
export PYTHONPATH=$PYTHONPATH:$CARLA_ROOT/PythonAPI/carla/dist/carla-0.9.10-py3.7-linux-x86_64.egg
export PYTHONPATH=$PYTHONPATH:leaderboard
export PYTHONPATH=$PYTHONPATH:leaderboard/leaderboard/agents
export PYTHONPATH=$PYTHONPATH:scenario_runner

export LEADERBOARD_ROOT=leaderboard

export CHECKPOINT_ENDPOINT=${DATA_ROOT}/weather-attack/results/routes_town05_long.json
export SAVE_PATH=${DATA_ROOT}/weather-attack/data
export TEAM_CONFIG=${YAML_ROOT}/weather.yaml
export TRAFFIC_SEED=20000
export CARLA_SEED=20000
export SCENARIOS=${LEADERBOARD_ROOT}/data/scenarios/town05_all_scenarios.json
export ROUTES=${LEADERBOARD_ROOT}/data/training_routes/routes_town05_long.xml
export TM_PORT=20500
export PORT=20000
export HOST=localhost
export CHALLENGE_TRACK_CODENAME=SENSORS
export DEBUG_CHALLENGE=0
export REPETITIONS=1 # multiple evaluation runs
export TEAM_AGENT=${LEADERBOARD_ROOT}/leaderboard/agents/auto_pilot.py # agent
export RESUME=True

python3 ${LEADERBOARD_ROOT}/leaderboard/leaderboard_evaluator.py \
--scenarios=${SCENARIOS}  \
--routes=${ROUTES} \
--repetitions=${REPETITIONS} \
--track=${CHALLENGE_TRACK_CODENAME} \
--checkpoint=${CHECKPOINT_ENDPOINT} \
--agent=${TEAM_AGENT} \
--agent-config=${TEAM_CONFIG} \
--debug=${DEBUG_CHALLENGE} \
--record=${RECORD_PATH} \
--resume=${RESUME} \
--port=${PORT} \
--host=${HOST} \
--trafficManagerPort=${TM_PORT} \
--carlaProviderSeed=${CARLA_SEED} \
--trafficManagerSeed=${TRAFFIC_SEED}
