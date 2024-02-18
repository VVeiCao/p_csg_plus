export CARLA_ROOT=${1:-/home/ubuntu/project/p_csg_plus/carla}
export WORK_DIR=${2:-/home/ubuntu/project/p_csg_plus}

export CARLA_SERVER=${CARLA_ROOT}/CarlaUE4.sh
export PYTHONPATH=$PYTHONPATH:${CARLA_ROOT}/PythonAPI
export PYTHONPATH=$PYTHONPATH:${CARLA_ROOT}/PythonAPI/carla
export PYTHONPATH=$PYTHONPATH:$CARLA_ROOT/PythonAPI/carla/dist/carla-0.9.10-py3.7-linux-x86_64.egg
export PYTHONPATH=$PYTHONPATH:${WORK_DIR}/leaderboard
export PYTHONPATH=$PYTHONPATH:${WORK_DIR}/leaderboard/leaderboard/agents
export PYTHONPATH=$PYTHONPATH:${WORK_DIR}/scenario_runner

export TEAM_AGENT=${WORK_DIR}/leaderboard/agents/p_csg.py
export AGENT_CONFIG=${WORK_DIR}/model_ckpts/
export SCENARIOS=${WORK_DIR}/leaderboard/data/scenarios/town05_all_scenarios.json
export ROUTES=${WORK_DIR}/leaderboard/data/evaluation_routes/routes_town05_long.xml
export CHECKPOINT=${WORK_DIR}/results/longest6.json
export SAVE_PATH=${WORK_DIR}/frames/

CUDA_VISIBLE_DEVICES=0 python3 leaderboard/leaderboard/leaderboard_evaluator.py \
--scenarios=${SCENARIOS} \
--routes=${ROUTES} \
--repetitions=1 \
--track=SENSORS \
--checkpoint=${CHECKPOINT} \
--agent=${TEAM_AGENT} \
--agent-config=${AGENT_CONFIG} \
--debug=0 \
--record=record \
--resume=True \
--port=20000 \
--trafficManagerPort=20500 \
--timeout=600.0
