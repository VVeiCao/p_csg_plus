export CARLA_ROOT=${1:-/home/slt/gitlab_file/transfuser/carla}
export WORK_DIR=${2:-/home/zhk/project/vae}

export CARLA_SERVER=${CARLA_ROOT}/CarlaUE4.sh
export PYTHONPATH=$PYTHONPATH:${CARLA_ROOT}/PythonAPI
export PYTHONPATH=$PYTHONPATH:${CARLA_ROOT}/PythonAPI/carla
export PYTHONPATH=$PYTHONPATH:$CARLA_ROOT/PythonAPI/carla/dist/carla-0.9.10-py3.7-linux-x86_64.egg
export PYTHONPATH=$PYTHONPATH:${WORK_DIR}/leaderboard
export PYTHONPATH=$PYTHONPATH:${WORK_DIR}/leaderboard/leaderboard/agents
export PYTHONPATH=$PYTHONPATH:${WORK_DIR}/leaderboard/scenario_runner

export TEAM_AGENT=${WORK_DIR}/leaderboard/agents/auto_pilot.py
export AGENT_CONFIG=${WORK_DIR}/model_ckpt/p_csg
export SCENARIOS=${WORK_DIR}/leaderboard/data/scenarios/town05_all_scenarios.json
export ROUTES=${WORK_DIR}/leaderboard/data/evaluation_routes/routes_town05_long.xml
export CHECKPOINT=${WORK_DIR}/results/auto_pilot_t1.json

CUDA_VISIBLE_DEVICES=1 python3 leaderboard/leaderboard/leaderboard_evaluator.py \
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
--port=2000 \
--trafficManagerPort=8000 \
--timeout=180.0