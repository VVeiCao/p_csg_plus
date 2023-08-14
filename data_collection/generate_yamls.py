import yaml
import os

d = {}
d[
    "waypoint_disturb"
] = 0.2  # in meters, a way of data augumentaion that randomly distrub the planned waypoints
d["waypoint_disturb_seed"] = 2020
d["destory_hazard_actors"] = True
d["save_skip_frames"] = 10  # skip 10 frames equals fps = 2
d["rgb_only"] = False

if not os.path.exists("yamls"):
    os.mkdir("yamls")

d["weather"] = -1
with open("yamls/weather-mixed.yaml", "w") as fw:
    yaml.dump(d, fw)

d["weather"] = 0
with open("yamls/clear-weather.yaml", "w") as fw:
    yaml.dump(d, fw)
