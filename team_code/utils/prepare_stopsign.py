from __future__ import print_function
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import os
import json
import sys
import math
import shapely

ROOT="/home/slt/gitlab_file/transfuser"
CARLA_ROOT=os.path.join(ROOT,"carla")
CARLA_SERVER=os.path.join(CARLA_ROOT,"CarlaUE4.sh")
sys.path.append(os.path.join(CARLA_ROOT,"PythonAPI"))
sys.path.append(os.path.join(CARLA_ROOT,"PythonAPI/carla"))
sys.path.append(os.path.join(CARLA_ROOT,"PythonAPI/carla/dist/carla-0.9.10-py3.7-linux-x86_64.egg"))
sys.path.append(os.path.join(ROOT,"leaderboard"))
sys.path.append(os.path.join(ROOT,"leaderboard/team_code"))
sys.path.append(os.path.join(ROOT,"scenario_runner"))

import traceback
import argparse
from argparse import RawTextHelpFormatter
from datetime import datetime
from distutils.version import LooseVersion
import importlib
# import os
# import sys
import gc
import pkg_resources
import sys
import carla
import copy
import signal

from srunner.scenariomanager.carla_data_provider import *
from srunner.scenariomanager.timer import GameTime
from srunner.scenariomanager.watchdog import Watchdog

from leaderboard.scenarios.scenario_manager import ScenarioManager
from leaderboard.scenarios.route_scenario import RouteScenario
from leaderboard.envs.sensor_interface import SensorInterface, SensorConfigurationInvalid
from leaderboard.autoagents.agent_wrapper import  AgentWrapper, AgentError
from leaderboard.utils.statistics_manager import StatisticsManager
from leaderboard.utils.route_indexer import RouteIndexer

# change to what you want
############################################################
DATA_PATH = 'carla_data/clear-weather/data'
PORT = 20002
TRAFFIC_PORT = 20502
NUM_LIST = [3, 4]# [1, 2, 6, 7, 10] 
TYPE_LIST = ['long']
############################################################


def rotate_point(point, angle):
    """
    rotate a given point by a given angle
    """
    x_ = math.cos(math.radians(angle)) * point.x - math.sin(math.radians(angle)) * point.y
    y_ = math.sin(math.radians(angle)) * point.x + math.cos(math.radians(angle)) * point.y
    return carla.Vector3D(x_, y_, point.z)

def point_inside_boundingbox(point, bb_center, bb_extent):
    """
    X
    :param point:
    :param bb_center:
    :param bb_extent:
    :return:
    """

    # pylint: disable=invalid-name
    A = np.array([bb_center[0] - bb_extent[0], bb_center[1] - bb_extent[1]])
    B = np.array([bb_center[0] + bb_extent[0], bb_center[1] - bb_extent[1]])
    D = np.array([bb_center[0] - bb_extent[0], bb_center[1] + bb_extent[1]])
    M = np.array([point[0], point[1]])

    AB = B - A
    AD = D - A
    AM = M - A
    am_ab = AM[0] * AB[0] + AM[1] * AB[1]
    ab_ab = AB[0] * AB[0] + AB[1] * AB[1]
    am_ad = AM[0] * AD[0] + AM[1] * AD[1]
    ad_ad = AD[0] * AD[0] + AD[1] * AD[1]

    return am_ab > 0 and am_ab < ab_ab and am_ad > 0 and am_ad < ad_ad

def is_actor_affected_by_stop(wp, stop, extent, multi_step=20):
    """
    Check if the given actor is affected by the stop
    """

    # first we run a fast coarse test
    current_location = wp.transform.location
    stop_location = carla.Location(stop[0], stop[1], 0)
    if stop_location.distance(current_location) > 50:
        return 0

    # stop_t = stop.get_transform()
    # transformed_tv = stop_t.transform(stop.trigger_volume.location)

    # slower and accurate test based on waypoint's horizon and geometric test
    list_locations = [np.array([current_location.x, current_location.y])]
    waypoint = _map.get_waypoint(current_location)
    for _ in range(multi_step):
        if waypoint:
            next_wps = waypoint.next(1.0)
            if not next_wps:
                break
            waypoint = next_wps[0]
            if not waypoint:
                break
            loc = waypoint.transform.location
            list_locations.append(np.array([loc.x, loc.y]))

    for idx, actor_location in enumerate(list_locations):
        if point_inside_boundingbox(actor_location, stop, extent):
            return idx + 1

    return 0

def scan_for_stop_sign(wp, list_stop_signs):
    target_stop_sign = None
    distance = 0

    for stop_sign in list_stop_signs:
        distance = is_actor_affected_by_stop(wp, [stop_sign["c_x"], stop_sign["c_y"]], [stop_sign["e_x"], stop_sign["e_y"]])
        if  distance > 0:
            # this stop sign is affecting the vehicle
            target_stop_sign = stop_sign
            break
            
    return target_stop_sign, distance


ego_vehicles = []

# Tunable parameters
client_timeout = 30.0  # in seconds
wait_for_world = 20.0  # in seconds
frame_rate = 20.0      # in Hz

record = ""
timeout = 600.0
# routes = os.path.join(ROOT, "leaderboard/data/training_routes/routes_town01_tiny.xml")
scenarios = os.path.join(ROOT, "leaderboard/data/scenarios/no_scenarios.json")
track = "SENSORS"
resume = 1
repetitions = 1

statistics_manager = StatisticsManager()

sensors = None
sensor_icons = []
_vehicle_lights = carla.VehicleLightState.Position | carla.VehicleLightState.LowBeam


info = {}
simple_info = {}
# num_list = [1, 2]
num_list = NUM_LIST
for num in num_list:
    if num == 5:
        routes = os.path.join(ROOT, "leaderboard/data/validation_routes/routes_town05_short.xml")
    else:
        routes = os.path.join(ROOT, "leaderboard/data/training_routes/routes_town" + str(num).zfill(2) +"_tiny.xml")

    client = carla.Client("localhost", PORT)
    client.set_timeout(client_timeout)
    traffic_manager = client.get_trafficmanager(TRAFFIC_PORT)

    route_indexer = RouteIndexer(routes, scenarios, repetitions)
    route_indexer.peek()
    # print(route_indexer)
    config = route_indexer.next()

    statistics_manager.set_route(config.name, config.index)

    world = client.load_world(config.town)
    settings = world.get_settings()
    settings.fixed_delta_seconds = 1.0 / frame_rate
    settings.synchronous_mode = True
    world.apply_settings(settings)

    world.reset_all_traffic_lights()

    CarlaDataProvider.set_client(client)
    CarlaDataProvider.set_world(world)
    CarlaDataProvider.set_traffic_manager_port(TRAFFIC_PORT)

#     traffic_manager.set_synchronous_mode(True)
#     traffic_manager.set_random_device_seed(0)

    _map = CarlaDataProvider.get_map()
    actor_list = world.get_actors()
    list_stops = []
    for _actor in actor_list:
        if 'traffic.stop' in _actor.type_id:
            list_stops.append(_actor)
    
#     info[num] = list_stops

    print("++++++++++++++++++++++++++++++++++++++++++")
    print("Town" + str(num))
    for idx_s in range(len(list_stops)):
        print("No." + str(idx_s) + "-------------------------")
        print(str(list_stops[idx_s].get_transform()))
        base_transform = list_stops[idx_s].get_transform()
        area_loc = base_transform.transform(list_stops[idx_s].trigger_volume.location)
        print(str(list_stops[idx_s].trigger_volume.location))
        print(str(area_loc))                           
        print(str(list_stops[idx_s].trigger_volume.extent))
        print("-----------------------------------------")
        
        
    stops_info = []
    for idx, stop in enumerate(list_stops):
        stop_info = {}
        trans = stop.get_transform()
        area_loc = trans.transform(stop.trigger_volume.location)
        tl = stop.trigger_volume.location
        te = stop.trigger_volume.extent
        A = carla.Location(tl.x - te.x, tl.y - te.y, 0)
        B = carla.Location(tl.x - te.x, tl.y + te.y, 0)
        C = carla.Location(tl.x + te.x, tl.y + te.y, 0)
        D = carla.Location(tl.x + te.x, tl.y - te.y, 0)
        trans = stop.get_transform()
        A_t = trans.transform(A)
        trans = stop.get_transform()
        B_t = trans.transform(B)
        trans = stop.get_transform()
        C_t = trans.transform(C)
        trans = stop.get_transform()
        D_t = trans.transform(D)
        stop_info["x"] = trans.location.x
        stop_info["y"] = trans.location.y
        stop_info["yaw"] = trans.rotation.yaw
        stop_info["c_x"] = area_loc.x
        stop_info["c_y"] = area_loc.y
        stop_info["e_x"] = stop.trigger_volume.extent.x
        stop_info["e_y"] = stop.trigger_volume.extent.y
        stop_info["lft"] = (B_t.x, B_t.y)
        stop_info["rgt"] = (C_t.x, C_t.y)
        print(str(trans))

        stops_info.append(stop_info)
    info[num] = stops_info



type_list = TYPE_LIST 
count  = 0
for num in num_list:
    if num == 5:
        routes = os.path.join(ROOT, "leaderboard/data/validation_routes/routes_town05_short.xml")
    else:
        routes = os.path.join(ROOT, "leaderboard/data/training_routes/routes_town" + str(num).zfill(2) +"_short.xml")

    client = carla.Client("localhost", PORT)
    client.set_timeout(client_timeout)
    traffic_manager = client.get_trafficmanager(TRAFFIC_PORT)

    route_indexer = RouteIndexer(routes, scenarios, repetitions)
    route_indexer.peek()
    # print(route_indexer)
    config = route_indexer.next()

    statistics_manager.set_route(config.name, config.index)

    world = client.load_world(config.town)
    settings = world.get_settings()
    settings.fixed_delta_seconds = 1.0 / frame_rate
    settings.synchronous_mode = True
    world.apply_settings(settings)

    # world.reset_all_traffic_lights()

    CarlaDataProvider.set_client(client)
    CarlaDataProvider.set_world(world)
    CarlaDataProvider.set_traffic_manager_port(TRAFFIC_PORT)

#     traffic_manager.set_synchronous_mode(True)
#     traffic_manager.set_random_device_seed(0)

    _map = CarlaDataProvider.get_map()
    
    for town_type in type_list:
        if num in [7, 10] and town_type == "long":
            continue
        print("Process Town" + str(num).zfill(2) + "_" + town_type)
        root = os.path.join(DATA_PATH, "Town" + str(num).zfill(2) + "_" + str(town_type))
        folders = [ os.path.join(root, names) for names in os.listdir(root) if names[:6] == "routes"]
        for folder in folders:
            print("Process " + folder)
            # if "routes_town07_junctions_10_21_07_10_09" not in folder:
            #     continue
            stop_folder = os.path.join(folder, "stop")
            if not os.path.isdir(stop_folder):
                os.makedirs(stop_folder)
#             else:
#                 for file in os.listdir(light_folder):
#                     # print(file)
#                     if file[-10:]==".json.json":
#                         os.remove(os.path.join(light_folder, file))
#                         print(os.path.join(light_folder, file))

            mea_folder = os.path.join(folder, "measurements")
            seg_folder = os.path.join(folder, "seg_front")
            bev_folder = os.path.join(folder, "topdown")
            files = sorted([ file[:-5] for file in sorted(os.listdir(mea_folder)) if file[-4:] == "json" ])

            for idx, file in enumerate(files):
                
                output = {}
                if num in [1,2]:
                    output["stop"] = 0
                    outfile = os.path.join(stop_folder,file)+".json"
                    with open(outfile, 'w') as f:
                        json.dump(output, f, sort_keys=True, indent=4)
                    continue
                # print("id :" + str(idx).zfill(4))

                # seg_front = np.unique(np.array(Image.open(os.path.join(seg_folder,file)+".png"))) 
                seg= np.unique(np.array(Image.open(os.path.join(seg_folder,file)+".png"))) 
                # if 23 in seg_front or 23 in bev:
                if 26 in seg or 27 in seg:
                    stop = 1                
                else:
                    output["stop"] = 0
                    outfile = os.path.join(stop_folder,file)+".json"
                    with open(outfile, 'w') as f:
                        json.dump(output, f, sort_keys=True, indent=4)
                    continue
                
                with open(os.path.join(mea_folder,file)+".json") as f:
                    mea = json.load(f)
                location = carla.Location(mea["gps_y"], -mea["gps_x"], 0)
                rot_deg = mea["theta"] / np.pi * 180 - 90
                rot_add = carla.Transform(carla.Location(0, 0, 0), carla.Rotation(yaw=rot_deg))
                # loc_add = rot_add.transform(carla.Location(-2.45, 0, 0))
                try:
                    wp = _map.get_waypoint(location)
                except:
                    output["stop"] = 0
                    outfile = os.path.join(stop_folder,file)+".json"
                    with open(outfile, 'w') as f:
                        json.dump(output, f, sort_keys=True, indent=4)
                    continue
                    
                fw = wp.transform.get_forward_vector()

                stop_info, distance = scan_for_stop_sign(wp, info[num])

                if stop_info is not None:
                    count += 1
#                         if flag:
#                             print(os.path.join(mea_folder,file)+".json"())
#                             print(flag)
#                             print("+++++++++++++++++")
#                             print(light)
#                        flag = light
                    output["stop"] = 1
                    output["distance"] = distance
                    output["wp_direction"] = (fw.x, fw.y)
                    output["x"] = stop_info["x"]
                    output["y"] = stop_info["y"]
                    output["yaw"] = stop_info["yaw"]
                    output["c_x"] = stop_info["c_x"]
                    output["c_y"] = stop_info["c_y"]
                    output["e_x"] = stop_info["e_x"]
                    output["e_y"] = stop_info["e_y"]
                    output["lft"] = stop_info["lft"]
                    output["rgt"] = stop_info["rgt"]


                    outfile = os.path.join(stop_folder,file)+".json"
                    print(outfile)
                    with open(outfile, 'w') as f:
                        json.dump(output, f, sort_keys=True, indent=4)
#                         print(mea)
#                         print()
#                         print(outfile)
#                         print(output)
#                         print("===================================")
                else:
                    output["stop"] = 0
                    outfile = os.path.join(stop_folder,file)+".json"
                    with open(outfile, 'w') as f:
                        json.dump(output, f, sort_keys=True, indent=4)
            
            