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


def rotate_point(point, angle):
    """
    rotate a given point by a given angle
    """
    x_ = math.cos(math.radians(angle)) * point.x - math.sin(math.radians(angle)) * point.y
    y_ = math.sin(math.radians(angle)) * point.x + math.cos(math.radians(angle)) * point.y
    return carla.Vector3D(x_, y_, point.z)

def get_traffic_light_waypoints(traffic_light):
    """
    get area of a given traffic light
    """

    base_transform = traffic_light.get_transform()
    base_rot = base_transform.rotation.yaw
    area_loc = base_transform.transform(traffic_light.trigger_volume.location)

    # Discretize the trigger box into points
    area_ext = traffic_light.trigger_volume.extent
    x_values = np.arange(-0.9 * area_ext.x, 0.9 * area_ext.x, 1.0)  # 0.9 to avoid crossing to adjacent lanes

    area = []
    for x in x_values:
        point = rotate_point(carla.Vector3D(x, 0, area_ext.z), base_rot)
        point_location = area_loc + carla.Location(x=point.x, y=point.y)
        area.append(point_location)

    # Get the waypoints of these points, removing duplicates
    ini_wps = []
    for pt in area:
        wpx = _map.get_waypoint(pt)
        # As x_values are arranged in order, only the last one has to be checked
        if not ini_wps or ini_wps[-1].road_id != wpx.road_id or ini_wps[-1].lane_id != wpx.lane_id:
            ini_wps.append(wpx)

    # Advance them until the intersection
    wps = []
    for wpx in ini_wps:
        while not wpx.is_intersection: # is_junction
            next_wp = wpx.next(0.5)[0]
            if next_wp and not next_wp.is_intersection:
                wpx = next_wp
            else:
                break
        wps.append(wpx)

    return area_loc, wps


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
num_list = [5]  # [1, 2, 3, 4, 6, 7, 10, 5]
# num_list = [1]
for num in num_list:
    if num == 5:
        routes = os.path.join(ROOT, "leaderboard/data/validation_routes/routes_town05_short.xml")
        # routes = os.path.join(ROOT, "leaderboard/data/evaluation_routes/routes_town05_long.xml")

    else:
        routes = os.path.join(ROOT, "leaderboard/data/training_routes/routes_town" + str(num).zfill(2) +"_tiny.xml")

    client = carla.Client("localhost", 20000)
    client.set_timeout(client_timeout)
    traffic_manager = client.get_trafficmanager(20500)

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
    CarlaDataProvider.set_traffic_manager_port(20500)

    traffic_manager.set_synchronous_mode(True)
    traffic_manager.set_random_device_seed(0)

    _map = CarlaDataProvider.get_map()
    actor_list = world.get_actors()

    list_traffic_lights = []
    for _actor in actor_list:
        if 'traffic_light' in _actor.type_id:
            center, waypoints = get_traffic_light_waypoints(_actor)
            list_traffic_lights.append((_actor, center, waypoints))


    lights_info = []
    lights_simple_info = []
    for idx, light in enumerate(list_traffic_lights):
        light_s_info = {}
        print("id: " + str(idx + 1))
        print(str(light[0].get_transform()))
        trans = light[0].get_transform()
        print()
        print(light[0].get_transform().get_forward_vector())
        print()
        print(str(light[0].trigger_volume))
        print()
        print(str(light[1]))
        print()
        light_s_info = {}
        light_s_info["x"] = trans.location.x
        light_s_info["y"] = trans.location.y
        light_s_info["yaw"] = trans.rotation.yaw
        light_s_info["c_x"] = light[1].x
        light_s_info["c_y"] = light[1].y
        for idy, wp in enumerate(light[2]):
            light_info = {}
            light_info["x"] = trans.location.x
            light_info["y"] = trans.location.y
            light_info["yaw"] = trans.rotation.yaw
            light_info["c_x"] = light[1].x
            light_info["c_y"] = light[1].y
            light_info["w_x"] = wp.transform.location.x
            light_info["w_y"] = wp.transform.location.y
            light_info["w_yaw"] = wp.transform.rotation.yaw

            w_type = round(wp.transform.rotation.yaw / 90) % 4
            light_info["orignal_theta"] = wp.transform.rotation.yaw 
            light_info["theta"] = (w_type - 1) * np.pi / 2
            if w_type == 0:
                light_info["direction_1"] = 0 # "x"
                light_info["direction_2"] = 1
            elif w_type == 1:
                light_info["direction_1"] = 1 # "y"
                light_info["direction_2"] = 1
            elif w_type == 2:
                light_info["direction_1"] = 0 # "x"
                light_info["direction_2"] = -1
            elif w_type == 3:
                light_info["direction_1"] = 1 # "y"
                light_info["direction_2"] = -1
            else:
                raise "ERROR direction"


            light_info["forward"] = [wp.transform.get_forward_vector().x,
                        wp.transform.get_forward_vector().y,
                        wp.transform.get_forward_vector().z]
            light_info["road_id"] = wp.road_id
            light_info["lane_id"] = wp.lane_id

            print("waypoint id :" + str(idy + 1))
            print(str(wp))
            print()
            print(wp.transform.get_forward_vector())
            print()
            print("road: " + str(wp.road_id))
            print("lane: " + str(wp.lane_id))



            yaw_wp = wp.transform.rotation.yaw
            lane_width = wp.lane_width
            location_wp = wp.transform.location

            lft_lane_wp = rotate_point(carla.Vector3D(0.4 * lane_width, 0.0, location_wp.z), yaw_wp + 90)
            lft_lane_wp = location_wp + carla.Location(lft_lane_wp)
            rgt_lane_wp = rotate_point(carla.Vector3D(0.4 * lane_width, 0.0, location_wp.z), yaw_wp - 90)
            rgt_lane_wp = location_wp + carla.Location(rgt_lane_wp)
            print("left: " + str(lft_lane_wp))
            print("right: " + str(rgt_lane_wp))
            light_info["lft"] = (lft_lane_wp.x, lft_lane_wp.y)
            light_info["rgt"] = (rgt_lane_wp.x, rgt_lane_wp.y)
            print("===================================")
            if light_info["direction_1"] == 0: # "x"
                light_info["value"] = (lft_lane_wp.x + rgt_lane_wp.x) / 2
            elif light_info["direction_1"] == 1: # "y"
                light_info["value"] = (lft_lane_wp.y + rgt_lane_wp.y) / 2
            else:
                raise "ERROR lane"

            lights_info.append(light_info)
            lights_simple_info.append(light_s_info)
        info[num] = lights_info
        simple_info[num] = lights_simple_info

type_list = ["long"]
# num_list = [1, 2, 3, 4, 5, 6, 7, 10]
# num_list = [6, 7, 10]
count  = 0
for num in num_list:
    if num == 5:
        routes = os.path.join(ROOT, "leaderboard/data/validation_routes/routes_town05_short.xml")
    else:
        routes = os.path.join(ROOT, "leaderboard/data/training_routes/routes_town" + str(num).zfill(2) +"_short.xml")

    client = carla.Client("localhost", 20000)
    client.set_timeout(client_timeout)
    traffic_manager = client.get_trafficmanager(20500)

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
    CarlaDataProvider.set_traffic_manager_port(20500)

    traffic_manager.set_synchronous_mode(True)
    traffic_manager.set_random_device_seed(0)

    _map = CarlaDataProvider.get_map()
    
    for town_type in type_list:
        print("Process Town" + str(num).zfill(2) + "_" + town_type)
        root = os.path.join("/home/zhk/project/vae/carla_data/weather-attack", "data")
        folders = [ os.path.join(root, names) for names in os.listdir(root) if names[:6] == "routes"]
        for folder in folders:
            # if "routes_town05_long_intersections_subsample_10_20_18_05_08" not in folder:
            #     continue
            print("Process " + folder)
            light_folder = os.path.join(folder, "light")
            if not os.path.isdir(light_folder):
                os.makedirs(light_folder)
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
            old = None
            old_light = None
            for idx, file in enumerate(files):
                # print("id :" + str(idx).zfill(4))
                output = {}
                light_v = 0
                # seg_front = np.unique(np.array(Image.open(os.path.join(seg_folder,file)+".png"))) % 26 # ignore stop sign
                bev = np.unique(np.array(Image.open(os.path.join(bev_folder,file)+".png"))) 
                # if 23 in seg_front or 23 in bev:
                if 23 in bev:
                    light_v = 23
                # elif 24 in seg_front or 24 in bev:
                elif 24 in bev:
                    light_v = 24
                # elif 25 in seg_front or 25 in bev:
                elif 25 in bev:
                    light_v = 25                
                elif old is None:
                    output["light"] = 0
                    output["light_crossing"] = 0
                    old = None
                    old_light = None
                    outfile = os.path.join(light_folder,file)+".json"
                    with open(outfile, 'w') as f:
                        json.dump(output, f, sort_keys=True, indent=4)
                    continue
                
                with open(os.path.join(mea_folder,file)+".json") as f:
                    mea = json.load(f)
                location = carla.Location(mea["gps_y"], -mea["gps_x"], 0)
                rot_deg = mea["theta"] / np.pi * 180 - 90
                rot_add = carla.Transform(carla.Location(0, 0, 0), carla.Rotation(yaw=rot_deg))
                loc_add = rot_add.transform(carla.Location(-2.45, 0, 0))
                # print(str(location))
                # print(light_v)
                try:
                    wp = _map.get_waypoint(location + loc_add)
                except:
                    output["light"] = -1
                    output["light_crossing"] = -1
                    old = None
                    old_light = None
                    outfile = os.path.join(light_folder,file)+".json"
                    with open(outfile, 'w') as f:
                        json.dump(output, f, sort_keys=True, indent=4)
                    # print(outfile)
                    print("Not found wp")
                    print(outfile)
                    print("===================================")
                    continue
                    
                fw = wp.transform.get_forward_vector()
                flag = None
                for light in info[num]:
                    if wp.road_id == light["road_id"] and wp.lane_id == light["lane_id"] and \
                    fw.x * light['forward'][0] + fw.y * light['forward'][1] + fw.z * light['forward'][2] >= 0:
                        count += 1
#                         if flag:
#                             print(os.path.join(mea_folder,file)+".json"())
#                             print(flag)
#                             print("+++++++++++++++++")
#                             print(light)
#                        flag = light
                        
                        output["light"] = light_v
                        seg_result = False
                        if light_v == 0 and old_light is not None:
                            seg_result = True
                            seg_front = np.unique(np.array(Image.open(os.path.join(seg_folder,file)+".png")))
                            if old_light in seg_front:
                                light_v = old_light
                            if 23 in seg_front:
                                light_v = 23
                            elif 24 in seg_front:
                                light_v = 24
                            elif 25 in seg_front:
                                light_v = 25                
                            else:
                                light_v = 0
                                                         
                        output["light_crossing"] = light_v
                        if light_v != 0:
                            output["value"] = light["value"]
                            output["direction_1"] = light["direction_1"]
                            output["direction_2"] = light["direction_2"]
                            output["lft"] = light["lft"]
                            output["rgt"] = light["rgt"]
                            output["wp_direction"] = (fw.x, fw.y)
                            output["crossing"] = 0
                            old = (fw.x, fw.y)
                            old_light = light_v
                            if output["direction_1"] == 0:
                                output["distance"] = abs(light["x"] - light["w_x"])
                            else:
                                output["distance"] = abs(light["y"] - light["w_y"])
                            outfile = os.path.join(light_folder,file)+".json"
                            with open(outfile, 'w') as f:
                                json.dump(output, f, sort_keys=True, indent=4)
    #                         print(mea)
    #                         print()
                            if seg_result:
                                print("seg_front find")
                                print(output)
                                print(outfile)
                                print("===================================")
                            break
                        else:
                            output["light"] = 0
                            output["light_crossing"] = 0
                            old = None
                            old_light = None
                            outfile = os.path.join(light_folder,file)+".json"
                            if seg_result:
                                print("seg_front not find")
                                print(output)
                                print(outfile)
                                print("===================================")
                            with open(outfile, 'w') as f:
                                json.dump(output, f, sort_keys=True, indent=4)
                            break
                else:
                    if old is not None:
                        loc_sub = carla.Location((-3) * old[0], (-3) * old[1], 0)
                        try:
                            wp = _map.get_waypoint(location + loc_add + loc_sub)
                        except:
                            outfile = os.path.join(light_folder,file)+".json"
                            print("Not found wp")
                            print(outfile)
                            # import pdb; pdb.set_trace()
                            print("===================================")
                            output["light"] = -1
                            output["light_crossing"] = -1
                            old = None
                            old_light = None
                            with open(outfile, 'w') as f:
                                json.dump(output, f, sort_keys=True, indent=4)
                        for light in info[num]:
                            if wp.road_id == light["road_id"] and wp.lane_id == light["lane_id"] and \
                            fw.x * light['forward'][0] + fw.y * light['forward'][1] + fw.z * light['forward'][2] >= 0:
                                flag = light
                                output["light"] = light_v
                                if light_v == 0 and old_light is not None:
                                    seg_front = np.unique(np.array(Image.open(os.path.join(seg_folder,file)+".png")))
                                    if old_light in seg_front:
                                        light_v = old_light
                                    if 23 in seg_front:
                                        light_v = 23
                                    elif 24 in seg_front:
                                        light_v = 24
                                    elif 25 in seg_front:
                                        light_v = 25                
                                    elif old is None:
                                        light_v = 0
                                output["light_crossing"] = light_v
                                if light_v != 0:
                                    output["value"] = light["value"]
                                    output["direction_1"] = light["direction_1"]
                                    output["direction_2"] = light["direction_2"]
                                    output["lft"] = light["lft"]
                                    output["rgt"] = light["rgt"]
                                    output["wp_direction"] = (fw.x, fw.y)
                                    output["crossing"] = 1
                                    old = (fw.x, fw.y)
                                    old_light = light_v
                                    if output["direction_1"] == 0:
                                        output["distance"] = abs(light["x"] - light["w_x"])
                                    else:
                                        output["distance"] = abs(light["y"] - light["w_y"])
                                    outfile = os.path.join(light_folder,file)+".json"
                                    with open(outfile, 'w') as f:
                                        json.dump(output, f, sort_keys=True, indent=4)
            #                         print(mea)
            #                         print()
                                    print("crossing find")
                                    print(output)
                                    print("===================================")
                                    break
                                else:
                                    output["light"] = 0
                                    output["light_crossing"] = 0
                                    old = None
                                    old_light = None
                                    outfile = os.path.join(light_folder,file)+".json"
                                    with open(outfile, 'w') as f:
                                        json.dump(output, f, sort_keys=True, indent=4)
                                    print("crossing not find")
                                    print(output)
                                    print("===================================")
                                    break

                        
                    if flag is None:

                        outfile = os.path.join(light_folder,file)+".json"
                        print(outfile)
                        print("Not found")
                        # import pdb; pdb.set_trace()
                        print("===================================")
                        output["light"] = 0
                        output["light_crossing"] = -1
                        old = None
                        old_light = None
                        with open(outfile, 'w') as f:
                            json.dump(output, f, sort_keys=True, indent=4)

                    



                    
