import numpy as np
from PIL import Image, ImageDraw

from srunner.scenariomanager.carla_data_provider import CarlaDataProvider

from base_agent import BaseAgent
from planner import RoutePlanner
import carla


class MapAgent(BaseAgent):
    def sensors(self):
        result = super().sensors()
        result.append({
            'type': 'sensor.camera.semantic_segmentation',
            'x': 0.0, 'y': 0.0, 'z': 100.0,
            'roll': 0.0, 'pitch': -90.0, 'yaw': 0.0,
            'width': 512, 'height': 512, 'fov': 5 * 10.0,
            'id': 'map'
            })

        return result

    def set_global_plan(self, global_plan_gps, global_plan_world_coord):
        super().set_global_plan(global_plan_gps, global_plan_world_coord)

        self._plan_HACK = global_plan_world_coord
        self._plan_gps_HACK = global_plan_gps

    def _init(self):
        super()._init()

        self._vehicle = CarlaDataProvider.get_hero_actor()
        self._world = self._vehicle.get_world()

        self._waypoint_planner = RoutePlanner(4.0, 50)
        self._waypoint_planner.set_route(self._plan_gps_HACK, True)

        self._traffic_lights = list()

    def tick(self, input_data):
        self._actors = self._world.get_actors()
        self._traffic_lights = get_nearby_lights(self._vehicle, self._actors.filter('*traffic_light*'))
        self._stop_signs = get_nearby_lights(self._vehicle, self._actors.filter('*stop*'))

        self._light_near = []
        if self._traffic_lights:
            for light in self._traffic_lights:
                info = {}
                trans = light.get_transform()
                # velocity = walker.get_velocity()
                info["location"] = (trans.location.x, trans.location.y, trans.location.z)
                info["rotation"] = (trans.rotation.pitch, trans.rotation.yaw, trans.rotation.roll)
                info["trigget_location"] = (light.trigger_volume.location.x, light.trigger_volume.location.y,
                                            light.trigger_volume.location.z)
                info["trigget_extent"] = (light.trigger_volume.extent.x, light.trigger_volume.extent.y,
                                          light.trigger_volume.extent.z)
                # info["velocity"] = (velocity.x, velocity.y, velocity.z)
                info["state"] = light_state_str(light.state)
                self._light_near.append(info)

        self._stop_sign_near = []
        if self._stop_signs:
            for stop_sign in self._stop_signs:
                info = {}
                trans = stop_sign.get_transform()
                # velocity = walker.get_velocity()
                info["location"] = (trans.location.x, trans.location.y, trans.location.z)
                info["rotation"] = (trans.rotation.pitch, trans.rotation.yaw, trans.rotation.roll)
                info["trigget_location"] = (stop_sign.trigger_volume.location.x, stop_sign.trigger_volume.location.y,
                                            stop_sign.trigger_volume.location.z)
                info["trigget_extent"] = (stop_sign.trigger_volume.extent.x, stop_sign.trigger_volume.extent.y,
                                          stop_sign.trigger_volume.extent.z)
                # info["velocity"] = (velocity.x, velocity.y, velocity.z)
                # info["state"] = light_state_str(light.state)
                self._stop_sign_near.append(info)

        topdown = input_data['map'][1][:, :, 2]
        topdown = draw_traffic_lights(topdown, self._vehicle, self._traffic_lights)
        topdown = draw_stop_signs(topdown, self._vehicle, self._stop_signs)

        result = super().tick(input_data)
        result['topdown'] = topdown

        return result


def get_nearby_lights(vehicle, lights, pixels_per_meter=5.5, size=512, radius=5):
    result = list()

    transform = vehicle.get_transform()
    pos = transform.location
    theta = np.radians(90 + transform.rotation.yaw)
    R = np.array([
        [np.cos(theta), -np.sin(theta)],
        [np.sin(theta),  np.cos(theta)],
        ])

    for light in lights:
        delta = light.get_transform().location - pos

        target = R.T.dot([delta.x, delta.y])
        target *= pixels_per_meter
        target += size // 2

        if min(target) < 0 or max(target) >= size:
            continue

        trigger = light.trigger_volume
        light.get_transform().transform(trigger.location)
        dist = trigger.location.distance(vehicle.get_location())
        a = np.sqrt(
                trigger.extent.x ** 2 +
                trigger.extent.y ** 2 +
                trigger.extent.z ** 2)
        b = np.sqrt(
                vehicle.bounding_box.extent.x ** 2 +
                vehicle.bounding_box.extent.y ** 2 +
                vehicle.bounding_box.extent.z ** 2)

        if dist > a + b:
            continue

        result.append(light)

    return result


def draw_traffic_lights(image, vehicle, lights, pixels_per_meter=5.5, size=512, radius=5):
    image = Image.fromarray(image)
    draw = ImageDraw.Draw(image)

    transform = vehicle.get_transform()
    pos = transform.location
    theta = np.radians(90 + transform.rotation.yaw)
    R = np.array([
        [np.cos(theta), -np.sin(theta)],
        [np.sin(theta),  np.cos(theta)],
        ])

    for light in lights:
        delta = light.get_transform().location - pos

        target = R.T.dot([delta.x, delta.y])
        target *= pixels_per_meter
        target += size // 2

        if min(target) < 0 or max(target) >= size:
            continue

        trigger = light.trigger_volume
        light.get_transform().transform(trigger.location)
        dist = trigger.location.distance(vehicle.get_location())
        a = np.sqrt(
                trigger.extent.x ** 2 +
                trigger.extent.y ** 2 +
                trigger.extent.z ** 2)
        b = np.sqrt(
                vehicle.bounding_box.extent.x ** 2 +
                vehicle.bounding_box.extent.y ** 2 +
                vehicle.bounding_box.extent.z ** 2)

        if dist > a + b:
            continue

        x, y = target
        draw.ellipse((x-radius, y-radius, x+radius, y+radius), 23 + light.state.real) # 13 changed to 23 for carla 0.9.10

    return np.array(image)


def draw_stop_signs(image, vehicle, lights, pixels_per_meter=5.5, size=512, radius=5):
    image = Image.fromarray(image)
    draw = ImageDraw.Draw(image)

    transform = vehicle.get_transform()
    pos = transform.location
    theta = np.radians(90 + transform.rotation.yaw)
    R = np.array([
        [np.cos(theta), -np.sin(theta)],
        [np.sin(theta),  np.cos(theta)],
        ])

    for light in lights:
        delta = light.get_transform().location - pos

        target = R.T.dot([delta.x, delta.y])
        target *= pixels_per_meter
        target += size // 2

        if min(target) < 0 or max(target) >= size:
            continue

        trigger = light.trigger_volume
        light.get_transform().transform(trigger.location)
        dist = trigger.location.distance(vehicle.get_location())
        a = np.sqrt(
                trigger.extent.x ** 2 +
                trigger.extent.y ** 2 +
                trigger.extent.z ** 2)
        b = np.sqrt(
                vehicle.bounding_box.extent.x ** 2 +
                vehicle.bounding_box.extent.y ** 2 +
                vehicle.bounding_box.extent.z ** 2)

        if dist > a + b:
            continue

        x, y = target
        draw.ellipse((x-radius, y-radius, x+radius, y+radius), 26)

    return np.array(image)

def light_state_str(state):
    if state == carla.TrafficLightState.Red:
        return "red"
    if state == carla.TrafficLightState.Green:
        return "green"
    if state == carla.TrafficLightState.Yellow:
        return "yellow"
    if state == carla.TrafficLightState.Off:
        return "off"
    if state == carla.TrafficLightState.Unknown:
        return "unknown"
    return "error"