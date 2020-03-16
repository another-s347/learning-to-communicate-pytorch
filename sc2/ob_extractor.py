import copy
import numpy as np
from collections import defaultdict

extra_distance = defaultdict(int, {
    36: 12  # Command Center
})


class LocalAgentExtractor:
    def __init__(self, id, summary_writer=None, resolution=24, map_boundary=48):
        self.inited = False
        self.done = False
        self.id = id
        self.unit_tag = None
        self.x = 0
        self.y = 0
        self.raw_unit = None
        self.resolution = resolution
        self.map_boundary = map_boundary
        self.last_total_damage_dealt = 0
        self.last_total_damage_taken = 0
        self.enemies = []
        self.action_step = 0
        self.summary_writer = summary_writer
        pass

    def reset(self):
        self.done = False
        self.inited = False

    def get_init_info(self, ob):
        feature_units = [
            unit for unit in ob.feature_units if unit.alliance == 1 and unit.unit_type == 35]
        if len(feature_units) <= self.id:
            # raise NotImplementedError("len(feature_units) < extractor id")
            return None
        unit = feature_units[self.id]
        if unit.tag == 0:
            raise NotImplementedError("unit tag == 0")
        self.unit_tag = unit.tag
        self.inited = True
        return 1

    def get_unit(self, ob):
        for unit in ob.feature_units:
            if unit.tag == self.unit_tag:
                return unit
        return None

    def get_raw_unit(self, ob):
        for unit in ob.raw_units:
            if unit.tag == self.unit_tag:
                return unit
        return None

    def get_enemy(self, ob):
        self.enemies = [
            unit for unit in ob.raw_units if unit.alliance == 4]

    def get_action(self, action_id, action_arg):
        if action_id != 2:
            return action_id, action_arg, 0
        xy = action_arg[2]
        for e in self.enemies:
            x_diff = xy[0] - e.x
            y_diff = xy[1] - e.y
            distance = y_diff**2 + x_diff**2
            if distance < 16+extra_distance[e.unit_type]:
                print(f"hit. Distance = {distance}")
                self.action_step += 1
                if self.summary_writer:
                    self.summary_writer.add_scalar(
                        "hit_distance", distance, self.action_step)
                return 3, [action_arg[0], action_arg[1], [e.tag]], 5
            elif distance < 36+extra_distance[e.unit_type]:
                print(f"not hit. Distance = {distance}")
        return action_id, action_arg, 0

    def get_reward(self, obs):
        reward = 0
        cur_damage_dealt = obs.score_by_vital.total_damage_dealt.life + \
            obs.score_by_vital.total_damage_dealt.shields
        reward += (cur_damage_dealt - self.last_total_damage_dealt)*10
        if reward < 0:
            self.reward = 0
            return 0
        self.last_total_damage_dealt = cur_damage_dealt
        cur_damage_taken = obs.score_by_vital.total_damage_taken.life
        self.last_total_damage_taken = cur_damage_taken
        # reward = reward // 10
        self.reward = reward
        return reward

    def extract(self, obs):
        observation = obs.observation
        # Init if needed
        if not self.inited:
            if self.get_init_info(observation) is None:
                return None

        if self.done:
            return None
        unit = self.get_unit(observation)
        if unit is None:
            self.done = True
            return None
        self.raw_unit = self.get_raw_unit(observation)
        self.get_enemy(observation)
        x = unit.x
        y = unit.y
        self.x = x
        self.y = y
        x_left = x - self.resolution // 2
        x_right = x + self.resolution // 2
        y_up = y - self.resolution // 2
        y_down = y + self.resolution // 2
        if x_left <= 0 and y_up <= 0:
            feature_screen = observation.feature_screen[:, :y_down, :x_right]
            feature_screen = np.pad(feature_screen, pad_width=(
                (0, 0), (-y_up, 0), (-x_left, 0)), mode='constant', constant_values=0)
        elif x_left <= 0 and y_down >= self.map_boundary:
            feature_screen = observation.feature_screen[:,
                                                        y_up:, :x_right]
            feature_screen = np.pad(feature_screen, pad_width=(
                (0, 0), (0, y_down-self.map_boundary), (-x_left, 0)), mode='constant', constant_values=0)
        elif x_right >= self.map_boundary and y_up <= 0:
            feature_screen = observation.feature_screen[:,
                                                        :y_down, x_left:]
            feature_screen = np.pad(feature_screen, pad_width=(
                (0, 0), (-y_up, 0), (0, x_right-self.map_boundary)), mode='constant', constant_values=0)
        elif x_right >= self.map_boundary and y_down >= self.map_boundary:
            feature_screen = observation.feature_screen[:,
                                                        y_up:, x_left:]
            feature_screen = np.pad(feature_screen, pad_width=(
                (0, 0), (0, y_down-self.map_boundary), (0, x_right-self.map_boundary)), mode='constant', constant_values=0)
        elif x_left <= 0:
            feature_screen = observation.feature_screen[:,
                                                        y_up:y_down, :x_right]
            feature_screen = np.pad(feature_screen, pad_width=(
                (0, 0), (0, 0), (-x_left, 0)), mode='constant', constant_values=0)
        elif y_down >= self.map_boundary:
            feature_screen = observation.feature_screen[:,
                                                        y_up:, x_left:x_right]
            feature_screen = np.pad(feature_screen, pad_width=(
                (0, 0), (0, y_down-self.map_boundary), (0, 0)), mode='constant', constant_values=0)
        elif y_up <= 0:
            feature_screen = observation.feature_screen[:,
                                                        :y_down, x_left:x_right]
            feature_screen = np.pad(feature_screen, pad_width=(
                (0, 0), (-y_up, 0), (0, 0)), mode='constant', constant_values=0)
        elif x_right >= self.map_boundary:
            feature_screen = observation.feature_screen[:,
                                                        y_up:y_down, x_left:]
            feature_screen = np.pad(feature_screen, pad_width=(
                (0, 0), (0, 0), (0, x_right-self.map_boundary)), mode='constant', constant_values=0)
        else:
            feature_screen = observation.feature_screen[:,
                                                        y_up:y_down, x_left:x_right]

        # TODO: calc reward
        reward = self.get_reward(observation)
        if feature_screen.shape != (27, self.resolution, self.resolution):
            return {
            "feature_screen": np.zeros((27, 24, 24)),
            "reward": 0,
            "done": False,
            "unit": np.zeros((46,)),
            "x": 0,
            "y": 0
        }
        return {
            "feature_screen": feature_screen,
            "reward": reward,
            "done": self.done,
            "unit": self.raw_unit,
            "x": self.x,
            "y": self.y
        }


# from nsnn.obsextractor import BaseObsExtractor
# import copy
# import numpy as np


# class LocalAgentExtractor(BaseObsExtractor):
#     def __init__(self, id, resolution = 30):
#         self.inited = False
#         self.done = False
#         self.id = id
#         self.unit_tag = None
#         self.x = 0
#         self.y = 0
#         self.raw_unit = None
#         self.resolution = resolution
#         self.last_total_damage_dealt = 0
#         self.last_total_damage_taken = 0
#         self.enemy_num = 0
#         pass

#     def reset(self):
#         self.done = False
#         self.inited = False

#     def get_init_info(self, ob):
#         feature_units = [
#             unit for unit in ob.feature_units if unit.alliance == 1 and unit.unit_type == 35]
#         self.enemy_num = len([unit for unit in ob.feature_units if unit.alliance == 4])
#         if len(feature_units) <= self.id:
#             # raise NotImplementedError("len(feature_units) < extractor id")
#             return None
#         unit = feature_units[self.id]
#         if unit.tag == 0:
#             raise NotImplementedError("unit tag == 0")
#         self.unit_tag = unit.tag
#         self.inited = True
#         return 1

#     def get_unit(self, ob):
#         for unit in ob.feature_units:
#             if unit.tag == self.unit_tag:
#                 return unit
#         return None

#     def get_raw_unit(self, ob):
#         for unit in ob.raw_units:
#             if unit.tag == self.unit_tag:
#                 return unit
#         return None

#     def get_reward(self, obs):
#         reward = 0
#         cur_damage_dealt = obs.score_by_vital.total_damage_dealt.life
#         self.last_total_damage_dealt = cur_damage_dealt
#         cur_damage_taken = obs.score_by_vital.total_damage_taken.life
#         reward += (cur_damage_dealt - cur_damage_taken)
#         self.last_total_damage_taken = cur_damage_taken
#         enemy_num = len([unit for unit in obs.feature_units if unit.alliance == 4])
#         if enemy_num - self.enemy_num > 0:
#             reward += 100*(enemy_num - self.enemy_num)
#             self.enemy_num = enemy_num
#         if reward == 0:
#             reward -= 10
#         self.reward = reward
#         return reward

#     def extract(self, obs):
#         observation = obs.observation
#         # Init if needed
#         if not self.inited:
#             if self.get_init_info(observation) is None:
#                 return None

#         if self.done:
#             return None
#         unit = self.get_unit(observation)
#         if unit is None:
#             self.done = True
#             return None
#         self.raw_unit = self.get_raw_unit(observation)
#         x = unit.x
#         y = unit.y
#         self.x = x
#         self.y = y
#         x_left = x - 15
#         x_right = x + 15
#         y_up = y - 15
#         y_down = y + 15
#         if x_left <= 0 and y_up <= 0:
#             feature_screen = observation.feature_screen[:, :y_down, :x_right]
#             feature_screen = np.pad(feature_screen, pad_width=(
#                 (0, 0), (-y_up, 0), (-x_left, 0)), mode='constant', constant_values=0)
#         elif x_left <= 0 and y_down >= self.map_boundary:
#             feature_screen = observation.feature_screen[:,
#                                                         y_up:, :x_right]
#             feature_screen = np.pad(feature_screen, pad_width=(
#                 (0, 0), (0, y_down-self.map_boundary), (-x_left, 0)), mode='constant', constant_values=0)
#         elif x_right >= self.map_boundary and y_up <= 0:
#             feature_screen = observation.feature_screen[:,
#                                                         :y_down, x_left:]
#             feature_screen = np.pad(feature_screen, pad_width=(
#                 (0, 0), (-y_up, 0), (0, x_right-self.map_boundary)), mode='constant', constant_values=0)
#         elif x_right >= self.map_boundary and y_down >= self.map_boundary:
#             feature_screen = observation.feature_screen[:,
#                                                         y_up:, x_left:]
#             feature_screen = np.pad(feature_screen, pad_width=(
#                 (0, 0), (0, y_down-self.map_boundary), (0, x_right-self.map_boundary)), mode='constant', constant_values=0)
#         elif x_left <= 0:
#             feature_screen = observation.feature_screen[:,
#                                                         y_up:y_down, :x_right]
#             feature_screen = np.pad(feature_screen, pad_width=(
#                 (0, 0), (0, 0), (-x_left, 0)), mode='constant', constant_values=0)
#         elif y_down >= self.map_boundary:
#             feature_screen = observation.feature_screen[:,
#                                                         y_up:, x_left:x_right]
#             feature_screen = np.pad(feature_screen, pad_width=(
#                 (0, 0), (0, y_down-self.map_boundary), (0, 0)), mode='constant', constant_values=0)
#         elif y_up <= 0:
#             feature_screen = observation.feature_screen[:,
#                                                         :y_down, x_left:x_right]
#             feature_screen = np.pad(feature_screen, pad_width=(
#                 (0, 0), (-y_up, 0), (0, 0)), mode='constant', constant_values=0)
#         elif x_right >= self.map_boundary:
#             feature_screen = observation.feature_screen[:,
#                                                         y_up:y_down, x_left:]
#             feature_screen = np.pad(feature_screen, pad_width=(
#                 (0, 0), (0, 0), (0, x_right-self.map_boundary)), mode='constant', constant_values=0)
#         else:
#             feature_screen = observation.feature_screen[:,
#                                                         y_up:y_down, x_left:x_right]

#         obs.observation.feature_screen = feature_screen
#         # TODO: calc reward
#         reward = self.get_reward(observation)
#         return {
#             "feature_screen": feature_screen,
#             "reward": reward,
#             "done": self.done,
#             "unit": self.raw_unit
#         }