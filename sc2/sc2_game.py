"""
Switch game

This class manages the state of the Switch game among multiple agents.

RIAL Actions:

1 = Nothing
2 = Tell
3 = On
4 = Off
"""

import numpy as np
import torch

from utils.dotdic import DotDic
from sc2.make_env import make_env
from sc2.ob_extractor import LocalAgentExtractor
from pysc2.lib.actions import FunctionCall, FUNCTIONS, RAW_FUNCTIONS

localaction_table = [
    13,
    2,
    2
]

class Sc2Game:

    def __init__(self, opt):
        self.opt = opt
        if opt.bs is not 1:
            raise NotImplementedError()

        # Set game defaults
        opt_game_default = DotDic({
            'render': True,
            'feature_screen_size': 48,
            'feature_minimap_size': 48,
            'rgb_screen_size': None,
            'rgb_minimap_size': None,
            'action_space':"RAW",
            'use_feature_units':True,
            'use_raw_units':True,
            'disable_fog':True,
            'max_agent_step':0,
            'game_steps_per_episode':None,
            'max_episodes':0,
            'step_mul':4,
            'agent':'pysc2.agents.random_agent.RandomAgent',
            'agent_name':None,
            'agent_race':'random',
            'agent2':'Bot',
            'agent2_name':None,
            'agent2_race':'random',
            'difficulty':'very_easy',
            'bot_build':'random',
            'save_replay':False,
            'map':'1',
            'battle_net_map':False
        })
        for k in opt_game_default:
            if k not in self.opt:
                self.opt[k] = opt_game_default[k]

        self.env = make_env(self.opt)

        self.reset()

    def reset(self):
        # Step count
        self.step_count = 0

        # Rewards
        self.reward = torch.zeros(self.opt.bs, self.opt.game_nagents)

        # Who has been in the room?
        self.has_been = torch.zeros(
            self.opt.bs, self.opt.nsteps, self.opt.game_nagents)

        # Terminal state
        self.terminal = torch.zeros(self.opt.bs, dtype=torch.long)

        # Active agent
        # self.active_agent = torch.zeros(
        #     self.opt.bs, self.opt.nsteps, dtype=torch.long)  # 1-indexed agents
        # for b in range(self.opt.bs):
        #     for step in range(self.opt.nsteps):
        #         agent_id = 1 + np.random.randint(self.opt.game_nagents)
        #         self.active_agent[b][step] = agent_id
        #         self.has_been[b][step][agent_id - 1] = 1
        
        self.extractors = [LocalAgentExtractor( 
            id=x) for x in range(self.opt.game_nagents)]

        state = self.env.reset()
        self.states = np.array([[e.extract(state[0]) for e in self.extractors]])

        return self

    def get_action_range(self, step, agent_id):
        """
        Return 1-indexed indices into Q vector for valid actions and communications (so 0 represents no-op)
        """
        opt = self.opt
        action_dtype = torch.long
        action_range = torch.zeros((self.opt.bs, 2), dtype=action_dtype)
        action_arg_range = torch.zeros((self.opt.bs, 2), dtype=action_dtype)
        comm_range = torch.zeros((self.opt.bs, 2), dtype=action_dtype)
        for b in range(self.opt.bs):
            # if self.active_agent[b][step] == agent_id:
            #     action_range[b] = torch.tensor(
            #         [1, opt.game_action_space], dtype=action_dtype)
            comm_range[b] = torch.tensor(
                    [1, 4], dtype=action_dtype)
            # else:
            action_range[b] = torch.tensor([0, 1], dtype=action_dtype)
            action_arg_range[b] = torch.tensor([1, 24*24], dtype=action_dtype)

        return action_range, action_arg_range, comm_range

    def get_comm_limited(self, step, agent_id):
        # if self.opt.game_comm_limited:
        # 	comm_lim = torch.zeros(self.opt.bs, dtype=torch.long)
        # 	for b in range(self.opt.bs):
        # 		if step > 0 and agent_id == self.active_agent[b][step]:
        # 			comm_lim[b] = self.active_agent[b][step - 1]
        # 	return comm_lim
        return None

    def get_reward(self, a_t):
        # Return reward for action a_t by active agent
        for b in range(self.opt.bs):
            active_agent_idx = self.active_agent[b][self.step_count].item() - 1
            if a_t[b][active_agent_idx].item() == self.game_actions.TELL and not self.terminal[b].item():
                has_been = self.has_been[b][:self.step_count +
                                            1].sum(0).gt(0).sum(0).item()
                if has_been == self.opt.game_nagents:
                    self.reward[b] = self.reward_all_live
                else:
                    self.reward[b] = self.reward_all_die
                self.terminal[b] = 1
            elif self.step_count == self.opt.nsteps - 1 and not self.terminal[b]:
                self.terminal[b] = 1

        return self.reward.clone(), self.terminal.clone()

    def localactions_to_pysc2(self, actions, extractor):
        """Convert agent action representation to FunctionCall representation."""
        height, width = (24, 24)
        fn_id, world_pt = actions
        actions_list = []
        reward_addon = 0
        a_0 = fn_id.item()
        a_0 = localaction_table[a_0]
        # print(extractor.raw_unit.x,extractor.raw_unit.y,world_pt.item() // width,world_pt.item() % width)
        x = extractor.raw_unit.x - (world_pt.item() // width - 11)
        y = extractor.raw_unit.y - (world_pt.item() % height - 11)
        if x < 0:
            x = 0
        if y < 0:
            y = 0
        arg = [x, y]
        a_0, args, r = extractor.get_action(
            a_0, [[0], [extractor.unit_tag], arg])
        reward_addon += r
        action = FunctionCall(a_0, args)
        # actions_list.append(action)
        return reward_addon, action

    def step(self, a_t, a_a_t):
        actions = []
        reward = torch.zeros(self.opt.bs, self.opt.game_nagents)
        for i in range(self.opt.game_nagents):
            policy = (a_t[0][i], a_a_t[0][i])
            r_a, a = self.localactions_to_pysc2(policy, self.extractors[i])
            reward[0][i] = r_a
            actions.append(a)
        terminal = torch.zeros(self.opt.bs)
        next_state = self.env.step([actions])
        self.states = np.array([[e.extract(next_state[0]) for e in self.extractors]])
        self.step_count += 1

        return reward, terminal

    def get_state(self):
        # state = torch.zeros(
            # self.opt.bs, self.opt.game_nagents, dtype=torch.long)

        # Get the state of the game
        # for b in range(self.opt.bs):
        #     for a in range(1, self.opt.game_nagents + 1):
        #         if self.active_agent[b][self.step_count] == a:
        #             state[b][a - 1] = self.game_states.INSIDE
        return self.states

        # return state

    def god_strategy_reward(self, steps):
        reward = torch.zeros(self.opt.bs)
        for b in range(self.opt.bs):
            has_been = self.has_been[b][:self.opt.nsteps].sum(
                0).gt(0).sum().item()
            if has_been == self.opt.game_nagents:
                reward[b] = self.reward_all_live

        return reward

    def naive_strategy_reward(self):
        pass

    def get_stats(self, steps):
        stats = DotDic({})
        stats.god_reward = self.god_strategy_reward(steps)
        return stats

    def describe_game(self, b=0):
        print('has been:', self.has_been[b])
        print('num has been:', self.has_been[b].sum(0).gt(0).sum().item())
        print('active agents: ', self.active_agent[b])
        print('reward:', self.reward[b])
