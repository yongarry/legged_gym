from time import time
import numpy as np
import os

from isaacgym.torch_utils import *
from isaacgym import gymtorch, gymapi, gymutil

import torch
from typing import Tuple, Dict
from legged_gym.envs import LeggedRobot

class Bolt6D(LeggedRobot):
    def _init_buffers(self):
        super()._init_buffers()
        self.commands_scale = torch.tensor([self.obs_scales.lin_vel, self.obs_scales.lin_vel], device=self.device, requires_grad=False,) # TODO change this

    def _reward_no_fly(self):
        contacts = self.contact_forces[:, self.feet_indices, 2] > 0.1
        
        
        single_contact = torch.sum(1.*contacts, dim=1)==1
        return 1.*single_contact
    
    
    def _reward_energy(self):
        #veltorque, value  scale or sum
        energy=(self.torques, self.dof_vel)
        positive_energy=(self.torques*self.dof_vel).clip(min=0.)
        if self.cfg.rewards.positive_energy_reward:
            return torch.sum(torch.square(positive_energy), dim=1)
        else:
            return torch.sum(torch.square(energy), dim=1)
        ### sum over all columns of the square values for energy (using joint velocities)

    def compute_observations(self):
        super().compute_observations()
        self.obs_buf = torch.cat((  self.base_lin_vel * self.obs_scales.lin_vel,
                                    self.base_ang_vel  * self.obs_scales.ang_vel,
                                    self.projected_gravity,
                                    self.commands[:, :2] * self.commands_scale,
                                    (self.dof_pos - self.default_dof_pos) * self.obs_scales.dof_pos,
                                    self.dof_vel * self.obs_scales.dof_vel,
                                    self.actions
                                    ),dim=-1)
        # print(self.obs_buf)

    def _resample_commands(self, env_ids):
        # super()._resample_commands()
        self.commands[env_ids, 0] = torch_rand_float(self.command_ranges["lin_vel_x"][0], self.command_ranges["lin_vel_x"][1], (len(env_ids), 1), device=self.device).squeeze(1)
        self.commands[env_ids, 1] = torch_rand_float(self.command_ranges["lin_vel_y"][0], self.command_ranges["lin_vel_y"][1], (len(env_ids), 1), device=self.device).squeeze(1)
        
        
