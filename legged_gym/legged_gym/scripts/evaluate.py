# SPDX-FileCopyrightText: Copyright (c) 2021 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# 
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
# list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
# this list of conditions and the following disclaimer in the documentation
# and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
# contributors may be used to endorse or promote products derived from
# this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#
# Copyright (c) 2021 ETH Zurich, Nikita Rudin

from legged_gym import LEGGED_GYM_ROOT_DIR
import os
import code

import isaacgym
from legged_gym.envs import *
from legged_gym.utils import  get_args, export_policy_as_jit, task_registry, Logger
from isaacgym import gymtorch, gymapi, gymutil
import numpy as np
import torch
import cv2
from collections import deque
import statistics
import faulthandler
from copy import deepcopy
import matplotlib.pyplot as plt
from time import time, sleep
from legged_gym.utils import webviewer
from tqdm import tqdm

def get_load_path(root, load_run=-1, checkpoint=-1, model_name_include="model"):
                    
    if checkpoint==-1:
        models = [file for file in os.listdir(root) if model_name_include in file]
        models.sort(key=lambda m: '{0:0>15}'.format(m))
        model = models[-1]
        checkpoint = model.split("_")[-1].split(".")[0]
        # code.interact(local=locals())
    # else:
    #     model = "model{}_jit.pt".format(checkpoint) 

    # load_path = root + model
    return model, checkpoint

def play(args):
    if args.web:
        web_viewer = webviewer.WebViewer()
    faulthandler.enable()
    exptid = args.exptid
    log_pth = "../../logs/{}/".format(args.proj_name) + args.exptid
    env_cfg, train_cfg = task_registry.get_cfgs(name=args.task)
    # override some parameters for testing
    if args.nodelay:
        env_cfg.domain_rand.action_delay_view = 0
    env_cfg.env.num_envs = 32
    env_cfg.env.episode_length_s = 60
    env_cfg.commands.resampling_time = 100000
    env_cfg.terrain.num_rows = 5
    env_cfg.terrain.num_cols = 5
    env_cfg.terrain.height = [0.02, 0.02]
    env_cfg.terrain.terrain_dict = {"smooth slope": 0., 
                                    "rough slope up": 0.0,
                                    "rough slope down": 0.0,
                                    "rough stairs up": 1., 
                                    "rough stairs down": 0., 
                                    "discrete": 0., 
                                    "stepping stones": 0.0,
                                    "gaps": 0., 
                                    "smooth flat": 0,
                                    "pit": 0.0,
                                    "wall": 0.0,
                                    "platform": 0.,
                                    "large stairs up": 0.,
                                    "large stairs down": 0.,
                                    "parkour": 0.0,
                                    "parkour_hurdle": 0.0,
                                    "parkour_flat": 0.,
                                    "parkour_step_up": 0.0,
                                    "parkour_gap": 0.0, 
                                    "demo": 0.0,
                                    "parkour_step_down": 0.0,
                                    "parkour_single_box": 0.0}
    # MAYBE CHANGE NUM_GOALS FOR THE SINGLE BOX ENV

    env_cfg.terrain.num_goals = 3
    env_cfg.terrain.terrain_length = 9.
    env_cfg.terrain.evaluate = True
    
    env_cfg.terrain.terrain_proportions = list(env_cfg.terrain.terrain_dict.values())
    env_cfg.terrain.curriculum = False
    env_cfg.terrain.max_difficulty = True  # MAYBE CHANGE THAT TO TRUE
    
    env_cfg.depth.angle = [0, 1]
    env_cfg.noise.add_noise = True
    env_cfg.domain_rand.randomize_friction = True
    env_cfg.domain_rand.push_robots = False
    env_cfg.domain_rand.push_interval_s = 6
    env_cfg.domain_rand.randomize_base_mass = False
    env_cfg.domain_rand.randomize_base_com = False

    depth_latent_buffer = []
    # prepare environment
    env: LeggedRobot
    env, _ = task_registry.make_env(name=args.task, args=args, env_cfg=env_cfg)
    obs = env.get_observations()

    if args.web:
        web_viewer.setup(env)

    # load policy
    train_cfg.runner.resume = True
    ppo_runner, train_cfg, log_pth = task_registry.make_alg_runner(log_root = log_pth, env=env, name=args.task, args=args, train_cfg=train_cfg, return_log_dir=True)
    
    policy = ppo_runner.get_inference_policy(device=env.device)
    if env.cfg.depth.use_camera:
        depth_encoder = ppo_runner.get_depth_encoder_inference_policy(device=env.device)
    
    total_steps = 2000
    rewbuffer = deque(maxlen=total_steps)
    lenbuffer = deque(maxlen=total_steps)
    num_waypoints_buffer = deque(maxlen=total_steps)
    time_to_fall_buffer = deque(maxlen=total_steps)
    #success_rate = deque(maxlen=total_steps)
    # TODO: add boolean buff that identifies the agents that are done with their first run to set those actions to 0

    cur_reward_sum = torch.zeros(env.num_envs, dtype=torch.float, device=env.device)
    cur_episode_length = torch.zeros(env.num_envs, dtype=torch.float, device=env.device)
    cur_time_from_start = torch.zeros(env.num_envs, dtype=torch.float, device=env.device)
    run_completed = torch.zeros(env.num_envs, dtype=bool, device=env.device)
    edge_violation = torch.zeros(env.num_envs, dtype=float, device=env.device)
    knees_usage = torch.zeros(env.num_envs, dtype=float, device=env.device)

    actions = torch.zeros(env.num_envs, 12, device=env.device, requires_grad=False)
    infos = {}
    infos["depth_image"] = env.depth_buffer.clone().to(ppo_runner.device)[:, -1] if ppo_runner.if_depth else None 
    for i in (range(2000)):
        if run_completed.sum() == env.num_envs: break

        if env.cfg.depth.use_camera:
            if infos["depth_image"] is not None:  
                obs_student = obs[:, :env.cfg.env.n_proprio]    # nb of proprioception inputs (3 + 2 + 3 + 4 + 36 + 5)
                obs_student[:, 6:8] = 0
                #if i == 0: print(obs_student.shape)
                with torch.no_grad():
                    depth_latent_and_yaw = depth_encoder(infos["depth_image"], obs_student)
                depth_latent = depth_latent_and_yaw[:, :-2]
                yaw = depth_latent_and_yaw[:, -2:]
            obs[:, 6:8] = 1.5*yaw
            
        else:
            depth_latent = None

        if hasattr(ppo_runner.alg, "depth_actor"):
            with torch.no_grad():
                actions = ppo_runner.alg.depth_actor(obs.detach(), hist_encoding=True, scandots_latent=depth_latent)
        else:
            actions = policy(obs.detach(), hist_encoding=True, scandots_latent=depth_latent)
            
        actions[run_completed] = 0
        
        cur_goal_idx = env.cur_goal_idx.clone()
        obs, _, rews, dones, infos = env.step(actions.detach())
        if args.web:
            web_viewer.render(fetch_results=True,
                        step_graphics=True,
                        render_all_camera_sensors=True,
                        wait_for_page_load=True)
        
        id = env.lookat_id
        # Log stuff

        edge_violation += env.feet_at_edge.any(dim=1).float() * env.dt #* env_cfg.control.decimation
        knees_usage += env.contact_knees.any(dim=1).float() * env.dt #* env_cfg.control.decimation

        cur_reward_sum += rews
        cur_episode_length += 1
        cur_time_from_start += 1
        run_completed = torch.logical_or(run_completed, dones)

        new_ids = (dones > 0).nonzero(as_tuple=False)
        killed_ids = ((dones > 0) & (~infos["time_outs"])).nonzero(as_tuple=False)
        rewbuffer.extend(cur_reward_sum[new_ids][:, 0].cpu().numpy().tolist())
        lenbuffer.extend(cur_episode_length[new_ids][:, 0].cpu().numpy().tolist())
        num_waypoints_buffer.extend(cur_goal_idx[new_ids][:, 0].cpu().numpy().tolist())
        time_to_fall_buffer.extend(cur_time_from_start[killed_ids][:, 0].cpu().numpy().tolist())

        #print(np.where(infos["time_outs"] == True)[0])
        #print(infos["time_outs"])

        cur_reward_sum[new_ids] = 0
        cur_episode_length[new_ids] = 0
        cur_time_from_start[killed_ids] = 0
    
    #compute buffer mean and std
    rew_mean = statistics.mean(rewbuffer)
    rew_std = statistics.stdev(rewbuffer)

    len_mean = statistics.mean(lenbuffer)
    len_std = statistics.stdev(lenbuffer)

    num_waypoints_mean = np.mean(np.array(num_waypoints_buffer).astype(float)/7.0)
    num_waypoints_std = np.std(np.array(num_waypoints_buffer).astype(float)/7.0)

    if len(time_to_fall_buffer) > 1:
        time_to_fall_mean = statistics.mean(time_to_fall_buffer)
        time_to_fall_std = statistics.stdev(time_to_fall_buffer)
    elif len(time_to_fall_buffer) == 1:
        time_to_fall_mean = statistics.mean(time_to_fall_buffer)
        time_to_fall_std = 0.
    else:
        time_to_fall_mean = 0.
        time_to_fall_std = 0.

    time = infos["success"] # in time, the >0 values are the time to succeed the task, ==0 are jammed agent, < 0 are agent that failed
    success_rate = 100 * (time > 0).sum() / time.shape[0]
    failure_rate = 100 * (time < 0).sum() / time.shape[0]
    jamming_rate = 100 * (time == 0).sum() / time.shape[0]

    edge_violation_percent = 100 * edge_violation[time > 0] / time[time > 0]
    knees_usage_percent = 100 * edge_violation[time > 0] / time[time > 0]

    print("Mean reward: {:.2f}$\pm${:.2f}".format(rew_mean, rew_std))
    print("Mean episode length: {:.2f}$\pm${:.2f}".format(len_mean, len_std))
    print("Mean number of waypoints: {:.2f}$\pm${:.2f}".format(num_waypoints_mean, num_waypoints_std))
    print("Mean time to fall: {:.2f}$\pm${:.2f}".format(time_to_fall_mean, time_to_fall_std))
    print("Mean edge violation: {:.2f}$\pm${:.2f}s".format(edge_violation.mean(), edge_violation.std()))
    print("Mean edge violation percentage: {:.2f}$\pm${:.2f}%".format(edge_violation_percent.mean(), edge_violation_percent.std()))
    print("Mean knee usage: {:.2f}$\pm${:.2f}s".format(knees_usage.mean(), knees_usage.std()))
    print("Mean knee usage percentage: {:.2f}$\pm${:.2f}%".format(knees_usage_percent.mean(), knees_usage_percent.std()))
    print("Success rate: {:.2f}%".format(success_rate))
    print("Failure rate: {:.2f}%".format(failure_rate))
    print("Jamming rate: {:.2f}%".format(jamming_rate))
    #print(infos["success"])
    #print(infos["goals_pos_x"])
    #print(infos["commands"])
    distance = infos["goals_pos_x"] # distance from origin to last goal in x
    speed = infos["commands"]   # input speed

    diff = time[time > 0] - distance[time > 0] / speed[time > 0]

    if len(diff) > 1:
        overtime_mean = diff.mean()
        overtime_std = diff.std()
    elif len(diff) == 1:
        overtime_mean = diff.mean()
        overtime_std = 0.
    else:
        overtime_mean = 0.
        overtime_std = 0.

    print("Mean overtime: {:.2f}$\pm${:.2f}".format(overtime_mean, overtime_std))



if __name__ == '__main__':
    EXPORT_POLICY = False
    RECORD_FRAMES = False
    MOVE_CAMERA = False
    args = get_args()
    play(args)


# 038-10 no feet edge
# 038-91 ours
# 043-21 non-inner
