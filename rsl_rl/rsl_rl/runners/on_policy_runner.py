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

import time
import os
from collections import deque
import statistics

# from torch.utils.tensorboard import SummaryWriter
import torch
import torch.optim as optim
import wandb
# import ml_runlog
import datetime

from rsl_rl.algorithms import PPO
from rsl_rl.modules import *
from rsl_rl.env import VecEnv
import sys
from copy import copy, deepcopy
import warnings

class OnPolicyRunner:

    def __init__(self,
                 env: VecEnv,
                 train_cfg,
                 log_dir=None,
                 init_wandb=True,
                 device='cpu', **kwargs):

        self.cfg=train_cfg["runner"]
        self.alg_cfg = train_cfg["algorithm"]
        self.policy_cfg = train_cfg["policy"]
        self.estimator_cfg = train_cfg["estimator"]
        self.depth_encoder_cfg = train_cfg["depth_encoder"]
        self.device = device
        self.env = env

        print("Using MLP and Priviliged Env encoder ActorCritic structure")
        actor_critic: ActorCriticRMA = ActorCriticRMA(self.env.cfg.env.n_proprio,
                                                      self.env.cfg.env.n_scan,
                                                      self.env.num_obs,
                                                      self.env.cfg.env.n_priv_latent,
                                                      self.env.cfg.env.n_priv,
                                                      self.env.cfg.env.history_len,
                                                      self.env.num_actions,
                                                      **self.policy_cfg).to(self.device)
        estimator = Estimator(input_dim=env.cfg.env.n_proprio, output_dim=env.cfg.env.n_priv, hidden_dims=self.estimator_cfg["hidden_dims"]).to(self.device)
        # Depth encoder
        self.if_depth = self.depth_encoder_cfg["if_depth"]
        if self.if_depth:
            depth_backbone = DepthOnlyFCBackbone58x87(env.cfg.env.n_proprio, 
                                                    self.policy_cfg["scan_encoder_dims"][-1], 
                                                    self.depth_encoder_cfg["hidden_dims"],
                                                    )
            depth_encoder = RecurrentDepthBackbone(depth_backbone, env.cfg).to(self.device) # (depth_backbone, 
            #depth_actor = deepcopy(actor_critic.actor) # Student: not needed anymore
            self.chunk_size = env.cfg.depth.chunk_size

        else:
            depth_encoder = None
            #depth_actor = None
        # self.depth_encoder = depth_encoder
        # self.depth_encoder_optimizer = optim.Adam(self.depth_encoder.parameters(), lr=self.depth_encoder_cfg["learning_rate"])
        # self.depth_encoder_paras = self.depth_encoder_cfg
        # self.depth_encoder_criterion = nn.MSELoss()
        # Create algorithm
        alg_class = eval(self.cfg["algorithm_class_name"]) # PPO
        self.alg: PPO = alg_class(actor_critic, 
                                  estimator, self.estimator_cfg, 
                                  depth_encoder, self.depth_encoder_cfg, #depth_actor,
                                  device=self.device, **self.alg_cfg)
        self.num_steps_per_env = self.cfg["num_steps_per_env"]
        self.save_interval = self.cfg["save_interval"]
        self.dagger_update_freq = self.alg_cfg["dagger_update_freq"]
        self.yaw_estimates_update_freq = self.alg_cfg["yaw_estimates_update_freq"]

        self.alg.init_storage(
            self.env.num_envs, 
            self.num_steps_per_env * self.depth_encoder_cfg["num_steps_per_env"], 
            [self.env.num_obs], 
            [self.env.num_privileged_obs], 
            [self.env.num_actions],
        )
    
        #print("$"*120)
        #print(self.env.num_envs)    # 4
        #print(self.num_steps_per_env)   # 24
        #print(self.env.num_obs)     # 753
        #print(self.env.num_privileged_obs)  # None
        #print(self.env.num_actions) # 12
        #print("%"*120)

        #self.learn = self.learn_RL if not self.if_depth else self.learn_vision
        self.learn = self.learn_anymal
            
        # Log
        self.log_dir = log_dir
        self.writer = None
        self.tot_timesteps = 0
        self.tot_time = 0
        self.current_learning_iteration = 0

    def learn_anymal(self, num_learning_iterations, init_at_random_ep_len=False):
        mean_value_loss = 0.
        mean_surrogate_loss = 0.
        mean_estimator_loss = 0.
        mean_disc_loss = 0.
        mean_disc_acc = 0.
        mean_hist_latent_loss = 0.
        mean_priv_reg_loss = 0. 
        priv_reg_coef = 0.
        entropy_coef = 0.

        if init_at_random_ep_len:
            self.env.episode_length_buf = torch.randint_like(self.env.episode_length_buf, high=int(self.env.max_episode_length))
        obs = self.env.get_observations()
        privileged_obs = self.env.get_privileged_observations()
        critic_obs = privileged_obs if privileged_obs is not None else obs
        obs, critic_obs = obs.to(self.device), critic_obs.to(self.device)
        infos = {}
        infos["depth"] = self.env.depth_buffer.clone().to(self.device)[:, -1] if self.if_depth else None
        if self.if_depth:
            #delta_yaw_ok = torch.zeros(self.env.num_envs, dtype=torch.bool, device=self.device)
            #delta_next_yaw_ok = torch.zeros(self.env.num_envs, dtype=torch.bool, device=self.device)
            self.alg.depth_encoder.train()
            #self.alg.depth_actor.train()   # No depth_actor = student here

        
        self.alg.actor_critic.train() # switch to train mode (for dropout for example)

        ep_infos = []
        rewbuffer = deque(maxlen=100)
        rew_explr_buffer = deque(maxlen=100)
        rew_entropy_buffer = deque(maxlen=100)
        lenbuffer = deque(maxlen=100)
        cur_reward_sum = torch.zeros(self.env.num_envs, dtype=torch.float, device=self.device)
        cur_reward_explr_sum = torch.zeros(self.env.num_envs, dtype=torch.float, device=self.device)
        cur_reward_entropy_sum = torch.zeros(self.env.num_envs, dtype=torch.float, device=self.device)
        cur_episode_length = torch.zeros(self.env.num_envs, dtype=torch.float, device=self.device)

        tot_iter = self.current_learning_iteration + num_learning_iterations
        self.start_learning_iteration = copy(self.current_learning_iteration)

        assert self.depth_encoder_cfg["num_steps_per_env"] == 5
        assert self.num_steps_per_env == 5
        assert self.if_depth

        # This block is only for the edge case it == 0
        yaw_loss = 0
        delta_yaw_ok_buffer = []
        delta_next_yaw_ok_buffer = []
        yaw_buffer_student = []
        yaw_buffer_teacher = []
        ####

        for it in range(self.current_learning_iteration, tot_iter):
            learn_time = 0
            collection_time = 0
            start = time.time()

            hist_encoding = it % self.dagger_update_freq == 0
            if self.if_depth:
                update_yaw_estimates = it % self.yaw_estimates_update_freq == 0
                if it % self.yaw_estimates_update_freq == 1:    # accumulate through time before updating yaw estimates
                    yaw_loss = 0
                    delta_yaw_ok_buffer = []
                    delta_next_yaw_ok_buffer = []
                    yaw_buffer_student = []
                    yaw_buffer_teacher = []
                    #depth_latent_buffer = []       

            for j in range(self.depth_encoder_cfg["num_steps_per_env"]): # (= 5) new image every time
                if self.if_depth:
                    assert infos["depth"] != None # it is None when self.global_counter % self.cfg.depth.update_interval != 0
                    if infos["depth"] != None:
                        obs_prop_depth = obs[:, :self.env.cfg.env.n_proprio].clone()
                        obs_prop_depth[:, 6:8] = 0

                        start_learning = time.time()
                        delta_yaw_grad, delta_yaw = self.alg.compute_delta_yaw_estimates(infos["depth"], obs_prop_depth, self.chunk_size)
                        stop_learning = time.time()
                        learn_time += stop_learning - start_learning

                        #depth_latent_and_yaw = self.alg.depth_encoder(infos["depth"].clone(), obs_prop_depth)  # clone is crucial to avoid in-place operation
                                
                        # The model now outputs only 2 values anyway: so [:, :-2] is basically the whole output
                        #depth_latent = depth_latent_and_yaw[:, :-2] #   Could use them as input in the model later on

                        #depth_latent_buffer.append(depth_latent)
                        yaw_buffer_student.append(delta_yaw_grad)#.cpu())
                        yaw_buffer_teacher.append(obs[:self.chunk_size, 6:8])
                            
                        # TEST WITHOUT DELTA_YAW_OK TO RECOVER PERFORMANCE
                        delta_yaw_ok = torch.abs(delta_yaw[:, 0] - obs[:, 6]) < 0.26
                        delta_next_yaw_ok = torch.abs(delta_yaw[:, 1] - obs[:, 7]) < 0.26

                        delta_yaw_ok_buffer.append(torch.nonzero(delta_yaw_ok).size(0) / delta_yaw_ok.numel())
                        delta_next_yaw_ok_buffer.append(torch.nonzero(delta_next_yaw_ok).size(0) / delta_next_yaw_ok.numel())
                # Rollout
                with torch.inference_mode():
                    for i in range(self.num_steps_per_env):     # takes num_steps_per_env (5) steps on each images
                        if self.if_depth:
                            if i != 0: assert infos["depth"] == None   # should be None because the update happens above (depth image is updated each j-th iteration)
                            
                            # OBSERVATIONS
                            # change the observation everytime even if the image has not updated
                            obs[delta_yaw_ok, 6] = delta_yaw[delta_yaw_ok, 0].detach()
                            obs[delta_next_yaw_ok, 7] = delta_yaw[delta_next_yaw_ok, 1].detach()
                        
                        actions = self.alg.act(obs, critic_obs, infos, hist_encoding)  # Default: hist_encoding
                        obs, privileged_obs, rewards, dones, infos = self.env.step(actions)  # obs has changed to next_obs !! if done obs has been reset
                        critic_obs = privileged_obs if privileged_obs is not None else obs
                        obs, critic_obs, rewards, dones = obs.to(self.device), critic_obs.to(self.device), rewards.to(self.device), dones.to(self.device)
                        total_rew = self.alg.process_env_step(rewards, dones, infos)
                        
                        if self.log_dir is not None:
                            # Book keeping
                            if 'episode' in infos:
                                ep_infos.append(infos['episode'])
                            cur_reward_sum += total_rew
                            cur_reward_explr_sum += 0
                            cur_reward_entropy_sum += 0
                            cur_episode_length += 1
                            
                            new_ids = (dones > 0).nonzero(as_tuple=False)
                            
                            rewbuffer.extend(cur_reward_sum[new_ids][:, 0].cpu().numpy().tolist())
                            rew_explr_buffer.extend(cur_reward_explr_sum[new_ids][:, 0].cpu().numpy().tolist())
                            rew_entropy_buffer.extend(cur_reward_entropy_sum[new_ids][:, 0].cpu().numpy().tolist())
                            lenbuffer.extend(cur_episode_length[new_ids][:, 0].cpu().numpy().tolist())
                            
                            cur_reward_sum[new_ids] = 0
                            cur_reward_explr_sum[new_ids] = 0
                            cur_reward_entropy_sum[new_ids] = 0
                            cur_episode_length[new_ids] = 0

            with torch.inference_mode():
                stop = time.time()
                collection_time = stop - start - learn_time
            
                # Learning step
                start = stop
                self.alg.compute_returns(critic_obs)
            
            # From the collected rollout above, compute PPO update
        
            if self.if_depth:
                delta_yaw_ok_percentage = sum(delta_yaw_ok_buffer) / len(delta_yaw_ok_buffer)
                delta_next_yaw_ok_percentage = sum(delta_next_yaw_ok_buffer) / len(delta_next_yaw_ok_buffer)

                depth_encoder_loss = 0.
                
                if update_yaw_estimates and it > 0:    # This is updated every 5*5*5 = 125 steps (about 120 as in learn_vision(), which corresponds to 5*5=25 images)
                    print("Updating delta yaw estimates...")
                    yaw_buffer_student = torch.cat(yaw_buffer_student, dim=0)#.to(self.device)
                    yaw_buffer_teacher = torch.cat(yaw_buffer_teacher, dim=0)#.to(self.device)
                    self.alg.update_delta_yaw_estimates(yaw_buffer_student, yaw_buffer_teacher)
                    del yaw_buffer_student, yaw_buffer_teacher
            

            #del delta_yaw_ok, delta_next_yaw_ok
            #delta_yaw_grad = delta_yaw_grad.cpu()
            #delta_yaw = delta_yaw.cpu()
            #self.alg.depth_encoder.cpu()
            #torch.cuda.empty_cache()
            
            mean_value_loss, mean_surrogate_loss, mean_estimator_loss, mean_disc_loss, mean_disc_acc, mean_priv_reg_loss, priv_reg_coef = self.alg.update() # This is updated every 5*5 steps (about 24 as in learn_RL())            

            #delta_yaw_grad = delta_yaw_grad.cuda()
            #delta_yaw = delta_yaw.cuda()
            #self.alg.depth_encoder.cuda()
            
            if hist_encoding and it > 0:
                print("Updating DAgger...")
                mean_hist_latent_loss = self.alg.update_dagger()
            
            stop = time.time()
            learn_time += stop - start

            if self.if_depth:
                self.alg.depth_encoder.detach_hidden_states()
            
            if self.log_dir is not None:
                if self.if_depth:
                    self.log_vision(locals())
                else:
                    self.log(locals())
            if it % self.save_interval == 0:
                self.save(os.path.join(self.log_dir, 'model_{}.pt'.format(it)))
            ep_infos.clear()
        
        # self.current_learning_iteration += num_learning_iterations
        self.save(os.path.join(self.log_dir, 'model_{}.pt'.format(self.current_learning_iteration)))
        

    def x_learn_anymal(self, num_learning_iterations, init_at_random_ep_len=False):
        mean_value_loss = 0.
        mean_surrogate_loss = 0.
        mean_estimator_loss = 0.
        mean_disc_loss = 0.
        mean_disc_acc = 0.
        mean_hist_latent_loss = 0.
        mean_priv_reg_loss = 0. 
        priv_reg_coef = 0.
        entropy_coef = 0.

        if init_at_random_ep_len:
            self.env.episode_length_buf = torch.randint_like(self.env.episode_length_buf, high=int(self.env.max_episode_length))
        obs = self.env.get_observations()
        privileged_obs = self.env.get_privileged_observations()
        critic_obs = privileged_obs if privileged_obs is not None else obs
        obs, critic_obs = obs.to(self.device), critic_obs.to(self.device)
        infos = {}
        #infos["depth"] = self.env.depth_buffer.clone().to(self.device)[:, -1] if self.if_depth else None
        if self.if_depth:
            #delta_yaw_ok = torch.zeros(self.env.num_envs, dtype=torch.bool, device=self.device)
            #delta_next_yaw_ok = torch.zeros(self.env.num_envs, dtype=torch.bool, device=self.device)
            self.alg.depth_encoder.train()
            #self.alg.depth_actor.train()   # No depth_actor = student here

        
        self.alg.actor_critic.train() # switch to train mode (for dropout for example)

        ep_infos = []
        rewbuffer = deque(maxlen=100)
        rew_explr_buffer = deque(maxlen=100)
        rew_entropy_buffer = deque(maxlen=100)
        lenbuffer = deque(maxlen=100)
        cur_reward_sum = torch.zeros(self.env.num_envs, dtype=torch.float, device=self.device)
        cur_reward_explr_sum = torch.zeros(self.env.num_envs, dtype=torch.float, device=self.device)
        cur_reward_entropy_sum = torch.zeros(self.env.num_envs, dtype=torch.float, device=self.device)
        cur_episode_length = torch.zeros(self.env.num_envs, dtype=torch.float, device=self.device)

        tot_iter = self.current_learning_iteration + num_learning_iterations
        self.start_learning_iteration = copy(self.current_learning_iteration)

        assert self.depth_encoder_cfg["num_steps_per_env"] == 5
        assert self.num_steps_per_env == 5
        assert self.if_depth

        # This block is only for the edge case it == 0
        yaw_loss = 0
        delta_yaw_ok_buffer = []
        delta_next_yaw_ok_buffer = []
        yaw_buffer_student = []
        yaw_buffer_teacher = []
        ####

        for it in range(self.current_learning_iteration, tot_iter):
            learn_time = 0
            collection_time = 0
            start = time.time()

            hist_encoding = it % self.dagger_update_freq == 0
            if self.if_depth:
                update_yaw_estimates = it % self.yaw_estimates_update_freq == 0
                if it % self.yaw_estimates_update_freq == 1:    # accumulate through time before updating yaw estimates
                    yaw_loss = 0
                    delta_yaw_ok_buffer = []
                    delta_next_yaw_ok_buffer = []
                    yaw_buffer_student = []
                    yaw_buffer_teacher = []
                    #depth_latent_buffer = []       

            for j in range(self.depth_encoder_cfg["num_steps_per_env"]): # (= 5) new image every time
                if self.if_depth:
                    #assert infos["depth"] != None # it is None when self.global_counter % self.cfg.depth.update_interval != 0
                    #if infos["depth"] != None:
                    obs_prop_depth = obs[:, :self.env.cfg.env.n_proprio].clone()
                    obs_prop_depth[:, 6:8] = 0
                    scandots = obs[:, self.env.cfg.env.n_proprio:self.env.cfg.env.n_proprio + self.env.cfg.env.n_scan].clone()

                    start_learning = time.time()
                    delta_yaw_grad, delta_yaw = self.alg.compute_delta_yaw_estimates(scandots, obs_prop_depth)
                    stop_learning = time.time()
                    learn_time += stop_learning - start_learning

                    #depth_latent_and_yaw = self.alg.depth_encoder(infos["depth"].clone(), obs_prop_depth)  # clone is crucial to avoid in-place operation
                            
                    # The model now outputs only 2 values anyway: so [:, :-2] is basically the whole output
                    #depth_latent = depth_latent_and_yaw[:, :-2] #   Could use them as input in the model later on

                    #depth_latent_buffer.append(depth_latent)
                    yaw_buffer_student.append(delta_yaw_grad)#.cpu())
                    yaw_buffer_teacher.append(obs[:16, 6:8])
                        
                    # TEST WITHOUT DELTA_YAW_OK TO RECOVER PERFORMANCE
                    delta_yaw_ok = torch.abs(delta_yaw[:, 0] - obs[:, 6]) < 0.26
                    delta_next_yaw_ok = torch.abs(delta_yaw[:, 1] - obs[:, 7]) < 0.26

                    delta_yaw_ok_buffer.append(torch.nonzero(delta_yaw_ok).size(0) / delta_yaw_ok.numel())
                    delta_next_yaw_ok_buffer.append(torch.nonzero(delta_next_yaw_ok).size(0) / delta_next_yaw_ok.numel())
                # Rollout
                with torch.inference_mode():
                    for i in range(self.num_steps_per_env):     # takes num_steps_per_env (5) steps on each images
                        if self.if_depth:
                            #if i != 0: assert infos["depth"] == None   # should be None because the update happens above (depth image is updated each j-th iteration)
                            
                            # OBSERVATIONS
                            # change the observation everytime even if the image has not updated
                            obs[delta_yaw_ok, 6] = delta_yaw[delta_yaw_ok, 0].detach()
                            obs[delta_next_yaw_ok, 7] = delta_yaw[delta_next_yaw_ok, 1].detach()
                        
                        actions = self.alg.act(obs, critic_obs, infos, hist_encoding)  # Default: hist_encoding
                        obs, privileged_obs, rewards, dones, infos = self.env.step(actions)  # obs has changed to next_obs !! if done obs has been reset
                        critic_obs = privileged_obs if privileged_obs is not None else obs
                        obs, critic_obs, rewards, dones = obs.to(self.device), critic_obs.to(self.device), rewards.to(self.device), dones.to(self.device)
                        total_rew = self.alg.process_env_step(rewards, dones, infos)
                        
                        if self.log_dir is not None:
                            # Book keeping
                            if 'episode' in infos:
                                ep_infos.append(infos['episode'])
                            cur_reward_sum += total_rew
                            cur_reward_explr_sum += 0
                            cur_reward_entropy_sum += 0
                            cur_episode_length += 1
                            
                            new_ids = (dones > 0).nonzero(as_tuple=False)
                            
                            rewbuffer.extend(cur_reward_sum[new_ids][:, 0].cpu().numpy().tolist())
                            rew_explr_buffer.extend(cur_reward_explr_sum[new_ids][:, 0].cpu().numpy().tolist())
                            rew_entropy_buffer.extend(cur_reward_entropy_sum[new_ids][:, 0].cpu().numpy().tolist())
                            lenbuffer.extend(cur_episode_length[new_ids][:, 0].cpu().numpy().tolist())
                            
                            cur_reward_sum[new_ids] = 0
                            cur_reward_explr_sum[new_ids] = 0
                            cur_reward_entropy_sum[new_ids] = 0
                            cur_episode_length[new_ids] = 0

            with torch.inference_mode():
                stop = time.time()
                collection_time = stop - start - learn_time
            
                # Learning step
                start = stop
                self.alg.compute_returns(critic_obs)
            
            # From the collected rollout above, compute PPO update
        
            if self.if_depth:
                delta_yaw_ok_percentage = sum(delta_yaw_ok_buffer) / len(delta_yaw_ok_buffer)
                delta_next_yaw_ok_percentage = sum(delta_next_yaw_ok_buffer) / len(delta_next_yaw_ok_buffer)

                depth_encoder_loss = 0.
                
                if update_yaw_estimates and it > 0:    # This is updated every 5*5*5 = 125 steps (about 120 as in learn_vision(), which corresponds to 5*5=25 images)
                    print("Updating delta yaw estimates...")
                    yaw_buffer_student = torch.cat(yaw_buffer_student, dim=0)#.to(self.device)
                    yaw_buffer_teacher = torch.cat(yaw_buffer_teacher, dim=0)#.to(self.device)
                    self.alg.update_delta_yaw_estimates(yaw_buffer_student, yaw_buffer_teacher)
                    del yaw_buffer_student, yaw_buffer_teacher
            

            #del delta_yaw_ok, delta_next_yaw_ok
            #delta_yaw_grad = delta_yaw_grad.cpu()
            #delta_yaw = delta_yaw.cpu()
            #self.alg.depth_encoder.cpu()
            #torch.cuda.empty_cache()
            
            mean_value_loss, mean_surrogate_loss, mean_estimator_loss, mean_disc_loss, mean_disc_acc, mean_priv_reg_loss, priv_reg_coef = self.alg.update() # This is updated every 5*5 steps (about 24 as in learn_RL())            

            #delta_yaw_grad = delta_yaw_grad.cuda()
            #delta_yaw = delta_yaw.cuda()
            #self.alg.depth_encoder.cuda()
            
            if hist_encoding and it > 0:
                print("Updating DAgger...")
                mean_hist_latent_loss = self.alg.update_dagger()
            
            stop = time.time()
            learn_time += stop - start

            if self.if_depth:
                self.alg.depth_encoder.detach_hidden_states()
            
            if self.log_dir is not None:
                if self.if_depth:
                    self.log_vision(locals())
                else:
                    self.log(locals())
            if it % self.save_interval == 0:
                self.save(os.path.join(self.log_dir, 'model_{}.pt'.format(it)))
            ep_infos.clear()
        
        # self.current_learning_iteration += num_learning_iterations
        self.save(os.path.join(self.log_dir, 'model_{}.pt'.format(self.current_learning_iteration)))
         
    
    def learn_RL(self, num_learning_iterations, init_at_random_ep_len=False):
        mean_value_loss = 0.
        mean_surrogate_loss = 0.
        mean_estimator_loss = 0.
        mean_disc_loss = 0.
        mean_disc_acc = 0.
        mean_hist_latent_loss = 0.
        mean_priv_reg_loss = 0. 
        priv_reg_coef = 0.
        entropy_coef = 0.
        # initialize writer
        # if self.log_dir is not None and self.writer is None:
        #     self.writer = SummaryWriter(log_dir=self.log_dir, flush_secs=10)
        if init_at_random_ep_len:
            self.env.episode_length_buf = torch.randint_like(self.env.episode_length_buf, high=int(self.env.max_episode_length))
        obs = self.env.get_observations()
        privileged_obs = self.env.get_privileged_observations()
        critic_obs = privileged_obs if privileged_obs is not None else obs
        obs, critic_obs = obs.to(self.device), critic_obs.to(self.device)
        infos = {}
        infos["depth"] = self.env.depth_buffer.clone().to(self.device) if self.if_depth else None   # I think this never happens becuase if self.depth is true then the function used is learn_vision() and not learn_RL()
        self.alg.actor_critic.train() # switch to train mode (for dropout for example)

        ep_infos = []
        rewbuffer = deque(maxlen=100)
        rew_explr_buffer = deque(maxlen=100)
        rew_entropy_buffer = deque(maxlen=100)
        lenbuffer = deque(maxlen=100)
        cur_reward_sum = torch.zeros(self.env.num_envs, dtype=torch.float, device=self.device)
        cur_reward_explr_sum = torch.zeros(self.env.num_envs, dtype=torch.float, device=self.device)
        cur_reward_entropy_sum = torch.zeros(self.env.num_envs, dtype=torch.float, device=self.device)
        cur_episode_length = torch.zeros(self.env.num_envs, dtype=torch.float, device=self.device)

        tot_iter = self.current_learning_iteration + num_learning_iterations
        self.start_learning_iteration = copy(self.current_learning_iteration)

        assert self.depth_encoder_cfg["num_steps_per_env"] == 5
        assert self.num_steps_per_env == 24
        assert not self.if_depth

        for it in range(self.current_learning_iteration, tot_iter):
            start = time.time()
            hist_encoding = it % self.dagger_update_freq == 0

            # Rollout
            with torch.inference_mode():
                for i in range(self.num_steps_per_env):
                    actions = self.alg.act(obs, critic_obs, infos, hist_encoding)
                    obs, privileged_obs, rewards, dones, infos = self.env.step(actions)  # obs has changed to next_obs !! if done obs has been reset
                    critic_obs = privileged_obs if privileged_obs is not None else obs
                    obs, critic_obs, rewards, dones = obs.to(self.device), critic_obs.to(self.device), rewards.to(self.device), dones.to(self.device)
                    total_rew = self.alg.process_env_step(rewards, dones, infos)
                    
                    if self.log_dir is not None:
                        # Book keeping
                        if 'episode' in infos:
                            ep_infos.append(infos['episode'])
                        cur_reward_sum += total_rew
                        cur_reward_explr_sum += 0
                        cur_reward_entropy_sum += 0
                        cur_episode_length += 1
                        
                        new_ids = (dones > 0).nonzero(as_tuple=False)
                        
                        rewbuffer.extend(cur_reward_sum[new_ids][:, 0].cpu().numpy().tolist())
                        rew_explr_buffer.extend(cur_reward_explr_sum[new_ids][:, 0].cpu().numpy().tolist())
                        rew_entropy_buffer.extend(cur_reward_entropy_sum[new_ids][:, 0].cpu().numpy().tolist())
                        lenbuffer.extdnd(cur_episode_length[new_ids][:, 0].cpu().numpy().tolist())
                        
                        cur_reward_sum[new_ids] = 0
                        cur_reward_explr_sum[new_ids] = 0
                        cur_reward_entropy_sum[new_ids] = 0
                        cur_episode_length[new_ids] = 0

                stop = time.time()
                collection_time = stop - start

                # Learning step
                start = stop
                self.alg.compute_returns(critic_obs)
            
            mean_value_loss, mean_surrogate_loss, mean_estimator_loss, mean_disc_loss, mean_disc_acc, mean_priv_reg_loss, priv_reg_coef = self.alg.update()
            if hist_encoding:
                print("Updating dagger...")
                mean_hist_latent_loss = self.alg.update_dagger()
            
            stop = time.time()
            learn_time = stop - start
            if self.log_dir is not None:
                self.log(locals())
            if it % self.save_interval == 0:
                self.save(os.path.join(self.log_dir, 'model_{}.pt'.format(it)))
            ep_infos.clear()
        
        # self.current_learning_iteration += num_learning_iterations
        self.save(os.path.join(self.log_dir, 'model_{}.pt'.format(self.current_learning_iteration)))

    def learn_vision(self, num_learning_iterations, init_at_random_ep_len=False):
        tot_iter = self.current_learning_iteration + num_learning_iterations
        self.start_learning_iteration = copy(self.current_learning_iteration)

        ep_infos = []
        rewbuffer = deque(maxlen=100)
        lenbuffer = deque(maxlen=100)
        cur_reward_sum = torch.zeros(self.env.num_envs, dtype=torch.float, device=self.device)
        cur_episode_length = torch.zeros(self.env.num_envs, dtype=torch.float, device=self.device)

        obs = self.env.get_observations()
        infos = {}
        infos["depth"] = self.env.depth_buffer.clone().to(self.device)[:, -1] if self.if_depth else None
        infos["delta_yaw_ok"] = torch.zeros(self.env.num_envs, dtype=torch.bool, device=self.device)
        infos["delta_next_yaw_ok"] = torch.zeros(self.env.num_envs, dtype=torch.bool, device=self.device)
        self.alg.depth_encoder.train()
        self.alg.depth_actor.train()

        num_pretrain_iter = 0
        for it in range(self.current_learning_iteration, tot_iter):
            start = time.time()
            depth_latent_buffer = []
            scandots_latent_buffer = []
            actions_teacher_buffer = []
            actions_student_buffer = []
            #yaw_buffer_student = []
            #yaw_buffer_teacher = []
            delta_yaw_ok_buffer = []
            delta_next_yaw_ok_buffer = []
            for i in range(self.depth_encoder_cfg["num_steps_per_env"]):
                if infos["depth"] != None: # it is None when self.global_counter % self.cfg.depth.update_interval != 0
                    with torch.no_grad():
                        scandots_latent = self.alg.actor_critic.actor.infer_scandots_latent(obs)
                    scandots_latent_buffer.append(scandots_latent)
                    obs_prop_depth = obs[:, :self.env.cfg.env.n_proprio].clone()
                    obs_prop_depth[:, 6:8] = 0
                    depth_latent_and_yaw = self.alg.depth_encoder(infos["depth"].clone(), obs_prop_depth)  # clone is crucial to avoid in-place operation
                    
                    depth_latent = depth_latent_and_yaw[:, :-2]
                    yaw = 1.5*depth_latent_and_yaw[:, -2:]  # why is this 1.5 there..? Also it's called yaw but it's actually delta_yaw..
                    
                    depth_latent_buffer.append(depth_latent)
                    #yaw_buffer_student.append(yaw)
                    #yaw_buffer_teacher.append(obs[:, 6:8])
                
                # COMPUTE DELTA_YAW_OK HERE: outside the if() because infos["depth"] is None when self.global_counter % self.cfg.depth.update_interval != 0
                infos["delta_yaw_ok"] = torch.abs(yaw[:, 0] - obs[:, 6]) < 0.26
                infos["delta_next_yaw_ok"] = torch.abs(yaw[:, 1] - obs[:, 7]) < 0.26
               
                with torch.no_grad():
                    actions_teacher = self.alg.actor_critic.act_inference(obs, hist_encoding=True, scandots_latent=None)
                    actions_teacher_buffer.append(actions_teacher)

                obs_student = obs.clone()
                # obs_student[:, 6:8] = yaw.detach()
                obs_student[infos["delta_yaw_ok"], 6] = yaw[infos["delta_yaw_ok"], 0].detach() # I think, during the first iteration, infos["delta_yaw_ok"] are allllll true (because the buffer is updated only after step()), which might be bad
                obs_student[infos["delta_next_yaw_ok"], 7] = yaw[infos["delta_next_yaw_ok"], 1].detach()
                delta_yaw_ok_buffer.append(torch.nonzero(infos["delta_yaw_ok"]).size(0) / infos["delta_yaw_ok"].numel())
                delta_next_yaw_ok_buffer.append(torch.nonzero(infos["delta_next_yaw_ok"]).size(0) / infos["delta_next_yaw_ok"].numel())
                actions_student = self.alg.depth_actor(obs_student, hist_encoding=True, scandots_latent=depth_latent)
                actions_student_buffer.append(actions_student)

                # detach actions before feeding the env
                if it < num_pretrain_iter:  # always false because num_pretrain_it is set to 0 above
                    obs, privileged_obs, rewards, dones, infos = self.env.step(actions_teacher.detach())  # obs has changed to next_obs !! if done obs has been reset
                else:   # Always use student actions
                    obs, privileged_obs, rewards, dones, infos = self.env.step(actions_student.detach())  # obs has changed to next_obs !! if done obs has been reset
                critic_obs = privileged_obs if privileged_obs is not None else obs
                obs, critic_obs, rewards, dones = obs.to(self.device), critic_obs.to(self.device), rewards.to(self.device), dones.to(self.device)

                if self.log_dir is not None:
                        # Book keeping
                        if 'episode' in infos:
                            ep_infos.append(infos['episode'])
                        cur_reward_sum += rewards
                        cur_episode_length += 1
                        new_ids = (dones > 0).nonzero(as_tuple=False)
                        rewbuffer.extend(cur_reward_sum[new_ids][:, 0].cpu().numpy().tolist())
                        lenbuffer.extend(cur_episode_length[new_ids][:, 0].cpu().numpy().tolist())
                        cur_reward_sum[new_ids] = 0
                        cur_episode_length[new_ids] = 0
                
            stop = time.time()
            collection_time = stop - start
            start = stop

            delta_yaw_ok_percentage = sum(delta_yaw_ok_buffer) / len(delta_yaw_ok_buffer)
            delta_next_yaw_ok_percentage = sum(delta_next_yaw_ok_buffer) / len(delta_next_yaw_ok_buffer)
            scandots_latent_buffer = torch.cat(scandots_latent_buffer, dim=0)
            depth_latent_buffer = torch.cat(depth_latent_buffer, dim=0)
            depth_encoder_loss = 0
            # depth_encoder_loss = self.alg.update_depth_encoder(depth_latent_buffer, scandots_latent_buffer)

            actions_teacher_buffer = torch.cat(actions_teacher_buffer, dim=0)
            actions_student_buffer = torch.cat(actions_student_buffer, dim=0)
            yaw_buffer_student = torch.cat(yaw_buffer_student, dim=0)
            yaw_buffer_teacher = torch.cat(yaw_buffer_teacher, dim=0)
            depth_actor_loss, yaw_loss = self.alg.update_depth_actor(actions_student_buffer, actions_teacher_buffer, yaw_buffer_student, yaw_buffer_teacher)    # this happens every 24 images (5*24 steps in the main loop)

            # depth_encoder_loss, depth_actor_loss = self.alg.update_depth_both(depth_latent_buffer, scandots_latent_buffer, actions_student_buffer, actions_teacher_buffer)
            stop = time.time()
            learn_time = stop - start

            self.alg.depth_encoder.detach_hidden_states()

            if self.log_dir is not None:
                self.log_vision(locals())
            if it % self.save_interval == 0:
               self.save(os.path.join(self.log_dir, 'model_{}.pt'.format(it)))
            ep_infos.clear()
    
    def log_vision(self, locs, width=80, pad=35):
        self.tot_timesteps += self.num_steps_per_env * self.env.num_envs
        self.tot_time += locs['collection_time'] + locs['learn_time']
        iteration_time = locs['collection_time'] + locs['learn_time']

        ep_string = f''
        wandb_dict = {}
        if locs['ep_infos']:
            for key in locs['ep_infos'][0]:
                infotensor = torch.tensor([], device=self.device)
                for ep_info in locs['ep_infos']:
                    # handle scalar and zero dimensional tensor infos
                    if not isinstance(ep_info[key], torch.Tensor):
                        ep_info[key] = torch.Tensor([ep_info[key]])
                    if len(ep_info[key].shape) == 0:
                        ep_info[key] = ep_info[key].unsqueeze(0)
                    infotensor = torch.cat((infotensor, ep_info[key].to(self.device)))
                value = torch.mean(infotensor)
                wandb_dict['Episode_rew/' + key] = value
                ep_string += f"""{f'Mean episode {key}:':>{pad}} {value:.4f}\n"""
        mean_std = self.alg.actor_critic.std.mean()
        fps = int(self.num_steps_per_env * self.env.num_envs / (locs['collection_time'] + locs['learn_time']))

        wandb_dict['Loss_depth/delta_yaw_ok_percent'] = locs['delta_yaw_ok_percentage']
        wandb_dict['Loss_depth/delta_next_yaw_ok_percent'] = locs['delta_next_yaw_ok_percentage']
        wandb_dict['Loss_depth/depth_encoder'] = locs['depth_encoder_loss']
        #wandb_dict['Loss_depth/depth_actor'] = locs['depth_actor_loss']
        wandb_dict['Loss_depth/yaw'] = locs['yaw_loss']
        wandb_dict['Policy/mean_noise_std'] = mean_std.item()
        wandb_dict['Perf/total_fps'] = fps
        wandb_dict['Perf/collection time'] = locs['collection_time']
        wandb_dict['Perf/learning_time'] = locs['learn_time']
        if len(locs['rewbuffer']) > 0:
            wandb_dict['Train/mean_reward'] = statistics.mean(locs['rewbuffer'])
            wandb_dict['Train/mean_episode_length'] = statistics.mean(locs['lenbuffer'])
        
        wandb.log(wandb_dict, step=locs['it'])

        str = f" \033[1m Learning iteration {locs['it']}/{self.current_learning_iteration + locs['num_learning_iterations']} \033[0m "

        if len(locs['rewbuffer']) > 0:
            log_string = (f"""{'#' * width}\n"""
                          f"""{str.center(width, ' ')}\n\n"""
                          f"""{'Computation:':>{pad}} {fps:.0f} steps/s (collection: {locs[
                            'collection_time']:.3f}s, learning {locs['learn_time']:.3f}s)\n"""
                          f"""{'Mean action noise std:':>{pad}} {mean_std.item():.2f}\n"""
                          f"""{'Mean reward (total):':>{pad}} {statistics.mean(locs['rewbuffer']):.2f}\n"""
                          f"""{'Mean episode length:':>{pad}} {statistics.mean(locs['lenbuffer']):.2f}\n"""
                          f"""{'Depth encoder loss:':>{pad}} {locs['depth_encoder_loss']:.4f}\n"""
                          #f"""{'Depth actor loss:':>{pad}} {locs['depth_actor_loss']:.4f}\n"""
                          f"""{'Yaw loss:':>{pad}} {locs['yaw_loss']:.4f}\n"""
                          f"""{'Delta yaw ok percentage:':>{pad}} {locs['delta_yaw_ok_percentage']:.4f}\n"""
                          f"""{'Delta next yaw ok percentage:':>{pad}} {locs['delta_next_yaw_ok_percentage']:.4f}\n""")
        else:
            log_string = (f"""{'#' * width}\n""")

        log_string += f"""{'-' * width}\n"""
        log_string += ep_string
        curr_it = locs['it'] - self.start_learning_iteration
        eta = self.tot_time / (curr_it + 1) * (locs['num_learning_iterations'] - curr_it)
        mins = eta // 60
        secs = eta % 60
        log_string += (f"""{'-' * width}\n"""
                       f"""{'Total timesteps:':>{pad}} {self.tot_timesteps}\n"""
                       f"""{'Iteration time:':>{pad}} {iteration_time:.2f}s\n"""
                       f"""{'Total time:':>{pad}} {self.tot_time:.2f}s\n"""
                       f"""{'ETA:':>{pad}} {mins:.0f} mins {secs:.1f} s\n""")
        print(log_string)

    def log(self, locs, width=80, pad=35):
        self.tot_timesteps += self.num_steps_per_env * self.env.num_envs
        self.tot_time += locs['collection_time'] + locs['learn_time']
        iteration_time = locs['collection_time'] + locs['learn_time']

        ep_string = f''
        wandb_dict = {}
        if locs['ep_infos']:
            for key in locs['ep_infos'][0]:
                infotensor = torch.tensor([], device=self.device)
                for ep_info in locs['ep_infos']:
                    # handle scalar and zero dimensional tensor infos
                    if not isinstance(ep_info[key], torch.Tensor):
                        ep_info[key] = torch.Tensor([ep_info[key]])
                    if len(ep_info[key].shape) == 0:
                        ep_info[key] = ep_info[key].unsqueeze(0)
                    infotensor = torch.cat((infotensor, ep_info[key].to(self.device)))
                value = torch.mean(infotensor)
                wandb_dict['Episode_rew/' + key] = value
                ep_string += f"""{f'Mean episode {key}:':>{pad}} {value:.4f}\n"""
        mean_std = self.alg.actor_critic.std.mean()
        fps = int(self.num_steps_per_env * self.env.num_envs / (locs['collection_time'] + locs['learn_time']))

        wandb_dict['Loss/value_function'] = ['mean_value_loss']
        wandb_dict['Loss/surrogate'] = locs['mean_surrogate_loss']
        wandb_dict['Loss/estimator'] = locs['mean_estimator_loss']
        wandb_dict['Loss/hist_latent_loss'] = locs['mean_hist_latent_loss']
        wandb_dict['Loss/priv_reg_loss'] = locs['mean_priv_reg_loss']
        wandb_dict['Loss/priv_ref_lambda'] = locs['priv_reg_coef']
        wandb_dict['Loss/entropy_coef'] = locs['entropy_coef']
        wandb_dict['Loss/learning_rate'] = self.alg.learning_rate
        wandb_dict['Loss/discriminator'] = locs['mean_disc_loss']
        wandb_dict['Loss/discriminator_accuracy'] = locs['mean_disc_acc']

        wandb_dict['Policy/mean_noise_std'] = mean_std.item()
        wandb_dict['Perf/total_fps'] = fps
        wandb_dict['Perf/collection time'] = locs['collection_time']
        wandb_dict['Perf/learning_time'] = locs['learn_time']
        if len(locs['rewbuffer']) > 0:
            wandb_dict['Train/mean_reward'] = statistics.mean(locs['rewbuffer'])
            wandb_dict['Train/mean_reward_explr'] = statistics.mean(locs['rew_explr_buffer'])
            wandb_dict['Train/mean_reward_task'] = wandb_dict['Train/mean_reward'] - wandb_dict['Train/mean_reward_explr']
            wandb_dict['Train/mean_reward_entropy'] = statistics.mean(locs['rew_entropy_buffer'])
            wandb_dict['Train/mean_episode_length'] = statistics.mean(locs['lenbuffer'])
            # wandb_dict['Train/mean_reward/time', statistics.mean(locs['rewbuffer']), self.tot_time)
            # wandb_dict['Train/mean_episode_length/time', statistics.mean(locs['lenbuffer']), self.tot_time)

        wandb.log(wandb_dict, step=locs['it'])

        str = f" \033[1m Learning iteration {locs['it']}/{self.current_learning_iteration + locs['num_learning_iterations']} \033[0m "

        if len(locs['rewbuffer']) > 0:
            log_string = (f"""{'#' * width}\n"""
                          f"""{str.center(width, ' ')}\n\n"""
                          f"""{'Computation:':>{pad}} {fps:.0f} steps/s (collection: {locs[
                            'collection_time']:.3f}s, learning {locs['learn_time']:.3f}s)\n"""
                          f"""{'Value function loss:':>{pad}} {locs['mean_value_loss']:.4f}\n"""
                          f"""{'Surrogate loss:':>{pad}} {locs['mean_surrogate_loss']:.4f}\n"""
                          f"""{'Discriminator loss:':>{pad}} {locs['mean_disc_loss']:.4f}\n"""
                          f"""{'Discriminator accuracy:':>{pad}} {locs['mean_disc_acc']:.4f}\n"""
                          f"""{'Mean action noise std:':>{pad}} {mean_std.item():.2f}\n"""
                          f"""{'Mean reward (total):':>{pad}} {statistics.mean(locs['rewbuffer']):.2f}\n"""
                          f"""{'Mean reward (task):':>{pad}} {statistics.mean(locs['rewbuffer']) - statistics.mean(locs['rew_explr_buffer']):.2f}\n"""
                          f"""{'Mean reward (exploration):':>{pad}} {statistics.mean(locs['rew_explr_buffer']):.2f}\n"""
                          f"""{'Mean reward (entropy):':>{pad}} {statistics.mean(locs['rew_entropy_buffer']):.2f}\n"""
                          f"""{'Mean episode length:':>{pad}} {statistics.mean(locs['lenbuffer']):.2f}\n""")
                        #   f"""{'Mean reward/step:':>{pad}} {locs['mean_reward']:.2f}\n"""
                        #   f"""{'Mean episode length/episode:':>{pad}} {locs['mean_trajectory_length']:.2f}\n""")
        else:
            log_string = (f"""{'#' * width}\n"""
                          f"""{str.center(width, ' ')}\n\n"""
                          f"""{'Computation:':>{pad}} {fps:.0f} steps/s (collection: {locs[
                            'collection_time']:.3f}s, learning {locs['learn_time']:.3f}s)\n"""
                          f"""{'Value function loss:':>{pad}} {locs['mean_value_loss']:.4f}\n"""
                          f"""{'Surrogate loss:':>{pad}} {locs['mean_surrogate_loss']:.4f}\n"""
                          f"""{'Estimator loss:':>{pad}} {locs['mean_estimator_loss']:.4f}\n"""
                          f"""{'Mean action noise std:':>{pad}} {mean_std.item():.2f}\n""")
                        #   f"""{'Mean reward/step:':>{pad}} {locs['mean_reward']:.2f}\n"""
                        #   f"""{'Mean episode length/episode:':>{pad}} {locs['mean_trajectory_length']:.2f}\n""")

        log_string += f"""{'-' * width}\n"""
        log_string += ep_string
        curr_it = locs['it'] - self.start_learning_iteration
        eta = self.tot_time / (curr_it + 1) * (locs['num_learning_iterations'] - curr_it)
        mins = eta // 60
        secs = eta % 60
        log_string += (f"""{'-' * width}\n"""
                       f"""{'Total timesteps:':>{pad}} {self.tot_timesteps}\n"""
                       f"""{'Iteration time:':>{pad}} {iteration_time:.2f}s\n"""
                       f"""{'Total time:':>{pad}} {self.tot_time:.2f}s\n"""
                       f"""{'ETA:':>{pad}} {mins:.0f} mins {secs:.1f} s\n""")
        print(log_string)

    def save(self, path, infos=None):
        state_dict = {
            'model_state_dict': self.alg.actor_critic.state_dict(),
            'estimator_state_dict': self.alg.estimator.state_dict(),
            'optimizer_state_dict': self.alg.optimizer.state_dict(),
            'iter': self.current_learning_iteration,
            'infos': infos,
            }
        if self.if_depth:
            state_dict['depth_encoder_state_dict'] = self.alg.depth_encoder.state_dict()
            # No need for that now
            #state_dict['depth_actor_state_dict'] = self.alg.depth_actor.state_dict()
        torch.save(state_dict, path)

    def load(self, path, load_optimizer=True):
        print("*" * 80)
        print("Loading model from {}...".format(path))
        loaded_dict = torch.load(path, map_location=self.device)
        self.alg.actor_critic.load_state_dict(loaded_dict['model_state_dict'])
        self.alg.estimator.load_state_dict(loaded_dict['estimator_state_dict'])
        # TODO: pretrain with camera and load depth_*encoder* rather than depth_actor (which doesn't exist anymore)
        if self.if_depth:
            if 'depth_encoder_state_dict' not in loaded_dict:
                warnings.warn("'depth_encoder_state_dict' key does not exist, not loading depth encoder...")
            else:
                print("Saved depth encoder detected, loading...")
                self.alg.depth_encoder.load_state_dict(loaded_dict['depth_encoder_state_dict'])
        # No depth_actor anymore
        #    if 'depth_actor_state_dict' in loaded_dict:
        #        print("Saved depth actor detected, loading...")
        #        self.alg.depth_actor.load_state_dict(loaded_dict['depth_actor_state_dict'])
        #    else:
        #        print("No saved depth actor, Copying actor critic actor to depth actor...")
        #        self.alg.depth_actor.load_state_dict(self.alg.actor_critic.actor.state_dict())
        if load_optimizer:
            self.alg.optimizer.load_state_dict(loaded_dict['optimizer_state_dict'])
        # self.current_learning_iteration = loaded_dict['iter']
        print("*" * 80)
        return loaded_dict['infos']

    def get_inference_policy(self, device=None):
        self.alg.actor_critic.eval() # switch to evaluation mode (dropout for example)
        if device is not None:
            self.alg.actor_critic.to(device)
        return self.alg.actor_critic.act_inference
    
    def get_depth_actor_inference_policy(self, device=None):
        self.alg.depth_actor.eval() # switch to evaluation mode (dropout for example)
        if device is not None:
            self.alg.depth_actor.to(device)
        return self.alg.depth_actor
    
    def get_actor_critic(self, device=None):
        self.alg.actor_critic.eval() # switch to evaluation mode (dropout for example)
        if device is not None:
            self.alg.actor_critic.to(device)
        return self.alg.actor_critic
    
    def get_estimator_inference_policy(self, device=None):
        self.alg.estimator.eval() # switch to evaluation mode (dropout for example)
        if device is not None:
            self.alg.estimator.to(device)
        return self.alg.estimator.inference

    def get_depth_encoder_inference_policy(self, device=None):
        self.alg.depth_encoder.eval()
        if device is not None:
            self.alg.depth_encoder.to(device)
        return self.alg.depth_encoder
    
    def get_disc_inference_policy(self, device=None):
        self.alg.discriminator.eval() # switch to evaluation mode (dropout for example)
        if device is not None:
            self.alg.discriminator.to(device)
        return self.alg.discriminator.inference
