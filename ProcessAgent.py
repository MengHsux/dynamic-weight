# Copyright (c) 2016, NVIDIA CORPORATION. All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions
# are met:
#  * Redistributions of source code must retain the above copyright
#    notice, this list of conditions and the following disclaimer.
#  * Redistributions in binary form must reproduce the above copyright
#    notice, this list of conditions and the following disclaimer in the
#    documentation and/or other materials provided with the distribution.
#  * Neither the name of NVIDIA CORPORATION nor the names of its
#    contributors may be used to endorse or promote products derived
#    from this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
# EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
# PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
# CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
# EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
# PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
# PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
# OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
# (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

from datetime import datetime
from multiprocessing import Process, Queue, Value

import numpy as np
import time
import csv
import random

from Config import Config
from Environment import Environment
from Experience import Experience


class ProcessAgent(Process):
    def __init__(self, id, prediction_q, training_q, episode_log_q):
        super(ProcessAgent, self).__init__()

        self.id = id
        self.prediction_q = prediction_q
        self.training_q = training_q
        self.episode_log_q = episode_log_q

        self.env = Environment()
        self.num_actions = self.env.get_num_actions()
        self.actions = np.arange(self.num_actions)
        self.weight = np.arange(2)
        self.gate = np.zeros(2).astype(np.float32)

        self.discount_factor = Config.DISCOUNT
        # one frame at a time
        self.wait_q = Queue(maxsize=1)
        self.exit_flag = Value('i', 0)
        # sampling rate
        self.kexi = Config.KEXI_START
        self.kexi1 = Config.KEXI_START


    def _accumulate_rewards(self, experiences, discount_factor, terminal_reward, terminal_reward_hit, terminal_reward_big):
        reward_sum = terminal_reward
        reward_sum_hit = terminal_reward_hit
        reward_sum_big = terminal_reward_big
        for t in reversed(range(0, len(experiences)-1)):
            r = np.clip(experiences[t].reward, Config.REWARD_MIN, Config.REWARD_MAX)
            reward_sum = discount_factor * reward_sum + r
            experiences[t].reward = reward_sum

            r_hit = np.clip(experiences[t].reward_hit_ball, Config.REWARD_MIN, Config.REWARD_MAX)
            r_big = np.clip(experiences[t].reward_big_angle, Config.REWARD_MIN, Config.REWARD_MAX)
            reward_sum_hit = discount_factor * reward_sum_hit + r_hit
            reward_sum_big = discount_factor * reward_sum_big + r_big

            self.gate[experiences[t].gate] = 1
            experiences[t].reward_hit_ball = self.gate[0] * (reward_sum_hit - (self.kexi*0.6122365479)) / (self.kexi*1.4315440129)
            experiences[t].reward_big_angle = self.gate[1] * (reward_sum_big - (self.kexi1*0.1634638122)) / (self.kexi1*0.2670975099)
            self.gate[experiences[t].gate] = 0
        return experiences[:-1]

    def convert_data(self, experiences):
        x_ = np.array([exp.state for exp in experiences])
        r_ = np.array([exp.reward for exp in experiences])
        a_ = np.eye(self.num_actions)[np.array([exp.action for exp in experiences])].astype(np.float32)
        gate_ = np.eye(2)[np.array([exp.gate for exp in experiences])].astype(np.float32)
        A_hit = np.array([exp.reward_hit_ball for exp in experiences])
        A_big = np.array([exp.reward_big_angle for exp in experiences])
        return x_, r_, a_, gate_, A_hit, A_big

    def predict(self, state):
        # put the state in the prediction q
        self.prediction_q.put((self.id, state))
        # wait for the prediction to come back
        p, gate_p, v, v_hit, v_big = self.wait_q.get()
        return p, gate_p, v, v_hit, v_big

    def select_gate(self, gate_p):
        if Config.PLAY_MODE:
            # action = np.random.choice(self.actions, p=prediction)
            gate = np.argmax(gate_p)
        else:
            gate = np.random.choice(self.weight, p=gate_p)
        return gate

    def select_action(self, prediction):
        if Config.PLAY_MODE:
            # action = np.random.choice(self.actions, p=prediction)
            action = np.argmax(prediction)
        else:
            action = np.random.choice(self.actions, p=prediction)
        return action

    def run_episode(self):
        self.env.reset()
        done = False
        experiences = []

        time_count = 0
        reward_sum = 0.0

        hit_ball_flag = 0
        ball_x1 = 0
        ball_y1 = 0

        while not done:
            # very first few frames
            if self.env.current_state is None:
                self.env.step(0, hit_ball_flag, ball_x1, ball_y1)  # 0 == NOOP
                continue

            prediction, gate_p, value, v_hit, v_big = self.predict(self.env.current_state)
            action = self.select_action(prediction)
            gate = self.select_gate(gate_p)
            reward, reward_hit_ball, reward_big_angle, done, hit_ball_flag, ball_x1, ball_y1 = self.env.step(action, hit_ball_flag, ball_x1, ball_y1)
            reward_sum += reward

            exp = Experience(self.env.previous_state, action, prediction, gate, gate_p, reward, reward_hit_ball, reward_big_angle, v_hit, v_big, done)
            experiences.append(exp)

            if done or time_count == Config.TIME_MAX:
                terminal_reward = 0 if done else value

                terminal_reward_hit = 0 if done else v_hit
                terminal_reward_big = 0 if done else v_big

                updated_exps = self._accumulate_rewards(experiences, self.discount_factor, terminal_reward, terminal_reward_hit, terminal_reward_big)
                x_, r_, a_, gate_, A_hit, A_big = self.convert_data(updated_exps)
                yield x_, r_, a_, gate_, A_hit, A_big, reward_sum

                # reset the tmax count
                time_count = 0
                # keep the last experience for the next batch
                experiences = [experiences[-1]]
                reward_sum = 0.0

            time_count += 1

    def run(self):
        # randomly sleep up to 1 second. helps agents boot smoothly.
        time.sleep(np.random.rand())
        np.random.seed(np.int32(time.time() % 1 * 1000 + self.id * 10))

        while self.exit_flag.value == 0:
            total_reward = 0
            total_length = 0
            for x_, r_, a_, gate_, A_hit, A_big, reward_sum in self.run_episode():
                total_reward += reward_sum
                total_length += len(a_) + 1  # +1 for last frame that we drop
                self.training_q.put((x_, r_, a_, gate_, A_hit, A_big))
            self.episode_log_q.put((datetime.now(), total_reward, total_length))