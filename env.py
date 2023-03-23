# imports
import gym
import logging
import math
import numpy as np
import matplotlib.pyplot as plt
import cv2
import scipy.signal
import torch
import skimage.exposure

from gym import error, spaces, utils
from gym.utils import seeding
from scipy.fft import fft2, ifft2
from PIL import Image, ImageFilter
from numpy.fft import fft, fftshift, ifft, ifftshift
from math import sin, cos
from numpy.random import default_rng
from env_aux import *

# Choose the MODE (2D removes the z actions)
mode_mono_d = False
mode_xz_yz_actions = True
mode_2D = False  # False
easy_diff_map = False  # False
map_type = 'blobs' #'original'  # 'lines', 'blobs', 'boxes'


class Env():

    def __init__(self, args):
        self.__version__ = "1.0.1"
        self.device = args.device
        self.map_type = map_type
        # Hyperparameter definition
        ## Field of  134 x 240 m
        # Observations
        self.w_field, self.h_field = 84, 84  # 01/12/2022 - assert error size
        self.x_min, self.y_min = float(0), float(0)
        self.x_max, self.y_max = float(self.w_field), float(self.h_field)  # [m]
        self.min_height, self.max_height = float(2 + 4), float(20 + 4)  # [m/s]
        # Actions
        '''
        self.min_displacement_xy, self.max_displacement_xy = -4.0, 4.0  # [m/s]
        self.min_displacement_h, self.max_displacement_h = -4.0, 4.0  # [m/s]
        '''
        self.min_displacement_xy, self.max_displacement_xy = -1.0, 1.0  # [m/s]
        self.min_displacement_h, self.max_displacement_h = -1.0, 1.0  # [m/s]
        self.denormalize_action_factor = 4.0  # so max/min speed is -4 to 4
        ## For the Agent
        self.max_step = 4000 # 2000 #4000  # 2000

        # Some initialization
        self.max_reach_reward = -1
        self.out_of_range = 0
        self.lower_height_achieved = 0

        self.time_punishment = 2  # 50 # 2 # 50 # 2 # 1 # 0.5# 0.1
        self.range_punishment = 10
        self.lower_height_punishment_decrease = 0.8  # decrease time_punishment by x% given a lower height

        self.reward_list = []
        self.reward_list.append(0)
        self.cv2_save_render = 1 # 0
        self.cv2_show_render = True
        self.cv2_save_number = 1

        # TODO: what is the observation space for x,y,h,last_r (reward? -inf to inf?)
        self.observation_space = spaces.Box(
            low=np.array([self.y_min, self.x_min, self.min_height, -self.time_punishment - self.range_punishment]),
            high=np.array([self.y_max, self.x_max, self.max_height, float('inf')]),
            # TODO: fov size at the maximum height as the reward
            dtype=np.float32)  # 30/01/2023

        # Action space
        # Used by the agent:

        self.action_space = spaces.Box(
            low=np.array([self.min_displacement_xy, self.min_displacement_xy, self.min_displacement_h]),
            high=np.array([self.max_displacement_xy, self.max_displacement_xy, self.max_displacement_h]),
            dtype=np.float32)

    def seed(self, seed):
        np.random.seed(seed)

    def train(self):
        self.training = True

    def eval(self):
        self.training = False

    def CalculateInputImage(self, state):

        # input:    (current_pos,map_unc,flew_map) # 19/01/2023
        # output:   (map_unc, fov,flew_map) # 19/01/2023

        # fov_h_min, fov_h_max, fov_v_min, fov_v_max = self.calculate_fov(state[0][2], [state[0][0], state[0][1]])
        self.calculate_fov(state[0][2], [state[0][0], state[0][1]])

        # start_point = (int(self.fov_h_max) - 1, int(fov_v_max) - 1)
        # end_point = (int(fov_h_min), int(fov_v_min))
        # TODO: I am not sure if I should use the "selfies" or the interpolated size of the fov (i's)
        # TODO: I think the self is correct because I should only use 'i's' to update the unc map

        start_point = (int(self.fov_h_max) - 1, int(self.fov_v_max) - 1)
        end_point = (int(self.fov_h_min), int(self.fov_v_min))

        fov_image = np.zeros(np.shape(self.map_unc))
        fov_image = cv2.rectangle(fov_image, start_point, end_point, 1.,
                                  -1)  # calculate thickness acording to the image size

        return cv2.merge((state[1], fov_image, state[2]))

    def initialize_maps(self):

        self.flew_map = np.zeros((self.h_field, self.w_field))
        self.min_height_map = np.ones((self.h_field, self.w_field))

        ## Difficulty map
        I_d = np.random.randint(2, size=(self.h_field, self.w_field))
        ## Decay factor map
        I_f = np.random.randint(2, size=(self.h_field, self.w_field))
        # Generate n-by-n grid of spatially correlated noise:
        correlation_scale = 15  # 51# 500 # 51
        # For the difficulty map:0,0
        noise_d = cv2.GaussianBlur(np.double(I_d), (0, 0), correlation_scale)
        noise_f = cv2.GaussianBlur(np.double(I_f), (0, 0), correlation_scale)
        beta_decay = 0.7

        if 'original' in map_type:
            self.f_decay = np.float64(beta_decay + (1 - beta_decay) * cv2.normalize(noise_f,
                                                                                    None,
                                                                                    alpha=0,
                                                                                    beta=255,
                                                                                    norm_type=cv2.NORM_MINMAX,
                                                                                    dtype=cv2.CV_32F) / 255)  # 11/01/2023
        elif 'lines' in map_type:
            self.f_decay = generate_unc_map_lines(self.h_field, self.w_field, beta_decay)
        elif 'blobs' in map_type:
            self.f_decay = generate_unc_map_blobs(self.h_field, self.w_field, beta_decay)
        elif 'boxes' in map_type:
            self.f_decay = generate_random_boxes(self.h_field, self.w_field, beta_decay)

        self.max_reach_reward = np.sum(np.sum(2 * noise_d - 1))  # R - (1 - R)

        # Calculate "optimal" unc map (at min height)
        # ROI_diff = self.map_diff
        ROI_diff = 1
        ROI_f = self.f_decay

        # Randomly:
        ## Generate uncertaintly map
        self.map_unc = np.ones((self.h_field, self.w_field))  ## generated at the end of this function
        self.map_unc_min_h = np.ones((self.h_field, self.w_field))
        self.map_unc_min_h = 1 - ROI_diff * ROI_f ** (self.min_height - self.min_height)

    def calculate_fov(self, h, uav_coord):  # <--
        # TODO: Remove this
        if mode_2D:
            h = self.min_height

        # RealSense
        ##RGB sensor FOV (H × V): 69° × 42°
        # TODO: use a class to define these variables! Class RGBcamera. Something like that
        alpha_h = np.deg2rad(69)
        alpha_v = np.deg2rad(42)

        # The higher the height, the uav obtains larger areas from the uncertainty map, but the uncertainty is worse
        # The lower the height, the uav obtains smaller areas from the uncertainty map, but the uncertainty is better
        h_b = h
        h_a = np.minimum(np.maximum(h_b, self.min_height), self.max_height)

        # TODO: remove rounding?
        # Horizontal view:
        self.o_h = np.ceil(h_a * np.tan(alpha_h / 2))  # be CAREFUL! ROUNDING

        # TODO: remove rounding?
        # Vertical view:
        self.o_v = np.ceil(h_a * np.tan(alpha_v / 2))  # be CAREFUL! ROUNDING

        # uav_coord: uav coordinates in x,y
        fov_h_min_b, fov_h_max_b = uav_coord[1] - self.o_h, uav_coord[1] + self.o_h
        fov_v_min_b, fov_v_max_b = uav_coord[0] - self.o_v, uav_coord[0] + self.o_v

        self.fov_h_min = np.minimum(np.maximum(fov_h_min_b, self.x_min), self.x_max)
        self.fov_h_max = np.minimum(np.maximum(fov_h_max_b, self.x_min), self.x_max)

        self.fov_v_min = np.minimum(np.maximum(fov_v_min_b, self.y_min), self.y_max)
        self.fov_v_max = np.minimum(np.maximum(fov_v_max_b, self.y_min), self.y_max)

        # TODO: this function is being called three times! I should optimize to be called only once! Problem: reset()
        # print("uav_coord: {}".format(uav_coord))

        # return int(fov_h_min_a), int(fov_h_max_a), int(fov_v_min_a), int(fov_v_max_a)
        # return fov_h_min_a, fov_h_max_a, fov_v_min_a, fov_v_max_a

    def update_uncertainty(self, h, uav_coord):

        # self.fov_h_min, self.fov_h_max, self.fov_v_min, self.fov_v_max = self.calculate_fov(h, uav_coord)
        self.calculate_fov(h, uav_coord)
        # print("           y={} to y={}".format(self.fov_v_min, self.fov_v_max))
        # print("           x={} to x={}".format(self.fov_h_min, self.fov_h_max))

        # For the complete update:
        y_min_c, y_max_c = np.ceil(self.fov_v_min), np.floor(self.fov_v_max)
        x_min_c, x_max_c = np.ceil(self.fov_h_min), np.floor(self.fov_h_max)
        # For the interpolated update:
        y_min_i, y_max_i = np.floor(self.fov_v_min), np.ceil(self.fov_v_max)
        x_min_i, x_max_i = np.floor(self.fov_h_min), np.ceil(self.fov_h_max)
        # Update for the pixels with x AND y intersection
        y_min, y_max = (y_min_c - self.fov_v_min), (self.fov_v_max - y_max_c)
        x_min, x_max = (x_min_c - self.fov_h_min), (self.fov_h_max - x_max_c)
        A, B, C, D = x_min * y_min, x_max * y_min, x_min * y_max, x_max * y_max
        # Update for the pixels with x XOR y intersection
        alpha, beta, gamma, delta = y_min, x_min, x_max, y_max
        '''
        print('----------------------------------')
        print("uav_coord: {}".format(uav_coord))
        print("           y={} to y={}".format(self.fov_v_min, self.fov_v_max))
        print("           x={} to x={}".format(self.fov_h_min, self.fov_h_max))
        print("y_c: {} à {}".format(y_min_c, y_max_c))
        print("x_c: {} à {}".format(x_min_c, x_max_c))
        print("y_i: {} à {}".format(y_min_i, y_max_i))
        print("x_i: {} à {}".format(x_min_i, x_max_i))

        print("A: {:.2f}*{:.2f} = {:.2f}".format(x_min, y_min, A))
        print("B: {:.2f}*{:.2f} = {:.2f}".format(x_max, y_min, B))
        print("C: {:.2f}*{:.2f} = {:.2f}".format(x_min, y_max, C))
        print("D: {:.2f}*{:.2f} = {:.2f}".format(x_max, y_max, D))

        print("alpha: {:.2f}".format(y_min))
        print("beta: {:.2f}".format(x_min))
        print("gamma: {:.2f}".format(x_max))
        print("delta: {:.2f}".format(y_max))
        print('----------------------------------')
        '''

        ROI_diff = 1  # self.map_diff[self.fov_v_min:self.fov_v_max, self.fov_h_min:self.fov_h_max] # limitar as bordas

        # ROI_f = self.f_decay[self.fov_v_min:self.fov_v_max, self.fov_h_min:self.fov_h_max]  # limitar as bordas
        # We use the "interpolated" coordinates because it covers the entire region rounded up/down

        # Convert to integer to be used as indices for the matrices
        y_min_i, y_max_i, x_min_i, x_max_i = int(y_min_i), int(y_max_i), int(x_min_i), int(x_max_i)
        ROI_f = self.f_decay[y_min_i:y_max_i, x_min_i:x_max_i]

        # base: difficulty to measure
        # h: uavs height
        # h_min: minimum uavs height
        # f: decay factor
        # MV[0,0] = A * MV[0,0] + 1-A
        # Preparing the matrices for the interpolation

        matrix_values = 1 - ROI_diff * ROI_f ** (h - self.min_height)
        # matrix_values[0, 0]   =  partialupdate(A, matrix_values[0, 0])  # Update Pixel A
        # matrix_values[0, -1]  =  partialupdate(B, matrix_values[0, -1])  # Update Pixel B
        # matrix_values[-1, 0]  = partialupdate(C, matrix_values[-1, 0])  # Update Pixel C
        # matrix_values[-1, -1] = partialupdate(D, matrix_values[-1, -1])  # Update Pixel D

        # matrix_values[0, 1:-1]  = partialupdate(alpha, matrix_values[0, 1:-1])  # Update alpha region
        # matrix_values[1:-1, 0]  = partialupdate(beta, matrix_values[1:-1, 0])  # Update beta region
        # matrix_values[1:-1, -1] = partialupdate(gamma, matrix_values[1:-1, -1])  # Update gamma region
        # matrix_values[-1, 1:-1] = partialupdate(delta, matrix_values[-1, 1:-1])  # Update sigma region

        # cv2.imshow("matrix_values", matrix_values)
        # print(matrix_values)

        # Again, we use the "interpolated" coordinates because it covers the entire region rounded up/down
        # Update uncertainty
        self.map_unc[y_min_i:y_max_i, x_min_i:x_max_i] = \
            np.minimum(self.map_unc[y_min_i:y_max_i, x_min_i:x_max_i], matrix_values)

        self.min_height_map[y_min_i:y_max_i, x_min_i:x_max_i] = \
            np.minimum(self.min_height_map[y_min_i:y_max_i, x_min_i:x_max_i], h / self.max_height)

        prev_flew_map = np.copy(self.flew_map)
        self.flew_map[y_min_i:y_max_i, x_min_i:x_max_i] = 1

        # Backup
        '''
        self.map_unc[self.fov_v_min:self.fov_v_max, self.fov_h_min:self.fov_h_max] = np.minimum(
            self.map_unc[self.fov_v_min:self.fov_v_max, self.fov_h_min:self.fov_h_max], matrix_values)

        self.min_height_map[self.fov_v_min:self.fov_v_max, self.fov_h_min:self.fov_h_max] = np.minimum(
            self.min_height_map[self.fov_v_min:self.fov_v_max, self.fov_h_min:self.fov_h_max], h / self.max_height)
        # Update heightmaps (pra nao chamar o calculate_fov 2 vezes!)
        prev_flew_map = np.copy(self.flew_map)
        self.flew_map[self.fov_v_min:self.fov_v_max, self.fov_h_min:self.fov_h_max] = 1  # se sobrevoou, recebem 1
        '''

    ### Decode action
    def interpretate_action(self, action):

        # print("action = {}".format(action))
        uav_coord = self.current_pos[0:2]
        h = self.current_pos[2]
        self.calculate_fov(h, uav_coord)

        '''
        # TODO: Remove multiplier
        fov_speed_multiplier_y = 1.0 # self.o_v
        fov_speed_multiplier_x = 1.0 # self.o_h

        # TODO: find a better name for this variable. Speed? Inst position?
        inst_position = np.multiply(action, [fov_speed_multiplier_y, fov_speed_multiplier_x, 1])
        '''
        inst_position = np.multiply(action, [self.denormalize_action_factor, self.denormalize_action_factor,
                                             self.denormalize_action_factor])

        inst_position = np.minimum(np.maximum(inst_position, self.action_space.low), self.action_space.high)

        self.prev_pos = self.current_pos  # save last position

        self.current_pos += inst_position  # update last position

        current_pos_b = self.current_pos

        self.current_pos = np.minimum(np.maximum(self.current_pos,
                                                 self.observation_space.low[0:-1]),
                                      self.observation_space.high[0:-1])

        if not np.array_equal(current_pos_b, self.current_pos):
            self.out_of_range = 1
        else:
            self.out_of_range = 0

    def step(self, action):

        done = False
        self.current_step += 1

        if self.current_step > self.max_step:
            done = True

        """
            The agent (drone) takes a step (flies somewhere) in the environment.
            Parameters
            ----------
            action : (int,int,int) - the uav displacement in x,y, and h
            Returns: (float32) - uncertaintly map (observation), (float32) reward, (bool) episode_over, (int,int) - coords
            -------
            ob, reward, episode_over, info : tuple
                ob (object) :
                    an environment-specific object representing your observation of
                    the environment.
                reward (float) :
                    amount of reward achieved by the previous action. The scale
                    varies between environments, but the goal is always to increase
                    your total reward. (This reward per step is normalised to 1.)
                episode_over (bool) :
                    whether it's time to reset the environment again. Most (but not
                    all) tasks are divided up into well-defined episodes, and done
                    being True indicates the episode has terminated. (For example,
                    perhaps the pole tipped too far, or you lost your last life.)re
                info (dict) :
                     diagnostic information useful for debugging. It can sometimes
                     be useful for learning (for example, it might contain the raw
                     probabilities behind the environment's last state change).
                     However, official evaluations of your agent are not allowed to
                     use this for learning.
            """

        # TODO: Choose appropriate done (check if this one is correct!)
        # if (action < [0.1,0.1,0.1]).all() and (action > [-0.1,-0.1,-0.1]).all():  # Implement done
        #    done = True

        # Take a step, and observe environment.
        self.interpretate_action(action)  # returns self.current_pos

        uav_coord = self.current_pos[0:2]
        h = self.current_pos[2]
        prev_map = np.copy(self.map_unc)

        self.update_uncertainty(h, uav_coord)

        state_ = (np.copy(self.current_pos), np.copy(self.map_unc), np.copy(self.flew_map))
        input_image = self.CalculateInputImage(state_)
        self.render_state = np.copy(input_image)

        self.last_reward = self.reward
        self.reward = self.calculate_reward(action, prev_map, done)

        # The state is y,x,h,last_r
        # self.state = np.append(self.current_pos,self.last_reward)
        self.state = np.append(self.current_pos, self.reward)

        if done == True:
            self.cv2_save_render = 1
        else:
            self.cv2_save_render = 0

        self.reward_list[-1] += self.reward

        return self.state, self.reward, done, 0

    def calculate_reward(self, action, prev_map, done):

        #done_reward = 0 * np.sum(np.sum(self.map_unc_min_h - self.map_unc))  # 10/01/2023
        # normal_reward = np.sum(np.sum(prev_map - self.map_unc)) - self.time_punishment*(1 - (self.lower_height_achieved*self.lower_height_punishment_decrease)) - self.out_of_range*self.range_punishment # tempo igual a cada passo. Se ja viu, a reward é o prpoprio -time_punishment

        #self.done_reward = np.sum(np.sum(self.map_unc_min_h - self.map_unc))
        self.done_reward = 0*np.sum(np.sum(self.map_unc_min_h - self.map_unc))
        self.normal_reward = np.sum(np.sum(
            prev_map - self.map_unc)) - self.time_punishment - self.out_of_range * self.range_punishment  # tempo igual a cada passo. Se ja viu, a reward é o prpoprio -time_punishment

        '''
        if (action < [0.1, 0.1, 0.1]).all() and (action > [-0.1, -0.1, -0.1]).all():
            return done_reward
        else:
            return normal_reward
        '''

        if done == True:
            return self.done_reward
        else:
            return self.normal_reward



    def reset(self):
        # reset should always run at the end of an episode and before the first run.

        self.reward_list = []
        self.reward_list.append(0)

        self.episode_over = False

        # self.current_pos = [self.h_field//2, self.w_field//2, self.min_height]
        # self.prev_pos = [self.h_field//2, self.w_field//2, self.min_height]

        self.current_pos = [0, 0, self.min_height]
        self.prev_pos = [0, 0, self.min_height]

        # Redundancia para evitar bug (mesmo efeito de initialize maps!)
        self.flew_map = np.zeros((self.h_field, self.w_field))
        self.map_unc = np.ones((self.h_field, self.w_field))
        self.min_height_map = np.ones((self.h_field, self.w_field))

        self.initialize_maps()  # reset map

        state_ = (np.copy(self.current_pos), np.copy(self.map_unc), np.copy(self.flew_map))
        input_image = self.CalculateInputImage(state_)
        self.render_state = np.copy(input_image)

        self.current_step = 0
        self.reward = 0
        self.last_reward = 0
        self.state = np.append(self.current_pos, self.last_reward)

        # The state is y,x,h,last_r
        return self.state

    def render(self):
        unc_map_plt = ConvertToColorMap(self.render_state[:, :, 0])
        fov_plt = ConvertToColor(self.render_state[:, :, 1])
        aug_factor = 4  # 5 #20
        thickness = np.uint8(np.shape(unc_map_plt)[0] / 200)
        unc_map_plt = cv2.resize(unc_map_plt, (0, 0), fx=aug_factor * 1.0, fy=aug_factor * 1.0,
                                 interpolation=cv2.INTER_AREA)
        fov_plt = cv2.resize(fov_plt, (0, 0), fx=aug_factor * 1.0, fy=aug_factor * 1.0, interpolation=cv2.INTER_AREA)

        state = (np.copy(self.current_pos), np.copy(self.map_unc), np.copy(self.flew_map))

        img = np.zeros((np.shape(unc_map_plt)))
        img = np.uint8(img)

        uav_coord = self.current_pos[0:2]
        h = self.current_pos[2]
        # fov_h_min_a, fov_h_max_a, fov_v_min_a, fov_v_max_a = self.calculate_fov(h, uav_coord)
        # self.calculate_fov(h, uav_coord)

        ''' Backup
        start_point = (int(self.fov_h_max) - 1, int(self.fov_v_max) - 1)
        end_point = (int(self.fov_h_min), int(self.fov_v_min))
        '''

        start_point = (int(aug_factor * self.fov_h_max), int(aug_factor * self.fov_v_max))
        end_point = (int(aug_factor * self.fov_h_min), int(aug_factor * self.fov_v_min))
        img = cv2.rectangle(img, start_point, end_point, (0, 0, 255),
                            thickness)  # calculate thickness acording to the image size

        y_, x_, h_ = self.current_pos
        img = cv2.putText(img, 'y,x,h= {:.0f},{:.0f},{:.0f} [m]'.format(y_, x_, h_),
                          (aug_factor + np.uint8(np.shape(img)[0] / 20),
                           5 + aug_factor + np.uint8(np.shape(img)[1] / 20)),
                          cv2.FONT_HERSHEY_SIMPLEX,
                          1, (255, 255, 255), thickness, cv2.LINE_AA)

        img = cv2.putText(img, 'sum_r={:.2f}'.format(self.reward_list[-1]),
                          (aug_factor + np.uint8(np.shape(img)[0] / 20),
                           35 + aug_factor + np.uint8(np.shape(img)[1] / 20)),
                          cv2.FONT_HERSHEY_SIMPLEX,
                          1, (255, 255, 255), thickness, cv2.LINE_AA)

        img = cv2.putText(img, 'r={:.2f}'.format(self.reward),
                          (aug_factor + np.uint8(np.shape(img)[0] / 20),
                           70 + aug_factor + np.uint8(np.shape(img)[1] / 20)),
                          cv2.FONT_HERSHEY_SIMPLEX,
                          1, (255, 255, 255), thickness, cv2.LINE_AA)

        img = cv2.putText(img, 'step={}'.format(self.current_step), (
            aug_factor + np.uint8(np.shape(img)[0] / 20), 105 + aug_factor + np.uint8(np.shape(img)[1] / 20)),
                          cv2.FONT_HERSHEY_SIMPLEX,
                          1, (255, 255, 255), thickness, cv2.LINE_AA)

        img = cv2.putText(img, 'out_of_range={}'.format(self.out_of_range), (
            aug_factor + np.uint8(np.shape(img)[0] / 20), 140 + aug_factor + np.uint8(np.shape(img)[1] / 20)),
                          cv2.FONT_HERSHEY_SIMPLEX,
                          1, (255, 255, 255), thickness, cv2.LINE_AA)

        img = cv2.putText(img, 'done_r={:.2f}'.format(np.sum(np.sum(self.map_unc_min_h - self.map_unc))), (
            aug_factor + np.uint8(np.shape(img)[0] / 20), 175 + aug_factor + np.uint8(np.shape(img)[1] / 20)),
                          cv2.FONT_HERSHEY_SIMPLEX,
                          1, (255, 255, 255), thickness, cv2.LINE_AA)

        # TODO: interpolate
        cX = int(np.round(aug_factor * uav_coord[1]))
        cY = int(np.round(aug_factor * uav_coord[0]))

        img = cv2.putText(img, ".", (cX, cY), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 4 * thickness)

        input_image = self.CalculateInputImage(state)

        flew_map_1 = cv2.resize(input_image[:, :, 2], (0, 0), fx=aug_factor * 1.0, fy=aug_factor * 1.0,
                                interpolation=cv2.INTER_AREA)
        flew_map_1 = ConvertToColor(flew_map_1)

        min_height_map = cv2.resize(self.min_height_map, (0, 0), fx=aug_factor * 1.0, fy=aug_factor * 1.0,
                                    interpolation=cv2.INTER_AREA)
        min_height_map = ConvertToColorMap(min_height_map)

        map_1 = cv2.resize(self.f_decay, (0, 0), fx=aug_factor * 1.0, fy=aug_factor * 1.0, interpolation=cv2.INTER_AREA)
        map_1 = ConvertToColorMap(map_1)

        map_2 = cv2.normalize(self.f_decay, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F) / 255
        map_2 = cv2.resize(map_2, (0, 0), fx=aug_factor * 1.0, fy=aug_factor * 1.0, interpolation=cv2.INTER_AREA)
        map_2 = ConvertToColorMap(map_2)

        second_window_1 = cv2.hconcat([map_1[:, :, ::-1], map_2[:, :, ::-1]])  # inverti para o cv2 imshow
        second_window_2 = cv2.hconcat([flew_map_1, min_height_map])

        display1 = cv2.hconcat([unc_map_plt, img])
        display2 = cv2.vconcat([second_window_1, second_window_2])

        if self.cv2_show_render:
            cv2.imshow("f_decay / f_decay (norm) / flew_map / h_min", display2)
            cv2.imshow("screen", display1)

        if self.cv2_save_render or self.current_step==self.max_step:
            both_displays = cv2.vconcat([display1, display2])
            save_path = 'results/testing_images/screen_f_decay_f_decay_norm_flew_map_h_min_{}_sum_r_{:.2f}.jpg'.format(
                self.cv2_save_number,self.reward_list[-1])
            print('saving results to: {}'.format(save_path))

            cv2.imwrite(save_path,both_displays)
            self.cv2_save_number += 1

        cv2.waitKey(1)

    def close(self):
        cv2.destroyAllWindows()


def make_pytorch_env(args):
    env = Env(args)
    return env
