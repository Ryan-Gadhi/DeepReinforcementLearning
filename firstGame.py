import gym
import random
# import keras
from statistics import mean, median

import logging
import sys

# for info about this project http://gym.openai.com/docs/

gameNames = ['CartPole-v0', 'MountainCar-v0', 'MsPacman-v0']
env = gym.make(gameNames[0])

# The process gets started by calling reset(), which returns an initial observation


# goal_steps = 500
# the score when reached we are going to accept the action
# score_requirement = 50
# inital_games = 10000

num_games = 50  # to play

def initial_play():
    for episode in range(num_games):
        env.reset()
        for i in range(1000):
            # env.render()
            observation, reward, done, info = env.step(env.action_space.sample())
            if done:
                print(' - Game Num ', episode, ' Finished after ', i, 'actions ')
                break
    env.close()

initial_play()

# print('\nis action space:', env.action_space)
# print('\nis observation space:', env.observation_space)
# print('\nis upper limit: ', env.observation_space.high)
# print('\nis upper limit: ', env.observation_space.low)

