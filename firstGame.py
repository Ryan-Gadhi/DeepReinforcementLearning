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


min_allowed_score = 45  # to save to data
num_games = 200  # to play


def game_info():
    print('\nis action space:', env.action_space)
    print('\nis observation space:', env.observation_space)
    print('\nis upper limit: ', env.observation_space.high)
    print('\nis upper limit: ', env.observation_space.low)


def initial_play():
    for episode in range(num_games):
        env.reset()
        for i in range(1000):
            env.render()
            observation, reward, done, info = env.step(env.action_space.sample())
            if done:
                print(' - Game Num ', episode, ' Finished after ', i, 'actions ')
                break
    env.close()


def collect_good_data():
    games_data = []

    for game in range(num_games):
        env.reset()
        game_data = []
        trail_reached = 0  # number of successful steps taken
        previous_observation = []
        for trial in range(250):
            trail_reached = trial
            action = env.action_space.sample()
            observation, reward, done, info = env.step(action)

            if previous_observation != []:
                game_data.append([action, observation])

            if done:  # i.e. lost the game
                break
            # one trial finished
            previous_observation = observation

        # one game finished
        games_data.append([trail_reached, game_data])

    # all games finished
    good_data = []

    for game_data_with_trail_reached in games_data:
        if game_data_with_trail_reached[0] >= min_allowed_score:
            for action_observation_pair in game_data_with_trail_reached[1]:
                good_data.append(action_observation_pair)

    return good_data

good_data = collect_good_data()
print(good_data[0])


#   we have:   observation: [x,x,x,x] we took action: 0 (save what we did,
#   and see if we get the min requirements)
#   we can save (observation, taken action)
#   or
#   we can save (previous_observation, taken action)

# def initial_play_2():
#     for episode in range(3):
#         env.reset()
#         for i in range(10):
#             # env.render()
#             action = env.action_space.sample()
#             print(action, ' is the action taken')
#             observation, reward, done, info = env.step(action)
#             print(observation, ' then we saw the pole reacted this way')
#             if done:
#                 print(' - Game Num ', episode, ' Finished after ', i, 'actions ')
#                 break
#     env.close()
# initial_play_2()

