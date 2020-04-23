import gym
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout
from keras.utils import to_categorical
from statistics import mean, median

# for info about this project http://gym.openai.com/docs/

gameNames = ['CartPole-v0', 'MountainCar-v0', 'MsPacman-v0']
env = gym.make(gameNames[0])

# The process gets started by calling reset(), which returns an initial observation

min_allowed_score = 55  # to save to data
num_games = 9000  # to play


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


def collect_good_data(show_info=True, save_data=False):
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
    trails_record = []
    good_trails_record = []
    for game_data_with_trail_reached in games_data:
        trails_record.append(game_data_with_trail_reached[0])
        if game_data_with_trail_reached[0] >= min_allowed_score:
            good_trails_record.append(game_data_with_trail_reached[0])
            for action_observation_pair in game_data_with_trail_reached[1]:
                good_data.append(action_observation_pair)
    if show_info:
        print(mean(trails_record), ' is the mean of all games, played: ', len(trails_record))
        print(mean(good_trails_record), ' is the mean of the good trails, played: ', len(good_trails_record))

    if save_data:
        np_array_to_save = np.array(good_data)
        np.save('training_data.npy', np_array_to_save)

    return good_data


def create_dnn_model(input_dimension, batch_size):
    model = Sequential()
    # 1
    model.add(Dense(128, input_shape=(input_dimension, ), activation='relu'))
    model.add(Dropout(0.2))
    # 2
    model.add(Dense(256, activation='relu'))
    model.add(Dropout(0.2))
    # 3
    model.add(Dense(512, activation='relu'))
    model.add(Dropout(0.2))
    # 4
    model.add(Dense(256, activation='relu'))
    model.add(Dropout(0.2))
    # 5
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.2))

    # output
    model.add(Dense(2, activation='softmax'))

    # compiling the model
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    return model


def train_dnn_model(model, training_data, labels, epochs, batch_size, verbose):
    # Convert labels to categorical one-hot encoding
    one_hot_labels = to_categorical(labels, num_classes=2)
    model.fit(training_data, one_hot_labels, epochs=epochs, batch_size=batch_size, verbose = verbose)


def train_game():
    training_data = np.array(collect_good_data())
    # train_x = np.array(data[1] for data in training_data)
    # train_y = np.array(data[0] for data in training_data)
    train_x = []
    train_y = []
    for data in training_data:
        train_x.append(data[1])
        train_y.append(data[0])

    train_x = np.array(train_x)
    train_y = np.array(train_y)

    batch_size = 512
    input_dimension = len(train_x[0])
    print(input_dimension, ' : input_dimension')
    model = create_dnn_model(input_dimension, batch_size)
    train_dnn_model(model, train_x, train_y, 10, batch_size, True)

train_game()
#   we have:   observation: [x,x,x,x] we took action: 0 (save what we did,
#   and see if we get the min requirements)
#   we can save (observation, taken action)
#   or
#   we can save (previous_observation, taken action), in this code we chose the later
