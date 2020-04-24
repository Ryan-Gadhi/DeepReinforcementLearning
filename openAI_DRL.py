import gym
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout
from keras.utils import to_categorical
from statistics import mean, median

# for info about this project http://gym.openai.com/docs/

# The process gets started by calling reset(), which returns an initial observation


num_games = 9000  # to play


def game_info():
    print('\nis action space:', env.action_space)
    print('\nis action count:', env.action_space.n)
    print('\nis observation space:', env.observation_space)
    print('\nis upper obs limit: ', env.observation_space.high)
    print('\nis lower obs limit: ', env.observation_space.low)


def initial_play(verbose=False):
    for episode in range(num_games):
        env.reset()
        for i in range(1000):
            env.render()
            action = env.action_space.sample()
            observation, reward, done, info = env.step(action)
            if verbose:
                print('action:', action)
                print('observation:', reward)
                print('done:', done)
                print('info:', info)
                print("- - - - - - - - -")

            if done:
                if verbose:
                    print(' - Game Num ', episode, ' Finished after ', i, 'actions ')
                break
    env.close()


def collect_good_data(verbose=True, save_data=False):
    games_data = []
    max_steps = 1000  # max num of steps the agent can take in the env
    min_allowed_score = -1  # to save to data

    for game in range(num_games):
        env.reset()
        game_data = []
        step_reached = 0  # number of successful steps taken
        previous_observation = []
        for step in range(max_steps):
            step_reached = step
            action = env.action_space.sample()
            observation, reward, done, info = env.step(action)

            if previous_observation != []:
                game_data.append([action, observation])

            if done:  # i.e. lost the game
                break
            # one step finished
            previous_observation = observation

        # one game finished
        games_data.append([step_reached, game_data])

    # all games finished
    good_data = []
    min_allowed_score = sorted(list(i[0] for i in games_data))[int(len(games_data) * 0.95)]
    print(min_allowed_score, ': min_allowed_score')
    steps_record = []
    good_steps_record = []
    for game_data_with_trail_reached in games_data:
        steps_record.append(game_data_with_trail_reached[0])
        if game_data_with_trail_reached[0] >= min_allowed_score:
            good_steps_record.append(game_data_with_trail_reached[0])
            for action_observation_pair in game_data_with_trail_reached[1]:
                good_data.append(action_observation_pair)
    if verbose:
        print(mean(steps_record), ' is the mean of all games, played: ', len(steps_record))
        print(mean(good_steps_record), ' is the mean of the good trails, played: ', len(good_steps_record))

    if save_data:
        np_array_to_save = np.array(good_data)
        np.save('training_data.npy', np_array_to_save)

    num_classes = env.action_space.n
    return good_data, num_classes


def create_dnn_model(input_dimension, num_classes):
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
    model.add(Dense(num_classes, activation='softmax'))

    # compiling the model
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    return model


def train_dnn_model(model, training_data, labels, epochs, batch_size, verbose, num_classes):
    # Convert labels to categorical one-hot encoding
    one_hot_labels = to_categorical(labels, num_classes=num_classes)
    model.fit(training_data, one_hot_labels, epochs=epochs, batch_size=batch_size, verbose = verbose)


def train_model_by_game_data():
    training_data, num_classes = np.array(collect_good_data())

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
    model = create_dnn_model(input_dimension, num_classes)
    train_dnn_model(model, train_x, train_y, 10, batch_size, True, num_classes)
    return model


def play_using_model(model):
    games_data = []

    for game in range(num_games):
        env.reset()
        game_data = []
        previous_observation = []

        for trial in range(250):
            env.render()
            if previous_observation != []:
                input = np.array(previous_observation).reshape((1, -1))
                prediction = model.predict(input)
                action = np.argmax(prediction)
            else:
                action = env.action_space.sample()

            observation, reward, done, info = env.step(action)
            previous_observation = observation

            if done:  # i.e. lost the game
                break
            # one trial finished


if __name__ == "__main__":
    gameNames = ['CartPole-v0', 'MountainCar-v0', 'MsPacman-v0']
    env = gym.make(gameNames[0])

    # game_info()
    # initial_play()
    model = train_model_by_game_data()
    play_using_model(model)
