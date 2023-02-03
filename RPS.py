# The example function below keeps track of the opponent's history and plays whatever the opponent played two plays ago. It is not a very good player so you will need to change the code to pass the challenge.
import tensorflow as tf
from collections import deque
import numpy as np
from tensorflow import keras
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


# Look at input shape

rpsMap = {
    'R': 0,
    'P': 1,
    'S': 2,
}

rpsVector = ['R', 'P', 'S']


def winnerAction(counter):
    return rpsVector[(rpsMap[counter] + 1) % 3]


def rewardVector(opponentPlay):
    vector = [0.0, 0.0, 0.0]
    if (opponentPlay == ''):
        opponentPlay = 'R'
    for i in range(-1, 2):
        vector[(rpsMap[opponentPlay]+i) % 3] = i*1
    return vector


def player(prev_play, state=[]):
    n = 3
    if len(state) == 0:
        state.append(initState(n))
    else:
        if prev_play == '':
            state[0] = initState(n)
    return play(prev_play, state[0])


def initState(n):

    markov = np.zeros((2, 3, 3))
    lenMarkov = len(markov.ravel())

    learningRate = 0.0007

    model = keras.Sequential()

    model.add(keras.layers.Dense(
        96,  activation='relu',
        input_shape=(2*n+lenMarkov,),
        kernel_regularizer=tf.keras.regularizers.L2(0.002)
    ))

    # model.add(keras.layers.Dense(8, activation='relu',
    #                             kernel_regularizer=tf.keras.regularizers.L1(0.03)))

    model.add(keras.layers.Dense(3, activation='linear',
                                 kernel_regularizer=tf.keras.regularizers.L1(0.035))
              )

    prev_moves_you = deque(maxlen=n)
    prev_moves_opponent = deque(maxlen=n)
    optimizer = tf.optimizers.Adam(learning_rate=learningRate)
    model.build()
    model.compile(optimizer=optimizer, loss='mean_squared_error')
    model.summary()

    print('Learning Rate: ', learningRate)
    return {'prev_moves_you': prev_moves_you,
            'prev_moves_opponent': prev_moves_opponent,
            'markov': markov,
            'model': model,
            'memory': n,
            'moves': 0,
            'prediction': [],
            'optimizer': optimizer}


def play(prev, state):
    if prev == '':
        nextMove = rpsVector[np.random.randint(0, 3)]
    else:

        input_data_tensor = inputFromState(state)  # should be -1

        # tf.print(input_data_tensor)
        # tf.print(input_data_tensor.shape)

        if (state['moves'] > state['memory']):
            with tf.GradientTape() as tape:

                tape.watch(state['model'].trainable_weights)

                # Predictions represents the expectation of reward per action per state (previous plays)
                state['prediction'] = state['model'](input_data_tensor)
                # print('Predition')
                # tf.print(state['prediction'])
                # tf.print(state['prediction'].shape)
                # target = tf.identity(state['prediction'])

                target = tf.expand_dims(rewardVector(prev), 0)
                # tf.print(target)
                # tf.print(target.shape)

                loss = keras.losses.mean_squared_error(
                    target, state['prediction'])
                # tf.print(loss)
                # inputTensor = tf.convert_to_tensor( input_data,dtype=tf.float32)

                grads = tape.gradient(
                    loss, state['model'].trainable_weights, unconnected_gradients='none')
                # tf.print(state['model'].trainable_weights)

                # tf.print(grads)

                state['optimizer'].apply_gradients(
                    zip(grads, state['model'].trainable_weights))

            state['prev_moves_opponent'].append(rpsMap[prev])
            input_data_tensor = inputFromState(state)
            state['prediction'] = state['model'](input_data_tensor)

            nextMove = rpsVector[np.argmax(state['prediction'][0])]
        else:
            state['prev_moves_opponent'].append(rpsMap[prev])

            nextMove = rpsVector[np.random.randint(0, 3)]

    if state['moves'] >= 2:
        state['markov'][0,
                        state['prev_moves_opponent'][-2],
                        state['prev_moves_opponent'][-1]] += 1
        state['markov'][1,
                        state['prev_moves_you'][-2],
                        state['prev_moves_you'][-1]] += 1
    state['prev_moves_you'].append(rpsMap[nextMove])
    state['moves'] += 1
    return nextMove


def inputFromState(state):
    input_data = np.concatenate((state['prev_moves_you'],
                                 state['prev_moves_opponent'],
                                 state['markov'].ravel() / (state['moves'])))
    input_data = tf.expand_dims(input_data, 0)

    return input_data
