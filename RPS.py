# The example function below keeps track of the opponent's history and plays whatever the opponent played two plays ago. It is not a very good player so you will need to change the code to pass the challenge.
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf 
from tensorflow import keras

import numpy as np

from collections import deque

rpsMap = {
    'R':0,
    'P':1,
    'S':2,
}

rpsVector = ['R', 'P', 'S']

def winnerAction(counter):
            return rpsVector[(rpsMap[counter] + 1 )% 3 ]

def rewardVector(opponentPlay):
    vector = [0.0,0.0,0.0]
    for i in range(-1,2):
        vector[(rpsMap[opponentPlay]+i) % 3] = i*1
    return vector



def player(prev_play, state=[]):
    n=3
    if len(state) == 0:
        state.append(initState(n))
    return play(prev_play, state[0])
     

  



def initState(n):
    

    model = keras.Sequential()
    model.add(keras.layers.Dense(64, input_shape=(2*n,), activation='relu'))
    model.add(keras.layers.Dense(64, activation='relu'))
    model.add(keras.layers.Dense(3, activation='linear'))

    prev_moves_you = deque(maxlen=n)
    prev_moves_opponent = deque(maxlen=n)
    optimizer = tf.optimizers.Adam()
    model.compile(optimizer= optimizer, loss='mean_squared_error')
    model.build()
    return {'prev_moves_you': prev_moves_you,
            'prev_moves_opponent': prev_moves_opponent,
            'model': model,
            'memory':n,
            'moves': 0,
            'prediction' : [],
            'optimizer': optimizer}



def play(prev,state):
    if(state['moves']> state['memory']):
        with tf.GradientTape() as tape:
            input_data = np.concatenate((state['prev_moves_you'], state['prev_moves_opponent']))
            
            if len(state['prediction'])> 0:
                target = state['prediction'].copy()
            

                target[0] = rewardVector(prev)
                #Predictions represents the expectation of reward per action per state 

                loss = keras.losses.mean_squared_error(target, state['prediction'])
                tf.print(loss)
                tape.watch(state['model'].trainable_weights)
                grads = tape.gradient(loss[0], state['model'].trainable_weights)
                tf.print(grads)
                tf.print(state['model'].trainable_weights)
                state['optimizer'].apply_gradients(zip(grads, state['model'].trainable_weights))



        state['prev_moves_opponent'].append(rpsMap[prev])
        input_data = np.concatenate((np.array(state['prev_moves_opponent']), np.array(state['prev_moves_you'])))
        input_data = input_data.reshape(1, 2*state['memory'])
        state['prediction'] = state['model'].predict(np.array(input_data))
        
        nextMove = rpsVector[np.argmax(state['prediction'][0])]
    else:
        if prev != '' :
            state['prev_moves_opponent'].append(rpsMap[prev])
        nextMove = rpsVector[np.random.randint(0,3)]
    
    state['prev_moves_you'].append(rpsMap[nextMove])
    state['moves'] += 1

    return nextMove