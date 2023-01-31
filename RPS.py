# The example function below keeps track of the opponent's history and plays whatever the opponent played two plays ago. It is not a very good player so you will need to change the code to pass the challenge.
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf 
from tensorflow import keras

import numpy as np

from collections import deque


#Look at input shape

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
    if(opponentPlay==''):
        opponentPlay='R'
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
    #model.add(keras.layers.Dense(64, input_shape=(2*n), activation='relu'))
    model.add(keras.layers.Dense(64,  activation='relu', input_shape=(2*n,)))

    model.add(keras.layers.Dense(64, activation='relu'))
    model.add(keras.layers.Dense(3, activation='linear'))

    prev_moves_you = deque(maxlen=n)
    prev_moves_opponent = deque(maxlen=n)
    optimizer = tf.optimizers.Adam()
    model.build()
    model.compile(optimizer= optimizer, loss='mean_squared_error')
    model.summary()
    return {'prev_moves_you': prev_moves_you,
            'prev_moves_opponent': prev_moves_opponent,
            'model': model,
            'memory':n,
            'moves': 0,
            'prediction' : [],
            'optimizer': optimizer}



def play(prev,state):
    if prev != '' :
        state['prev_moves_opponent'].append(rpsMap[prev])

    input_data = np.concatenate((state['prev_moves_you'], state['prev_moves_opponent']))
    #input_data_tensor = tf.constant(input_data, dtype=tf.float32)
    input_data_tensor = tf.expand_dims(input_data, 0)
    tf.print(input_data_tensor)
    if(state['moves']> state['memory']):
        with tf.GradientTape() as tape:
            tape.watch(state['model'].trainable_weights)
            
            state['prediction'] = state['model'](input_data_tensor)
            print('Predition')
            tf.print(state['prediction'])
            tf.print(state['prediction'].shape)
            #target = tf.identity(state['prediction'])
            

            target = tf.expand_dims(rewardVector(prev),0)
            tf.print(target)
            tf.print(target.shape)
            #Predictions represents the expectation of reward per action per state 

            loss = keras.losses.mean_squared_error(target, state['prediction'])
            tf.print(target)
            tf.print(target)
            
            tf.print(loss)
            #inputTensor = tf.convert_to_tensor( input_data,dtype=tf.float32)
            #tape.watch(inputTensor)
        
            grads = tape.gradient(loss, state['model'].trainable_weights, unconnected_gradients='none')
            #tf.print(state['model'].trainable_weights)

            tf.print(grads)
            #tf.print(state['model'].trainable_weights)
            state['optimizer'].apply_gradients(zip(grads, state['model'].trainable_weights))



        state['prev_moves_opponent'].append(rpsMap[prev])
        input_data = np.concatenate((np.array(state['prev_moves_opponent']), np.array(state['prev_moves_you'])))
        input_data = input_data.reshape(1, 2*state['memory'])
        
        nextMove = rpsVector[np.argmax(state['prediction'][0])]
    else:
        nextMove = rpsVector[np.random.randint(0,3)]
    
    state['prev_moves_you'].append(rpsMap[nextMove])
    state['moves'] += 1

    return nextMove