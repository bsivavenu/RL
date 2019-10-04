# -*- coding: utf-8 -*-
"""
Created on Fri May 31 19:48:14 2019

@author: win10
"""

import gym
import numpy as np
env = gym.make("MountainCar-v0")

LEARNING_RATE = 0.5
DISCOUNT = 0.95
EPISODES = 25000
SHOW_EVERY = 10

discrete_os_size = [20]*len(env.observation_space.high)
discrete_os_win_size = (env.observation_space.high - env.observation_space.low)/(discrete_os_size)

epsilon = 0.5
start_epsilon_decaying =  1
end_epsilon_decaying = EPISODES//2

epsilon_decay_value = epsilon/(end_epsilon_decaying - start_epsilon_decaying)
q_table = np.random.uniform(low=-2, high = 0, size = (discrete_os_size+[env.action_space.n]))

def get_discreate_state(state): #this function is to convert continuos value to discreate value of a state
	discrete_state = (state - env.observation_space.low)/ discrete_os_win_size
	return  tuple(discrete_state.astype(int))

# print(env.reset())
# print(discrete_os_win_size)

for episode in range(EPISODES):

	if episode % SHOW_EVERY == 0:

		print(episode)
		render = True
	
	else:

		render = False
 	
	discrete_state = get_discreate_state(env.reset()) #this gets intial state env.reset()
	# print('discrete_state.............',discrete_state)
	# print((q_table[discrete_state]))
	# print(np.argmax(q_table[discrete_state])) #this shows to which value we have to take
	done = False
	while not done:
	    # action = 2
	    action = np.argmax(q_table[discrete_state])
	    new_state, reward, done, _  = env.step(action)
	    new_discreate_state = get_discreate_state(new_state)
	    # print('new_discreate_state: ',new_discreate_state)
	    # print(reward,done)
	    if render:
	    	env.render()
	    if not done:
	    	max_future_q = np.max(q_table[new_discreate_state])
	    	# print('max_future_q: ',max_future_q)
	    	current_q = q_table[discrete_state+(action,)]
	    	# print('current_q: ', current_q)

	    	# new_q_value = (1-learning_rate)*(current_q_value) + (learning_rate)*(reward + discount_factor *(estimate_of_optimal_q_value))

	    	new_q = (1 - LEARNING_RATE)*(current_q)+LEARNING_RATE*(reward+DISCOUNT*max_future_q)
	    	q_table[discrete_state+(action,)] = new_q

	    elif new_state[0] >= env.goal_position:
	    	print(f"we made it on episode{episode}")
	    	q_table[discrete_state+(action,)] = 0


      #   elif new_state[0] >= env.goal_position:
	    	# q_table[discrete_state+(action,)] = 0
	    discrete_state = new_discreate_state
	if end_epsilon_decaying >= episode >= start_epsilon_decaying:
		epsilon -= epsilon_decay_value

	env.close()

