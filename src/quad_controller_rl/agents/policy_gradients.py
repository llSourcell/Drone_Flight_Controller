"""Policy search agent."""

import numpy as np
from quad_controller_rl.agents.base_agent import BaseAgent

import tensorflow as tf 
import numpy as np 
import gym
import random
from collections import deque

class DDPG(BaseAgent):
	"""Sample agent that searches for optimal policy deterministically."""

	def __init__(self, task):
		# Task (environment) information
		self.task = task
		state_dim = int(np.prod(self.task.observation_space.shape))
		# self.action_dim = int(np.prod(self.task.action_space.shape))
		
		# use only z-coordinate
		self.action_dim = 1
		self.action_space_low = self.task.action_space.low[3]
		self.action_space_high = self.task.action_space.high[3]

		# set seeds to 0
		np.random.seed(0)


		# Network parameters
		gamma = 0.99			
		h1_actor = 8			
		h2_actor = 8			
		h3_actor = 8			
		h1_critic = 8			
		h2_critic = 8			
		h3_critic = 8			
		lr_actor = 1e-3			
		lr_critic = 1e-3		
		lr_decay = 1			
		l2_reg_actor = 1e-6		
		l2_reg_critic = 1e-6	
		dropout_actor = 0		
		dropout_critic = 0		
		tau = 1e-2				
		
		self.train_every = 1		
		self.minibatch_size = 1024
		self.initial_noise_scale = 10
		self.noise_decay = 0.99	
		self.exploration_mu = 0.0
		self.exploration_theta = 0.15
		self.exploration_sigma = 0.2
		self.ep = 0 			
		self.total_steps = 0	
		self.log_file = open("log_file" + str(time.time()))

		
		replay_memory_capacity = int(1e5)	# capacity of experience replay memory
		self.replay_memory = deque(maxlen=replay_memory_capacity)			# used for O(1) popleft() operation

		## Tensorflow

		tf.reset_default_graph()

		# placeholders
		self.state_ph = tf.placeholder(dtype=tf.float32, shape=[None,state_dim])
		self.action_ph = tf.placeholder(dtype=tf.float32, shape=[None,self.action_dim])
		self.reward_ph = tf.placeholder(dtype=tf.float32, shape=[None])
		self.next_state_ph = tf.placeholder(dtype=tf.float32, shape=[None,state_dim])
		self.is_not_terminal_ph = tf.placeholder(dtype=tf.float32, shape=[None]) # indicators (go into target computation)
		self.is_training_ph = tf.placeholder(dtype=tf.bool, shape=()) # for dropout

		# episode counter
		episodes = tf.Variable(0.0, trainable=False, name='episodes')
		self.episode_inc_op = episodes.assign_add(1)

		# will use this to initialize both the actor network its slowly-changing target network with same structure
		def generate_actor_network(s, trainable, reuse):
			hidden = tf.layers.dense(s, h1_actor, activation = tf.nn.relu, trainable = trainable, name = 'dense', reuse = reuse)
			hidden_drop = tf.layers.dropout(hidden, rate = dropout_actor, training = trainable & self.is_training_ph)
			hidden_2 = tf.layers.dense(hidden_drop, h2_actor, activation = tf.nn.relu, trainable = trainable, name = 'dense_1', reuse = reuse)
			hidden_drop_2 = tf.layers.dropout(hidden_2, rate = dropout_actor, training = trainable & self.is_training_ph)
			hidden_3 = tf.layers.dense(hidden_drop_2, h3_actor, activation = tf.nn.relu, trainable = trainable, name = 'dense_2', reuse = reuse)
			hidden_drop_3 = tf.layers.dropout(hidden_3, rate = dropout_actor, training = trainable & self.is_training_ph)
			actions_unscaled = tf.layers.dense(hidden_drop_3, self.action_dim, trainable = trainable, name = 'dense_3', reuse = reuse)
			actions = self.action_space_low + tf.nn.sigmoid(actions_unscaled)*(self.action_space_high - self.action_space_low) # bound the actions to the valid range
			return actions

		# actor network
		with tf.variable_scope('actor'):
			# Policy's outputted action for each self.state_ph (for generating actions and training the critic)
			self.actions = generate_actor_network(self.state_ph, trainable = True, reuse = False)

		# slow target actor network
		with tf.variable_scope('slow_target_actor', reuse=False):
			# Slow target policy's outputted action for each self.next_state_ph (for training the critic)
			# use stop_gradient to treat the output values as constant targets when doing backprop
			slow_target_next_actions = tf.stop_gradient(generate_actor_network(self.next_state_ph, trainable = False, reuse = False))

		# will use this to initialize both the critic network its slowly-changing target network with same structure
		def generate_critic_network(s, a, trainable, reuse):
			state_action = tf.concat([s, a], axis=1)
			hidden = tf.layers.dense(state_action, h1_critic, activation = tf.nn.relu, trainable = trainable, name = 'dense', reuse = reuse)
			hidden_drop = tf.layers.dropout(hidden, rate = dropout_critic, training = trainable & self.is_training_ph)
			hidden_2 = tf.layers.dense(hidden_drop, h2_critic, activation = tf.nn.relu, trainable = trainable, name = 'dense_1', reuse = reuse)
			hidden_drop_2 = tf.layers.dropout(hidden_2, rate = dropout_critic, training = trainable & self.is_training_ph)
			hidden_3 = tf.layers.dense(hidden_drop_2, h3_critic, activation = tf.nn.relu, trainable = trainable, name = 'dense_2', reuse = reuse)
			hidden_drop_3 = tf.layers.dropout(hidden_3, rate = dropout_critic, training = trainable & self.is_training_ph)
			q_values = tf.layers.dense(hidden_drop_3, 1, trainable = trainable, name = 'dense_3', reuse = reuse)
			return q_values

		with tf.variable_scope('critic') as scope:
			# Critic applied to self.state_ph and a given action (for training critic)
			q_values_of_given_actions = generate_critic_network(self.state_ph, self.action_ph, trainable = True, reuse = False)
			# Critic applied to self.state_ph and the current policy's outputted actions for self.state_ph (for training actor via deterministic policy gradient)
			q_values_of_suggested_actions = generate_critic_network(self.state_ph, self.actions, trainable = True, reuse = True)

		# slow target critic network
		with tf.variable_scope('slow_target_critic', reuse=False):
			# Slow target critic applied to slow target actor's outputted actions for self.next_state_ph (for training critic)
			slow_q_values_next = tf.stop_gradient(generate_critic_network(self.next_state_ph, slow_target_next_actions, trainable = False, reuse = False))

		# isolate vars for each network
		actor_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='actor')
		slow_target_actor_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='slow_target_actor')
		critic_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='critic')
		slow_target_critic_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='slow_target_critic')

		# update values for slowly-changing targets towards current actor and critic
		update_slow_target_ops = []
		for i, slow_target_actor_var in enumerate(slow_target_actor_vars):
			update_slow_target_actor_op = slow_target_actor_var.assign(tau*actor_vars[i]+(1-tau)*slow_target_actor_var)
			update_slow_target_ops.append(update_slow_target_actor_op)

		for i, slow_target_var in enumerate(slow_target_critic_vars):
			update_slow_target_critic_op = slow_target_var.assign(tau*critic_vars[i]+(1-tau)*slow_target_var)
			update_slow_target_ops.append(update_slow_target_critic_op)

		self.update_slow_targets_op = tf.group(*update_slow_target_ops, name='update_slow_targets')

		# One step TD targets y_i for (s,a) from experience replay
		# = r_i + gamma*Q_slow(s',mu_slow(s')) if s' is not terminal
		# = r_i if s' terminal
		targets = tf.expand_dims(self.reward_ph, 1) + tf.expand_dims(self.is_not_terminal_ph, 1) * gamma * slow_q_values_next

		# 1-step temporal difference errors
		td_errors = targets - q_values_of_given_actions

		# critic loss function (mean-square value error with regularization)
		critic_loss = tf.reduce_mean(tf.square(td_errors))
		for var in critic_vars:
			if not 'bias' in var.name:
				critic_loss += l2_reg_critic * 0.5 * tf.nn.l2_loss(var)

		# critic optimizer
		self.critic_train_op = tf.train.AdamOptimizer(lr_critic*lr_decay**episodes).minimize(critic_loss)

		# actor loss function (mean Q-values under current policy with regularization)
		actor_loss = -1*tf.reduce_mean(q_values_of_suggested_actions)
		for var in actor_vars:
			if not 'bias' in var.name:
				actor_loss += l2_reg_actor * 0.5 * tf.nn.l2_loss(var)

		# actor optimizer
		# the gradient of the mean Q-values wrt actor params is the deterministic policy gradient (keeping critic params fixed)
		self.actor_train_op = tf.train.AdamOptimizer(lr_actor*lr_decay**episodes).minimize(actor_loss, var_list=actor_vars)

		# initialize session
		self.sess = tf.Session()	
		self.sess.run(tf.global_variables_initializer())

		
		# Episode variables
		self.reset_episode_vars()


	def add_to_memory(self, experience):
		self.replay_memory.append(experience)

	def sample_from_memory(self, minibatch_size):
		return random.sample(self.replay_memory, minibatch_size)


	def reset_episode_vars(self):
		self.total_reward = 0
		self.steps_in_ep = 0

		# Initialize exploration noise process
		self.noise_process = np.zeros(self.action_dim)
		self.noise_scale = (self.initial_noise_scale * self.noise_decay**self.ep) * (self.action_space_high - self.action_space_low)
		self.ep += 1

		self.last_observation = None
		self.last_action = None
		
	def step(self, observation, reward, done):
		# choose action based on deterministic policy
		action_for_state, = self.sess.run(self.actions, feed_dict = {
			self.state_ph: observation, 
			self.is_training_ph: False
		})

		# add temporally-correlated exploration noise to action (using an Ornstein-Uhlenbeck process)
		# print(action_for_state)
		self.noise_process = self.exploration_theta*(self.exploration_mu - self.noise_process) + self.exploration_sigma*np.random.randn(self.action_dim)
		# print(self.noise_scale*self.noise_process)
		action_for_state += self.noise_scale*self.noise_process

		self.total_reward += reward

		
		# Save experience / reward
		if self.last_observation is not None and self.last_action is not None:
			self.add_to_memory((self.last_observation[0], self.last_action, reward, observation[0], 
				# is next_observation a terminal state?
				# 0.0 if done and not env.env._past_limit() else 1.0))
				0.0 if done else 1.0))

		# update network weights to fit a minibatch of experience
		if self.total_steps%self.train_every == 0 and len(self.replay_memory) >= self.minibatch_size:

			# grab N (s,a,r,s') tuples from replay memory
			minibatch = self.sample_from_memory(self.minibatch_size)

			# update the critic and actor params using mean-square value error and deterministic policy gradient, respectively
			_, _ = self.sess.run([self.critic_train_op, self.actor_train_op], 
				feed_dict = {
					self.state_ph: np.asarray([elem[0] for elem in minibatch]),
					self.action_ph: np.asarray([elem[1] for elem in minibatch]),
					self.reward_ph: np.asarray([elem[2] for elem in minibatch]),
					self.next_state_ph: np.asarray([elem[3] for elem in minibatch]),
					self.is_not_terminal_ph: np.asarray([elem[4] for elem in minibatch]),
					self.is_training_ph: True})

			# update slow actor and critic targets towards current actor and critic
			_ = self.sess.run(self.update_slow_targets_op)


		self.last_observation = observation
		self.last_action = action_for_state
		self.total_steps += 1
		self.steps_in_ep += 1
		

		if done:
			_ = self.sess.run(self.episode_inc_op)
			print('Reward: {:.3f}, Steps: {:.3f}, Final noise scale: {:.3f}, Z: {:.3f}'.format(self.total_reward, self.steps_in_ep, self.noise_scale, state[0]))
			print(self.ep, self.total_reward)
			self.reset_episode_vars()

		return np.array([0,0] + list(action_for_state) + [0,0,0])


