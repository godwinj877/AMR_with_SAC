import time
import os
import math
import numpy as np
import tensorflow as tf
import tensorflow.keras as keras
import tensorflow_probability as tfp
from tensorflow.keras.layers import Dense
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential

# Importing ros libraries
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
from sensor_msgs.msg import LaserScan
from nav_msgs.msg import Odometry
from std_srvs.srv import Empty

class RBuffer:
    def __init__(self, max_size = int, input_shape = int, no_of_actions = int):
        self.mem_size = max_size
        self.ith_index = 0
        self.current_states = np.zeros((self.mem_size, 14))
        self.next_states = np.zeros((self.mem_size, 14))
        self.action_taken = np.zeros((self.mem_size, 2))
        self.reward = np.zeros((self.mem_size))
        self.terminal = np.zeros((self.mem_size))
        self.prev_action = np.zeros((self.mem_size, 2))
        
    def transition(self, s, a, r, nxt_s, t):
        i = self.ith_index % self.mem_size
        self.current_states[i] = s
        self.next_states[i] =nxt_s
        self.action_taken[i] = a
        self.reward[i] = r
        self.terminal[i] = t
        
        self.ith_index +=1
    
    def sample_buffer(self, batch_size):
        max_mem = min(self.mem_size, self.ith_index)
        indexes_with_ones = np.where(self.reward > 190)[0]
        indexes_with_neg = np.where(self.reward < -190)[0]
        indexes_with_small_pos = np.where(self.reward > 1)[0]
        #Out of 1024 Batch size, 75 Sampled from the Goal state transitions, 75 sampled from collisions, 150 from the good path taken by the agent.
        #Helps in learning the environment better due to vast state spaces.
        if len(indexes_with_small_pos) == 0: 
            last_150_small_pos = np.array[(0)]
        else:
            repeated_small_pos_indexes = np.tile(indexes_with_small_pos, max(1, 150 // len(indexes_with_small_pos) + 1))
            last_150_small_pos = repeated_small_pos_indexes[-150:]

        if len(indexes_with_ones) == 0:
            last_75_pos_indexes = np.array([0])
        else:
            repeated_pos_indexes = np.tile(indexes_with_ones, max(1, 75 // len(indexes_with_ones) + 1))
            last_75_pos_indexes = repeated_pos_indexes[-75:]

        if len(indexes_with_neg) == 0:
            last_75_neg_indexes = np.array([0])
        else:   
            repeated_neg_indexes = np.tile(indexes_with_neg, max(1, 75 // len(indexes_with_neg) + 1))
            last_75_neg_indexes = repeated_neg_indexes[-75:]

        batch = np.random.choice(max_mem, batch_size - len(last_75_neg_indexes) - len(last_75_pos_indexes) - len(last_150_small_pos))
        batch = np.concatenate([last_75_pos_indexes, batch, last_75_neg_indexes, last_150_small_pos])
        c_states = self.current_states[batch]
        n_states = self.next_states[batch]
        actions = self.action_taken[batch]   
        rewards = self.reward[batch]
        terminal = self.terminal[batch]
        return  c_states, actions, rewards, n_states, terminal
    
class CriticNet(keras.Model): #Critic Network Architecture 
    def __init__(self, name='critic'): 
        super(CriticNet, self).__init__()
        self.q_value = Sequential([              
            Dense(units=512, activation='relu'),            
            Dense(units=512, activation='relu'),  
            Dense(units=512, activation='relu'),          
            Dense(units=1, activation='linear'),
        ])
        self.model_name = name
    def call(self, state, action):
        return self.q_value(tf.concat([state, action], axis=1))
    
class ValueNet(keras.Model): #Value Network Architecture 
    def __init__(self, name='value'):
        super(ValueNet, self).__init__()
        self.v_value = Sequential([              
            Dense(units=512, activation='relu'),            
            Dense(units=512, activation='relu'),  
            Dense(units=512, activation='relu'),          
            Dense(units=1, activation='linear'),
        ])
        self.model_name = name
    def call(self, state):
        return self.v_value(state)
    
class ActorNetwork(keras.Model): #Actor Network Architecture 
    def __init__(self, name='actor'):
        super(ActorNetwork, self).__init__()
        self.noise = 1e-6
        self.log_std_min = -20
        self.log_std_max = 2
        self.a_values = Sequential([
            Dense(units=512, activation='relu'),            
            Dense(units=512, activation='relu'),  
            Dense(units=512, activation='relu'),          
            Dense(units=4, activation='relu'),
        ])
        self.model_name = name
  
    def call(self,state):
        called = self.a_values(state)
        mu = called[:, :2]  
        sigma = called[:, 2:]
        return mu, sigma
    
    def sample_normal(self, state):  #Normal sampling and Re-Parametrisation layer (Refer Documentation for better understanding)
        mu, sigma = self.call(state)
        log_sigma = tf.clip_by_value(sigma, self.log_std_min, self.log_std_max)
        std = tf.exp(log_sigma)
        normal = tfp.distributions.Normal(0, 1)
        z = normal.sample()
        z = tf.convert_to_tensor(z)
        actions = tf.math.tanh(mu + std * z)
        actions_tf0 = (tf.math.sigmoid(actions[:, 0])) * 0.3 #Linear velocity is limited to [~0.09, 0.3]. Can be tuned further based on performance check
        actions_tf1 = (actions[:, 1]) * 0.5 #Angular velocity is limited to [-0.5, 0.5]
        a= tf.stack([actions_tf0, actions_tf1], axis=-1)
        log_probs =  tfp.distributions.Normal(mu, std).log_prob(mu + std * z)
        log_probs -= tf.math.log(1-tf.math.pow(a,2)+self.noise)
        log_probs = tf.math.reduce_sum(log_probs, axis=1, keepdims=True) #131 to 133 is the calculation of Entropy
        return a, log_probs
    
class Agent():
    def __init__(self, alpha=0.0003, beta=0.0003, input_shape=[14],
                gamma=0.99, no_of_actions=2, max_size=1000000, tau=0.005, 
                batch_size=1024): # Batch sizes and learning rates can be changed here. Refer Documentary for technical details on
        self.gamma = gamma        # the functions of Neural networks and their loss functions
        self.tau = tau
        self.memory = RBuffer(max_size, input_shape, no_of_actions)
        self.actor = ActorNetwork(name='actor')
        self.critic_1 = CriticNet(name='critic_1')
        self.critic_2 = CriticNet(name='critic_2')
        self.value_network = ValueNet(name='value_network')
        self.target_network = ValueNet(name='target_value_network')
        self.batch_size = batch_size
        
        self.actor.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=alpha))
        self.critic_1.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=beta))
        self.critic_2.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=beta))
        self.value_network.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=beta))
        self.target_network.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=beta))

        #Checkpoint manager from Tensorflow
        self.actor_ckpt = tf.train.Checkpoint(actor=self.actor)
        self.critic_1_ckpt = tf.train.Checkpoint(critic_1=self.critic_1)
        self.critic_2_ckpt = tf.train.Checkpoint(critic_2=self.critic_2)
        self.value_network_ckpt = tf.train.Checkpoint(value_network=self.value_network)
        self.target_network_ckpt = tf.train.Checkpoint(target_network=self.target_network)

        self.actor_manager = tf.train.CheckpointManager(self.actor_ckpt, 'actor_checkpoints', max_to_keep=None)
        self.critic_1_manager = tf.train.CheckpointManager(self.critic_1_ckpt, 'critic_1_checkpoints', max_to_keep=None)
        self.critic_2_manager = tf.train.CheckpointManager(self.critic_2_ckpt, 'critic_2_checkpoints', max_to_keep=None)
        self.value_network_manager = tf.train.CheckpointManager(self.value_network_ckpt, 'value_network_checkpoints', max_to_keep=None)
        self.target_network_manager = tf.train.CheckpointManager(self.target_network_ckpt, 'target_network_checkpoints', max_to_keep=None)

    def choose_action(self, obs):
        state = tf.convert_to_tensor([obs])
        a,log_probs = self.actor.sample_normal(state)
        return a
    
    def update_target_network(self,tau):
        target_weights = []
        target_value_network_wts = self.target_network.weights
        for i, value_network_wts in enumerate(self.value_network.weights):
            target_weights.append(value_network_wts*tau + target_value_network_wts[i]*(1-tau))
        
        self.target_network.set_weights(target_weights)
        
    def learn(self):
        if self.memory.ith_index < self.batch_size:
            return
        
        print("Started learning")
        current_state, action, reward, next_state, done = self.memory.sample_buffer(self.batch_size)
        
        current_state = tf.convert_to_tensor(current_state, dtype=tf.float32)
        action_from_buffer = tf.convert_to_tensor(action, dtype=tf.float32)
        reward = tf.convert_to_tensor(reward, dtype=tf.float32)
        next_state = tf.convert_to_tensor(next_state, dtype=tf.float32)
        
        with tf.GradientTape() as tape:
            value_network_v = tf.squeeze(self.value_network(current_state),axis=1)
            target_value_network_v = tf.squeeze(self.target_network(current_state),axis=1)
            action_from_actor, log_probs = self.actor.sample_normal(current_state)
            q1_critic_1 = self.critic_1(current_state, action_from_actor)
            q2_critic_2 = self.critic_2(current_state, action_from_actor)
            min_q = tf.squeeze(tf.math.minimum(q1_critic_1,q2_critic_2),1)
            min_q_with_entropy = min_q - log_probs
            loss_func_v = 0.5* keras.losses.MSE(value_network_v, min_q_with_entropy)
            
        value_network_gradient = tape.gradient(loss_func_v,
                                            self.value_network.trainable_variables)
        self.value_network.optimizer.apply_gradients(zip(value_network_gradient,
                                                        self.value_network.trainable_variables))
        
        with tf.GradientTape() as tape:
            
            action_from_actor,log_probs = self.actor.sample_normal(current_state)
            q1_critic_1 = self.critic_1(current_state, action_from_actor)
            q2_critic_2 = self.critic_2(current_state, action_from_actor)
            min_q = tf.squeeze(tf.math.minimum(q1_critic_1,q2_critic_2),1)
            min_q_with_entropy = min_q - log_probs
            loss_a = -min_q_with_entropy
            loss_func_a = tf.math.reduce_mean(loss_a)
        
        actor_network_gradient = tape.gradient(loss_func_a, self.actor.trainable_variables)
        self.actor.optimizer.apply_gradients(zip(actor_network_gradient, self.actor.trainable_variables))
        
        with tf.GradientTape(persistent=True) as tape:
            value_network_vNext = tf.squeeze(self.target_network(next_state),1)
            q_from_env = reward + self.gamma * value_network_vNext
            q1_critic_1 = self.critic_1(current_state, action_from_buffer)
            q2_critic_2 = self.critic_2(current_state, action_from_buffer)  
            c1_loss = 0.5 * keras.losses.MSE(q_from_env, q1_critic_1)
            c2_loss = 0.5 * keras.losses.MSE(q_from_env, q2_critic_2)
        
        c1_grad = tape.gradient(c1_loss, self.critic_1.trainable_variables)
        c2_grad = tape.gradient(c2_loss, self.critic_2.trainable_variables)

        max_grad_norm = 0.5
        c1_grad, _ = tf.clip_by_global_norm(c1_grad, max_grad_norm)
        c2_grad, _ = tf.clip_by_global_norm(c2_grad, max_grad_norm)
        
        self.critic_1.optimizer.apply_gradients(zip(c1_grad,
                                                    self.critic_1.trainable_variables))
        self.critic_2.optimizer.apply_gradients(zip(c2_grad,
                                                    self.critic_2.trainable_variables))
        
        self.update_target_network(0.005)
    
            
    def store_transition(self,current_state,action,reward,next_state,t):
        self.memory.transition(current_state,action,reward,next_state,t) 
        
    def save_models(self):
        print('... saving models ...')
        self.actor_manager.save()
        self.critic_1_manager.save()
        self.critic_2_manager.save()
        self.value_network_manager.save()
        self.target_network_manager.save()

    def load_models(self):
        print('... loading models ...')
        self.actor_manager.restore_or_initialize()
        self.critic_1_manager.restore_or_initialize()
        self.critic_2_manager.restore_or_initialize()
        self.value_network_manager.restore_or_initialize()
        self.target_network_manager.restore_or_initialize()

def normalize_heading_error(delta_phi):
    if delta_phi > np.pi:
        normalized_phi = delta_phi - 2 * np.pi
    elif delta_phi < -np.pi:
        normalized_phi = delta_phi + 2 * np.pi
    else:
        normalized_phi = delta_phi
    return normalized_phi

def pos_state(position, goal_pos, moving_angle):
    a = np.sqrt(np.sum((position - goal_pos)**2))
    delta_y = goal_pos[1] - position[1]
    delta_x = goal_pos[0] - position[0]
    delta_phi = np.arctan2(delta_y, delta_x) - moving_angle
    b = normalize_heading_error(delta_phi)
    position_obs = np.array([a,b[-1]])
    return position_obs   

def get_reward(p_prev, p_now, goal_state, action, termination):    
    r1 = np.sqrt(np.sum((p_prev - goal_state)**2))
    r2 = np.sqrt(np.sum((p_now - goal_state)**2))
    Rh = (r1 - r2)*50
    print("r2: ", r2)
    if(termination):
        Rcollision = -200
    else:
        Rcollision = 0
    if(r2 < 0.2):  #Considered as goal reached if its within 0.2m radius to the goal position
        Rgoal = 200
    else:
        Rgoal = 0
    reward =  Rh + Rcollision + Rgoal 
    return reward

# Velocity Publisher
class VelocityPublisher(Node):
    def __init__(self):
        super().__init__("velocity_publisher")
        self.velocity_publisher = self.create_publisher(Twist, "/cmd_vel", 10)

    def publish_velocity(self, linear_x, angular_z):
        twist_msg = Twist()
        twist_msg.linear.x = linear_x
        twist_msg.angular.z = angular_z
        self.velocity_publisher.publish(twist_msg)

# Laser Subscriber
class LaserSubscriber(Node):
    def __init__(self):
        super().__init__("laser_subscriber")
        self.laser_obs = []
        self.laser_sub = self.create_subscription(LaserScan, "/scan", self.laser_callback, 10)

    def laser_callback(self, msg):
        self.laser_obs = (msg.ranges)

# Position Subscriber
class PositionSubscriber(Node):
    def __init__(self):
        super().__init__("position_subscriber")
        self.position_obs = []
        self.moving_angle = []
        self.position_sub = self.create_subscription(Odometry, "/odom", self.pos_callback, 10)

    def pos_callback(self, msg):
        self.position_obs = ([msg.pose.pose.position.x, msg.pose.pose.position.y])
        # Orientation
        x = msg.pose.pose.orientation.x
        y = msg.pose.pose.orientation.y
        z = msg.pose.pose.orientation.z
        w = msg.pose.pose.orientation.w
        t3 = 2.0*(w*z + x*y)
        t4 = 1.0 - 2.0*(y*y + z*z)

        moving_angle = math.atan2(t3, t4)
        self.moving_angle = (moving_angle)

# Gazebo Resetter
class GazeboWorldResetter(Node):
    def __init__(self):
        super().__init__('gazebo_world_resetter')
        # Create a client for the reset_simulation service
        self.reset_simulation_client = self.create_client(Empty, '/reset_simulation')
        # Wait for the service to be available
        while not self.reset_simulation_client.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('Service /reset_simulation not available, waiting...')
        self.call_reset_simulation_service()

    def call_reset_simulation_service(self):
        # Create an empty request
        request = Empty.Request()
        # Call the reset_simulation service
        future = self.reset_simulation_client.call_async(request)
        # Wait for the service call to complete
        rclpy.spin_until_future_complete(self, future)
        # Check if the service call was successful
        if future.result() is not None:
            self.get_logger().info('Gazebo world reset successful')
        else:
            self.get_logger().error('Failed to reset Gazebo world')


if __name__ == '__main__':
    goal_state = np.array([3.0, 1.0])
    agent = Agent()
    agent.load_models()
    prev_action = np.array([0,0])
    n_steps = 200
    n_games = 500
    
    # Running ros
    rclpy.init()

    # Initialize the ROS nodes
    laser_node = LaserSubscriber()
    pos_node = PositionSubscriber()
    vel_node = VelocityPublisher()

    score_history = []

    for i in range(n_games):

        prev_action = np.array([0,0])     
        # Running laser node
        rclpy.spin_once(laser_node)
        laser_obs = np.array(laser_node.laser_obs)
        laser_obs = laser_obs.flatten()
        laser_obs = laser_obs

        # Running position node
        rclpy.spin_once(pos_node)
        position = np.array(pos_node.position_obs)
        position = position.flatten()

        moving_angle = np.array(pos_node.moving_angle)
        moving_angle = moving_angle.flatten()
        moving_angle = moving_angle

        position_obs = pos_state(position, goal_state, moving_angle) #Initial State is ready
        termination = 0
        score = 0

        for j in range(n_steps):
            laser_obs = laser_obs.flatten()
            for idx in range(len(laser_obs)):

                if(laser_obs[idx] > 10): #Limited the LiDAR output to 10, incase there's no obstacle in the respective direction
                    laser_obs[idx] = 10

            state = np.concatenate((prev_action, laser_obs[:], position_obs[:]))
            print(agent.target_network(tf.convert_to_tensor([state])))
            print(agent.value_network(tf.convert_to_tensor([state])))
            action = agent.choose_action(state)
            action = action.numpy()
            action = action.reshape(-1)
            print("Action: ", action)
            ### we sent state and got action from our network
            ### Publish this action into ROS, our robot should move according to the action we sent
            vel_node.publish_velocity(float(action[0]), float(action[1]))
            
            prev_action = action
            rclpy.spin_once(laser_node)
            laser_obs = np.array(laser_node.laser_obs)
            laser_obs = laser_obs.flatten()
            laser_obs = laser_obs

            for idx in range(len(laser_obs)):
                if(laser_obs[idx] > 10):
                    laser_obs[idx] = 10

            rclpy.spin_once(pos_node)
            position_data = np.array(pos_node.position_obs)
            position_data = position_data.flatten()
            next_position = position_data

            for idx in range(10):
                if(laser_obs[idx] < 0.35): #If the any obstacle gets closer than 0.35m within LiDAR radius. It was considered as a collision
                    termination = 1
                    break
                termination = 0

            reward = get_reward(position, next_position, goal_state, action, termination)
            print("Reward: ", reward)
            position = next_position


            moving_angle = np.array(pos_node.moving_angle)
            moving_angle = moving_angle.flatten()
            moving_angle = moving_angle

            position_obs = pos_state(position, goal_state, moving_angle)
            next_state = np.concatenate((prev_action, laser_obs[:], position_obs[:]))
            ### we got next state
            
            agent.store_transition(state,action,reward,next_state,termination)
            ### we saved
            score+=reward   

            time.sleep(0.15)       

            if(termination):
                break

            # Resetting the gazebo world when the goal is reached
            if reward > 190:
                print("Reached")
                resetter_after_goal = GazeboWorldResetter()
                rclpy.spin_once(resetter_after_goal)
                resetter_after_goal.destroy_node()
                time.sleep(5)
                break
            
        score_history.append(score)
        avg_score = np.mean(score_history[-100:]) 
        if i%2 == 0:       
            for j in range(50):
                agent.learn()
            agent.save_models()
        print('episode ', i+1, 'score %.1f' % score, 'avg_score %.1f' % avg_score)

        # Reseting world
        resetter = GazeboWorldResetter()
        rclpy.spin_once(resetter)
        resetter.destroy_node()
        time.sleep(2)

    # Destroying the nodes
    laser_node.destroy_node()
    pos_node.destroy_node()
    vel_node.destroy_node()








