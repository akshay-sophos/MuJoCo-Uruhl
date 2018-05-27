#!/usr/bin/env python
import gym
import math
import random
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from matplotlib import cm

#### Learning related constants ####
MIN_EXPLORE_RATE = 0.01 #The min exploration rate; The max is 1
PULL_UP_EXPLORE_LINE = 3 #Increase this to decrease the rate of decrease of epsilon

START_LEARNING_RATE = 1 #The max learning_rate ITS FOR Q FUNCTION AND NOT GRADIENT DESCENT
MIN_LEARNING_RATE = 0.01   #The min learning_rate
PULL_UP_LEARN_RATE = 1

START_DISCOUNT_FACTOR = 0 #The min discount_factor
MAX_DISCOUNT_FACTOR = 0.99  #The max discount_factor
PULL_UP_DISC_FACTOR = 1

TF_LEARN_RATE = 0.01 #Learning Rate for Gradient Descent

#### Defining the simulation related constants ####

#Defines the number of episodes it should perform the increment/decrement of values
NUM_EPISODES = 5000
NUM_EPISODES_PLATEAU_EXPLORE =  3000#*3/5
NUM_EPISODES_PLATEAU_LEARNING = 2000#*3/5
NUM_EPISODES_PLATEAU_DISCOUNT = 2000#*3/5

STREAK_TO_END = 120
SOLVED_T = 500          # anything more than this returns Done = true for the openAI Gym

NEG_REW = -50 #negative reward for fallen pole
DISPLAY_RATES = False #To display the rates as a graph over time
DISPLAY_ENV = True  #To display the render for enviroment
if DISPLAY_ENV ==True:
    from time import sleep

# number of neurons in each layer
input_num_units = 10
hidden_num_units1 = 100
hidden_num_units2 = 100
hidden_num_units3 = 100
output_num_units = 1

#def pcom(s):
    #print(s, end='', flush=True)

def get_explore_rate(t):
    maxValReached = math.log10(NUM_EPISODES_PLATEAU_EXPLORE)
    return max( min(1, 1.0 - math.log10((t+0.1)/PULL_UP_EXPLORE_LINE)/maxValReached), MIN_EXPLORE_RATE)
def get_learning_rate(t):
    maxValReached = math.log10(NUM_EPISODES_PLATEAU_LEARNING)
    return max(min(START_LEARNING_RATE, 1.0 - (math.log10(t+0.1)/PULL_UP_LEARN_RATE)/maxValReached), MIN_LEARNING_RATE)
def get_discount_factor(t):
    maxValReached = math.log10(NUM_EPISODES_PLATEAU_DISCOUNT)
    return min(max(START_DISCOUNT_FACTOR, (math.log10(t+0.1)/PULL_UP_DISC_FACTOR)/maxValReached), MAX_DISCOUNT_FACTOR)
    # return MAX_DISCOUNT_FACTOR


if DISPLAY_RATES:
    numPoints = 100;
    a = np.linspace(0,NUM_EPISODES,numPoints);
    e = np.zeros(numPoints);
    l = np.zeros(numPoints);
    d = np.zeros(numPoints);
    for i in range(numPoints):
        e[i]= get_explore_rate(a[i])
        l[i]= get_learning_rate(a[i])
        d[i]= get_discount_factor(a[i])

    f, axarr = plt.subplots(3, sharex=True)
    axarr[0].plot(a,e)
    axarr[0].set_title('Exploration Factor')

    axarr[1].plot(a,l)
    axarr[1].set_title('Learning Rate of Q function')

    axarr[2].plot(a,d)
    axarr[2].set_title('Discount factor for Q function')
    plt.show()

# define placeholders
tf_x = tf.placeholder(tf.float32, [None, input_num_units],name="Input")
#y = tf.placeholder(tf.float32, [1, output_num_units],name="Output")
# tf_qval = tf.placeholder(tf.float32,[1,1],name="Q_value")
tf_exp_q =  tf.placeholder(tf.float32,[None,1],name="Expected_Q_value")

if 0:
    weights = {
    'hidden1': tf.Variable(tf.random_normal([input_num_units, hidden_num_units1], seed=seed)),
    'hidden2': tf.Variable(tf.random_normal([hidden_num_units1, hidden_num_units2], seed=seed)),
    'output': tf.Variable(tf.random_normal([hidden_num_units2, output_num_units], seed=seed))
    }

    biases = {
        'hidden1': tf.Variable(tf.random_normal([hidden_num_units1], seed=seed)),
        'hidden2': tf.Variable(tf.random_normal([hidden_num_units2], seed=seed)),
        'output': tf.Variable(tf.random_normal([output_num_units], seed=seed))
    }
    hidden_layer1 = tf.layers.dense(tf_x, hidden_num_units1, tf.nn.tanh)
    hidden_layer2 = tf.layers.dense(hidden_layer1, hidden_num_units2, tf.nn.relu)
    output_layer = tf.layers.dense(hidden_layer2, output_num_units)
else:
    hidden_layer1 = tf.layers.dense(tf_x, hidden_num_units1, tf.nn.tanh)
    hidden_layer2 = tf.layers.dense(hidden_layer1, hidden_num_units2, tf.nn.relu)
    hidden_layer3 = tf.layers.dense(hidden_layer2, hidden_num_units3, tf.nn.relu)
    output_layer = tf.layers.dense(hidden_layer3, output_num_units)

cost = tf.losses.mean_squared_error(tf_exp_q, output_layer)
optimizer = tf.train.AdamOptimizer(TF_LEARN_RATE)
# optimizer = tf.train.GradientDescentOptimizer(learning_rate=TF_LEARN_RATE)
train_op = optimizer.minimize(cost)


env = gym.make('Uruhl-v0')
cost_plot = [0]
reward_plot = [0]
observation = np.ndarray(shape=(8,1))
with tf.Session() as sess:
    # create initialized variables
    sess.run(tf.global_variables_initializer())
    ep = 0
    #for ep in range(NUM_EPISODES):
    while ep<=NUM_EPISODES:
        explore_rate = get_explore_rate(ep)
        learning_rate = get_learning_rate(ep)
        discount_factor = get_discount_factor(ep)
        observa = env.reset()
        np.copyto(observation,observa)
        np.put(observation,[6,7],[((observa[6])*(180/math.pi))%180,((observa[7])*(180/math.pi))%180])
        if DISPLAY_ENV == True and ep > NUM_EPISODES-200:
            env.render()
        tot_cost = 0
        tot_rew = 0
        # Qlearning is off-policy.
        # if max=True, return the (maxQ, bestAction)
        # if max = False, return the (bestQ, correspondingAction) based on explore_rate
#######################################################################################33
        def Q(observation,max):
            #array returned, make scalar
            acto = np.arange(0,200)
            for i in range(-100,100):
                acto[i+100] = sess.run(output_layer,feed_dict={tf_x:(np.append(observation,[i,-i]))[np.newaxis]})[0][0]

            maxA = np.argmax(acto)
            maxQ = acto[maxA]
            act = maxA -100
            maxA = [act,-act]

            if (max ==True):
                return (maxQ, maxA)
            else:
                if(random.random()<explore_rate): # EXPLORE high explore rate => more exploration
                    act = random.randrange(200)-100
                    return(acto[act],[act,-act])
                else:                             # DONT EXPLORE
                    return (maxQ, maxA)

################################################################################################
        for t in range(SOLVED_T):
            pobs = observation
            curQval,action = Q(pobs,False)
            # reward is 1 for all steps except those that are called after a done=True is returned
            # done is True when the pole has fell
            observa,reward,done,_ = env.step(action)
            np.copyto(observation,observa)
            np.put(observation,[6,7],[((observa[6])*(180/math.pi))%180,((observa[7])*(180/math.pi))%180])
            if DISPLAY_ENV == True:
                env.render()
            # pcom(action)
            # print ("curQval ", curQval, " action ", action)
            nextMaxQval,_ = Q(observation, True)
            # if done == True and tot_rew<199:
            #     reward = NEG_REW
            exp_qVal = (1-learning_rate)* curQval  + learning_rate*( reward + discount_factor*nextMaxQval )
            action_array = np.asarray(action).reshape([1,2])
            exp_qVal_array = np.asarray(exp_qVal).reshape([1,1])
            inpu = np.append(pobs,action_array)[np.newaxis]
            if t==0:
                I = inpu
                Z = exp_qVal_array
            else:
                I = np.vstack([I,inpu])
                Z = np.vstack([Z,exp_qVal_array])
            #_,c = sess.run([train_op,cost], {tf_x: np.append(pobs, action_array)[np.newaxis], tf_exp_q: exp_qVal_array})
            #c1 = sess.run([cost], {tf_x: np.append(pobs, action_array)[np.newaxis], tf_exp_q: exp_qVal_array})
            # print('c c1 ',c, ' ', c1)
            #tot_cost += c
            tot_rew +=reward
            if done == True:
                break
        _,c = sess.run([train_op,cost], {tf_x: I, tf_exp_q: Z})

        if(ep%10 == 0):
            cost_plot = np.append(cost_plot,c)#tot_cost)
            reward_plot = np.append(reward_plot, tot_rew)
        print(ep, "T_Cost:%.4f" %c,  "T_Reward:%d" %tot_rew)
        ep = ep+1

    saver = tf.train.Saver()
    saver.save(sess, './save/model.ckpt')
    print("\n Training Over")

# To plot Reward and Cost w.r.t time
    f, axarr = plt.subplots(2, sharex=True)
    axarr[0].plot(cost_plot)
    axarr[0].set_title('cost_plot')
    axarr[0].set_ylim([0, 1])
    axarr[1].plot(reward_plot)
    axarr[1].set_title('reward_plot')
    plt.savefig('./cost.png')
    #plt.show()
