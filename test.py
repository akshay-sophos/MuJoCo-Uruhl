#! /usr/bin/python
import gym
import math
import random
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from matplotlib import cm
###############################################################################
#### Learning related constants ####
MIN_EXPLORE_RATE = 0.01 #The min exploration rate; The max is 1
PULL_UP_EXPLORE_LINE = 16 #Increase this to decrease the rate of decrease of epsilon

START_LEARNING_RATE = 1 #The max learning_rate ITS FOR Q FUNCTION AND NOT GRADIENT DESCENT...
MIN_LEARNING_RATE = 0.01   #The min learning_rate
PULL_UP_LEARN_RATE = 16

START_DISCOUNT_FACTOR = 0 #The min discount_factor
MAX_DISCOUNT_FACTOR = 0.99  #The max discount_factor
PULL_UP_DISC_FACTOR = 16

#    exp_qVal = (1-learning_rate)* curQval  + learning_rate*( reward + discount_factor*nextMaxQval )

TF_LEARN_RATE = 0.005 #Learning Rate for Gradient Descent

#### Defining the simulation related constants ####

#Defines the number of episodes it should perform the increment/decrement of values
NUM_EPISODES = 3000
NUM_EPISODES_PLATEAU_EXPLORE =  3000*NUM_EPISODES/5000
NUM_EPISODES_PLATEAU_LEARNING = 2000*NUM_EPISODES/5000
NUM_EPISODES_PLATEAU_DISCOUNT = 2000*NUM_EPISODES/5000

STREAK_TO_END = 120
SOLVED_T = 200          # anything more than this returns Done = true for the openAI Gym
#IMU_sensor_count = 6
#NEG_REW = -5 #negative reward for fallen pole
DISPLAY_RATES =False #To display the rates as a graph over time
DISPLAY_ENV = True #To display the render for enviroment
# if DISPLAY_ENV ==True:
#     from time import sleep
###############################################################################
# number of neurons in each layer
input_num_units = 3
hidden_num_units1 = 15#8
hidden_num_units2 = 15#8
hidden_num_units3 = 10#8
output_num_units = 1

################################################################################
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
##############################################################################
# define placeholders
tf_x = tf.placeholder(tf.float32, [None, input_num_units],name="Input")
tf_exp_q =  tf.placeholder(tf.float32,[None,1],name="Expected_Q_value")
hidden_layer1 = tf.layers.dense(tf_x, hidden_num_units1, tf.nn.tanh)
hidden_layer2 = tf.layers.dense(hidden_layer1, hidden_num_units2, tf.nn.relu)
hidden_layer3 = tf.layers.dense(hidden_layer2, hidden_num_units3, tf.nn.relu)
output_layer = tf.layers.dense(hidden_layer3, output_num_units)
cost = tf.losses.mean_squared_error(tf_exp_q, output_layer)
optimizer = tf.train.AdamOptimizer()
train_op = optimizer.minimize(cost)
###############################################################################
env = gym.make('Uruhl-v0')
cost_plot = []
reward_plot = []
MOV_WIN = [0]
vall = 0
rew = 0
t = 0
angle = 0
IM_INTERVAL = 100
MAX_STACK = 2000
WINDOW_SIZE = 100
start = 0
###############################################################################
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    ep = 0
    observation = env.reset()
    #########################################
    while ep<=NUM_EPISODES:
        explore_rate = get_explore_rate(ep)
        learning_rate = get_learning_rate(ep)
        discount_factor = get_discount_factor(ep)
        tot_cost = 0
        tot_rew = 0

################################################################################
    # Qlearning is off-policy.
    # if max=True, return the (maxQ, bestAction)
    # if max = False, return the (bestQ, correspondingAction) based on explore_rate
        def Q(observation,max):
            action_scaling = 5
            act_num = 10
            acto = np.zeros(2*act_num+1)
            V = np.append(observation,[-act_num*action_scaling,act_num*action_scaling])[np.newaxis]
            for i in range((-act_num)+1,act_num+1):
                V = np.vstack([V,(np.append(observation,[i*action_scaling,-i*action_scaling]))[np.newaxis]])
            acto = sess.run(output_layer,feed_dict={tf_x:V})#[0][0]
            act = np.argmax(acto)
            maxQ = acto[act]
            act = act-act_num
            maxA = [act*action_scaling,-act*action_scaling]
            #Plotting Bargraph
            if t==10 and ep%IM_INTERVAL == 0:
                plt.rcdefaults()
                plt.bar(np.arange(-act_num,act_num+1,1),np.concatenate( acto, axis=0 ))#,align='center',alpha=1)
                #plt.yticks(acto, np.arange(-5,5))
                plt.xlabel('Action')
                x = 'Q value for Angle = '+str(observation)+ 'with Angular Velocity ='+str(av)
                plt.title(x)
                x = 'fig'+str(int(ep/IM_INTERVAL))+'.png'
                plt.savefig('./bargraph_angle/'+x)
                plt.clf()
            if (max ==True):
                return (maxQ, maxA)
            else:
                if(random.random()<explore_rate): # EXPLORE high explore rate => more exploration
                    act = random.randrange(2*act_num+1)-act_num
                    #print "RANDOM"
                    return(acto[act+act_num],[act*action_scaling,-act*action_scaling])
                else:                             # DONT EXPLORE
                    #print "not random"
                    return (maxQ, maxA)
################################################################################
        for t in range(SOLVED_T): #Window
            pobs = observation
            curQval,action = Q(pobs,False)
            observation,reward,done,av = env.step(action)
            if (DISPLAY_ENV == True):# and ep > (NUM_EPISODES-500):
                env.render()
            nextMaxQval,_ = Q(observation, True)
            exp_qVal = (1-learning_rate)* curQval  + learning_rate*( reward + discount_factor*nextMaxQval )
            action_array = np.asarray(action).reshape([1,2])
            if start==0:
                I = np.append(pobs,action_array)[np.newaxis]
                Z = np.asarray(exp_qVal).reshape([1,1])
            else:
                I = np.vstack([I,np.append(pobs,action_array)[np.newaxis]])
                Z = np.vstack([Z,np.asarray(exp_qVal).reshape([1,1])])
            start = 1
            if len(I)>MAX_STACK:
                I = np.delete(I,0,axis=0)#np.s_[0:int(MOV_WIN[0])],axis=0)
                Z = np.delete(Z,0,axis=0)#np.s_[0:int(MOV_WIN[0])],axis=0)
            if len(I)>WINDOW_SIZE and len(I)<=MAX_STACK:
                I_train = I[len(I)-1]
                Z_train = Z[len(Z)-1]
                samp = random.sample(range(0,len(I)),WINDOW_SIZE-1)
                for x in samp:
                    I_train = np.vstack([I_train,I[x]])
                    Z_train = np.vstack([Z_train,Z[x]])
                _,c = sess.run([train_op,cost], {tf_x: I_train, tf_exp_q: Z_train})
            # elif len(I)<=WINDOW_SIZE:
            #     print I
            #     _,c = sess.run([train_op,cost], {tf_x: I, tf_exp_q: Z})

            tot_rew +=reward
            rew += reward
            if done == True:
                print("Reward: %.5f"%rew)
                reward_plot = np.append(reward_plot,rew)
                rew = 0
                observation = env.reset()
        cost_plot = np.append(cost_plot,c)#tot_cost)
        print(ep, "T_Cost:%.4f" %c)#,  "T_Reward:%d" %tot_rew)
        ep = ep+1

##############################################################################

    saver = tf.train.Saver()
    saver.save(sess, './save/model.ckpt')
    print("\n Training Over")
    # To plot Reward and Cost w.r.t time
    f, axarr = plt.subplots(2, sharex=True)
    axarr[0].plot(cost_plot)
    axarr[0].set_title('cost_plot')
    #axarr[0].set_ylim([0, 1])
    axarr[1].plot(reward_plot)
    axarr[1].set_title('reward_plot')
    plt.savefig('./bargraph_angle/cost.png')
    #plt.show()
                    ###############################
                    ###############################
