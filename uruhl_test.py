import gym
import time
import math
env = gym.make('Uruhl-v0')
a = [-90,90]
b = [90,-90]
action = a
for i_episode in range(100):
    observation = env.reset()
    for t in range(3000):
        env.render()
        action = [10, -10]#t/30]
        if (0 and t %100==0):
            if (action == a):
                action = b
            else:
                action = a
	if (0 and t <= 100):
	    action = [-10,10]
	if (0 and t>100 and t < 300):
	    action = [130,-130]
	if (0 and t==300 ):
	    action = [-10,10]
	    print "HHHHHHHHHHHHHHHHHHHHHHHHEEEEYYYY"
	if (0 and t>300):
	    action = [0,0]
        observation, reward, done, info = env.step(action)
	time.sleep(0.09)
	#print(reward,done)
	print "Angle",((observation[6])*(180/math.pi))%180
        if done:
            print("Episode {} finished after {} episodes".format(i_episode,t+1))
            time.sleep(1.2)
            break
