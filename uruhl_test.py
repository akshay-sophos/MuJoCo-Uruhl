import gym
import time
env = gym.make('Uruhl-v0')
a = [-10,0]
b = [-10,0]
action = a
for i_episode in range(1):
    observation = env.reset()
    for t in range(1000):
        env.render()
        if (1 and t %100==0):
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
	time.sleep(0.05) 
	#print(reward,done)
	#print observation
        if done:
            print("Episode {} finished after {} episodes".format(i_episode,t+1))
            time.sleep(1.2)
            break
