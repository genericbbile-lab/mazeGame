
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.neural_network import MLPRegressor
import matplotlib.pyplot as plt
from IPython.display import clear_output
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import gymnasium as gym
import time
import random
import warnings
import cv2
from typing import Optional
import pygame
list2=[]
agent_cool=[]
target_cool=[]
is_training=True
rng = np.random.default_rng()
matrix = rng.integers(low=0, high=2, size=(10,10))
x=random.randint(0,matrix.shape[0]-1)
y=random.randint(0,matrix.shape[1]-1)
episode_over=False
total_reward=0


monster=pd.DataFrame({
    "coordinates": [9,9],
    "state":[0,1]
})
monster2=pd.DataFrame({
    "coordinates": [0,9],
    "state":[0,1]
})

class GridWorldEnv(gym.Env):
    global matrix
    global coordinates1
    global x
    global list2
    global is_training
    global y
    def __init__(self, size: int = 10):
        learning_rate = 0.02
        start_epsilon = 1.0         
        epsilon_decay = 0.01  
        final_epsilon = 0.1 
        self.size = size
        self.window_size=512
        self._agent_location = np.array([-1,-1], dtype=np.int32)
        self._target_location = np.array([-1,-1], dtype=np.int32)
        
        self.observation_space = gym.spaces.Dict(
            {
                "agent": gym.spaces.Box(0, size - 1, shape=(2,), dtype=int),   
                "target": gym.spaces.Box(0, size - 1, shape=(2,), dtype=int),  
            }
        )
        self.action_space = gym.spaces.Discrete(4)

        self._action_to_direction = {
            0: np.array([0, 1]),  
            1: np.array([-1, 0]),  
            2: np.array([0, -1]),  
            3: np.array([1, 0]),   
        }
        
        self.q_values = np.zeros((size**2,size**2,4))
        self.list=[]
        for i in range(0,10):
            num=[random.randint(0,size-1),random.randint(0,size-1)]
            num2=num[0]*10+num[1]
            self.list.append(num2)
            list2.append([num[0],num[1]])
            self.q_values[num2]=-1
        learning_rate = 0.01
        epsilon_decay = 0.01
        final_epsilon = 0.01
        discount_factor = 0.95,
        self.lr = learning_rate
        self.discount_factor = discount_factor  
        self.epsilon = start_epsilon
        self.epsilon_decay = epsilon_decay
        self.final_epsilon = final_epsilon

        self.training_error = []
        self.window = None
        self.clock = None
    def decay_epsilon(self):
        self.epsilon = max(self.final_epsilon, self.epsilon - self.epsilon_decay)
    def update(
        self,
        obs: tuple[int, int,bool],
        action: int,
        reward: float,
        terminated: bool,
        next_obs: tuple[int, int,bool],
    ):
        future_q_value = (not terminated) * np.max(self.q_values[next_obs['agent'],next_obs['target']])

        target = reward + self.discount_factor[0] * (future_q_value - self.q_values[obs['agent'],obs['target']][action])
        temporal_difference = self.lr * target
        self.q_values[obs['agent'],obs['target']][action] = (
            self.q_values[next_obs['agent'],obs['target']][action] + temporal_difference
        )
        self.training_error.append(temporal_difference)

    def _get_obs(self):
        global agent_cool
        global target_cool
        agent_cool=self._agent_location[0]*10+self._agent_location[1]
        target_cool=self._target_location[0]*10+self._target_location[1]
        return{"agent": agent_cool,"target":target_cool}
    def get_action(self, obs: tuple[int, int,bool]) -> int:
        if np.random.random() < self.epsilon:
            return self.action_space.sample()
        else:
            return(int(np.argmax(self.q_values[obs['agent'],obs['target']])))
    def addon(self,number):
        number2=number[0]*10+number[1]
        self.list.append(number2)
        list2.append([number[0],number[1]])
        self.q_values[number2]=-1
            
    def _get_info(self):
        
        return {
            "distance": np.linalg.norm(
                self._agent_location - self._target_location, ord=1
            )
        }
    def reset(self, monstersub, fsub, seed: Optional[int] = None, options: Optional[dict] = None):
        
        super().reset(seed=seed)
        
        if not is_training:
            self._target_location=np.array([int(fsub[0]),int(fsub[1])])
            self._agent_location=np.array([monstersub[0],monstersub[1]])
        else:
          self._agent_location = self.np_random.integers(0, self.size, size=2, dtype=int)

          self._target_location = self._agent_location
          while np.array_equal(self._target_location, self._agent_location):
              self._target_location = self.np_random.integers(
                  0, self.size, size=2, dtype=int
              )

        observation = self._get_obs()
        info = self._get_info()
        for item in self.list:
            if item!=0:
                self.q_values[item-1,observation['target'],0]=-1000
            if item<=89:
                self.q_values[item+10,observation['target'],1]=-1000
            if item!=99:
                self.q_values[item+1,observation['target'],2]=-1000
            if item>=10:
                self.q_values[item-10,observation['target'],3]=-1000
        return observation, info
    def step(self, action,steps):
        direction = self._action_to_direction[action]
        self._agent_location = np.clip(
            self._agent_location + direction, 0, self.size - 1
        )

        terminated = np.array_equal(self._agent_location, self._target_location)
        info = self._get_info()
        distance = np.linalg.norm(self._agent_location - self._target_location)
        reward = 2000 if terminated else -1*distance+(-0.1*steps)
        observation = self._get_obs()
        truncated=True if steps>=200 else False
        if(observation['agent'] in self.list):
            reward=-2000
            truncated=True
        return observation,reward,terminated,truncated,info, self._agent_location

gym.register(
    id="gymnasium_env/GridWorld-v0",
    entry_point=GridWorldEnv,
    max_episode_steps=300,
)
env=GridWorldEnv()

episodes=15000
total_steps=0

for i in range(1,episodes):
    observation,info=env.reset(monster['coordinates'],[x,y])
    terminated = False
    truncated=False
    steps=0
    while not terminated and not truncated:
        steps+=1
        action = env.get_action(observation)
        next_observation,reward,terminated,truncated,info,monster['coordinates'] = env.step(action,steps)
        env.update(observation, action, reward, terminated, next_observation)
        observation=next_observation
    env.decay_epsilon()
    print('episode',i,'steps',steps)
    total_steps+=steps
    print('avg_steps:',total_steps/i)
is_training=False



rng = np.random.default_rng()
coins=0
turn=0

int_matrix = rng.integers(low=0, high=2, size=(10,10))
x=random.randint(0,int_matrix.shape[0]-1)
y=random.randint(0,int_matrix.shape[1]-1)
monster['coordinates']=[random.randint(0,int_matrix.shape[0]-1),random.randint(0,int_matrix.shape[1]-1)]

monster2['coordinates']=[random.randint(0,int_matrix.shape[0]-1),random.randint(0,int_matrix.shape[1]-1)]
from collections import defaultdict

def limit(z):
    if (z[0]>9):
      z[0]=9
    elif (z[0]<0):
      z[0]=0
    if (z[1]>9):
      z[1]=9
    elif (z[1]<0):
      z[1]=0
    return z


easy=input("Normal or Hard Mode")

movements=np.array([])
timess=np.array([])
taken=[]

reg=LinearRegression()
mlp=MLPRegressor(max_iter=100,learning_rate='adaptive', learning_rate_init=0.01,random_state=42)
def pathfinding(movement,time):
    global model_matrix
    global monster
    global f
    movement=movement[-16:]
    global reg
    global mlp
    time=time[-16:]
    movements = movement.reshape(-1, 2)
    times = time.reshape(-1, 2)
    X_train,X_test,y_train,y_test=train_test_split(times,movements,test_size=3,random_state=42,shuffle=False)
  
    reg.fit(X_train,y_train)
    mlp.fit(X_train,y_train)
    f=np.round(reg.predict(X_test))
    if monster['state'][0]==0:
        element=limit(f[1])
        observation,info=env.reset(monster['coordinates'],element)
        action = env.get_action(observation)
        next_observation,reward,terminated,truncated,info, monster['coordinates'] = env.step(action,steps)
        env.update(observation, action, reward, terminated, next_observation)
        observation=next_observation

        if int_matrix[monster['coordinates'][0],monster['coordinates'][1]]==-1:
            monster.loc[0,'state']+=2
            model_matrix[monster['coordinates'][0],monster['coordinates'][1]]=-2
    else:
        monster.loc[0,'state']-=1
        model_matrix[monster['coordinates'][0],monster['coordinates'][1]]=-2
        if (monster['state'][0]==0):
            monster.loc[0,'state']+=10
            int_matrix[monster['coordinates'][0],monster['coordinates'][1]]=0

    f=np.round(mlp.predict(X_test))
    
    if monster2['state'][0]==0:
        element=limit(f[1])
        observation,info=env.reset(monster2['coordinates'],element)
            
        action = env.get_action(observation)
        next_observation,reward,terminated,truncated,info, monster2['coordinates'] = env.step(action,steps)
        env.update(observation, action, reward, terminated, next_observation)
        observation=next_observation
        
        if int_matrix[monster2['coordinates'][0],monster2['coordinates'][1]]==-1:
            monster2.loc[0,'state']+=2
            model_matrix[monster2['coordinates'][0],monster2['coordinates'][1]]=-2
    else:
        monster2.loc[0,'state']-=1
        model_matrix[monster2['coordinates'][0],monster2['coordinates'][1]]=-2
        if (monster2['state'][0]==0):
            monster2.loc[0,'state']+=10
            int_matrix[monster2['coordinates'][0],monster2['coordinates'][1]]=0
radar=False
if easy=='Normal':
    coins=20
def loop2():
    goal=0
    for item in list2:
        int_matrix[item[0],item[1]]=2
    for r in range(len(int_matrix)):
        for c in range(len(int_matrix[r])):
            if (int_matrix[r][c]==1):
               goal+=1
    if goal==0:
       print('You Win')
       pygame.mixer.init()
       pygame.mixer.music.load("victoryff.swf.mp3")
       pygame.mixer.music.play()
       
       time.sleep(4)
       return
    global movements
    global model_matrix
    global x
    global y
    global easy
    global coins
    global radar
    global turn
    global timess
    model_matrix=int_matrix.copy()
    model_matrix[x,y]=-3
    print("\n" * 100)
    print("WASD to move")
    print("Radar(3 coins): Allows you to detect where the monsters are for one turn(key:r)")
    print("Trap(3 coins): Place a trap to delay a monster for 3 turns(key:t)")
    print("Travel(4 coins): Travel back 5 steps(key:g)")
    print("Blockade(1 coin): Build a wall, but its unbreakable(key:b)")
    print("Turn",turn)
    print(f"Your Coordinates:",x,y)
    print(f"Coins: ",coins)
    if easy=='Normal':
      model_matrix[monster['coordinates'][0],monster['coordinates'][1]]=-2
      model_matrix[monster2['coordinates'][0],monster2['coordinates'][1]]=-2
    if radar:
      model_matrix[monster['coordinates'][0],monster['coordinates'][1]]=-2
      model_matrix[monster2['coordinates'][0],monster2['coordinates'][1]]=-2
      radar=False
    if monster['state'][0]>0:
      model_matrix[monster['coordinates'][0],monster['coordinates'][1]]=-2
      if monster['state'][0]==10:
        monster.loc[0,'state']=0
    if monster2['state'][0]>0:
      model_matrix[monster2['coordinates'][0],monster2['coordinates'][1]]=-2
      if monster2['state'][0]==10:
        monster2.loc[0,'state']=0
    
    sns.heatmap(model_matrix, vmin=-3, vmax=3, cmap="Greys",cbar=False)
    plt.savefig('my_heatmap.png', dpi=100, bbox_inches='tight')
    img=cv2.imread('my_heatmap.png')
    cv2.imshow('Image view',img)
    key=cv2.waitKey(0)
    if(key==119 and x!=0 and not [x-1,y] in list2):
      x-=1
    elif(key==115 and x!=int_matrix.shape[0]-1 and not [x+1,y] in list2):
      x+=1
    elif(key==97 and y!=0 and not [x,y-1] in list2):
      y-=1
    elif(key==100 and y!=int_matrix.shape[1]-1 and not [x,y+1] in list2):
      y+=1
    elif(key==114 and coins>=3 and easy!='Normal'):
      coins-=3
      radar=True
      loop2()
    elif(key==116 and coins>=3 and int_matrix[x,y]!=-1 and int_matrix[x,y]!=2):
      coins-=3
      int_matrix[x,y]=-1
    elif(key==98 and coins>=1 and int_matrix[x,y]!=2 and int_matrix[x,y]!=-1):
       env.addon([x,y])
       coins-=1
    elif(key==103 and coins>=4 and turn>=12):
      x=int(movements[-12])
      y=int(movements[-11])
      coins-=4
    else:
      loop2()
    timess=np.append(timess,turn)
    timess=np.append(timess+1,turn)
    turn+=2
    movements=np.append(movements,x)
    movements=np.append(movements,y)
    if(model_matrix[x,y]==1):
      int_matrix[x,y]=0
      coins+=1
    if ((monster['coordinates'][0]==x and monster['coordinates'][1]==y)or(monster2['coordinates'][0]==x and monster2['coordinates'][1]==y)):
        pygame.mixer.init()
        pygame.mixer.music.load("fnaf-6-jumpscare-sound-effect.mp3")
        pygame.mixer.music.play()
        print("You Lose")
                
        time.sleep(2)
        return
    if turn>=20:
      pathfinding(movements,timess)
    
    if ((monster['coordinates'][0]==x and monster['coordinates'][1]==y)or(monster2['coordinates'][0]==x and monster2['coordinates'][1]==y)):
      pygame.mixer.init()
      pygame.mixer.music.load("fnaf-6-jumpscare-sound-effect.mp3")
      pygame.mixer.music.play()
      print("You Lose")
      
      time.sleep(2)
      return

    loop2()


def main() -> None:
    warnings.simplefilter(action='ignore')
    warnings.simplefilter(action='ignore', category=FutureWarning)
    pygame.mixer.init()
    pygame.mixer.music.load("horror_music.mp3.mp3")
    pygame.mixer.music.play()
    print("Your goal is to navigate through the maze and collect all the coins.")
    print("Beware, there are monsters who will hunt you down")
    print("Also don't spam keys, that will cause the game to lag behind")
    time.sleep(8)
    loop2()
if __name__ == "__main__":
    main()
