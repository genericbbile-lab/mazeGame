# TODO: Update the main function to your needs or remove it.
import numpy as np
import pandas as pd
import seaborn as sns
import os
from sklearn.neural_network import MLPRegressor
import matplotlib.pyplot as plt
from IPython.display import clear_output
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from IPython.display import Audio, display
from sklearn.linear_model import LogisticRegression
from pynput import keyboard
import tkinter as tk
import random
import warnings
from tkinter import filedialog,messagebox
from PIL import Image,ImageFilter
import cv2

# Read the image

x=9
y=0

easy=input("Baby Mode or Hard Mode")
monster=pd.DataFrame({
    "coordinates": [9,9],
    "state":[0,1]
})
monster2=pd.DataFrame({
    "coordinates": [0,9],
    "state":[0,1]
})
movements=np.array([])
time=np.array([])
taken=[]
rng = np.random.default_rng()
coins=0
turn=0

int_matrix = rng.integers(low=0, high=2, size=(10,10))
reg=LinearRegression()
mlp=MLPRegressor(max_iter=100,learning_rate='adaptive', learning_rate_init=0.01,random_state=42)
def pathfinding(movement,time):
  global model_matrix
  global monster
  global f # Make f global to be accessible in loop
  movement=movement[-12:]
  global reg
  global mlp
  time=time[-12:]
  movements = movement.reshape(-1, 2)
  times = time.reshape(-1, 2)
  X_train,X_test,y_train,y_test=train_test_split(times,movements,test_size=3,random_state=42,shuffle=False)
  #scaler = StandardScaler()
  #X_train_scaled=scaler.fit_transform(X_train)
  #X_test_scaled=scaler.fit_transform(X_test)
  
  reg.fit(X_train,y_train)
  mlp.fit(X_train,y_train)
  f=np.round(reg.predict(X_test))
  # f is of shape (1, 2), so f[0] is [x_pred, y_pred]
  if monster['state'][0]==0:
    if (monster['coordinates'][0]>f[1][0]and monster['coordinates'][0]!=0):
      monster.loc[0,'coordinates']-=1
    elif( monster['coordinates'][0]<f[1][0] and monster['coordinates'][0]!=model_matrix.shape[0]-1):
      monster.loc[0,'coordinates']+=1
    else:
      if (monster['coordinates'][1]>f[1][1]and monster['coordinates'][1]!=0):
         monster.loc[1,'coordinates']-=1
      elif(monster['coordinates'][1]<f[1][1]and monster['coordinates'][1]!=model_matrix.shape[1]-1):
         monster.loc[1,'coordinates']+=1
    if model_matrix[monster['coordinates'][0],monster['coordinates'][1]]==-1:
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
    if (monster2['coordinates'][1]>f[1][1]and monster2['coordinates'][1]!=0):
      monster2.loc[1,'coordinates']-=1
    elif( monster2['coordinates'][1]<f[1][1]and monster2['coordinates'][1]!=model_matrix.shape[1]-1):
      monster2.loc[1,'coordinates']+=1
    else:
      if (monster2['coordinates'][0]>f[1][0]and monster2['coordinates'][0]!=0):
        monster2.loc[0,'coordinates']-=1
      elif(monster2['coordinates'][0]<f[1][0]and monster2['coordinates'][0]!=model_matrix.shape[0]-1):
        monster2.loc[0,'coordinates']+=1

    if model_matrix[monster2['coordinates'][0],monster2['coordinates'][1]]==-1:
      monster2.loc[0,'state']+=2
      model_matrix[monster2['coordinates'][0],monster2['coordinates'][1]]=-2
  else:
    monster2.loc[0,'state']-=1
    model_matrix[monster2['coordinates'][0],monster2['coordinates'][1]]=-2
    if (monster2['state'][0]==0):
        monster2.loc[0,'state']+=10
        int_matrix[monster2['coordinates'][0],monster2['coordinates'][1]]=0
radar=False
x=random.randint(0,int_matrix.shape[0]-1)
y=random.randint(0,int_matrix.shape[1]-1)
monster['coordinates']=[random.randint(0,int_matrix.shape[0]-1),random.randint(0,int_matrix.shape[1]-1)]

monster2['coordinates']=[random.randint(0,int_matrix.shape[0]-1),random.randint(0,int_matrix.shape[1]-1)]
def loop2():
    global movements
    global model_matrix
    global x
    global y
    global easy
    global coins
    global radar
    
    model_matrix=int_matrix.copy()
    model_matrix[x,y]=-3
    if easy=='Baby':
      coins=69
      model_matrix[monster['coordinates'][0],monster['coordinates'][1]]=-2
      model_matrix[monster2['coordinates'][0],monster2['coordinates'][1]]=-2
    #display(Audio(filename="/content/sample_data/baby-crying-phonk.mp3",autoplay=True))
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
    global turn
    global time
    
    sns.heatmap(model_matrix, cmap="Grays",cbar=False)
    plt.savefig('my_heatmap.png', dpi=100, bbox_inches='tight')
    img=cv2.imread('my_heatmap.png')
    cv2.imshow('Image view',img)
    key=cv2.waitKey(0)
    
    if(key==119 and x!=0):
      x-=1
    elif(key==115 and x!=int_matrix.shape[0]-1):
      x+=1
    elif(key==97 and y!=0):
      y-=1
    elif(key==100 and y!=int_matrix.shape[1]-1):
      y+=1
    elif(key==114 and coins>=8 and easy!='Baby'):
      coins-=8
      radar=True
    elif(key==116 and coins>=3 and int_matrix[x,y]!=-1):
      coins-=3
      int_matrix[x,y]=-1
    elif(key==103 and coins>=6 and turn>=12):
      x=int(movements[-12])
      y=int(movements[-11])
      coins-=6
    else:
      loop2()
    print("Turn",turn)
    print(f"Your Coordinates:",x,y)
    print(f"Coins: ",coins)
    time=np.append(time,turn)
    time=np.append(time+1,turn)
    turn+=2
    movements=np.append(movements,x)
    movements=np.append(movements,y)
    if(model_matrix[x,y]==1):
      int_matrix[x,y]=0
      coins+=1
    if turn>=12:
      pathfinding(movements,time)

    # f is now global and has shape (1, 2), so f[0] is [x_pred, y_pred]
    if ((monster['coordinates'][0]==x and monster['coordinates'][1]==y)or(monster2['coordinates'][0]==x and monster2['coordinates'][1]==y)):
      print("You Lose")
      return # Changed break to return

    loop2()

def main() -> None:
    warnings.simplefilter(action='ignore')
    warnings.simplefilter(action='ignore', category=FutureWarning)
    print("Start coding in Python today!")
    loop2()
  

if __name__ == "__main__":
    main()
