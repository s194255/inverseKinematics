import torch
import AMCParsermaster.amc_parser_pytorch as amc
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.animation as animation
import os
import pickle
import glob
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

#Skeleton
asf_path = '../data/Walking/35.asf'

joints = amc.parse_asf(asf_path)

### Script for making a GIF of a single pickle-file ###

#Directory of file:
filedir = "..."
#File name:
name = "..."

#This loads the pkl-file
with open(filedir+name, "rb") as f:
    frame = pickle.load(f)

joints["root"].set_motion(frame)
  
fig = plt.figure()
ax = Axes3D(fig)

ax.set_xlim3d(-15, 15)
ax.set_ylim3d(-15, 15)
ax.set_zlim3d(-15, 15)

xs, ys, zs = [], [], []
for joint in joints.values():
  xs.append(joint.coordinate[0, 0])
  ys.append(joint.coordinate[1, 0])
  zs.append(joint.coordinate[2, 0])
plt.plot(zs, xs, ys, 'b.')

for joint in joints.values():
  child = joint
  if child.parent is not None:
    parent = child.parent
    xs = [child.coordinate[0, 0], parent.coordinate[0, 0]]
    ys = [child.coordinate[1, 0], parent.coordinate[1, 0]]
    zs = [child.coordinate[2, 0], parent.coordinate[2, 0]]
    plt.plot(zs, xs, ys, 'r')

# from https://stackoverflow.com/questions/51457738/animating-a-3d-scatterplot-with-matplotlib-to-gif-ends-up-empty
def rotate(angle):
    ax.view_init(azim=angle)

print("Making animation")
rot_animation = animation.FuncAnimation(fig, rotate, frames=np.arange(-90, 270, 2), interval=100)
## Uncomment to save animation as GIF!
# rot_animation.save('experiment1/rotation.gif', dpi=80, writer='imagemagick')



#%% Experiment 1 GIFS

#Input the correct file directory for Gauss and NF 
# from glob, * makes sure to load ALL files
paths = (glob.glob('experiment1/Gauss/*'), glob.glob('experiment1/NF/*'))


#Loops over all data paths and saves pkl.files as GIFs
for i in range(len(paths)):
    for j in range(len(paths[i])):
        
        filedir = paths[i][j].split('\\')[0] + "/"
        name = paths[i][j].split('\\')[1]
        
        with open(filedir+name, "rb") as f:
            frame = pickle.load(f)
            
        joints["root"].set_motion(frame)
          
        fig = plt.figure()
        ax = Axes3D(fig)
        
        ax.set_xlim3d(-15, 15)
        ax.set_ylim3d(-15, 15)
        ax.set_zlim3d(-15, 15)
        
        xs, ys, zs = [], [], []
        for joint in joints.values():
          xs.append(joint.coordinate[0, 0])
          ys.append(joint.coordinate[1, 0])
          zs.append(joint.coordinate[2, 0])
        plt.plot(zs, xs, ys, 'b.')
        
        for joint in joints.values():
          child = joint
          if child.parent is not None:
            parent = child.parent
            xs = [child.coordinate[0, 0], parent.coordinate[0, 0]]
            ys = [child.coordinate[1, 0], parent.coordinate[1, 0]]
            zs = [child.coordinate[2, 0], parent.coordinate[2, 0]]
            plt.plot(zs, xs, ys, 'r')
        
        # from https://stackoverflow.com/questions/51457738/animating-a-3d-scatterplot-with-matplotlib-to-gif-ends-up-empty
        def rotate(angle):
            ax.view_init(azim=angle)
        
        print("Making animation")
        rot_animation = animation.FuncAnimation(fig, rotate, frames=np.arange(-90, 270, 2), interval=100)
        ## Uncomment to save animation as GIF:
        # rot_animation.save("experiment1/Gifs/"+filedir.split("/")[2]+"/"+name+'.gif', dpi=80, writer='imagemagick')
      
#%% EXPERIMENT 2 GIFS

#Input the correct file directory for Gauss and NF 
# from glob, * makes sure to load ALL files
paths = (glob.glob('experiment2/Gauss/*'), glob.glob('experiment2/NF/*'), glob.glob('experiment2/run5/startpose/*'), glob.glob('experiment2/run5/rigtig/*'))

#Loops over all data paths and saves pkl.files as GIFs
for i in range(len(paths)):
    for j in range(len(paths[i])):
        
        filedir = paths[i][j].split('\\')[0] + "/"
        name = paths[i][j].split('\\')[1]
        
        with open(filedir+name, "rb") as f:
            frame = pickle.load(f)
            
        joints["root"].set_motion(frame)
          
        fig = plt.figure()
        ax = Axes3D(fig)
        
        ax.set_xlim3d(-15, 15)
        ax.set_ylim3d(-15, 15)
        ax.set_zlim3d(-15, 15)
        
        xs, ys, zs = [], [], []
        for joint in joints.values():
          xs.append(joint.coordinate[0, 0])
          ys.append(joint.coordinate[1, 0])
          zs.append(joint.coordinate[2, 0])
        plt.plot(zs, xs, ys, 'b.')
        
        for joint in joints.values():
          child = joint
          if child.parent is not None:
            parent = child.parent
            xs = [child.coordinate[0, 0], parent.coordinate[0, 0]]
            ys = [child.coordinate[1, 0], parent.coordinate[1, 0]]
            zs = [child.coordinate[2, 0], parent.coordinate[2, 0]]
            plt.plot(zs, xs, ys, 'r')
        
        # from https://stackoverflow.com/questions/51457738/animating-a-3d-scatterplot-with-matplotlib-to-gif-ends-up-empty
        def rotate(angle):
            ax.view_init(azim=angle)
        
        print("Making animation")
        rot_animation = animation.FuncAnimation(fig, rotate, frames=np.arange(-90, 270, 2), interval=100)
        ## Uncomment to save animation as GIF:
        # rot_animation.save("experiment2/Gifs/"+filedir.split("/")[2]+"/"+name+'.gif', dpi=80, writer='imagemagick')