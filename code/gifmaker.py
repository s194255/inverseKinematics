# import AMCParsermaster.amc_parser as amc
# import AMCParsermaster.ThreeDviewer as viewer
import torch
import AMCParsermaster.amc_parser_pytorch as amc
import AMCParsermaster.ThreeDviewer_pytorch as viewer
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.animation as animation
import os
import pickle
import glob
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

asf_path = '../data/subjects/09/09.asf'
amc_path = '../data/subjects/09/09_01.amc'

joints = amc.parse_asf(asf_path)
# motions = amc.parse_amc(amc_path)

picklerick = "experiment1/run5/NF/"
name = "walking_rhand_2"

with open(picklerick+name, "rb") as f:
    frame = pickle.load(f)

# joints["root"].set_motion(motions[0])
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
rot_animation.save('experiment1/run5/NF/rotation.gif', dpi=80, writer='imagemagick')



#%% GENERATED MOTIONS

paths = (glob.glob('experiment1/run5/Gauss/*'), glob.glob('experiment1/run5/NF/*'), glob.glob('experiment1/run5/zero/*'))



for i in range(len(paths)):
    for j in range(len(paths[i])):
        
        picklerick = paths[i][j].split('\\')[0] + "/"
        name = paths[i][j].split('\\')[1]
        
        with open(picklerick+name, "rb") as f:
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
        rot_animation.save(picklerick+name+'.gif', dpi=80, writer='imagemagick')
        
#%% REAL MOTIONS