import AMCParsermaster.amc_parser_pytorch as amc
import torch
import os
import glob
import copy
import pickle
from invKin import invKin
from priors import noRoot_gauss, noRoot_NF, zeroPrior


os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

#Hyperparameters
Root_path = 'experiment1/run6/'
logPath = 'experiment1/run6/logs.txt'
sigma = 0.1
lr = 0.5
NF_nitts = 1
invKin_nitts = 1
drawImages = False
log = True
K=7
flowtype = "MAF"



zero_motion = {"root": torch.tensor([  0.0,  0.0, 0.0,  0.0, 0.0, 0.0], requires_grad=False, device=device),
         'lowerback': torch.tensor([ 0.0, 0.0, 0.0], requires_grad=True, device=device),
         'upperback': torch.tensor([ 0.0, 0.0, 0.0], requires_grad=True, device=device),
         'thorax': torch.tensor([0.0, 0.0, 0.0], requires_grad=True, device=device),
         'lowerneck': torch.tensor([0.0, 0.0, 0.0], requires_grad=True, device=device),
         'upperneck': torch.tensor([0.0, 0.0, 0.0], requires_grad=True, device=device),
         'head': torch.tensor([0.0, 0.0, 0.0], requires_grad=True, device=device),
         'rclavicle': torch.tensor([0.0, 0.0], requires_grad=True, device=device),
         'rhumerus': torch.tensor([0.0, 0.0, 0.0], requires_grad=True, device=device),
         'rradius': torch.tensor([0.0], requires_grad=True, device=device),
         'rwrist': torch.tensor([0.0], requires_grad=True, device=device),
         'rhand': torch.tensor([0.0,0.0], requires_grad=True, device=device),
         'rfingers': torch.tensor([0.0], requires_grad=True, device=device),
         'rthumb': torch.tensor([ 0.0,0.0], requires_grad=True, device=device),
         'lclavicle': torch.tensor([0.0,0.0], requires_grad=True, device=device),
         'lhumerus': torch.tensor([ 0.0,0.0,0.0], requires_grad=True, device=device),
         'lradius': torch.tensor([0.0], requires_grad=True, device=device),
         'lwrist': torch.tensor([0.0], requires_grad=True, device=device),
         'lhand': torch.tensor([0.0,0.0], requires_grad=True, device=device),
         'lfingers': torch.tensor([0.0], requires_grad=True, device=device),
         'lthumb': torch.tensor([0.0,0.0], requires_grad=True, device=device),
         'rfemur': torch.tensor([0.0,0.0,0.0], requires_grad=True, device=device),
         'rtibia': torch.tensor([0.0], requires_grad=True, device=device),
         'rfoot': torch.tensor([0.0, 0.0], requires_grad=True, device=device),
         'rtoes': torch.tensor([0.0], requires_grad=True, device=device),
         'lfemur': torch.tensor([ 0.0,0.0,0.0], requires_grad=True, device=device),
         'ltibia': torch.tensor([0.0], requires_grad=True, device=device),
         'lfoot': torch.tensor([0.0,0.0], requires_grad=True, device=device),
         'ltoes': torch.tensor([0.0], requires_grad=True, device=device)
         }


# rhand target positions:
rhand = [[torch.tensor([[-4, 5, 5]], device=device).T], [torch.tensor([[-4, 0, 3]],device=device).T]]


# lfoot target position:
lfoot = [[torch.tensor([[2, -16, 4]], device=device).T], [torch.tensor([[2, -15,-1]],device=device).T]]

Goal_points = [rhand[0], rhand[1], lfoot[0], lfoot[1]]
Joint_names = ['rhand','rhand','lfoot','lfoot']

priorTypes = ["NF", "Gauss", "zero"]
movementTypes = ["walking", "running", "boxing"]


#Walking data
dataPath = glob.glob('../data/Walking/*.amc')
walkingData = amc.getFrameslist(dataPath)

#Running data
dataPath = glob.glob('../data/Running/*.amc')
runningData = amc.getFrameslist(dataPath)

#Boxing data
dataPath = glob.glob('../data/Boxing/*.amc')
boxingData = amc.getFrameslist(dataPath)

dataList = [walkingData, runningData, boxingData]

#ASF files for each dataset
walkingJoints = amc.parse_asf('../data/Walking/35.asf')
runningJoints = amc.parse_asf('../data/Running/16.asf')
boxingJoints = amc.parse_asf('../data/Boxing/14.asf')

asfList = [walkingJoints,runningJoints,boxingJoints]

# def priorchoice(priortype, datatype):
#     if datatype == "walking" and priortype == "NF":
#         return noRoot_NF(walkingData, K=K, flowtype="planar", nitts=NF_nitts)
#     elif datatype == "walking" and priortype == "Gauss":
#         return noRoot_gauss(walkingData)
    
#     elif datatype == "running" and priortype == "NF":
#         return noRoot_NF(runningData, K=K, flowtype="planar", nitts=NF_nitts)
#     elif datatype == "running" and priortype == "Gauss":
#         return noRoot_gauss(runningData)
    
#     elif datatype == "boxing" and priortype == "NF":
#         return noRoot_NF(boxingData, K=K, flowtype="planar", nitts=NF_nitts)
#     elif datatype == "boxing" and priortype == "Gauss":
#         return noRoot_gauss(boxingData)
    
#     elif priortype == "zero":
#         return zeroPrior()
#     else:
#         print("forkert navn")
 

Zero_Prior = zeroPrior()
NFwalking = noRoot_NF(walkingData, K=K, flowtype=flowtype, nitts=NF_nitts)
NFrunning = noRoot_NF(runningData, K=K, flowtype=flowtype, nitts=NF_nitts)
NFboxing = noRoot_NF(boxingData, K=K, flowtype=flowtype, nitts=NF_nitts)
Gausswalking = noRoot_gauss(walkingData)
Gaussrunning = noRoot_gauss(runningData)
Gaussboxing = noRoot_gauss(boxingData)
ZeroList = [Zero_Prior, Zero_Prior, Zero_Prior]
NFList = [NFwalking,NFrunning,NFboxing]
GaussList = [Gausswalking,Gaussrunning,Gaussboxing]
priorList = [NFList,GaussList,ZeroList]
 #%%
    
def saveMotion(motion, path, fileName):
    """
    This function saves the poses generated with invKin as pickle files. These files can be loaded and turned into Gifs with gifmaker.py

    Parameters
    ----------
    motion : Pose that is to be saved
    path : path the pose is to be saved at
    fileName : name of the resulting file
    
    Returns
    -------
    None.

    """
    motionClone =  {}
    for key in motion:
        tensorclone = motion[key].clone()
        motionClone[key] = tensorclone.detach().cpu()
    if os.path.isdir(path):
        pass
    else:
        print("direc no exist. Creating it ...")
        os.makedirs(path)
    with open(os.path.join(path, fileName), "wb") as f:
        pickle.dump(motionClone, f)
      
       

def Experiment1(Goal_points, Joint_names, motion, asfList, priorList, movementTypes, rootPath, drawImages, log, logpath, sigma, lr, nitts = 500):
    """
    Parameters
    ----------
    Goal_points : Target positions
    Joint_names : Represents which joints are supposed to reach the target positions
    motion : initial pose as a dictionary of tensors
    asfList : list containing all necessary skeletons loaded from asf-files
    priorList : list of trained priors to be used for MAP
    movementTypes : list of datasets that are supposed to be used
    rootPath : base path for multiple purposes such as saving poses and logs
    drawImages : Determines if poses should be plottet imidiately
    log : TToggles if logs of loss should be saved
    logpath : path for the log of loss values
    sigma : Hyperparameter that determines the priorities of likelihood and prior while doing MAP.
    lr : Learning rate for invKin
    nitts : Iterations for invKin

    Returns
    -------
    None.

    """
    #Create pose:
    for k in range(len(movementTypes)):
        for c in range(len(priorList)):
            for i in range(len(Joint_names)):
                #The following cloning is done to ensure each pose starts at the standart T-pose, and is independant of each other.
                motionClone =  {}
                for key in motion:
                    tensorclone = copy.deepcopy(motion[key])
                    motionClone[key] = tensorclone
                #Create pose    
                outputPose = invKin(prior=priorList[c][k], current_pose=motionClone, goal_pos=Goal_points[i],
                                    jointType=[Joint_names[i]], joints=asfList[k], logPath=logPath, sigma=sigma,
                                    lr=lr, nitts = nitts, log=True, verbose=True)
                #save pose
                saveMotion(outputPose, os.path.join(rootPath, priorTypes[c]), str(movementTypes[k] + "_" + Joint_names[i] + "_" + str(1+i%2)))
                if drawImages == True:
                    asfList[k]["root"].set_motion(outputPose)
                    asfList[k]["root"].draw()
                
if __name__ == "__main__":
    Experiment1(Goal_points, Joint_names, zero_motion, asfList, priorList, movementTypes, Root_path, drawImages, log, logPath,
                sigma=sigma, lr=lr, nitts = invKin_nitts )
