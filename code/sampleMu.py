from priors import noRoot_gauss, noRoot_NF, zeroPrior
import AMCParsermaster.amc_parser_pytorch as amc
import glob
import torch
import os
import pickle
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

K = 7
flowtype = "MAF"
nitts = 1
saveRoot = ""

# if the path don't exist, make it!
if os.path.isdir(saveRoot) == False:
    os.makedirs(saveRoot)
dataPaths = {"walking": (glob.glob('../data/Walking/*.amc'),
                             '../data/Walking/35.asf'),
                 "running": (glob.glob('../data/Running/*.amc'),
                             '../data/Running/16.asf'),
                 "boxing": (glob.glob('../data/Boxing/*.amc'),
                            '../data/Boxing/14.asf')}

def saveMotion(motion, path, fileName):
    """
    :param motion: A joint angle dictionary
    :param path: A path (wihtout filename)
    :param fileName: The filename
    :return:
    """
    # cloning the joint angle dictionary
    motionClone =  {}
    for key in motion:
        tensorclone = motion[key].clone()
        motionClone[key] = tensorclone.detach().cpu()
    # if the path doesnt exist already, create it
    if os.path.isdir(path):
        pass
    else:
        print("direc no exist. Creating it ...")
        os.makedirs(path)
    # dump joint angle dictionary is pickle
    with open(os.path.join(path, fileName), "wb") as f:
        pickle.dump(motionClone, f)

for activity in dataPaths:
    amc_filer, asf_fil = dataPaths[activity]
    frameslist = amc.getFrameslist(amc_filer)
    joints = amc.parse_asf(asf_fil)
    prior = noRoot_NF(frameslist=frameslist, K=K, flowtype=flowtype, nitts=nitts)

    # Get the mean vector of the NF's prior (it's usually the Standard Gauss)
    vector = prior.NF_model.prior.loc
    # Shape needs to be (1, 53), not (53,)
    vector = vector.unsqueeze(dim=0)

    #let the vector go in the sample direction
    samples =  prior.sampleDirection(vector)

    # save the joint angles as pickle
    saveMotion(samples[0], saveRoot, "{}_mu.pkl".format(activity))
    joints["root"].set_motion(samples[0])
    # save a snapshot of the pose
    joints["root"].draw(savePath=os.path.join(saveRoot, "{}_mu.jpg".format(activity)))