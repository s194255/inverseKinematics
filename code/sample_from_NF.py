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
num_samples = 5
saveRoot = ""


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


dataPaths = {"walking": (glob.glob('../data/Walking/*.amc'),
                             '../data/Walking/35.asf'),
                 "running": (glob.glob('../data/Running/*.amc'),
                             '../data/Running/16.asf'),
                 "boxing": (glob.glob('../data/Boxing/*.amc'),
                            '../data/Boxing/14.asf')}



for activity in dataPaths:
    amc_filer, asf_fil = dataPaths[activity]
    frameslist = amc.getFrameslist(amc_filer)
    joints = amc.parse_asf(asf_fil)
    prior = noRoot_NF(frameslist=frameslist, K=K, flowtype=flowtype, nitts=nitts)
    samples =  prior.sample(num_samples=num_samples)

    for i in range(len(samples)):
        #make the rotational coordinates zero.
        samples[i]["root"][3:6] = torch.tensor([0.0,0.0,0.0], device=device)
        #save the joint angle vector
        saveMotion(samples[i], os.path.join(saveRoot, activity), "{}.pkl".format(i))
        joints["root"].set_motion(samples[i])
        #save a snapshot of pose
        joints["root"].draw(savePath=os.path.join(saveRoot, activity, "{}.jpg".format(i)))
