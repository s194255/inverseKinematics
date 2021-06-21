from priors import noRoot_gauss, Root_gauss, noRoot_NF, Root_NF, zeroPrior
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
saveRoot = "../samples2"


def saveMotion(motion, path, fileName):
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


dataPaths = {"walking": (glob.glob('../data/subjects/Walking/*.amc'),
                             '../data/subjects/Walking/35.asf'),
                 "running": (glob.glob('../data/subjects/Running/*.amc'),
                             '../data/subjects/Running/16.asf'),
                 "boxing": (glob.glob('../data/subjects/Boxing/*.amc'),
                            '../data/subjects/Boxing/14.asf')}



for bevægelse in dataPaths:
    amc_filer, asf_fil = dataPaths[bevægelse]
    frameslist = amc.getFrameslist(amc_filer)
    joints = amc.parse_asf(asf_fil)
    prior = noRoot_NF(frameslist=frameslist, K=K, flowtype=flowtype, nitts=nitts)
    samples =  prior.sample(num_samples=num_samples)

    for i in range(len(samples)):
        samples[i]["root"][3:6] = torch.tensor([0.0,0.0,0.0], device=device)
        saveMotion(samples[i], os.path.join(saveRoot, bevægelse), "{}.pkl".format(i))
        joints["root"].set_motion(samples[i])
        joints["root"].draw(savePath=os.path.join(saveRoot, bevægelse, "{}.jpg".format(i)))
