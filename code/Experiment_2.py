# -*- coding: utf-8 -*-
"""
Created on Fri May 14 11:25:20 2021

@author: malth
"""
import AMCParsermaster.amc_parser_pytorch as amc
import torch
import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import glob
import pickle
from priors import noRoot_gauss, noRoot_NF, zeroPrior
from invKin import invKin

# hyperparametre
sigma = 1
K = 7
NF_training_nitts = 1
invKin_nitts = 1
learning_rate = 0.1
saveRootPath = "experiment2/run5"
drawImages = False
flowtype = "MAF"



device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
def Experiment_2(prior, inputPose, goal_pos, endPose, joints, jointType, saveRootPath, movementType, priorType, indeks):
    print("-----------------Initial pose------------------------")
    print("jointType = ", jointType)
    if drawImages:
        joints["root"].set_motion(inputPose)
        joints["root"].draw()
    saveMotion(inputPose, os.path.join(saveRootPath, "startpose"),
               movementType + "_" + str(jointType[0]) + "_" + str(indeks) + ".pkl")

    outputPose = invKin(prior, inputPose, goal_pos, jointType, joints, lr=learning_rate, sigma=sigma,
                        nitts=invKin_nitts, verbose=False)
    print("----------------Optimized pose -----------------------")
    print("jointType = ", jointType)
    if drawImages:
        joints["root"].set_motion(outputPose)
        joints["root"].draw()
    saveMotion(outputPose, os.path.join(saveRootPath, priorType),
               movementType + "_" + str(jointType[0]) + "_" + str(indeks) + ".pkl")

    print("----------------End pose (from data) -----------------------")
    print("jointType = ", jointType)
    if drawImages:
        joints["root"].set_motion(endPose)
        joints["root"].draw()
    saveMotion(endPose, os.path.join(saveRootPath, "rigtig"),
               movementType + "_" + str(jointType[0]) + "_" + str(indeks) + ".pkl")


def copyDict(dicct):
    dicct_ny = {}
    for key in dicct:
        dicct_ny[key] = dicct[key].detach().clone()
    return dicct_ny


def getPose(test_amc_path, indeks):
    motions = amc.parse_amc(test_amc_path)

    motions[indeks]['root'][:3] = torch.tensor([0.0, 0.0, 0.0], device=device)

    start_pose = copyDict(motions[indeks])

    for key in start_pose:
        if key != "root":
            start_pose[key].requires_grad = True
        start_pose[key] = start_pose[key].to(device)

    return start_pose


def getGoalposes(test_amc_path, test_asf_path, jointType, indekser):
    joints = amc.parse_asf(test_asf_path)
    motions = amc.parse_amc(test_amc_path)

    for motion in motions:
        motion['root'][:3] = torch.tensor([0.0, 0.0, 0.0], device=device)

    goal_pos_list = []
    goal_pos_temp = []

    print("Creating goal positions:")
    for joint in jointType:
        for indeks in indekser:
            goal_pos = amc.angle2pos(motions[indeks], joints, joint).detach()
            goal_pos_temp.append(goal_pos)
        goal_pos_list.append(goal_pos_temp)
        goal_pos_temp = []
    endPoses = [motions[indeks] for indeks in indekser]
    return goal_pos_list, endPoses


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


def getPrior(priorType, train_path):
    frameslist = amc.getFrameslist(train_path)
    if priorType == "NF":
        prior = noRoot_NF(frameslist, flowtype=flowtype, K=K,
                          nitts=NF_training_nitts)
    elif priorType == "Gauss":
        prior = noRoot_gauss(frameslist)
    elif priorType == "zero":
        prior = zeroPrior()
    else:
        raise NotImplementedError("this prior type no exist")
    del frameslist
    return prior

def changePathstyle(paths):
    changedPaths = []
    for path in paths:
        changedPath = path.replace("\\", "/")
        changedPaths.append(changedPath)
    return changedPaths

if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    priorTypes = ["NF", "Gauss", "zero"]
    indekser = {"walking": {"rhand": [(5, 25), (45, 65)], "lfoot": [(25, 45), (75, 95)]},
                "running": {"rhand": [(5, 25), (45, 65)], "lfoot": [(35, 55), (55, 75)]},
                "boxing": {"rhand": [(50, 70), (125, 145)], "lfoot": [(90, 110), (280, 300)]}
                }
    jointTyper = ["rhand", "lfoot"]
    bevægelser = ["walking", "running", "boxing"]

    dataPaths = {"walking": (glob.glob('../data/Walking/*.amc'), "../data/Walking/16_30.amc",
                             '../data/Walking/35.asf'),
                 "running": (glob.glob('../data/Running/*.amc'), "../data/Running/16_49.amc",
                             '../data/Running/16.asf'),
                 "boxing": (glob.glob('../data/Boxing/*.amc'), "../data/Boxing/13_18.amc",
                            '../data/Boxing/14.asf')}

    for bevægelse in bevægelser:
        dataPaths_train, dataPath_test, asf_path = dataPaths[bevægelse]
        dataPaths_train = changePathstyle(dataPaths_train)
        dataPaths_train.remove(dataPath_test)
        joints = amc.parse_asf(asf_path)
        for priorType in priorTypes:
            prior = getPrior(priorType, dataPaths_train)
            for jointType in jointTyper:
                for startIndeks, slutIndeks in indekser[bevægelse][jointType]:
                    startpose = getPose(dataPath_test, startIndeks)
                    goal_poses, endPoses = getGoalposes(dataPath_test, asf_path, [jointType], [slutIndeks])
                    Experiment_2(prior, startpose, goal_poses[0], endPoses[0], joints,
                                 [jointType], saveRootPath, bevægelse, priorType, slutIndeks)

