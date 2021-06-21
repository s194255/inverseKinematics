# -*- coding: utf-8 -*-
"""
Created on Mon Jun 21 10:12:58 2021

@author: malth
"""
import AMCParsermaster.amc_parser_pytorch as amc
import torch
import numpy as np


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def invKin(prior, current_pose, goal_pos, jointType, joints, logPath=None, verbose=True, log=False, sigma=10, lr=0.01, nitts=5*10**3+1):
    """
    Explanation of arguments for invKin:
        prior: the prior used in the maximum a posteriori loop
        current_pose: The starting skeleton as a dict with tensors
        goal_pos: the target position
        jointType: the joint that is supposed to reach the target destination
        Joints: The asf-file to be used
        logPath: path for the .txt file that saves loss-updates from the MAP loop
        verbose: toggles if the loss logs should be printed, if FALSE, logs cannot be written
        sigma: sigma^2 as discussed in the report. Changes the priority of likelihood compared to prior, low value means more priority to the likelihood.
        lr: learning rate for the MAP loop.
        nitts: iterations for the MAP loop.
        
    
    
    
    """
    sigma = sigma
    Sigma = torch.eye(3, device=device) * sigma
    Sigma_inv = torch.inverse(Sigma)
    optimizer = torch.optim.Adam([current_pose[key] for key in current_pose], lr=lr)
    for i in range(nitts):
        optimizer.zero_grad()
        curr_pos = amc.angle2poses(current_pose, joints, jointType)
        prior_loss = prior.eval(current_pose)
        likelihood = sum([(goal_pos[k] - curr_pos[k]).T @ Sigma_inv @ (goal_pos[k] - curr_pos[k]) for k in range(len(goal_pos))])
        loss = likelihood - prior_loss
        # loss = likelihood #without prior
        loss.backward()
        optimizer.step()

        if i % 1000 == 0:
            if verbose:
                print("Iterations:", i)
                print("-----------------")
                print("Posterior loss = ", round(loss.item(),4))
                print("Prior loss = ", round(-prior_loss.item(),4))
                print("Likelihood = ", round(likelihood.item(),4))
                print("Euclidean distance = ", round(np.sqrt(float(sum([torch.matmul((goal_pos[k] - curr_pos[k]).T, (goal_pos[k] - curr_pos[k])).item() for k in range(len(goal_pos))]))),4), "\n")
                
                lines = ["Iterations:" + str(i),"-----------------\n", "Posterior loss = " + str(round(loss.item(),4))+"\n"
                         ,"Prior loss = " + str(round(-prior_loss.item(),4))+"\n","Likelihood = " + str(round(likelihood.item(),4))+"\n"
                         ,"Likelihood = " + str(round(likelihood.item(),4))+"\n", "Euclidean distance = " + str(round(float(sum([torch.matmul((goal_pos[k] - curr_pos[k]).T, (goal_pos[k] - curr_pos[k])).item() for k in range(len(goal_pos))])),4))+"\n", "\n" ]
                if log:
                    with open(logPath, 'a') as f:
                        f.writelines(lines)
                        f.close()
    return current_pose