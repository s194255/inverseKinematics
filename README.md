# inverseKinematics
Hello. This is our repo for our fagprojekt. The repo lets the user make cool experiments with probabilistic inverse kinematics models. Enjoy!


The following is an explanation of each file/folder and what they do:
The Data folder contains the data-sets used for training the priors for the experiments.
The data used in this project was obtained from mocap.cs.cmu.edu.
The database was created with funding from NSF EIA-0196217.
Each subfolder contains multiple amc-files with motion capture data, as well as a single asf-file with the skeleton data. 

The code folder contains all the code needed to replicate our experiments.
Experiment_1.py & Experiment_2.py are the only files needed to actually run the experiments. Within them paths and parameters may need to be changed.
gifmaker.py is a script for creating a gif of a pose from a pickle file, which is obtained by running either of the experiments. The gif will be off the pose rotating 360 degrees at an elevated angle.
invKin.py contains the function invKin which performs the inverse kinematics training loop. It does not need to be opened or changed for the experiments to work.
priors.py containts functions for fitting the priors to data. Again, this file should not need to be opened or changed for the experiments to work.
sampleMu.py and sample_from_NF.py are files for creating the different plots of certain poses. sampleMu creates the mean pose of a given prior. It created the figure "expected poses" in our report. sample_from_NF.py draws a random sample from a standard gaussian distribution and lets it flow through the NF network in the sampling direction.

In the AMCParsermaster folder, the amc_parser and amc_parser_pytorch files do the same thing, parsing an amc-file for use in python.
ThreeDviewer.py makes it possible to view an amc-file animated. both amc_parser.py and ThreeDviewer.py was taken from the repository https://github.com/CalciferZh/AMCParser by Zhou, Yuxiao and Chiara, Anna.
the pytorch3d.py file contains functions from the pytorch3d.transforms library. This file only exists because we had trouble getting the library to work. The orinal library's website: https://pytorch3d.org/

In the Normalizing_flows folder, there are 3 files:
The flows.py file contains the functions for the normalizing flows that were used in this project. The linear and planar flows were written by us based on the blog post: http://akosiorek.github.io/ml/2018/04/03/norm_flows.html
MAF, NormalizingFlow and NormalizingFlowModel was taken from the repository: https://github.com/karpathy/pytorch-normalizing-flows
specifically the flows.py file. made.py and nets.py also originate there. The MAF function has undergone some small changes to make it work with the rest of our code.

## GIFs for the survey
In our project, GIFs was made using the files gifmaker.py
The first part of the survey (experiment 1) had users determine if a single pose was natural or not given a context pose. Example:
![](https://github.com/s194255/inverseKinematics/blob/main/data/Gifs/exp1_NF_boxing.gif)

The second part of the survey (experiment 2) had users determine if a single pose was natural or not. Example:
![](https://github.com/s194255/inverseKinematics/blob/main/data/Gifs/exp2_Gauss_running.gif)
