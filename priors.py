import torch
import numpy as np
from Normalizing_flows.flows import MAF, NormalizingFlowModel
from Normalizing_flows.homemade_flows import planar, normalizeModel
from torch.optim.lr_scheduler import StepLR
import copy

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

motion_dict_keys = {"root":3, "lowerback":3, "upperback":3, "thorax":3, "lowerneck":3, "upperneck":3, "head":3, "rclavicle":2, "rhumerus":3,
                    "rradius":1,"rwrist":1, "rhand":2, "rfingers":1, "rthumb":2, "lclavicle":2, "lhumerus":3, "lradius":1, "lwrist":1,
                   "lhand":2, "lfingers":1, "lthumb":2, "rfemur":3, "rtibia":1, "rfoot":2, "rtoes":1, "lfemur":3, "ltibia":1, "lfoot":2, "ltoes":1}

class noRoot_NF:
    def __init__(self, frameslist, K=7, flowtype="planar", nitts=8000):
        """
        :param frameslist: list of parsed amc-files
        :param K: number of layers in NF (int)
        :param flowtype: type of flow (either "planar" og "MAF")
        :param nitts: number of training iterations for NF
        """
        self.data = None
        self.mu = None
        self.cov = None
        self.frameslist = frameslist
        self.K = K
        self.flowtype = flowtype
        self.nitts = nitts
        # loading all frameslists
        for i in range(len(frameslist)):
            self.get_data(frameslist[i])
        self.fit()

    def get_data(self, frames):
        """
        :param frames: a parsed amc-file (a list of joint angles)
        :return:
        """
        # Create empty data tensor, 62 is because that is the total degrees of freedom
        if self.data == None:  # if no data is loaded yet
            data = torch.zeros(len(frames), 62, device=device)
            for i in range(len(frames)):
                data[i] = torch.tensor([item for sublist in list(frames[i].values()) for item in sublist],
                                       device=device)  # flatten frame, code credit:https://stackoverflow.com/a/952952
            self.data = data

            # removing idxs  [0, 1, 2, 24, 25, 33, 36, 37, 45]
            self.data = torch.hstack([self.data[:, 3:24], self.data[:, 26:33], self.data[:, 34:36],
                                      self.data[:, 38:45], self.data[:, 46:]])
        else:  # If one or more frames are already loaded
            data = torch.zeros(len(frames), 62, device=device)
            for i in range(len(frames)):
                data[i] = torch.tensor([item for sublist in list(frames[i].values()) for item in sublist],
                                       device=device)

            # removign idxs  [0, 1, 2, 24, 25, 33, 36, 37, 45]
            data = torch.hstack([data[:, 3:24], data[:, 26:33], data[:, 34:36],
                                 data[:, 38:45], data[:, 46:]])

            # stacking the v's
            self.data = torch.vstack((self.data, data))

    def fit(self):
        self.mu = torch.mean(self.data, dim=0)
        self.stds = self.data.std(dim=0, keepdim=True)
        self.data_standardized = (self.data - self.mu) / self.stds

        # if we want standard gauss as z0 (called prior) in the NF
        prior = torch.distributions.multivariate_normal.MultivariateNormal(
            loc=torch.zeros(self.data.shape[1], dtype=torch.float64,
                            device=device), covariance_matrix=torch.eye(self.data.shape[1],
                                                                        dtype=torch.float64,
                                                                        device=device))
        if self.flowtype == "MAF":
            #making a lists of flows
            self.flows = [MAF(dim=len(self.data[0]), parity=i % 2) for i in range(self.K)]
            self.NF_model = NormalizingFlowModel(prior, self.flows)
            self.NF_model.to(device)

        elif self.flowtype == "planar":
            # making a list of flows
            self.flows = [planar(dim=len(self.data[0])) for i in range(self.K)]
            self.NF_model = normalizeModel(prior, self.flows)
            self.NF_model.to(device)

        else:
            raise NotImplementedError("Flowtype is not implemented. Choose MAF or planar")

        print("\nModel:", next(self.NF_model.parameters()).device)

        # optimizer
        self.optimizer = torch.optim.Adam(self.NF_model.parameters(), lr=1e-5, weight_decay=1e-5)
        self.NF_model.train()

        # make data_standardized on device
        self.data_standardized = self.data_standardized.to(device)

        print("\nTraining the Normalizing flows: ")
        for k in range(self.nitts):
            #letting the dataflow through the NF in density estimation direction
            _, prior_logprob, log_det = self.NF_model(self.data_standardized)  # EDIT: from standardized to normal data
            logprob = prior_logprob + log_det
            loss = -torch.mean(logprob)  # NLL

            self.NF_model.zero_grad()
            loss.backward()
            self.optimizer.step()

            if k % 1000 == 0:
                print("NF Loss (itt {}):".format(k), loss.item())


        self.NF_model.eval()

    def dict_to_tensor(self, motion):
        """
        convert a dictionary of joint angles to a tensor of joint angles
        :param motion: A dictionary of joint angles
        :return: A tensor, with those coordinates.
        """
        tens = []
        # loop over all tensors, clone it and add to the list
        for joint in motion:
            # these keys will not be part of tensor
            if joint not in ["root", "rclavicle", "lclavicle", 'rfingers', 'lfingers']:
                # clone to keep track of differentiation graph
                t = motion[joint].clone()
                tens.append(t)
            if joint == "root":
                # clone to keep track of differentiation graph. For root, we don't want first 3 coordinates
                # they relate to translation of root.
                t = motion[joint][3:6].clone()
                tens.append(t)
        tens = torch.cat(tens)
        return tens

    def tensor_to_dict(self, tensorr):
        """
        converting a tensor of joint angles to a dictionary of joint angles
        :param tensorr: tensor (53,) of joint angles
        :return: a dictionary of joint angles
        """
        newDict = {}
        i = 0
        for key in motion_dict_keys:
            if key not in ["root", "rclavicle", "lclavicle", 'rfingers', 'lfingers']:
                n = motion_dict_keys[key]
                newDict[key] = copy.deepcopy(tensorr[i:i + n])
                i += n
            if key == "root":
                n = motion_dict_keys[key]
                t1 = torch.tensor([0.0, 0.0, 0.0], device=device)
                t2 = copy.deepcopy(tensorr[i:i + n])
                newDict[key] = torch.cat([t1, t2])
                i += n
        # the coordinates that are not part of the tensor, are set to zero.
        newDict["rclavicle"] = torch.tensor([0.0, 0.0])
        newDict["lclavicle"] = torch.tensor([0.0, 0.0])
        newDict["rfingers"] = torch.tensor([0.0])
        newDict["lfingers"] = torch.tensor([0.0])
        # getting requires_grad = True on all but root.
        for key in newDict:
            if key != "root":
                newDict[key].requires_grad = True
        return newDict


    def eval(self, motion):
        """
        :param motion: a joint angle dictionary
        :return: the density of that jont angle dictionary
        """
        #convertting the dictionary to a tensor
        tens = self.dict_to_tensor(motion) # .unsqueeze(dim=0) #for unstandardized
        # standardizing
        tens = (tens - self.mu) / self.stds
        _, prior_log, log_det = self.NF_model(tens)
        return log_det + prior_log

    def sample(self, num_samples=5):
        """
        :param num_samples: number of samples to be drawn
        :return: The samples as list of dictionaries
        """
        if self.flowtype == "planar":
            raise NotImplementedError("sample not implemented for planar")
        else:
            samples = self.NF_model.sample(num_samples=num_samples)
            #we only care about the vectors in the last layers
            samples = samples[-1]
            motionDicts = []
            for i in range(num_samples):
                #we need to de-standardize
                tensorr = samples[i, :].detach().clone()*self.stds + self.mu
                # converting the dicct to tens
                motionDict = self.tensor_to_dict(tensorr[0, :])
                motionDicts.append(motionDict)
            return motionDicts

    def sampleDirection(self, vectors):
        if self.flowtype == "planar":
            raise NotImplementedError("sample not implemented for planar")
        else:
            # let vector flow in sampling direction
            samples = self.NF_model.sampleDirection(vectors)
            # we only care about the vectors in the last layers
            samples = samples[-1]
            motionDicts = []
            for i in range(vectors.shape[0]):
                # we need to de-standardize
                tensorr = samples[i, :].detach().clone() * self.stds + self.mu
                # converting the dicct to tens
                motionDict = self.tensor_to_dict(tensorr[0, :])
                motionDicts.append(motionDict)
            return motionDicts


class noRoot_gauss:
    def __init__(self, frameslist):
        """
        :param frameslist: list of parsed amc-files
        """
        self.data = None
        self.mu = None
        self.cov = None
        self.frameslist = frameslist
        for i in range(len(frameslist)):
            self.get_data(frameslist[i])
        self.fit()

    def get_data(self, frames):
        """
        :param frames: a parsed amc-file (a list of joint angles)
        :return:
        """
        # Create empty data tensor, 62 is because that is the total degrees of freedom
        if self.data == None:  # if no data is loaded yet
            data = torch.zeros(len(frames), 62, device=device)
            for i in range(len(frames)):
                data[i] = torch.tensor([item for sublist in list(frames[i].values()) for item in sublist],
                                       device=device)  # flatten frame, code credit:https://stackoverflow.com/a/952952
            self.data = data

            # removing idxs  [24, 25, 33, 36, 37, 45]
            self.data = torch.hstack([self.data[:, 3:24], self.data[:, 26:33], self.data[:, 34:36],
                                      self.data[:, 38:45], self.data[:, 46:]])
        else:  # If one or more frames are already loaded
            data = torch.zeros(len(frames), 62, device=device)
            for i in range(len(frames)):
                data[i] = torch.tensor([item for sublist in list(frames[i].values()) for item in sublist],
                                       device=device)

            # fjerner idxs  [0, 24, 25, 33, 36, 37, 45]
            data = torch.hstack([data[:, 3:24], data[:, 26:33], data[:, 34:36],
                                 data[:, 38:45], data[:, 46:]])

            # stacking the v's
            self.data = torch.vstack((self.data, data))

    def fit(self):
        self.mu = torch.mean(self.data, dim=0)
        self.stds = self.data.std(dim=0, keepdim=True)
        self.data_standardized = (self.data - self.mu) / self.stds
        self.cov = torch.tensor(np.cov(self.data_standardized.cpu().detach().numpy().T))
        self.cov = self.cov.to(device)
        # we add a small value to the cov to make it invertible
        self.covinv = torch.inverse(self.cov + 0.00001)
        self.normdist = torch.distributions.multivariate_normal.MultivariateNormal(
            loc=torch.zeros(self.data.shape[1], dtype=torch.float64, device=device),
            covariance_matrix=self.cov)

    def dict_to_tensor(self, motion):
        """
        convert a dictionary of joint angles to a tensor of joint angles
        :param motion: A dictionary of joint angles
        :return: A tensor, with those coordinates.
        """
        tens = []
        # loop over all tensors, clone it and add to the list
        for joint in motion:
            # these keys will not be part of tensor
            if joint not in ["root", "rclavicle", "lclavicle", 'rfingers', 'lfingers']:
                # clone to keep track of differentiation graph
                t = motion[joint].clone()
                tens.append(t)
            if joint == "root":
                # clone to keep track of differentiation graph. For root, we don't want first 3 coordinates
                # they relate to translation of root.
                t = motion[joint][3:6].clone()
                tens.append(t)
        tens = torch.cat(tens)
        return tens

    def tensor_to_dict(self, tensorr):
        """
        converting a tensor of joint angles to a dictionary of joint angles
        :param tensorr: tensor (53,) of joint angles
        :return: a dictionary of joint angles
        """
        newDict = {}
        i = 0
        for key in motion_dict_keys:
            if key not in ["root", "rclavicle", "lclavicle", 'rfingers', 'lfingers']:
                n = motion_dict_keys[key]
                newDict[key] = copy.deepcopy(tensorr[i:i + n])
                i += n
            if key == "root":
                n = motion_dict_keys[key]
                t1 = torch.tensor([0.0, 0.0, 0.0], device=device)
                t2 = copy.deepcopy(tensorr[i:i + n])
                newDict[key] = torch.cat([t1, t2])
                i += n
        # the coordinates that are not part of the tensor, are set to zero.
        newDict["rclavicle"] = torch.tensor([0.0, 0.0])
        newDict["lclavicle"] = torch.tensor([0.0, 0.0])
        newDict["rfingers"] = torch.tensor([0.0])
        newDict["lfingers"] = torch.tensor([0.0])
        # getting requires_grad = True on all but root.
        for key in newDict:
            if key != "root":
                newDict[key].requires_grad = True
        return newDict

    def eval(self, motion):
        tens = self.dict_to_tensor(motion)
        tens = (tens - self.mu) / self.stds
        tens = tens[0, :]
        return self.normdist.log_prob(tens)

    def sample(self, num_sampes=5):
        raise NotImplementedError("sample has not been implemented for Gauss")

    
    
class zeroPrior:
    def __init__(self):
        pass
    
    def eval(self, motion):
        return torch.tensor(0)
        


        
