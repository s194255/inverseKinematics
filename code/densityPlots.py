from Normalizing_flows.flows import MAF, NormalizingFlowModel
import torch
import numpy as np
import matplotlib.pyplot as plt
from Normalizing_flows.homemade_flows import planar, normalizeModel
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# z_train, distribution we want to imitate 
normdist = torch.distributions.multivariate_normal.MultivariateNormal(
    loc=torch.tensor([5.0, 4.0]),
    covariance_matrix=torch.eye(2))


z_train = normdist.sample(torch.Size([80]))

## Random distribution (to get more complex data fit)
## NB: Change limits in plots to fit the probability density
# z_train = torch.tensor([list(10*np.random.rand(200)),list(8*np.random.rand(40))]).T.float()


# Prior = Unit Gaussian
prior = torch.distributions.multivariate_normal.MultivariateNormal(
    loc=torch.tensor([0.0, 0.0]),
    covariance_matrix=torch.eye(2))

# %% Run this cell to train a planar flow

# Everytime range(K) is changed to range(K+1), another layer is added to the flow
flows = [planar(dim=2) for K in range(10)]


planar_model = normalizeModel(prior, flows)


optimizer = torch.optim.Adam(
    planar_model.parameters(), lr=1e-4, weight_decay=1e-5)  # todo tune WD
print("number of params: ", sum(p.numel() for p in planar_model.parameters()))

planar_model.train()
for k in range(5000+1):
    zs, prior_log, log_det = planar_model(z_train)
    loss = -torch.sum(prior_log+log_det)  # NLL

    planar_model.zero_grad()
    loss.backward()
    optimizer.step()

    if k % 1000 == 0:
        print("Loss: ", loss.item())

planar_model = normalizeModel(prior, flows)

# %% Run this cell to train a MAF flow

# Everytime range(K) is changed to range(K+1), another layer is added to the flow
flows = [MAF(dim=2, parity=K % 2) for K in range(7)]


MAF_model = NormalizingFlowModel(prior, flows)

# optimizer
optimizer = torch.optim.Adam(
    MAF_model.parameters(), lr=1e-4, weight_decay=1e-5)  # todo tune WD
print("number of params: ", sum(p.numel() for p in MAF_model.parameters()))


MAF_model.train()

for k in range(5000+1):
    zs, prior_logprob, log_det = MAF_model(z_train)
    logprob = prior_logprob + log_det
    loss = -torch.sum(logprob)  # NLL

    MAF_model.zero_grad()
    loss.backward()
    optimizer.step()

    if k % 1000 == 0:
        print("Loss: ", loss.item())


MAF_model.eval()


# %% X, Y for prob plots (limits)
z_test_x = torch.linspace(0, 10, 100)
z_test_y = torch.linspace(0, 10, 100)
X, Y = torch.meshgrid(z_test_x, z_test_y)

Y2 = np.array(Y)
X2 = np.array(X)


# %% Run this cell to get probabilities for planar model
probs = torch.zeros((len(X2), len(X2)))

for i in range(len(X2)):
    for j in range(len(X2)):
        a = torch.tensor([[X[i, j]], [Y[i, j]]])
        _, prior_log, log_det = planar_model(a.T)
        probs[i, j] = prior_log+log_det

probs = probs.detach().numpy()

# %% Probabilities for Gaussian with mean = (5,4)

probs2 = torch.zeros((len(X2), len(X2)))

for i in range(len(X2)):
    for j in range(len(X2)):
        a = torch.tensor([X[i, j], Y[i, j]])
        probs2[i, j] = normdist.log_prob(a)

probs2 = probs2.detach().numpy()


# %% Planar contour plot
xmin, xmax = 0, 10
ymin, ymax = 0, 10

fig = plt.figure(figsize=(8, 8))
ax = fig.gca()
ax.set_xlim(xmin, xmax)
ax.set_ylim(ymin, ymax)
cfset = ax.contourf(X2, Y2, probs, cmap='coolwarm', levels=30)
ax.imshow(np.rot90(probs), cmap='coolwarm', extent=[xmin, xmax, ymin, ymax])
cset = ax.contour(X2, Y2, probs, colors='k', levels=30)
ax.clabel(cset, inline=1, fontsize=10)
ax.set_xlabel('X')
ax.set_ylabel('Y')
plt.title('2D Planar model density, K=10')

# %%3D surface plot for planar
fig = plt.figure(figsize=(13, 7))
ax = plt.axes(projection='3d')
surf = ax.plot_surface(X2, Y2, probs, rstride=1, cstride=1,
                       cmap='coolwarm', edgecolor='none')
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('PDF')
ax.set_title('Surface plot of Planar flow, K=10')
fig.colorbar(surf, shrink=0.5, aspect=5)  # add color bar indicating the PDF
ax.view_init(60, 35)

# %% Probs for unit Gaussian

probs0 = torch.zeros((len(X2), len(X2)))

for i in range(len(X2)):
    for j in range(len(X2)):
        a = torch.tensor([X[i, j], Y[i, j]])
        probs0[i, j] = prior.log_prob(a)

probs0 = probs0.detach().numpy()

# %% contour for Prior unit Gaussian
xmin, xmax = 0, 10
ymin, ymax = 0, 10

fig = plt.figure(figsize=(8, 8))
ax = fig.gca()
ax.set_xlim(xmin, xmax)
ax.set_ylim(ymin, ymax)
cfset = ax.contourf(X2, Y2, probs0, cmap='coolwarm')
ax.imshow(np.rot90(probs0), cmap='coolwarm', extent=[xmin, xmax, ymin, ymax])
cset = ax.contour(X2, Y2, probs0, colors='k')
ax.clabel(cset, inline=1, fontsize=10)
ax.set_xlabel('X')
ax.set_ylabel('Y')
plt.title('Prior unit Gaussian, mean = (0,0)')


# %%  contour plot for Gaussian distribution we want to imitate
xmin, xmax = 0, 10
ymin, ymax = 0, 10

fig = plt.figure(figsize=(8, 8))
ax = fig.gca()
ax.set_xlim(xmin, xmax)
ax.set_ylim(ymin, ymax)
cfset = ax.contourf(X2, Y2, probs2, cmap='coolwarm')
ax.imshow(np.rot90(probs2), cmap='coolwarm', extent=[xmin, xmax, ymin, ymax])
cset = ax.contour(X2, Y2, probs2, colors='k')
ax.clabel(cset, inline=1, fontsize=10)
ax.set_xlabel('X')
ax.set_ylabel('Y')
plt.title('Gaussian, mean = (5,4)')


# %% 3D surface for Gaussian distribution we want to imitate

fig = plt.figure(figsize=(13, 7))
ax = plt.axes(projection='3d')
surf = ax.plot_surface(X2, Y2, probs2, rstride=1,
                       cstride=1, cmap='coolwarm', edgecolor='none')
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('PDF')
ax.set_title('Gaussian, mean = (5,4)', fontsize=25)
fig.colorbar(surf, shrink=0.5, aspect=5)  # add color bar indicating the PDF
ax.view_init(60, 35)

# %% Now for MAF, probabilities:

probs3 = torch.zeros((100, 100))

for i in range(100):
    for j in range(100):
        a = torch.tensor([X[i, j], Y[i, j]]).unsqueeze(dim=0)
        _, priorprob, logdet = MAF_model(a)
        probs3[i, j] = priorprob+logdet

probs3 = probs3.detach().numpy()

# %% MAF PLOTS
# Contour:

#change limits to fit data best
xidxmin, xidxmax, yidxmin, yidxmax = 15, 85, 3, 75
xmin, xmax = np.min(X2[xidxmin:xidxmax, yidxmin:yidxmax]), np.max(
    X2[xidxmin:xidxmax, yidxmin:yidxmax])
ymin, ymax = np.min(Y2[xidxmin:xidxmax, yidxmin:yidxmax]), np.max(
    Y2[xidxmin:xidxmax, yidxmin:yidxmax])

fig = plt.figure(figsize=(8, 8))
ax = fig.gca()
ax.set_xlim(xmin, xmax)
ax.set_ylim(ymin, ymax)
levels=np.arange(-6.5,-3.5,0.25)
cfset = ax.contourf(X2[xidxmin:xidxmax, yidxmin:yidxmax], Y2[xidxmin:xidxmax,
                    yidxmin:yidxmax], probs3[xidxmin:xidxmax, yidxmin:yidxmax], cmap='coolwarm',levels=levels)
ax.imshow(np.rot90(probs3[xidxmin:xidxmax, yidxmin:yidxmax]),
          cmap='coolwarm', extent=[xmin, xmax, ymin, ymax])
cset = ax.contour(X2[xidxmin:xidxmax, yidxmin:yidxmax], Y2[xidxmin:xidxmax,
                  yidxmin:yidxmax], probs3[xidxmin:xidxmax, yidxmin:yidxmax], colors='k',levels=levels)
ax.clabel(cset, inline=1, fontsize=10)
ax.set_xlabel('X')
ax.set_ylabel('Y')
plt.title('2D MAF model density',fontsize=25)

# %%3D surface for MAF
#change limits to fit data best
xidxmin, xidxmax, yidxmin, yidxmax = 15, 85, 3, 75

fig = plt.figure(figsize=(13, 7))
ax = plt.axes(projection='3d')
surf = ax.plot_surface(X2[xidxmin:xidxmax, yidxmin:yidxmax], Y2[xidxmin:xidxmax, yidxmin:yidxmax],
                       probs3[xidxmin:xidxmax, yidxmin:yidxmax], rstride=1, cstride=1, cmap='coolwarm', edgecolor='none')
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('PDF')
ax.set_title('Surface plot of MAF flow',fontsize=25)
fig.colorbar(surf, shrink=0.5, aspect=5)  # add color bar indicating the PDF
ax.view_init(40, 35)
