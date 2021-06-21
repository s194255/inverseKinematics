import numpy as np
import matplotlib.pyplot as plt
#from transforms3d.euler import euler2mat
from mpl_toolkits.mplot3d import Axes3D
import torch
from AMCParsermaster.pytorch3d import euler_angles_to_matrix, euler2mat
import AMCParsermaster.amc_parser as amc2

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')



class Joint:
  def __init__(self, name, direction, length, axis, dof, limits):
    """
    Definition of basic joint. The joint also contains the information of the
    bone between it's parent joint and itself. Refer
    [here](https://research.cs.wisc.edu/graphics/Courses/cs-838-1999/Jeff/ASF-AMC.html)
    for detailed description for asf files.

    Parameter
    ---------
    name: Name of the joint defined in the asf file. There should always be one
    root joint. String.

    direction: Default direction of the joint(bone). The motions are all defined
    based on this default pose.

    length: Length of the bone.

    axis: Axis of rotation for the bone.

    dof: Degree of freedom. Specifies the number of motion channels and in what
    order they appear in the AMC file.

    limits: Limits on each of the channels in the dof specification

    """
    self.name = name
    self.direction = torch.reshape(direction, [3, 1])
    self.length = length
    axis = torch.deg2rad(axis)
    self.C = euler2mat(axis)
    self.Cinv = torch.inverse(self.C)
    self.limits = torch.zeros([3, 2], device=device)
    for lm, nm in zip(limits, dof):
      if nm == 'rx':
        self.limits[0] = torch.tensor([lm], device=device)
      elif nm == 'ry':
        self.limits[1] = torch.tensor([lm], device=device)
      else:
        self.limits[2] = torch.tensor([lm], device=device)
    self.parent = None
    self.children = []
    self.coordinate = None
    self.matrix = None
    self.num_angles = len(dof)

  def set_motion(self, motion):
    kø = [self]
    while len(kø) > 0:
      jointt = kø.pop(0)
      if jointt.name == 'root':
        jointt.coordinate = torch.reshape(motion['root'][:3], [3, 1])
        rotation = torch.deg2rad(motion['root'][3:])
        jointt.matrix = torch.chain_matmul(jointt.C, euler2mat(rotation), jointt.Cinv)
      else:
        idx = 0
        rotation = torch.zeros(3, device=device)
        for axis, lm in enumerate(jointt.limits):
          if not torch.equal(lm, torch.zeros(2, device=device)):
            rotation[axis] = torch.clamp(motion[jointt.name][idx],lm[0],lm[1])
            idx += 1
        rotation = torch.deg2rad(rotation)
        #jointt.matrix = jointt.parent.matrix.dot(jointt.C).dot(euler2mat(*rotation)).dot(jointt.Cinv)
        jointt.matrix = torch.chain_matmul(jointt.parent.matrix, jointt.C, euler2mat(rotation), jointt.Cinv)
        #jointt.coordinate = jointt.parent.coordinate + jointt.length * jointt.matrix.dot(jointt.direction)
        jointt.coordinate = jointt.parent.coordinate + jointt.length*torch.matmul(jointt.matrix, jointt.direction)
      for child in jointt.children:
        kø.append(child)

  def draw(self, mark=None, savePath=None):
    joints = self.to_dict()
    fig = plt.figure()
    ax = Axes3D(fig)

    # ax.set_xlim3d(-50, 10)
    # ax.set_ylim3d(-20, 40)
    # ax.set_zlim3d(-20, 40)
    
    ax.set_xlim3d(-15, 15)
    ax.set_ylim3d(-15, 15)
    ax.set_zlim3d(-15, 15)

    xs, ys, zs = [], [], []
    for joint in joints.values():
      if mark:
        if joint != mark:
          xs.append(joint.coordinate[0, 0].cpu().detach().numpy())
          ys.append(joint.coordinate[1, 0].cpu().detach().numpy())
          zs.append(joint.coordinate[2, 0].cpu().detach().numpy())
      else:
        xs.append(joint.coordinate[0, 0].cpu().detach().numpy())
        ys.append(joint.coordinate[1, 0].cpu().detach().numpy())
        zs.append(joint.coordinate[2, 0].cpu().detach().numpy())
    plt.plot(zs, xs, ys, 'b.')
    if mark:
      joint_obj = joints[mark]
      plt.plot(joint_obj.coordinate[2,0].cpu().detach().numpy(),
               joint_obj.coordinate[0,0].cpu().detach().numpy(),
               joint_obj.coordinate[1,0].cpu().detach().numpy(), c="g", marker= "o")

    for joint in joints.values():
      child = joint
      if child.parent is not None:
        parent = child.parent
        xs = [child.coordinate[0, 0].cpu().detach().numpy(), parent.coordinate[0, 0].cpu().detach().numpy()]
        ys = [child.coordinate[1, 0].cpu().detach().numpy(), parent.coordinate[1, 0].cpu().detach().numpy()]
        zs = [child.coordinate[2, 0].cpu().detach().numpy(), parent.coordinate[2, 0].cpu().detach().numpy()]
        plt.plot(zs, xs, ys, 'r')
    if savePath == None:
      plt.show()
    else:
      plt.savefig(savePath)

  def to_dict(self):
    ret = {self.name: self}
    for child in self.children:
      ret.update(child.to_dict())
    return ret

  def pretty_print(self):
    print('===================================')
    print('joint: %s' % self.name)
    print('direction:')
    print(self.direction)
    print('limits:', self.limits)
    print('parent:', self.parent)
    print('children:', self.children)


def read_line(stream, idx):
  if idx >= len(stream):
    return None, idx
  line = stream[idx].strip().split()
  idx += 1
  return line, idx


def parse_asf(file_path):
  '''read joint data only'''
  with open(file_path) as f:
    content = f.read().splitlines()

  for idx, line in enumerate(content):
    # meta infomation is ignored
    if line == ':bonedata':
      content = content[idx+1:]
      break

  # read joints
  joints = {'root': Joint('root', torch.zeros(3, device=device), 0, torch.zeros(3, device=device), [], [])}
  idx = 0
  while True:
    # the order of each section is hard-coded

    line, idx = read_line(content, idx)

    if line[0] == ':hierarchy':
      break

    assert line[0] == 'begin'

    line, idx = read_line(content, idx)
    assert line[0] == 'id'

    line, idx = read_line(content, idx)
    assert line[0] == 'name'
    name = line[1]

    line, idx = read_line(content, idx)
    assert line[0] == 'direction'
    direction = torch.tensor([float(axis) for axis in line[1:]], device=device)

    # skip length
    line, idx = read_line(content, idx)
    assert line[0] == 'length'
    length = float(line[1])

    line, idx = read_line(content, idx)
    assert line[0] == 'axis'
    assert line[4] == 'XYZ'

    axis = torch.tensor([float(axis) for axis in line[1:-1]], device=device)

    dof = []
    limits = []

    line, idx = read_line(content, idx)
    if line[0] == 'dof':
      dof = line[1:]
      for i in range(len(dof)):
        line, idx = read_line(content, idx)
        if i == 0:
          assert line[0] == 'limits'
          line = line[1:]
        assert len(line) == 2
        mini = float(line[0][1:])
        maxi = float(line[1][:-1])
        limits.append((mini, maxi))

      line, idx = read_line(content, idx)

    assert line[0] == 'end'
    joints[name] = Joint(
      name,
      direction,
      length,
      axis,
      dof,
      limits
    )

  # read hierarchy
  assert line[0] == ':hierarchy'

  line, idx = read_line(content, idx)

  assert line[0] == 'begin'

  while True:
    line, idx = read_line(content, idx)
    if line[0] == 'end':
      break
    assert len(line) >= 2
    for joint_name in line[1:]:
      joints[line[0]].children.append(joints[joint_name])
    for nm in line[1:]:
      joints[nm].parent = joints[line[0]]

  return joints


def parse_amc(file_path):
  with open(file_path) as f:
    content = f.read().splitlines()

  for idx, line in enumerate(content):
    if line == ':DEGREES':
      content = content[idx+1:]
      break

  frames = []
  idx = 0
  line, idx = read_line(content, idx)
  assert line[0].isnumeric(), line
  EOF = False
  while not EOF:
    joint_degree = {}
    while True:
      line, idx = read_line(content, idx)
      if line is None:
        EOF = True
        break
      if line[0].isnumeric():
        break
      joint_degree[line[0]] = torch.tensor([float(deg) for deg in line[1:]], requires_grad=True, device=device)
    frames.append(joint_degree)
  return frames


def angle2pos(motion, joints, jointName):
  joints["root"].set_motion(motion)
    
  return joints[str(jointName)].coordinate

def angle2poses(motion, joints, jointNames):
  joints["root"].set_motion(motion)

  return [joints[str(jointName)].coordinate for jointName in jointNames]

def getFrameslist(dataPath):
  frameslist = []
  for i in range(len(dataPath)):
    frameslist.append(amc2.parse_amc(dataPath[i]))
  return frameslist


def test_all():
  import os
  lv0 = '../data'
  lv1s = os.listdir(lv0)
  for lv1 in lv1s:
    lv2s = os.listdir('/'.join([lv0, lv1]))
    asf_path = '%s/%s/%s.asf' % (lv0, lv1, lv1)
    print('parsing %s' % asf_path)
    joints = parse_asf(asf_path)
    motions = parse_amc('./nopose.amc')
    joints['root'].set_motion(motions[0])
    joints['root'].draw()

    # for lv2 in lv2s:
    #   if lv2.split('.')[-1] != 'amc':
    #     continue
    #   amc_path = '%s/%s/%s' % (lv0, lv1, lv2)
    #   print('parsing amc %s' % amc_path)
    #   motions = parse_amc(amc_path)
    #   for idx, motion in enumerate(motions):
    #     print('setting motion %d' % idx)
    #     joints['root'].set_motion(motion)


if __name__ == '__main__':
    #test_all()
    asf_path = '../01.asf'
    amc_path = '../playground.amc'
    joints = parse_asf(asf_path)
    motions = parse_amc(amc_path)
    frame_idx = 0
    joints['root'].set_motion(motions[frame_idx])
    joints['root'].draw()
