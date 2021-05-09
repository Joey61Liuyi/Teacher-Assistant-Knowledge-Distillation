import torch
import torch.nn as nn
from operations import *
from torch.autograd import Variable
from utils import drop_path

#
# class Cell(nn.Module):
#
#   def __init__(self, genotype, C_prev_prev, C_prev, C, reduction, reduction_prev):
#     super(Cell, self).__init__()
#     print(C_prev_prev, C_prev, C)
#
#     if reduction_prev:
#       self.preprocess0 = FactorizedReduce(C_prev_prev, C)
#     else:
#       self.preprocess0 = ReLUConvBN(C_prev_prev, C, 1, 1, 0)
#     self.preprocess1 = ReLUConvBN(C_prev, C, 1, 1, 0)
#
#     if reduction:
#       op_names, indices = zip(*genotype.reduce)
#       concat = genotype.reduce_concat
#     else:
#       op_names, indices = zip(*genotype.normal)
#       concat = genotype.normal_concat
#     self._compile(C, op_names, indices, concat, reduction)
#
#   def _compile(self, C, op_names, indices, concat, reduction):
#     assert len(op_names) == len(indices)
#     self._steps = len(op_names) // 2
#     self._concat = concat
#     self.multiplier = len(concat)
#
#     self._ops = nn.ModuleList()
#     for name, index in zip(op_names, indices):
#       stride = 2 if reduction and index < 2 else 1
#       op = OPS[name](C, stride, True)
#       self._ops += [op]
#     self._indices = indices
#
#   def forward(self, s0, s1, drop_prob):
#     s0 = self.preprocess0(s0)
#     s1 = self.preprocess1(s1)
#
#     states = [s0, s1]
#     for i in range(self._steps):
#       h1 = states[self._indices[2*i]]
#       h2 = states[self._indices[2*i+1]]
#       op1 = self._ops[2*i]
#       op2 = self._ops[2*i+1]
#       h1 = op1(h1)
#       h2 = op2(h2)
#       if self.training and drop_prob > 0.:
#         if not isinstance(op1, Identity):
#           h1 = drop_path(h1, drop_prob)
#         if not isinstance(op2, Identity):
#           h2 = drop_path(h2, drop_prob)
#       s = h1 + h2
#       states += [s]
#     return torch.cat([states[i] for i in self._concat], dim=1)
#


class Cell(nn.Module):

  def __init__(self, genotype, C_prev_prev, C_prev, C, reduction, reduction_prev):
    super(Cell, self).__init__()
    print(C_prev_prev, C_prev, C)

    if reduction_prev:
      self.preprocess0 = FactorizedReduce(C_prev_prev, C)
    else:
      self.preprocess0 = ReLUConvBN(C_prev_prev, C, 1, 1, 0)
    self.preprocess1 = ReLUConvBN(C_prev, C, 1, 1, 0)

    if reduction:
      op_names, indices, values = zip(*genotype.reduce)
      concat = genotype.reduce_concat
    else:
      op_names, indices, values = zip(*genotype.normal)
      concat = genotype.normal_concat
    self._compile(C, op_names, indices, values, concat, reduction)

  def _compile(self, C, op_names, indices, values, concat, reduction):
    assert len(op_names) == len(indices)
    self._steps = len(op_names) // 2
    self._concat = concat
    self.multiplier = len(concat)

    self._ops = nn.ModuleList()
    for name, index in zip(op_names, indices):
      stride = 2 if reduction and index < 2 else 1
      op = OPS[name](C, C, stride, True, True)
      self._ops.append(op)
    self._indices = indices
    self._values = values

  def forward(self, s0, s1, drop_prob):
    s0 = self.preprocess0(s0)
    s1 = self.preprocess1(s1)

    states = [s0, s1]
    for i in range(self._steps):
      h1 = states[self._indices[2 * i]]
      h2 = states[self._indices[2 * i + 1]]
      op1 = self._ops[2 * i]
      op2 = self._ops[2 * i + 1]
      h1 = op1(h1)
      h2 = op2(h2)
      if self.training and drop_prob > 0.:
        if not isinstance(op1, Identity):
          h1 = drop_path(h1, drop_prob)
        if not isinstance(op2, Identity):
          h2 = drop_path(h2, drop_prob)

      s = h1 + h2

      states += [s]
    return torch.cat([states[i] for i in self._concat], dim=1)


class Transition(nn.Module):

  def __init__(self, C_prev_prev, C_prev, C, reduction_prev, multiplier=4):
    super(Transition, self).__init__()
    if reduction_prev:
      self.preprocess0 = FactorizedReduce(C_prev_prev, C)
    else:
      self.preprocess0 = ReLUConvBN(C_prev_prev, C, 1, 1, 0)
    self.preprocess1 = ReLUConvBN(C_prev, C, 1, 1, 0)
    self.multiplier = multiplier

    self.reduction = True
    self.ops1 = nn.ModuleList(
      [nn.Sequential(
        nn.ReLU(inplace=False),
        nn.Conv2d(C, C, (1, 3), stride=(1, 2), padding=(0, 1), groups=8, bias=False),
        nn.Conv2d(C, C, (3, 1), stride=(2, 1), padding=(1, 0), groups=8, bias=False),
        nn.BatchNorm2d(C, affine=True),
        nn.ReLU(inplace=False),
        nn.Conv2d(C, C, 1, stride=1, padding=0, bias=False),
        nn.BatchNorm2d(C, affine=True)),
        nn.Sequential(
          nn.ReLU(inplace=False),
          nn.Conv2d(C, C, (1, 3), stride=(1, 2), padding=(0, 1), groups=8, bias=False),
          nn.Conv2d(C, C, (3, 1), stride=(2, 1), padding=(1, 0), groups=8, bias=False),
          nn.BatchNorm2d(C, affine=True),
          nn.ReLU(inplace=False),
          nn.Conv2d(C, C, 1, stride=1, padding=0, bias=False),
          nn.BatchNorm2d(C, affine=True))])

    self.ops2 = nn.ModuleList(
      [nn.Sequential(
        nn.MaxPool2d(3, stride=2, padding=1),
        nn.BatchNorm2d(C, affine=True)),
        nn.Sequential(
          nn.MaxPool2d(3, stride=2, padding=1),
          nn.BatchNorm2d(C, affine=True))])

  def forward(self, s0, s1, drop_prob=-1):
    s0 = self.preprocess0(s0)
    s1 = self.preprocess1(s1)

    X0 = self.ops1[0](s0)
    X1 = self.ops1[1](s1)
    if self.training and drop_prob > 0.:
      X0, X1 = drop_path(X0, drop_prob), drop_path(X1, drop_prob)

    # X2 = self.ops2[0] (X0+X1)
    X2 = self.ops2[0](s0)
    X3 = self.ops2[1](s1)
    if self.training and drop_prob > 0.:
      X2, X3 = drop_path(X2, drop_prob), drop_path(X3, drop_prob)
    return torch.cat([X0, X1, X2, X3], dim=1)


class AuxiliaryHeadCIFAR(nn.Module):

  def __init__(self, C, num_classes):
    """assuming input size 8x8"""
    super(AuxiliaryHeadCIFAR, self).__init__()
    self.features = nn.Sequential(
      nn.ReLU(inplace=True),
      nn.AvgPool2d(5, stride=3, padding=0, count_include_pad=False), # image size = 2 x 2
      nn.Conv2d(C, 128, 1, bias=False),
      nn.BatchNorm2d(128),
      nn.ReLU(inplace=True),
      nn.Conv2d(128, 768, 2, bias=False),
      nn.BatchNorm2d(768),
      nn.ReLU(inplace=True)
    )
    self.classifier = nn.Linear(768, num_classes)

  def forward(self, x):
    x = self.features(x)
    x = self.classifier(x.view(x.size(0),-1))
    return x


class AuxiliaryHeadImageNet(nn.Module):

  def __init__(self, C, num_classes):
    """assuming input size 14x14"""
    super(AuxiliaryHeadImageNet, self).__init__()
    self.features = nn.Sequential(
      nn.ReLU(inplace=True),
      nn.AvgPool2d(5, stride=2, padding=0, count_include_pad=False),
      nn.Conv2d(C, 128, 1, bias=False),
      nn.BatchNorm2d(128),
      nn.ReLU(inplace=True),
      nn.Conv2d(128, 768, 2, bias=False),
      # NOTE: This batchnorm was omitted in my earlier implementation due to a typo.
      # Commenting it out for consistency with the experiments in the paper.
      # nn.BatchNorm2d(768),
      nn.ReLU(inplace=True)
    )
    self.classifier = nn.Linear(768, num_classes)

  def forward(self, x):
    x = self.features(x)
    x = self.classifier(x.view(x.size(0),-1))
    return x


# class NetworkCIFAR(nn.Module):
#
#   def __init__(self, C, num_classes, layers, auxiliary, genotype):
#     super(NetworkCIFAR, self).__init__()
#     self._layers = layers
#     self._auxiliary = auxiliary
#
#     stem_multiplier = 3
#     C_curr = stem_multiplier*C
#     self.stem = nn.Sequential(
#       nn.Conv2d(3, C_curr, 3, padding=1, bias=False),
#       nn.BatchNorm2d(C_curr)
#     )
#
#     C_prev_prev, C_prev, C_curr = C_curr, C_curr, C
#     self.cells = nn.ModuleList()
#     reduction_prev = False
#     for i in range(layers):
#       if i in [layers//3, 2*layers//3]:
#         C_curr *= 2
#         reduction = True
#       else:
#         reduction = False
#       cell = Cell(genotype, C_prev_prev, C_prev, C_curr, reduction, reduction_prev)
#       reduction_prev = reduction
#       self.cells += [cell]
#       C_prev_prev, C_prev = C_prev, cell.multiplier*C_curr
#       if i == 2*layers//3:
#         C_to_auxiliary = C_prev
#
#     if auxiliary:
#       self.auxiliary_head = AuxiliaryHeadCIFAR(C_to_auxiliary, num_classes)
#     self.global_pooling = nn.AdaptiveAvgPool2d(1)
#     self.classifier = nn.Linear(C_prev, num_classes)
#
#   def forward(self, input):
#     logits_aux = None
#     s0 = s1 = self.stem(input)
#     for i, cell in enumerate(self.cells):
#       s0, s1 = s1, cell(s0, s1, self.drop_path_prob)
#       if i == 2*self._layers//3:
#         if self._auxiliary and self.training:
#           logits_aux = self.auxiliary_head(s1)
#     out = self.global_pooling(s1)
#     logits = self.classifier(out.view(out.size(0),-1))
#     return logits, logits_aux

class NetworkCIFAR(nn.Module):

  def __init__(self, C, num_classes, layers, auxiliary, genotype):
    super(NetworkCIFAR, self).__init__()
    self._layers = layers

    stem_multiplier = 3
    C_curr = stem_multiplier * C
    self.stem = nn.Sequential(
      nn.Conv2d(3, C_curr, 3, padding=1, bias=False),
      nn.BatchNorm2d(C_curr)
    )

    C_prev_prev, C_prev, C_curr = C_curr, C_curr, C
    self.cells = nn.ModuleList()
    reduction_prev = False
    for i in range(layers):
      if i in [layers // 3, 2 * layers // 3]:
        C_curr *= 2
        reduction = True
      else:
        reduction = False
      if reduction and genotype.reduce is None:
        cell = Transition(C_prev_prev, C_prev, C_curr, reduction_prev)
      else:
        cell = Cell(genotype, C_prev_prev, C_prev, C_curr, reduction, reduction_prev)
      reduction_prev = reduction
      self.cells.append(cell)
      C_prev_prev, C_prev = C_prev, cell.multiplier * C_curr
      if i == 2 * layers // 3:
        C_to_auxiliary = C_prev

    if auxiliary:
      self.auxiliary_head = AuxiliaryHeadCIFAR(C_to_auxiliary, num_classes)
    else:
      self.auxiliary_head = None
    self.global_pooling = nn.AdaptiveAvgPool2d(1)
    self.classifier = nn.Linear(C_prev, num_classes)
    self.drop_path_prob = -1

  def update_drop_path(self, drop_path_prob):
    self.drop_path_prob = drop_path_prob

  def auxiliary_param(self):
    if self.auxiliary_head is None:
      return []
    else:
      return list(self.auxiliary_head.parameters())

  def forward(self, inputs):
    s0 = s1 = self.stem(inputs)
    for i, cell in enumerate(self.cells):
      s0, s1 = s1, cell(s0, s1, self.drop_path_prob)
      if i == 2 * self._layers // 3:
        if self.auxiliary_head and self.training:
          logits_aux = self.auxiliary_head(s1)
    out = self.global_pooling(s1)
    out = out.view(out.size(0), -1)
    logits = self.classifier(out)

    if self.auxiliary_head and self.training:
      return logits, logits_aux
    else:
      return logits


class NetworkImageNet(nn.Module):

  def __init__(self, C, num_classes, layers, auxiliary, genotype):
    super(NetworkImageNet, self).__init__()
    self._layers = layers
    self._auxiliary = auxiliary

    self.stem0 = nn.Sequential(
      nn.Conv2d(3, C // 2, kernel_size=3, stride=2, padding=1, bias=False),
      nn.BatchNorm2d(C // 2),
      nn.ReLU(inplace=True),
      nn.Conv2d(C // 2, C, 3, stride=2, padding=1, bias=False),
      nn.BatchNorm2d(C),
    )

    self.stem1 = nn.Sequential(
      nn.ReLU(inplace=True),
      nn.Conv2d(C, C, 3, stride=2, padding=1, bias=False),
      nn.BatchNorm2d(C),
    )

    C_prev_prev, C_prev, C_curr = C, C, C

    self.cells = nn.ModuleList()
    reduction_prev = True
    for i in range(layers):
      if i in [layers // 3, 2 * layers // 3]:
        C_curr *= 2
        reduction = True
      else:
        reduction = False
      cell = Cell(genotype, C_prev_prev, C_prev, C_curr, reduction, reduction_prev)
      reduction_prev = reduction
      self.cells += [cell]
      C_prev_prev, C_prev = C_prev, cell.multiplier * C_curr
      if i == 2 * layers // 3:
        C_to_auxiliary = C_prev

    if auxiliary:
      self.auxiliary_head = AuxiliaryHeadImageNet(C_to_auxiliary, num_classes)
    self.global_pooling = nn.AvgPool2d(7)
    self.classifier = nn.Linear(C_prev, num_classes)

  def forward(self, input):
    logits_aux = None
    s0 = self.stem0(input)
    s1 = self.stem1(s0)
    for i, cell in enumerate(self.cells):
      s0, s1 = s1, cell(s0, s1, self.drop_path_prob)
      if i == 2 * self._layers // 3:
        if self._auxiliary and self.training:
          logits_aux = self.auxiliary_head(s1)
    out = self.global_pooling(s1)
    logits = self.classifier(out.view(out.size(0), -1))
    return logits, logits_aux

