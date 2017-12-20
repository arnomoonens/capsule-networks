#!/usr/bin/env python
# -*- coding: utf8 -*-

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

import argparse
import pdb

m_plus = 0.9
m_min = 1.0 - m_plus

def batch_softmax(x):
    concatenated = torch.cat(x)
    s = F.softmax(concatenated)
    return s.view(*x.size())

class CapsuleLoss(nn.Module):
    def __init__(self, gamma=0.5):
        super(CapsuleLoss, self).__init__()
        self.gamma = gamma

    def forward(self, vectors, labels):
        v_norm = vectors.norm(2, dim=2)
        left = labels * F.relu(m_plus - v_norm) ** 2
        right = self.gamma * (1 - labels) * F.relu(v_norm - m_min)
        return torch.sum(left + right)

class CapsuleLayer(nn.Module):
    """Capsule layer of a Capsule net."""
    def __init__(self, num_capsules, num_route_nodes, in_channels, out_channels, kernel_size=None, stride=None):
        super(CapsuleLayer, self).__init__()
        self.num_capsules = num_capsules
        self.num_route_nodes = num_route_nodes

        if num_route_nodes == -1:  # Capsule layer with convolutional units
            self.capsules = nn.ModuleList(
                [nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride) for _ in range(num_capsules)]
            )
        else:
            self.routing_weights = nn.Parameter(torch.randn(num_capsules, num_route_nodes, in_channels, out_channels))

    # def forward(prediction_vectors, num_iterations, layer_idx):
        """
        for all capsule i in layer l and capsule j in layer (l + 1): b_{ij} ← 0
        for r iterations do:
          for all capsule i in layer l: c_i ← softmax(b_i)
          for all capsule j in layer (l+1): sj ← sum(c_{ij} * prediction_vector)
          for all capsule j in layer (l + 1): v_j ← squash(s_j)
          for all capsule i in layer l and capsule j in layer (l + 1): b_{ij} ← b_{ij} + prediction_vector * v_j
        return v_j
        """
        # pass

    def forward(self, x, num_iterations=None):
        if self.num_route_nodes == -1:
            outputs = [capsule(x).view(x.size(0), -1, 1) for capsule in self.capsules]
            outputs = torch.cat(outputs, dim=-1)
            outputs = CapsuleLayer.squash(outputs)
        else:
            prediction_vectors = x[None, :, :, None, :].matmul(self.routing_weights[:, None, :, :, :])
            logits = Variable(torch.zeros(*prediction_vectors.size()))

            for _ in range(num_iterations):
                coupling_coefficients = batch_softmax(logits)
                s = torch.sum(coupling_coefficients * prediction_vectors, dim=2, keepdim=True)
                v = CapsuleLayer.squash(s)
                logits = logits + prediction_vectors * v
            outputs = v

        return outputs

    def squash(v):
        n = v.norm()
        n_squared = n ** 2
        scale = n_squared / (1 + n_squared)
        return scale * v / n

class CapsuleNet(nn.Module):
    """Capsule Net."""
    def __init__(self, num_classes):
        super(CapsuleNet, self).__init__()

        self.conv = nn.Conv2d(in_channels=1, out_channels=256, kernel_size=9, stride=1)
        self.non_linearity = nn.ReLU()
        self.primary_capsules = CapsuleLayer(num_capsules=8, num_route_nodes=-1, in_channels=256, out_channels=32, kernel_size=9, stride=2)
        self.digit_capsules = CapsuleLayer(num_capsules=num_classes, num_route_nodes=32 * 6 * 6, in_channels=8, out_channels=16)

    def forward(self, x):
        x = self.non_linearity(self.conv(x))
        x = self.primary_capsules(x)
        x = self.digit_capsules(x, num_iterations=3).squeeze().transpose(0, 1)

        # classes = x.norm()
        # classes = F.softmax(classes)
        return x

parser = argparse.ArgumentParser()
parser.add_argument("--batch_size", type=int, default=100, help="Batch size.")
parser.add_argument("--num_epochs", type=int, default=500, help="Number of epochs to train.")
parser.add_argument("--gpu", dest="cuda", default=torch.cuda.is_available(), action="store_true", help="Run on GPU.")

def main():
    from torchvision import datasets, transforms
    args = parser.parse_args()
    train_loader = torch.utils.data.DataLoader(
        datasets.MNIST('../data', train=True,
                       transform=transforms.Compose([
                           transforms.ToTensor(),
                           transforms.Normalize((0.1307,), (0.3081,))
                       ])),
        batch_size=args.batch_size,
        shuffle=True
    )

    capsule_net = CapsuleNet(num_classes=10)
    capsule_loss = CapsuleLoss()
    optimizer = torch.optim.Adam(capsule_net.parameters())

    for epoch in range(args.num_epochs):
        for batch_idx, (data, target) in enumerate(train_loader):
            if args.cuda:
                data, target = data.cuda(), target.cuda()
            target = torch.sparse.torch.eye(10).index_select(dim=0, index=target)  # to one-hot vectors
            data, target = Variable(data), Variable(target)
            capsule_net.zero_grad()
            predictions = capsule_net(data)
            loss = capsule_loss(predictions, target)
            loss.backward()
            optimizer.step()
            if batch_idx % 10 == 0:
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    epoch, batch_idx * len(data), len(train_loader.dataset),
                    100. * batch_idx / len(train_loader), loss.data[0]))

if __name__ == '__main__':
    main()
