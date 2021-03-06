#!/usr/bin/env python
# -*- coding: utf8 -*-

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np

import argparse

m_plus = 0.9
m_min = 1.0 - m_plus


def conv2d_output_size(height, width, conv):
    """
    Calculate the height and with of the output of a 2-dimensional convolution,
    given the height and width of the input.

    Source for the formula: http://pytorch.org/docs/master/nn.html#conv2d
    """

    output_height = np.floor((height + 2 * conv.padding[0] - conv.dilation[0] * (conv.kernel_size[0] - 1) - 1) / conv.stride[0] + 1)
    output_width = np.floor((width + 2 * conv.padding[1] - conv.dilation[1] * (conv.kernel_size[1] - 1) - 1) / conv.stride[1] + 1)
    return int(output_height), int(output_width)

def batch_softmax(x):
    concatenated = torch.cat(x)
    s = F.softmax(concatenated)
    return s.view(*x.size())

def augmentation(x, max_shift=2):
    """
    Shift batch of images.
    Source: https://github.com/gram-ai/capsule-networks/blob/d054428af39c1fafd57a980f80be194308e220e2/capsule_network.py#L25

    """
    _, _, height, width = x.size()

    h_shift, w_shift = np.random.randint(-max_shift, max_shift + 1, size=2)
    source_height_slice = slice(max(0, h_shift), h_shift + height)
    source_width_slice = slice(max(0, w_shift), w_shift + width)
    target_height_slice = slice(max(0, -h_shift), -h_shift + height)
    target_width_slice = slice(max(0, -w_shift), -w_shift + width)

    shifted_image = torch.zeros(*x.size())
    shifted_image[:, :, source_height_slice, source_width_slice] = x[:, :, target_height_slice, target_width_slice]
    return shifted_image.float()

class CapsuleLoss(nn.Module):
    def __init__(self, gamma=0.5):
        super(CapsuleLoss, self).__init__()
        self.reconstruction_loss = nn.MSELoss(size_average=False)  # Losses of a batch are summed
        self.gamma = gamma

    def forward(self, vectors, labels, reconstructions, images):
        v_norm = vectors.norm(2, dim=2)
        left = labels * F.relu(m_plus - v_norm, inplace=True) ** 2
        right = self.gamma * (1 - labels) * F.relu(v_norm - m_min, inplace=True) ** 2
        margin_loss = torch.sum(left + right)  # Losses of a batch are summed
        reconstruction_loss = self.reconstruction_loss(reconstructions, images)

        return (margin_loss + 0.0005 * reconstruction_loss) / images.size(0)

class CapsuleLayer(nn.Module):
    """Capsule layer of a Capsule net."""
    def __init__(self, num_capsules, num_route_nodes, in_channels, out_channels, kernel_size=None, stride=None, use_cuda=False):
        super(CapsuleLayer, self).__init__()
        self.num_capsules = num_capsules
        self.num_route_nodes = num_route_nodes
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.use_cuda = use_cuda

        if num_route_nodes == -1:  # Capsule layer with convolutional units
            self.capsules = nn.ModuleList(
                [nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride) for _ in range(num_capsules)]
            )
        else:  # Fully connected layer
            self.routing_weights = nn.Parameter(torch.randn(num_capsules, num_route_nodes, in_channels, out_channels))

    def forward(self, x, num_iterations=None):
        """
        for all capsule i in layer l and capsule j in layer (l + 1): b_{ij} ← 0
        for r iterations do:
          for all capsule i in layer l: c_i ← softmax(b_i)
          for all capsule j in layer (l+1): sj ← sum(c_{ij} * prediction_vector)
          for all capsule j in layer (l + 1): v_j ← squash(s_j)
          for all capsule i in layer l and capsule j in layer (l + 1): b_{ij} ← b_{ij} + prediction_vector * v_j
        return v_j

        """
        if self.num_route_nodes == -1:
            outputs = [capsule(x).view(x.size(0), -1, 1) for capsule in self.capsules]
            outputs = torch.cat(outputs, dim=-1)
            outputs = CapsuleLayer.squash(outputs)
        else:
            prediction_vectors = x[None, :, :, None, :].matmul(self.routing_weights[:, None, :, :, :])
            logits = Variable(torch.zeros(*prediction_vectors.size()))
            if self.use_cuda:
                logits = logits.cuda()

            for _ in range(num_iterations):
                coupling_coefficients = batch_softmax(logits)
                s = torch.sum(coupling_coefficients * prediction_vectors, dim=2, keepdim=True)
                v = CapsuleLayer.squash(s)
                logits = logits + prediction_vectors * v
            outputs = v

        return outputs

    def squash(v):
        n = v.norm(2, dim=2, keepdim=True)
        n_squared = n ** 2
        scale = n_squared / (1 + n_squared)
        return scale * v / n

class CapsuleNet(nn.Module):
    """Capsule Net."""
    def __init__(self, num_classes, height, width, reconstr_hidden1=512, reconstr_hidden2=1024, use_cuda=False):
        super(CapsuleNet, self).__init__()
        self.num_classes = num_classes
        self.height = height
        self.width = width
        self.use_cuda = use_cuda

        self.conv = nn.Conv2d(in_channels=1, out_channels=256, kernel_size=9, stride=1)
        conv_size = conv2d_output_size(height, width, self.conv)
        self.non_linearity = nn.ReLU(inplace=True)
        self.primary_capsules = CapsuleLayer(num_capsules=8, num_route_nodes=-1, in_channels=256, out_channels=32, kernel_size=9, stride=2, use_cuda=use_cuda)
        # conv = self.primary_capsules.capsules[0]
        pc_h, pc_w = conv2d_output_size(*conv_size, self.primary_capsules.capsules[0])
        self.digit_capsules = CapsuleLayer(num_capsules=num_classes, num_route_nodes=32 * pc_h * pc_w, in_channels=8, out_channels=16, use_cuda=use_cuda)

        self.reconstruction = nn.Sequential(
            nn.Linear(num_classes * self.digit_capsules.out_channels, reconstr_hidden1),
            nn.ReLU(inplace=True),
            nn.Linear(reconstr_hidden1, reconstr_hidden2),
            nn.ReLU(inplace=True),
            nn.Linear(reconstr_hidden2, height * width),
            nn.Sigmoid()
        )

    def forward(self, x, labels=None):
        x = self.non_linearity(self.conv(x))
        x = self.primary_capsules(x)
        x = self.digit_capsules(x, num_iterations=3).squeeze().transpose(0, 1)

        classes = x.norm(2, dim=2)
        classes = F.softmax(classes)

        if labels is None:
            # In all batches, get the most active capsule.
            _, max_length_indices = classes.max(dim=1)
            mask = Variable(torch.sparse.torch.eye(self.num_classes))
            if self.use_cuda:
                mask = mask.cuda()
            mask = mask.index_select(dim=0, index=max_length_indices.data)
        else:  # During training, the capsules output is masked using correct labels
            mask = labels
        masked = mask[:, :, None] * x  # That None gives you an extra dimension, necessary for multiplication
        reconstructed = self.reconstruction(masked.contiguous().view(masked.size(0), -1))

        return x, masked, reconstructed

parser = argparse.ArgumentParser()
parser.add_argument("--data", type=str, default="./data", help="Path to directory with train/test data.")
parser.add_argument("--batch_size", type=int, default=100, help="Batch size.")
parser.add_argument("--num_epochs", type=int, default=30, help="Number of epochs to train.")
parser.add_argument("--gpu", dest="cuda", default=torch.cuda.is_available(), action="store_true", help="Run on GPU.")

def main():
    from torchvision import datasets, transforms
    from tensorboardX import SummaryWriter
    writer = SummaryWriter()
    args = parser.parse_args()
    data = datasets.MNIST(
        args.data,
        train=True,
        transform=transforms.Compose([
            transforms.ToTensor()
        ]))
    _, height, width = data[0][0].size()
    num_classes = len(set(data.train_labels))
    train_loader = torch.utils.data.DataLoader(
        data,
        batch_size=args.batch_size,
        shuffle=True
    )

    capsule_net = CapsuleNet(num_classes, height, width, use_cuda=args.cuda)
    if args.cuda:
        capsule_net = capsule_net.cuda()
    capsule_loss = CapsuleLoss()
    optimizer = torch.optim.Adam(capsule_net.parameters())

    train_steps = 0
    for epoch in range(args.num_epochs):
        for batch_idx, (data, target) in enumerate(train_loader):
            data = augmentation(data)
            target_onehot = torch.sparse.torch.eye(num_classes).index_select(dim=0, index=target)
            if args.cuda:
                data, target_onehot = data.cuda(), target_onehot.cuda()
            data, target_onehot = Variable(data), Variable(target_onehot)
            capsule_net.zero_grad()
            predictions, masked, reconstructions = capsule_net(data, target_onehot)
            loss = capsule_loss(predictions, target_onehot, reconstructions, data)
            loss.backward()
            optimizer.step()
            if batch_idx % 10 == 0:
                _, classes = F.softmax(predictions.norm(2, dim=2)).max(dim=1)
                accuracy = sum(classes.data.cpu() == target) / len(target)
                writer.add_scalar('model/accuracy', accuracy, train_steps)
                writer.add_scalar('model/loss', loss.data[0], train_steps)
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}\tAccuracy: {:.6f}'.format(
                    epoch, batch_idx * len(data), len(train_loader.dataset),
                    100. * batch_idx / len(train_loader), loss.data[0], accuracy))
            if batch_idx % 100 == 0:
                writer.add_image('original', data[-1][0].data, train_steps)
                writer.add_image('reconstruction', reconstructions[-1].view(height, width).data, train_steps)
                writer.add_text('vectors', str(list(masked[-1].view(-1).data)), train_steps)
            train_steps = train_steps + 1
        torch.save(capsule_net.state_dict(), 'epochs/epoch_%d.pt' % epoch)

if __name__ == '__main__':
    main()
