from copy import deepcopy

import torch
import torch.nn
import torch.nn.functional as F


def compute_fisher_trace(input, net, parameter_generator = None):
    '''
    Computes the trace of the Fisher matrix for a Categorical model. Expecting **linear** output.

    :param input: A (single) input suitable for net
    :param net: A nn.Module object with linear output
    :param parameter_generator: If None, all parameters will be considered (net.parameters() is taken)
    :return: The Fisher information for the input
    '''
    test_output = net(input)
    assert test_output.shape[0] == 1 and len(test_output.shape) == 2
    output_dim = test_output.shape[1]
    fisher_trace = 0
    if parameter_generator is None:
        parameter_generator = net.parameters()
    for parameter in parameter_generator:
        for j in range(output_dim):
            net.zero_grad()
            log_softmax_output = net(input)
            log_softmax_output[0,j].backward()
            log_softmax_grad = parameter.grad
            net.zero_grad()
            softmax_output = F.softmax(net(input), dim =1)
            softmax_output[0,j].backward()
            softmax_grad = parameter.grad
            fisher_trace += (log_softmax_grad * softmax_grad).sum()
    return fisher_trace

