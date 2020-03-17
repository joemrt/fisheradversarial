from copy import deepcopy

import torch
import torch.nn.functional as F


def numeric_fisher_form(x, net, v, eps = 1e-3):
    '''
    Computes the quadratic form of the Fisher matrix along v.
    :param x: Input for net
    :param net: A nn.Module object with log_softmax output
    :param v: A list of vectors along which to compute the directional derivative.
    :param eps: Used for numeric approximations
    :return: A 0-dimensional tensor, the quadratic Fisher form.
    '''
    test_output = net(x)
    assert test_output.shape[0] == 1 and len(test_output.shape) == 2
    fisher_sum = 0
    for i, par in enumerate(net.parameters()):
        old_data = deepcopy(par.data)
        par.data += eps * v[i]
        new_plus_log_softmax = net(x)
        new_plus_softmax = F.softmax(new_plus_log_softmax, dim=1)
        par.data -= 2 * eps * v[i]
        new_minus_log_softmax = net(x)
        new_minus_softmax = F.softmax(new_minus_log_softmax, dim=1)
        par.data = old_data
        fisher_sum += 1/(2*eps)**2 * ((new_plus_log_softmax-new_minus_log_softmax) * (new_plus_softmax-new_minus_softmax)).sum()
    return fisher_sum

def fisher_information_sensitivity(x, net , v, eps=1e-3, parameter_generator=None, pred_approx=-1):
    '''
    Computes fisher information sensitivity for x and v.  

    :param x: Input of net. Should have batch size 1
    :param net: A nn.Module object with log-softmax output
    :param v: List of tensors. The direction along which to to take the directional derivative..
    :param eps: Used to approximate the directional derivative
    :param parameter_generator: The parameters of net for which to take the directional derivative. Should iterate through len(v) elements.
    Default (None) is all parameters of net
    :param pred_approx: If None, all classes will be considered.
    If positive this will be interpreted as the prediction to reduce the model to two output nodes.
    If -1, argmax will be applied to the output of net and the output is reduced to two output nodes.
    :return: A torch tensor of the same dimension as input.
    '''
    test_output = net(x)
    output_dim = test_output.shape[1]
    assert output_dim>1
    assert test_output.shape[0] == 1 and len(test_output.shape) == 2
    if pred_approx is not None:
        # Reduce the output to two classes
        assert pred_approx<output_dim
        if pred_approx == -1:
            _,pred = torch.max(test_output, dim=1)
        else:
            # to match shapes below
            pred = torch.tensor([pred_approx])
        def log_softmax_output(x):
            full_linear_output = net(x)
            assert type(pred) is torch.Tensor
            pred_output = full_linear_output[:,pred]
            sum_without_pred = torch.logsumexp(full_linear_output[:, torch.arange(output_dim) != pred], dim=1, keepdim=True)
            reduced_linear_output = torch.cat((pred_output, sum_without_pred), 1)
            return reduced_linear_output
    else:
        linear_output = net
    softmax_output = lambda x: F.softmax(log_softmax_output(x), dim = 1)
    if parameter_generator is None:
        parameter_generator = net.parameters()
    fisher_sum = 0
    x = deepcopy(x.data)
    x.requires_grad = True
    def get_x_grad():
        # return the gradient w.r.t x and null all gradients
        assert x.grad is not None
        grad = deepcopy(x.grad.data)
        x.grad.zero_()
        net.zero_grad()
        return grad
    for i, par in enumerate(parameter_generator):
        for j in range(output_dim):
            if pred_approx is not None:
                if j not in [0,1]:
                    continue
            old_data = deepcopy(par.data)
            # TODO: Make this more efficient, no need to compute twice the derivative
            # plus eps
            par.data += eps * v[i]
            log_softmax_output(x)[0, j].backward()
            new_plus_linear_grad = get_x_grad()
            softmax_output(x)[0, j].backward()
            new_plus_softmax_grad = get_x_grad()
            # minus eps
            par.data -= 2 * eps * v[i]
            log_softmax_output(x)[0, j].backward()
            new_minus_linear_grad = get_x_grad()
            softmax_output(x)[0, j].backward()
            new_minus_softmax_grad = get_x_grad()
            # reset and evaluate
            par.data = old_data
            fisher_sum += 1/(2*eps)**2 * ((new_plus_linear_grad-new_minus_linear_grad)
                                          * (new_plus_softmax_grad - new_minus_softmax_grad))

    return fisher_sum

