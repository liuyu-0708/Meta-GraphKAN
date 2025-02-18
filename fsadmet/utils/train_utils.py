

import random
import numpy as np
import torch


def build_negative_edges(batch):

    font_list = batch.edge_index[0, ::2].tolist()
    back_list = batch.edge_index[1, ::2].tolist()

    all_edge = {}
    for count, front_e in enumerate(font_list):

        if front_e not in all_edge:
            all_edge[front_e] = [back_list[count]]
        else:

            all_edge[front_e].append(back_list[count])

    negative_edges = []

    for num in range(batch.x.size()[0]):

        if num in all_edge:

            for num_back in range(num, batch.x.size()[0]):
                if num_back not in all_edge[num] and num != num_back:
                    negative_edges.append((num, num_back))
        else:
            for num_back in range(num, batch.x.size()[0]):
                if num != num_back:
                    negative_edges.append((num, num_back))

    negative_edge_index = torch.tensor(np.array(
        random.sample(negative_edges, len(font_list))).T,
                                       dtype=torch.long)

    return negative_edge_index


def update_params(base_model, loss, update_lr):

    grads = torch.autograd.grad(loss, base_model.parameters(), allow_unused=True)

    # Replace None gradients with zeros
    grads = [g if g is not None else torch.zeros_like(p) for g, p in zip(grads, base_model.parameters())]


    grads_vector = parameters_to_vector(grads)
    params_vector = parameters_to_vector(base_model.parameters())

    updated_params_vector = params_vector - grads_vector * update_lr

    return grads_vector, updated_params_vector

def parameters_to_vector(parameters):
    """Convert parameters to a single vector."""
    # Check that parameters is an iterable
    if not isinstance(parameters, list):
        parameters = list(parameters)
    
    # Use reshape instead of view
    vec = []
    for param in parameters:
        vec.append(param.reshape(-1))
    return torch.cat(vec)

def vector_to_parameters(vector, parameters):
    """Convert a vector back to the parameters."""
    # Check that parameters is an iterable
    if not isinstance(parameters, list):
        parameters = list(parameters)
    
    # Use reshape instead of view
    pointer = 0
    for param in parameters:
        num_param = param.numel()
        param.copy_(vector[pointer:pointer + num_param].reshape(param.shape))
        pointer += num_param





def build_negative_edges1(batch):

    font_list = batch.edge_index1[0, ::2].tolist()
    back_list = batch.edge_index1[1, ::2].tolist()

    all_edge = {}
    for count, front_e in enumerate(font_list):

        if front_e not in all_edge:
            all_edge[front_e] = [back_list[count]]
        else:

            all_edge[front_e].append(back_list[count])

    negative_edges = []

    for num in range(batch.x1.size()[0]):

        if num in all_edge:

            for num_back in range(num, batch.x1.size()[0]):
                if num_back not in all_edge[num] and num != num_back:
                    negative_edges.append((num, num_back))
        else:
            for num_back in range(num, batch.x1.size()[0]):
                if num != num_back:
                    
                    negative_edges.append((num, num_back))

    negative_edge_index = torch.tensor(np.array(
        random.sample(negative_edges, len(font_list))).T,
                                       dtype=torch.long)

    return negative_edge_index


