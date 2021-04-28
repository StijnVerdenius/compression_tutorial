import math

import torch
import torch.nn as nn


class PrunableModel(nn.Module):

    def __init__(self, device='cpu'):
        super(PrunableModel, self).__init__()
        self.device = device

    def _post_init(self):
        """ initializes all pruning components """

        # define a mask for each parameter vector
        self.mask_dict = {
            name + ".weight": torch.ones_like(module.weight.data, device=self.device)
            for name, module in self.named_modules()
            if isinstance(module, (nn.Linear, nn.Conv2d))
        }

        # define vectors for each output shape. this shape is shared with the input shape from the next layer
        self.structured_vectors = {name: torch.ones(tens.shape[0], device=self.device) for name, tens in
                                   [x for x in self.mask][:-1]}

    def magnitude_prune_unstructured(self, percentage=0.0):
        """
        sets mask based on percentage (of remaining weights) and magnitude of weights
        does global magnitude pruning
        """

        percentage = percentage + self.unstructured_sparsity

        # get threshold
        all_weights = torch.cat(
            [torch.flatten(x) for name, x in self.named_parameters() if name in self.mask_dict]
        )
        count = len(all_weights)
        amount = int(count * percentage)
        limit = torch.topk(all_weights.abs(), amount, largest=False).values[-1]

        # prune
        for (name, weights) in self.named_parameters():
            if name in self.mask_dict:
                # prune on l1
                mask = weights.abs() > limit
                self.mask_dict[name] = mask
        self.apply_mask()

    def magnitude_prune_structured(self, percentage=0.0):
        """
        sets structured mask based on percentage (of remaining nodes) and magnitude of rows and columns
        does layer-wise magnitude pruning
        """

        percentage = percentage + self.structured_sparsity

        # determine how many nodes we are gonna prune per layer
        prunable_nodes = {
            name: math.ceil(percentage * x.shape[0]) for name, x in self.structured_vectors.items()
        }

        # count magnitudes
        weight_counts = {}
        last_name = None
        for name, param in self.named_parameters():
            if 'weight' in name:
                magnitude_output = param.abs().sum(dim=1)
                magnitude_input = param.abs().sum(dim=0)
                # input shape belongs to the magnitude of the previous layer
                if last_name is not None and last_name in weight_counts:
                    weight_counts[last_name] += magnitude_input
                if name in self.structured_vectors:
                    weight_counts[name] = magnitude_output
                    last_name = name

        for name, counts in weight_counts.items():
            limit = torch.topk(counts.abs(), prunable_nodes[name], largest=False).values[-1]
            self.structured_vectors[name] = (counts >= limit).float()

        self.apply_structured_vectors()
        self.apply_mask()

    def apply_structured_vectors(self):
        """ applies structured vectors to masks """
        last_name = None
        for name, mask_param in self.mask:
            # apply vector to input dimension
            if last_name is not None and last_name in self.structured_vectors:
                masker = self.structured_vectors[last_name]
                mask_param.data = mask_param.data * masker
            # apply vector to output dimension
            if name in self.structured_vectors:
                masker = self.structured_vectors[name]
                mask_param.data = (mask_param.data.t() * masker).t()
                last_name = name

    @property
    def mask(self):
        """ iterator as property """
        return self.mask_dict.items()

    @property
    def get_num_nodes(self):
        """ counts number of nodes in network """
        counter = 0
        last_addition = 0
        for i, (name, module) in enumerate(self.named_modules()):
            if (hasattr(module, "weight") or hasattr(module, "weights")) and not ("Norm" in str(module.__class__)):
                last_addition = module.weight.shape[0]  # add output shape each time
                counter += last_addition
        return counter - last_addition  # dont prune the output layer so remove last addition

    @property
    def get_num_nodes_unpruned(self):
        """ counts number of nodes in network which are still active """
        counter = 0
        last_addition = 0
        for i, (name, module) in enumerate(self.named_modules()):
            if (hasattr(module, "weight") or hasattr(module, "weights")) and not ("Norm" in str(module.__class__)):
                last_addition = (module.weight.abs().sum(dim=1) > 0).sum()  # add output shape each time of all nonzero
                counter += last_addition
        return counter - last_addition  # dont prune the output layer so remove last addition

    @property
    def get_num_weights(self):
        """ count number of weights in network """
        counter = 0
        for i, (name, module) in enumerate(self.named_modules()):
            if (hasattr(module, "weight") or hasattr(module, "weights")) and not ("Norm" in str(module.__class__)):
                counter += module.weight.shape[1] * module.weight.shape[0]
        return counter

    @property
    def get_num_weights_unpruned(self):
        """ count number of weights in network which are still active """

        counter = 0
        for i, mask_tensor in self.mask:
            counter += (mask_tensor > 0).sum()
        return counter

    @property
    def structured_sparsity(self):
        """ calculate structured sparsity """
        return (1.0 - ((self.get_num_nodes_unpruned + 1e-6) / (self.get_num_nodes + 1e-6))).item()

    @property
    def unstructured_sparsity(self):
        """ calculate unstructured sparsity """
        return (1.0 - ((self.get_num_weights_unpruned + 1e-6) / (self.get_num_weights + 1e-6))).item()

    def apply_mask(self):
        """ applies mask to both grads and weights """
        self._apply_grad_mask()
        self._apply_weight_mask()

    def _apply_weight_mask(self):
        """ applies mask to weights """

        with torch.no_grad():
            for name, tensor in self.named_parameters():
                if name in self.mask_dict:
                    tensor.data *= self.mask_dict[name]

    def _apply_grad_mask(self):
        """ applies mask to grads """

        with torch.no_grad():
            for name, tensor in self.named_parameters():
                if name in self.mask_dict and tensor.grad is not None:
                    tensor.grad.data *= self.mask_dict[name]
