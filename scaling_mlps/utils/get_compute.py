import torch
from fvcore.nn import FlopCountAnalysis


def get_compute(model, dataset_size, res):
    input = torch.randn(1, 3, res, res).cuda()
    flops = FlopCountAnalysis(model, input)

    return flops.total() * 3 * dataset_size
