from torch import topk, any, sum
import torch


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n

    def get_avg(self, percentage=False):
        return self.sum / self.count if not percentage else self.sum * 100 / self.count


def topk_acc(preds, targs, targs_perm=None, k=5, avg=False):
    if avg:
        num = preds.shape[0]
    else:
        num = 1
    _, top_k_inds = topk(preds, k)
    top_5 = 1 / num * sum(any(top_k_inds == targs.unsqueeze(dim=1), dim=1), dim=0)
    acc = 1 / num * sum(top_k_inds[:, 0].eq(targs), dim=0)

    if targs_perm is not None:
        top_5_perm = (1 / num * sum(any(top_k_inds == targs_perm.unsqueeze(dim=1), dim=1), dim=0))
        acc_perm = 1 / num * sum(top_k_inds[:, 0].eq(targs_perm), dim=0)

        return torch.maximum(acc, acc_perm), torch.maximum(top_5, top_5_perm)

    return acc.item(), top_5.item()


def real_acc(preds, targs, k, avg=False):
    if avg:
        num = preds.shape[0]
    else:
        num = 1
    _, top_k_inds = topk(preds, k)
    top_1_inds = top_k_inds[:, 0]
    acc_real = 1 / num * sum(any(top_1_inds.unsqueeze(dim=1).eq(targs), dim=1), dim=0)

    return acc_real
