from math import prod
from cola.ops.operator_base import LinearOperator
import torch
from torch.nn import functional as F
from mm.blockdiag_butterfly_multiply import blockdiag_butterfly_multiply
from mm.btt_mvm import btt_mvm, btt_mvmt

class Monarch(LinearOperator):
    def __init__(self, Ms, shape):
        self.Ms = Ms
        dtype = self.Ms[0].dtype
        num_blocks = self.Ms[0].shape[0]
        in_blksz, out_blksz = self.Ms[0].shape[-1], self.Ms[-1].shape[-1]
        self.d_in_ext = in_blksz * num_blocks
        self.d_out_ext = out_blksz * num_blocks
        super().__init__(dtype, shape)

    def _rmatmat(self, v):
        d_in, d_out = self.shape
        if v.shape[-1] < self.d_in_ext:
            v = F.pad(v, (0, self.d_in_ext - d_in))

        out = blockdiag_butterfly_multiply(v, self.Ms[0], self.Ms[1])

        if out.shape[-1] > d_out:
            out = out[..., :d_out]
        return out

class TT(LinearOperator):
    """
    Tensor-Train operator

    Args:
        *Ms (array_like): Cores of the TT-decomposition
    """
    def __init__(self, Ms):
        self.Ms = Ms  # (ki, ni, mi, ki+1) ki = rank_in, ki+1 = rank_out
        dtype = self.Ms[0].dtype
        n = prod([M.shape[1] for M in Ms])
        m = prod([M.shape[2] for M in Ms])
        shape = (n, m)
        super().__init__(dtype, shape)

    def _rmatmat(self, v):
        B = v.shape[0]
        v = v.view(B, *[Mi.shape[1] for Mi in self.Ms], -1)  # (B, n1, n2, ..., nd, k1)
        for M in self.Ms:
            v = torch.einsum('bn...k,knml->b...ml', v, M)  # flop count doesn't work for tensordot
        return v.view(B, -1)

    def to_dense(self):
        raise NotImplementedError

    def __str__(self):
        return "TT"


class BTT(LinearOperator):
    """
    BTT supporting any number of cores. Slower than FasterBTT.

    Args:
        *Ms (array_like): Cores
        transpose (bool): If True, reverse the output indices to be consistent with FasterBTT implementation
    """
    def __init__(self, Ms, transpose=True):
        self.Ms = Ms  # (r_i+1, ri, m[:i], mi, ni, n[i+1:])
        dtype = self.Ms[0].dtype
        self.ms = [M.shape[2 + i] for i, M in enumerate(Ms)]
        self.ns = [M.shape[3 + i] for i, M in enumerate(Ms)]
        self.m = prod(self.ms)
        self.n = prod(self.ns)
        self.rank_out = [M.shape[0] for M in Ms]  # (r, r, ..., 1)
        self.rank_in = [M.shape[1] for M in Ms]  # (1, r, r, ..., r)
        assert self.rank_in[1:] == self.rank_out[:-1], "Rank must match"
        assert self.rank_in[0] == 1, "First rank must be 1"
        assert self.rank_out[-1] == 1, "Last rank must be 1"
        shape = (self.n, self.m)
        self.transpose = transpose
        super().__init__(dtype, shape)

    def _rmatmat(self, v):
        # v: (B, N)
        b = v.shape[0]
        for idx, M in enumerate(self.Ms):
            r = self.rank_out[idx]
            t = self.rank_in[idx]
            m = self.ms[idx]
            n = self.ns[idx]
            p = prod(self.ms[:idx])
            q = prod(self.ns[idx + 1:])
            v = v.reshape(b, t, p, n, q)
            M = M.view(r, t, p, m, n, q)
            v = torch.einsum('rtpmnq,btpnq->brpmq', M, v)
        # v: (b, 1, m1, m2, ..., m_d, 1)
        if self.transpose:
            v = v.permute(0, 1, -2, *range(v.ndim - 3, 1, -1), -1)
        return v.reshape(b, -1)

    def to_dense(self):
        raise NotImplementedError

    def __str__(self):
        return "BTT"


class FasterBTT(LinearOperator):
    """
   Args:
        *Ms (array_like): Cores of the TT-decomposition
        shapes (tuple): Shapes of the cores
        normalize (bool): If True, normalize the cores
    """
    def __init__(self, Ms, shapes, normalize=False):
        self.Ms = Ms
        self.normalize = normalize
        dtype = self.Ms[0].dtype
        self.rs, self.ms, self.ns = shapes
        shape = (prod(self.ms), prod(self.ns))
        d_in, d_out = shape
        self.max_rms = (min(d_in, d_out) * d_out / (d_out * d_in * d_in))**0.5
        super().__init__(dtype, shape)

    def _rmatmat(self, v):  # x -> x @ A
        W0, W1 = self.Ms[0], self.Ms[1]
        if self.normalize:
            # rms_W0 = torch.sqrt(torch.mean(W0 ** 2.) + 1e-8).item()
            rms_W0 = torch.sqrt(torch.mean(W0**2.) + 1e-8)
            d_in, d_out = self.rs[0] * self.ms[0], self.rs[1] * self.ns[0]
            max_rms0 = (min(d_in, d_out) * d_out / (d_out * d_in * d_in))**0.5
            # W0 = self.gamma0 * W0 / max(1, rms_W0 / max_rms0)
            if len(self.Ms) > 2:
                W0 = self.Ms[2] * W0 / max(1, rms_W0 / max_rms0)
            else:
                W0 = W0 / max(1, rms_W0 / max_rms0)

            # rms_W1 = torch.sqrt(torch.mean(W1 ** 2.) + 1e-8).item()
            rms_W1 = torch.sqrt(torch.mean(W1**2.) + 1e-8)
            d_in, d_out = self.rs[1] * self.ms[1], self.rs[2] * self.ns[1]
            max_rms1 = (min(d_in, d_out) * d_out / (d_out * d_in * d_in))**0.5
            # W1 = self.gamma1 * W1 / max(1, rms_W1 / max_rms1)
            if len(self.Ms) > 2:
                W1 = self.Ms[3] * W1 / max(1, rms_W1 / max_rms1)
            else:
                W1 = W1 / max(1, rms_W1 / max_rms1)
        return btt_mvm(v, W0, W1, (self.rs, self.ms, self.ns))

    def _rmatmat_T(self, v):  # g -> g @ A^T
        return btt_mvmt(v, self.Ms[0], self.Ms[1], (self.rs, self.ms, self.ns))

    def _rmatmat_alt(self, V):
        batch_n = V.shape[0]
        y = V.reshape(batch_n, self.ms[1], self.ms[0], -1)
        y = y.transpose(0, 1)
        y = y.reshape(self.ms[1], batch_n, self.ms[0] * self.rs[0])
        out1 = torch.bmm(y, self.Ms[0])
        out1 = out1.reshape(self.ms[1], batch_n, self.ns[0], self.rs[1])
        out1 = out1.transpose(0, 2).contiguous()
        out1 = out1.reshape(self.ns[0], batch_n, self.ms[1] * self.rs[1])
        out2 = torch.bmm(out1, self.Ms[1])
        out2 = out2.reshape(self.ns[0], batch_n, self.ns[1], self.rs[2])
        out2 = out2.transpose(0, 1)
        out2 = out2.reshape(batch_n, -1)
        return out2

    def __str__(self):
        return "BTT"

class Permutation(LinearOperator):
    def __init__(self, indicies, dtype):
        self.indicies = indicies
        shape = (len(self.indicies), len(self.indicies))
        super().__init__(dtype, shape)

    def _rmatmat(self, v):
        return v[:, self.indicies]
