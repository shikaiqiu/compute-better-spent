import torch

# Based on https://github.com/HazyResearch/fly/blob/master/src/models/layers/blockdiag_butterfly_multiply.py

class BlockTeTr(torch.autograd.Function):
    @staticmethod
    @torch.cuda.amp.custom_fwd(
        cast_inputs=torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float16)
    def forward(ctx, x, W1, W2, shapes):
        rs, ms, ns = shapes
        batch_n = x.shape[0]
        out1 = torch.empty(batch_n, ms[1], ns[0] * rs[1], device=x.device, dtype=x.dtype).transpose(0, 1)
        out2 = torch.empty(batch_n, ns[0], ns[1] * rs[2], device=x.device, dtype=x.dtype).transpose(0, 1)
        ctx.shapes = shapes

        y = x.reshape(batch_n, ms[1], ms[0], -1)
        y = y.transpose(0, 1)
        y = y.reshape(ms[1], batch_n, ms[0] * rs[0])
        torch.bmm(y, W1, out=out1)
        out1 = out1.reshape(ms[1], batch_n, ns[0], rs[1])
        out1 = out1.transpose(0, 2).contiguous()
        out1 = out1.reshape(ns[0], batch_n, ms[1] * rs[1])
        torch.bmm(out1, W2, out=out2)
        out2 = out2.reshape(ns[0], batch_n, ns[1], rs[2])
        out2 = out2.transpose(0, 1)
        out2 = out2.reshape(batch_n, -1)
        ctx.save_for_backward(x, W1, W2, out1)
        # return out2
        return out2

    @staticmethod
    @torch.cuda.amp.custom_bwd
    def backward(ctx, grad):
        x, W1, W2, out1 = ctx.saved_tensors
        B = x.shape[0]
        rs, ms, ns = ctx.shapes
        grad_re = grad.reshape(B, ns[0], ns[1]).transpose(1, 0)
        aux = torch.empty(B, ns[0], rs[1] * ms[1], device=x.device, dtype=x.dtype).transpose(1, 0)
        dx = torch.empty(B, ms[1], ms[0], device=x.device, dtype=x.dtype).transpose(1, 0)

        torch.bmm(grad_re, W2.transpose(1, 2), out=aux)
        aux = aux.reshape(ns[0], B, ms[1], rs[1]).transpose(0, 2).contiguous()
        aux = aux.reshape(ms[1], B, ns[0] * rs[1])
        torch.bmm(aux, W1.transpose(1, 2), out=dx)
        dx = dx.reshape(ms[1], B, ms[0] * rs[0])
        dx = dx.transpose(1, 0).reshape(B, -1)

        x_res = x.reshape(B, ms[1], ms[0]).transpose(0, 1)
        dW1 = torch.bmm(aux.transpose(1, 2), x_res).transpose(1, 2)

        dW2 = torch.bmm(grad_re.transpose(1, 2), out1).transpose(1, 2)
        return dx, dW1, dW2, None

    @staticmethod
    @torch.cuda.amp.custom_fwd(
        cast_inputs=torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float16)
    def _pull_back(grad, W1, W2, shapes):
        B = grad.shape[0]
        rs, ms, ns = shapes
        grad_re = grad.reshape(B, ns[0], ns[1]).transpose(1, 0)
        aux = torch.empty(B, ns[0], rs[1] * ms[1], device=grad.device, dtype=grad.dtype).transpose(1, 0)
        dx = torch.empty(B, ms[1], ms[0], device=grad.device, dtype=grad.dtype).transpose(1, 0)

        torch.bmm(grad_re, W2.transpose(1, 2), out=aux)
        aux = aux.reshape(ns[0], B, ms[1], rs[1]).transpose(0, 2).contiguous()
        aux = aux.reshape(ms[1], B, ns[0] * rs[1])
        torch.bmm(aux, W1.transpose(1, 2), out=dx)
        dx = dx.reshape(ms[1], B, ms[0] * rs[0])
        dx = dx.transpose(1, 0).reshape(B, -1)
        return dx


btt_mvm = BlockTeTr.apply
btt_mvmt = BlockTeTr._pull_back
