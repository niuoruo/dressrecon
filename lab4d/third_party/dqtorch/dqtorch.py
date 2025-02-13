import torch
from torch.autograd import Function
from torch.autograd.function import once_differentiable
from torch.amp import custom_fwd, custom_bwd

import _dqtorch_cuda, _dqtorch_cpu

from typing import Tuple
DualQuaternions = Tuple[torch.Tensor, torch.Tensor]
QuaternionTranslation = Tuple[torch.Tensor, torch.Tensor]


# ===== Standardize Quaternion

class _Standardize_quaternion_bw(Function):
    @staticmethod
    @custom_fwd(cast_inputs=torch.half, device_type="cuda")
    def forward(ctx, out_grad: torch.Tensor, a: torch.Tensor):
        # out_grad: (..., 4), float
        # a: (..., 4), float
        # a_grad: (..., 4), float
        out_grad = out_grad.contiguous()
        a = a.contiguous()

        out_grad_shape = out_grad.shape
        a_shape = a.shape
        assert out_grad_shape[-1] == 4, out_grad_shape
        assert a_shape[-1] == 4, a_shape
        assert out_grad_shape[:-1] == a_shape[:-1], (out_grad_shape, a_shape)
        assert a.is_floating_point() and out_grad.dtype == a.dtype, (out_grad.dtype, a.dtype)
        assert out_grad.device == a.device, (out_grad.device, a.device)
        out_grad = out_grad.view(-1, out_grad_shape[-1])
        a = a.view(-1, a_shape[-1])
        
        B = out_grad.shape[0]
        alloc = torch.zeros(B * 4, dtype=a.dtype, device=a.device)
        a_grad = alloc[B * 0 : B * 4].view(B, 4)

        (_dqtorch_cuda if a.is_cuda else _dqtorch_cpu).standardize_quaternion_bw(out_grad, a, a_grad, B)
        a_grad = a_grad.view(a_shape)
        return a_grad

    @staticmethod
    @custom_bwd(device_type="cuda")
    def backward(*args):
        raise NotImplementedError

class _Standardize_quaternion_fw(Function):
    @staticmethod
    @custom_fwd(cast_inputs=torch.half, device_type="cuda")
    def forward(ctx, a: torch.Tensor):
        # a: (..., 4), float
        # out: (..., 4), float
        a = a.contiguous()

        a_shape = a.shape
        assert a_shape[-1] == 4, a_shape
        assert a.is_floating_point(), a.dtype
        a = a.view(-1, a_shape[-1])

        B = a.shape[0]
        alloc = torch.zeros(B * 4, dtype=a.dtype, device=a.device)
        out = alloc[B * 0 : B * 4].view(B, 4)

        (_dqtorch_cuda if a.is_cuda else _dqtorch_cpu).standardize_quaternion_fw(a, out, B)
        a = a.view(a_shape)
        out = out.view(a_shape[:-1] + (4,))
        ctx.save_for_backward(a)
        return out

    @staticmethod
    @custom_bwd(device_type="cuda")
    def backward(ctx, out_grad):
        a, = ctx.saved_tensors
        return _Standardize_quaternion_bw.apply(out_grad, a)

standardize_quaternion = _Standardize_quaternion_fw.apply


# ===== Quaternion Raw Multiply

class _Quaternion_raw_multiply_bw(Function):
    @staticmethod
    @custom_fwd(cast_inputs=torch.half, device_type="cuda")
    def forward(ctx, out_grad: torch.Tensor, a: torch.Tensor, b: torch.Tensor):
        # out_grad: (..., 4), float
        # a: (..., 3) or (..., 4), float
        # b: (..., 3) or (..., 4), float
        # a_grad: (..., 3) or (..., 4), float
        # b_grad: (..., 3) or (..., 4), float
        out_grad = out_grad.contiguous()
        a = a.contiguous()
        b = b.contiguous()

        out_grad_shape = out_grad.shape
        a_shape = a.shape
        b_shape = b.shape
        assert out_grad_shape[-1] == 4, out_grad_shape
        assert a_shape[-1] == 3 or a_shape[-1] == 4, a_shape
        assert b_shape[-1] == 3 or b_shape[-1] == 4, b_shape
        assert out_grad_shape[:-1] == a_shape[:-1] == b_shape[:-1], (out_grad_shape, a_shape, b_shape)
        assert a.is_floating_point() and out_grad.dtype == a.dtype == b.dtype, (out_grad.dtype, a.dtype, b.dtype)
        assert out_grad.device == a.device == b.device, (out_grad.device, a.device, b.device)
        out_grad = out_grad.view(-1, out_grad_shape[-1])
        a = a.view(-1, a_shape[-1])
        b = b.view(-1, b_shape[-1])
        
        B = out_grad.shape[0]
        Da = a.shape[1]
        Db = b.shape[1]
        alloc = torch.zeros(B * (Da + Db), dtype=a.dtype, device=a.device)
        a_grad = alloc[B * 0 : B * Da].view(B, Da)
        b_grad = alloc[B * Da : B * (Da + Db)].view(B, Db)

        (_dqtorch_cuda if a.is_cuda else _dqtorch_cpu).quaternion_raw_multiply_bw(out_grad, a, b, a_grad, b_grad, B, Da, Db)
        a_grad = a_grad.view(a_shape)
        b_grad = b_grad.view(b_shape)
        return a_grad, b_grad

    @staticmethod
    @custom_bwd(device_type="cuda")
    def backward(*args):
        raise NotImplementedError

class _Quaternion_raw_multiply_fw(Function):
    @staticmethod
    @custom_fwd(cast_inputs=torch.half, device_type="cuda")
    def forward(ctx, a: torch.Tensor, b: torch.Tensor):
        # a: (..., 3) or (..., 4), float
        # b: (..., 3) or (..., 4), float
        # out: (..., 4), float
        a = a.contiguous()
        b = b.contiguous()

        a_shape = a.shape
        b_shape = b.shape
        assert a_shape[-1] == 3 or a_shape[-1] == 4, a_shape
        assert b_shape[-1] == 3 or b_shape[-1] == 4, b_shape
        assert a_shape[:-1] == b_shape[:-1], (a_shape, b_shape)
        assert a.is_floating_point() and a.dtype == b.dtype, (a.dtype, b.dtype)
        assert a.device == b.device, (a.device, b.device)
        a = a.view(-1, a_shape[-1])
        b = b.view(-1, b_shape[-1])

        B = a.shape[0]
        Da = a.shape[1]
        Db = b.shape[1]
        alloc = torch.zeros(B * 4, dtype=a.dtype, device=a.device)
        out = alloc[B * 0 : B * 4].view(B, 4)

        (_dqtorch_cuda if a.is_cuda else _dqtorch_cpu).quaternion_raw_multiply_fw(a, b, out, B, Da, Db)
        a = a.view(a_shape)
        b = b.view(b_shape)
        out = out.view(a_shape[:-1] + (4,))
        ctx.save_for_backward(a, b)
        return out

    @staticmethod
    @custom_bwd(device_type="cuda")
    def backward(ctx, out_grad):
        a, b = ctx.saved_tensors
        return _Quaternion_raw_multiply_bw.apply(out_grad, a, b)

quaternion_raw_multiply = _Quaternion_raw_multiply_fw.apply


# ===== Quaternion Multiply

class _Quaternion_multiply_bw(Function):
    @staticmethod
    @custom_fwd(cast_inputs=torch.half, device_type="cuda")
    def forward(ctx, out_grad: torch.Tensor, a: torch.Tensor, b: torch.Tensor):
        # out_grad: (..., 4), float
        # a: (..., 3) or (..., 4), float
        # b: (..., 3) or (..., 4), float
        # a_grad: (..., 3) or (..., 4), float
        # b_grad: (..., 3) or (..., 4), float
        out_grad = out_grad.contiguous()
        a = a.contiguous()
        b = b.contiguous()

        out_grad_shape = out_grad.shape
        a_shape = a.shape
        b_shape = b.shape
        assert out_grad_shape[-1] == 4, out_grad_shape
        assert a_shape[-1] == 3 or a_shape[-1] == 4, a_shape
        assert b_shape[-1] == 3 or b_shape[-1] == 4, b_shape
        assert out_grad_shape[:-1] == a_shape[:-1] == b_shape[:-1], (out_grad_shape, a_shape, b_shape)
        assert a.is_floating_point() and out_grad.dtype == a.dtype == b.dtype, (out_grad.dtype, a.dtype, b.dtype)
        assert out_grad.device == a.device == b.device, (out_grad.device, a.device, b.device)
        out_grad = out_grad.view(-1, out_grad_shape[-1])
        a = a.view(-1, a_shape[-1])
        b = b.view(-1, b_shape[-1])
        
        B = a.shape[0]
        Da = a.shape[1]
        Db = b.shape[1]
        alloc = torch.zeros(B * (Da + Db), dtype=a.dtype, device=a.device)
        a_grad = alloc[B * 0 : B * Da].view(B, Da)
        b_grad = alloc[B * Da : B * (Da + Db)].view(B, Db)

        (_dqtorch_cuda if a.is_cuda else _dqtorch_cpu).quaternion_multiply_bw(out_grad, a, b, a_grad, b_grad, B, Da, Db)
        a_grad = a_grad.view(a_shape)
        b_grad = b_grad.view(b_shape)
        return a_grad, b_grad

    @staticmethod
    @custom_bwd(device_type="cuda")
    def backward(*args):
        raise NotImplementedError

class _Quaternion_multiply_fw(Function):
    @staticmethod
    @custom_fwd(cast_inputs=torch.half, device_type="cuda")
    def forward(ctx, a: torch.Tensor, b: torch.Tensor):
        # a: (..., 3) or (..., 4), float
        # b: (..., 3) or (..., 4), float
        # out: (..., 4), float
        a = a.contiguous()
        b = b.contiguous()

        a_shape = a.shape
        b_shape = b.shape
        assert a_shape[-1] == 3 or a_shape[-1] == 4, a_shape
        assert b_shape[-1] == 3 or b_shape[-1] == 4, b_shape
        assert a_shape[:-1] == b_shape[:-1], (a_shape, b_shape)
        assert a.is_floating_point() and a.dtype == b.dtype, (a.dtype, b.dtype)
        assert a.device == b.device, (a.device, b.device)
        a = a.view(-1, a_shape[-1])
        b = b.view(-1, b_shape[-1])

        B = a.shape[0]
        Da = a.shape[1]
        Db = b.shape[1]
        alloc = torch.zeros(B * 4, dtype=a.dtype, device=a.device)
        out = alloc[B * 0 : B * 4].view(B, 4)

        (_dqtorch_cuda if a.is_cuda else _dqtorch_cpu).quaternion_multiply_fw(a, b, out, B, Da, Db)
        a = a.view(a_shape)
        b = b.view(b_shape)
        out = out.view(a_shape[:-1] + (4,))
        ctx.save_for_backward(a, b)
        return out

    @staticmethod
    @custom_bwd(device_type="cuda")
    def backward(ctx, out_grad):
        a, b = ctx.saved_tensors
        return _Quaternion_multiply_bw.apply(out_grad, a, b)

quaternion_multiply = _Quaternion_multiply_fw.apply


# ===== Quaternion Conjugate

class _Quaternion_conjugate_bw(Function):
    @staticmethod
    @custom_fwd(cast_inputs=torch.half, device_type="cuda")
    def forward(ctx, out_grad: torch.Tensor, a: torch.Tensor):
        # out_grad: (..., 4), float
        # a: (..., 4), float
        # a_grad: (..., 4), float
        out_grad = out_grad.contiguous()
        a = a.contiguous()

        out_grad_shape = out_grad.shape
        a_shape = a.shape
        assert out_grad_shape[-1] == 4, out_grad_shape
        assert a_shape[-1] == 4, a_shape
        assert out_grad_shape[:-1] == a_shape[:-1], (out_grad_shape, a_shape)
        assert a.is_floating_point() and out_grad.dtype == a.dtype, (out_grad.dtype, a.dtype)
        assert out_grad.device == a.device, (out_grad.device, a.device)
        out_grad = out_grad.view(-1, out_grad_shape[-1])
        a = a.view(-1, a_shape[-1])
        
        B = out_grad.shape[0]
        alloc = torch.zeros(B * 4, dtype=a.dtype, device=a.device)
        a_grad = alloc[B * 0 : B * 4].view(B, 4)

        (_dqtorch_cuda if a.is_cuda else _dqtorch_cpu).quaternion_conjugate_bw(out_grad, a, a_grad, B)
        a_grad = a_grad.view(a_shape)
        return a_grad

    @staticmethod
    @custom_bwd(device_type="cuda")
    def backward(*args):
        raise NotImplementedError

class _Quaternion_conjugate_fw(Function):
    @staticmethod
    @custom_fwd(cast_inputs=torch.half, device_type="cuda")
    def forward(ctx, a: torch.Tensor):
        # a: (..., 4), float
        # out: (..., 4), float
        a = a.contiguous()

        a_shape = a.shape
        assert a_shape[-1] == 4, a_shape
        assert a.is_floating_point(), a.dtype
        a = a.view(-1, a_shape[-1])

        B = a.shape[0]
        alloc = torch.zeros(B * 4, dtype=a.dtype, device=a.device)
        out = alloc[B * 0 : B * 4].view(B, 4)

        (_dqtorch_cuda if a.is_cuda else _dqtorch_cpu).quaternion_conjugate_fw(a, out, B)
        a = a.view(a_shape)
        out = out.view(a_shape[:-1] + (4,))
        ctx.save_for_backward(a)
        return out

    @staticmethod
    @custom_bwd(device_type="cuda")
    def backward(ctx, out_grad):
        a, = ctx.saved_tensors
        return _Quaternion_conjugate_bw.apply(out_grad, a)

quaternion_conjugate = _Quaternion_conjugate_fw.apply


# ===== Quaternion Apply

class _Quaternion_apply_bw(Function):
    @staticmethod
    @custom_fwd(cast_inputs=torch.half, device_type="cuda")
    def forward(ctx, out_grad: torch.Tensor, a: torch.Tensor, b: torch.Tensor):
        # out_grad: (..., 3), float
        # a: (..., 4), float
        # b: (..., 3), float
        # a_grad: (..., 4), float
        # b_grad: (..., 3), float
        out_grad = out_grad.contiguous()
        a = a.contiguous()
        b = b.contiguous()

        out_grad_shape = out_grad.shape
        a_shape = a.shape
        b_shape = b.shape
        assert out_grad_shape[-1] == 3, out_grad_shape
        assert a_shape[-1] == 4, a_shape
        assert b_shape[-1] == 3, b_shape
        assert out_grad_shape[:-1] == a_shape[:-1] == b_shape[:-1], (out_grad_shape, a_shape, b_shape)
        assert a.is_floating_point() and out_grad.dtype == a.dtype == b.dtype, (out_grad.dtype, a.dtype, b.dtype)
        assert out_grad.device == a.device == b.device, (out_grad.device, a.device, b.device)
        out_grad = out_grad.view(-1, out_grad_shape[-1])
        a = a.view(-1, a_shape[-1])
        b = b.view(-1, b_shape[-1])
        
        B = a.shape[0]
        alloc = torch.zeros(B * 7, dtype=a.dtype, device=a.device)
        a_grad = alloc[B * 0 : B * 4].view(B, 4)
        b_grad = alloc[B * 4 : B * 7].view(B, 3)

        (_dqtorch_cuda if a.is_cuda else _dqtorch_cpu).quaternion_apply_bw(out_grad, a, b, a_grad, b_grad, B)
        a_grad = a_grad.view(a_shape)
        b_grad = b_grad.view(b_shape)
        return a_grad, b_grad

    @staticmethod
    @custom_bwd(device_type="cuda")
    def backward(*args):
        raise NotImplementedError

class _Quaternion_apply_fw(Function):
    @staticmethod
    @custom_fwd(cast_inputs=torch.half, device_type="cuda")
    def forward(ctx, a: torch.Tensor, b: torch.Tensor):
        # a: (..., 4), float
        # b: (..., 3), float
        # out: (..., 3), float
        a = a.contiguous()
        b = b.contiguous()

        a_shape = a.shape
        b_shape = b.shape
        assert a_shape[-1] == 4, a_shape
        assert b_shape[-1] == 3, b_shape
        assert a_shape[:-1] == b_shape[:-1], (a_shape, b_shape)
        assert a.is_floating_point() and a.dtype == b.dtype, (a.dtype, b.dtype)
        assert a.device == b.device, (a.device, b.device)
        a = a.view(-1, a_shape[-1])
        b = b.view(-1, b_shape[-1])

        B = a.shape[0]
        alloc = torch.zeros(B * 3, dtype=a.dtype, device=a.device)
        out = alloc[B * 0 : B * 3].view(B, 3)

        (_dqtorch_cuda if a.is_cuda else _dqtorch_cpu).quaternion_apply_fw(a, b, out, B)
        a = a.view(a_shape)
        b = b.view(b_shape)
        out = out.view(a_shape[:-1] + (3,))
        ctx.save_for_backward(a, b)
        return out

    @staticmethod
    @custom_bwd(device_type="cuda")
    def backward(ctx, out_grad):
        a, b = ctx.saved_tensors
        return _Quaternion_apply_bw.apply(out_grad, a, b)

quaternion_apply = _Quaternion_apply_fw.apply


# ===== Quaternion to Matrix

class _Quaternion_to_matrix_bw(Function):
    @staticmethod
    @custom_fwd(cast_inputs=torch.half, device_type="cuda")
    def forward(ctx, out_grad: torch.Tensor, a: torch.Tensor):
        # out_grad: (..., 3, 3), float
        # a: (..., 4), float
        # a_grad: (..., 4), float
        out_grad = out_grad.contiguous()
        a = a.contiguous()

        out_grad_shape = out_grad.shape
        a_shape = a.shape
        assert out_grad_shape[-1] == 3 and out_grad_shape[-2] == 3, out_grad_shape
        assert a_shape[-1] == 4, a_shape
        assert out_grad_shape[:-2] == a_shape[:-1], (out_grad_shape, a_shape)
        assert a.is_floating_point() and out_grad.dtype == a.dtype, (out_grad.dtype, a.dtype)
        assert out_grad.device == a.device, (out_grad.device, a.device)
        out_grad = out_grad.view(-1, out_grad_shape[-2] * out_grad_shape[-1])
        a = a.view(-1, a_shape[-1])
        
        B = out_grad.shape[0]
        alloc = torch.zeros(B * 4, dtype=a.dtype, device=a.device)
        a_grad = alloc[B * 0 : B * 4].view(B, 4)

        (_dqtorch_cuda if a.is_cuda else _dqtorch_cpu).quaternion_to_matrix_bw(out_grad, a, a_grad, B)
        a_grad = a_grad.view(a_shape)
        return a_grad

    @staticmethod
    @custom_bwd(device_type="cuda")
    def backward(*args):
        raise NotImplementedError

class _Quaternion_to_matrix_fw(Function):
    @staticmethod
    @custom_fwd(cast_inputs=torch.half, device_type="cuda")
    def forward(ctx, a: torch.Tensor):
        # a: (..., 4), float
        # out: (..., 3, 3), float
        a = a.contiguous()

        a_shape = a.shape
        assert a_shape[-1] == 4, a_shape
        assert a.is_floating_point(), a.dtype
        a = a.view(-1, a_shape[-1])

        B = a.shape[0]
        alloc = torch.zeros(B * 9, dtype=a.dtype, device=a.device)
        out = alloc[B * 0 : B * 9].view(B, 9)

        (_dqtorch_cuda if a.is_cuda else _dqtorch_cpu).quaternion_to_matrix_fw(a, out, B)
        a = a.view(a_shape)
        out = out.view(a_shape[:-1] + (3, 3))
        ctx.save_for_backward(a)
        return out

    @staticmethod
    @custom_bwd(device_type="cuda")
    def backward(ctx, out_grad):
        a, = ctx.saved_tensors
        return _Quaternion_to_matrix_bw.apply(out_grad, a)

quaternion_to_matrix = _Quaternion_to_matrix_fw.apply


# ===== Matrix to Raw Quaternion

class _Matrix_to_raw_quaternion_bw(Function):
    @staticmethod
    @custom_fwd(cast_inputs=torch.half, device_type="cuda")
    def forward(ctx, out_grad: torch.Tensor, a: torch.Tensor):
        # out_grad: (..., 4), float
        # a: (..., 3, 3), float
        # a_grad: (..., 3, 3), float
        out_grad = out_grad.contiguous()
        a = a.contiguous()

        out_grad_shape = out_grad.shape
        a_shape = a.shape
        assert out_grad_shape[-1] == 4, out_grad_shape
        assert a_shape[-2] == 3 and a_shape[-1] == 3, a_shape
        assert out_grad_shape[:-1] == a_shape[:-2], (out_grad_shape, a_shape)
        assert a.is_floating_point() and out_grad.dtype == a.dtype, (out_grad.dtype, a.dtype)
        assert out_grad.device == a.device, (out_grad.device, a.device)
        out_grad = out_grad.view(-1, out_grad_shape[-1])
        a = a.view(-1, a_shape[-2] * a_shape[-1])
        
        B = out_grad.shape[0]
        alloc = torch.zeros(B * 9, dtype=a.dtype, device=a.device)
        a_grad = alloc[B * 0 : B * 9].view(B, 9)

        (_dqtorch_cuda if a.is_cuda else _dqtorch_cpu).matrix_to_raw_quaternion_bw(out_grad, a, a_grad, B)
        a_grad = a_grad.view(a_shape)
        return a_grad

    @staticmethod
    @custom_bwd(device_type="cuda")
    def backward(*args):
        raise NotImplementedError

class _Matrix_to_raw_quaternion_fw(Function):
    @staticmethod
    @custom_fwd(cast_inputs=torch.half, device_type="cuda")
    def forward(ctx, a: torch.Tensor):
        # a: (..., 3, 3), float
        # out: (..., 4), float
        a = a.contiguous()

        a_shape = a.shape
        assert a_shape[-2] == 3 and a_shape[-1] == 3, a_shape
        assert a.is_floating_point(), a.dtype
        a = a.view(-1, a_shape[-2] * a_shape[-1])

        B = a.shape[0]
        alloc = torch.zeros(B * 4, dtype=a.dtype, device=a.device)
        out = alloc[B * 0 : B * 4].view(B, 4)

        (_dqtorch_cuda if a.is_cuda else _dqtorch_cpu).matrix_to_raw_quaternion_fw(a, out, B)
        a = a.view(a_shape)
        out = out.view(a_shape[:-2] + (4,))
        ctx.save_for_backward(a)
        return out

    @staticmethod
    @custom_bwd(device_type="cuda")
    def backward(ctx, out_grad):
        a, = ctx.saved_tensors
        return _Matrix_to_raw_quaternion_bw.apply(out_grad, a)

matrix_to_raw_quaternion = _Matrix_to_raw_quaternion_fw.apply


# ===== Matrix to Quaternion

class _Matrix_to_quaternion_bw(Function):
    @staticmethod
    @custom_fwd(cast_inputs=torch.half, device_type="cuda")
    def forward(ctx, out_grad: torch.Tensor, a: torch.Tensor):
        # out_grad: (..., 4), float
        # a: (..., 3, 3), float
        # a_grad: (..., 3, 3), float
        out_grad = out_grad.contiguous()
        a = a.contiguous()

        out_grad_shape = out_grad.shape
        a_shape = a.shape
        assert out_grad_shape[-1] == 4, out_grad_shape
        assert a_shape[-2] == 3 and a_shape[-1] == 3, a_shape
        assert out_grad_shape[:-1] == a_shape[:-2], (out_grad_shape, a_shape)
        assert a.is_floating_point() and out_grad.dtype == a.dtype, (out_grad.dtype, a.dtype)
        assert out_grad.device == a.device, (out_grad.device, a.device)
        out_grad = out_grad.view(-1, out_grad_shape[-1])
        a = a.view(-1, a_shape[-2] * a_shape[-1])
        
        B = out_grad.shape[0]
        alloc = torch.zeros(B * 9, dtype=a.dtype, device=a.device)
        a_grad = alloc[B * 0 : B * 9].view(B, 9)

        (_dqtorch_cuda if a.is_cuda else _dqtorch_cpu).matrix_to_quaternion_bw(out_grad, a, a_grad, B)
        a_grad = a_grad.view(a_shape)
        return a_grad

    @staticmethod
    @custom_bwd(device_type="cuda")
    def backward(*args):
        raise NotImplementedError

class _Matrix_to_quaternion_fw(Function):
    @staticmethod
    @custom_fwd(cast_inputs=torch.half, device_type="cuda")
    def forward(ctx, a: torch.Tensor):
        # a: (..., 3, 3), float
        # out: (..., 4), float
        a = a.contiguous()

        a_shape = a.shape
        assert a_shape[-2] == 3 and a_shape[-1] == 3, a_shape
        assert a.is_floating_point(), a.dtype
        a = a.view(-1, a_shape[-2] * a_shape[-1])

        B = a.shape[0]
        alloc = torch.zeros(B * 4, dtype=a.dtype, device=a.device)
        out = alloc[B * 0 : B * 4].view(B, 4)

        (_dqtorch_cuda if a.is_cuda else _dqtorch_cpu).matrix_to_quaternion_fw(a, out, B)
        a = a.view(a_shape)
        out = out.view(a_shape[:-2] + (4,))
        ctx.save_for_backward(a)
        return out

    @staticmethod
    @custom_bwd(device_type="cuda")
    def backward(ctx, out_grad):
        a, = ctx.saved_tensors
        return _Matrix_to_quaternion_bw.apply(out_grad, a)

matrix_to_quaternion = _Matrix_to_quaternion_fw.apply


# ===== Axis Angle to Quaternion

class _Axis_angle_to_quaternion_bw(Function):
    @staticmethod
    @custom_fwd(cast_inputs=torch.half, device_type="cuda")
    def forward(ctx, out_grad: torch.Tensor, a: torch.Tensor):
        # out_grad: (..., 4), float
        # a: (..., 3), float
        # a_grad: (..., 3), float
        out_grad = out_grad.contiguous()
        a = a.contiguous()

        out_grad_shape = out_grad.shape
        a_shape = a.shape
        assert out_grad_shape[-1] == 4, out_grad_shape
        assert a_shape[-1] == 3, a_shape
        assert out_grad_shape[:-1] == a_shape[:-1], (out_grad_shape, a_shape)
        assert a.is_floating_point() and out_grad.dtype == a.dtype, (out_grad.dtype, a.dtype)
        assert out_grad.device == a.device, (out_grad.device, a.device)
        out_grad = out_grad.view(-1, out_grad_shape[-1])
        a = a.view(-1, a_shape[-1])
        
        B = out_grad.shape[0]
        alloc = torch.zeros(B * 3, dtype=a.dtype, device=a.device)
        a_grad = alloc[B * 0 : B * 3].view(B, 3)

        (_dqtorch_cuda if a.is_cuda else _dqtorch_cpu).axis_angle_to_quaternion_bw(out_grad, a, a_grad, B)
        a_grad = a_grad.view(a_shape)
        return a_grad

    @staticmethod
    @custom_bwd(device_type="cuda")
    def backward(*args):
        raise NotImplementedError

class _Axis_angle_to_quaternion_fw(Function):
    @staticmethod
    @custom_fwd(cast_inputs=torch.half, device_type="cuda")
    def forward(ctx, a: torch.Tensor):
        # a: (..., 3), float
        # out: (..., 4), float
        a = a.contiguous()

        a_shape = a.shape
        assert a_shape[-1] == 3, a_shape
        assert a.is_floating_point(), a.dtype
        a = a.view(-1, a_shape[-1])

        B = a.shape[0]
        alloc = torch.zeros(B * 4, dtype=a.dtype, device=a.device)
        out = alloc[B * 0 : B * 4].view(B, 4)

        (_dqtorch_cuda if a.is_cuda else _dqtorch_cpu).axis_angle_to_quaternion_fw(a, out, B)
        a = a.view(a_shape)
        out = out.view(a_shape[:-1] + (4,))
        ctx.save_for_backward(a)
        return out

    @staticmethod
    @custom_bwd(device_type="cuda")
    def backward(ctx, out_grad):
        a, = ctx.saved_tensors
        return _Axis_angle_to_quaternion_bw.apply(out_grad, a)

axis_angle_to_quaternion = _Axis_angle_to_quaternion_fw.apply


# ===== Quaternion to Axis Angle

class _Quaternion_to_axis_angle_bw(Function):
    @staticmethod
    @custom_fwd(cast_inputs=torch.half, device_type="cuda")
    def forward(ctx, out_grad: torch.Tensor, a: torch.Tensor):
        # out_grad: (..., 3), float
        # a: (..., 4), float
        # a_grad: (..., 4), float
        out_grad = out_grad.contiguous()
        a = a.contiguous()

        out_grad_shape = out_grad.shape
        a_shape = a.shape
        assert out_grad_shape[-1] == 3, out_grad_shape
        assert a_shape[-1] == 4, a_shape
        assert out_grad_shape[:-1] == a_shape[:-1], (out_grad_shape, a_shape)
        assert a.is_floating_point() and out_grad.dtype == a.dtype, (out_grad.dtype, a.dtype)
        assert out_grad.device == a.device, (out_grad.device, a.device)
        out_grad = out_grad.view(-1, out_grad_shape[-1])
        a = a.view(-1, a_shape[-1])
        
        B = out_grad.shape[0]
        alloc = torch.zeros(B * 4, dtype=a.dtype, device=a.device)
        a_grad = alloc[B * 0 : B * 4].view(B, 4)

        (_dqtorch_cuda if a.is_cuda else _dqtorch_cpu).quaternion_to_axis_angle_bw(out_grad, a, a_grad, B)
        a_grad = a_grad.view(a_shape)
        return a_grad

    @staticmethod
    @custom_bwd(device_type="cuda")
    def backward(*args):
        raise NotImplementedError

class _Quaternion_to_axis_angle_fw(Function):
    @staticmethod
    @custom_fwd(cast_inputs=torch.half, device_type="cuda")
    def forward(ctx, a: torch.Tensor):
        # a: (..., 4), float
        # out: (..., 3), float
        a = a.contiguous()

        a_shape = a.shape
        assert a_shape[-1] == 4, a_shape
        assert a.is_floating_point(), a.dtype
        a = a.view(-1, a_shape[-1])

        B = a.shape[0]
        alloc = torch.zeros(B * 3, dtype=a.dtype, device=a.device)
        out = alloc[B * 0 : B * 3].view(B, 3)

        (_dqtorch_cuda if a.is_cuda else _dqtorch_cpu).quaternion_to_axis_angle_fw(a, out, B)
        a = a.view(a_shape)
        out = out.view(a_shape[:-1] + (3,))
        ctx.save_for_backward(a)
        return out

    @staticmethod
    @custom_bwd(device_type="cuda")
    def backward(ctx, out_grad):
        a, = ctx.saved_tensors
        return _Quaternion_to_axis_angle_bw.apply(out_grad, a)

quaternion_to_axis_angle = _Quaternion_to_axis_angle_fw.apply


# ===== Axis Angle to Matrix

class _Axis_angle_to_matrix_bw(Function):
    @staticmethod
    @custom_fwd(cast_inputs=torch.half, device_type="cuda")
    def forward(ctx, out_grad: torch.Tensor, a: torch.Tensor):
        # out_grad: (..., 3, 3), float
        # a: (..., 3), float
        # a_grad: (..., 3), float
        out_grad = out_grad.contiguous()
        a = a.contiguous()

        out_grad_shape = out_grad.shape
        a_shape = a.shape
        assert out_grad_shape[-2] == 3 and out_grad_shape[-1] == 3, out_grad_shape
        assert a_shape[-1] == 3, a_shape
        assert out_grad_shape[:-2] == a_shape[:-1], (out_grad_shape, a_shape)
        assert a.is_floating_point() and out_grad.dtype == a.dtype, (out_grad.dtype, a.dtype)
        assert out_grad.device == a.device, (out_grad.device, a.device)
        out_grad = out_grad.view(-1, out_grad_shape[-2] * out_grad_shape[-1])
        a = a.view(-1, a_shape[-1])
        
        B = out_grad.shape[0]
        alloc = torch.zeros(B * 3, dtype=a.dtype, device=a.device)
        a_grad = alloc[B * 0 : B * 3].view(B, 3)

        (_dqtorch_cuda if a.is_cuda else _dqtorch_cpu).axis_angle_to_matrix_bw(out_grad, a, a_grad, B)
        a_grad = a_grad.view(a_shape)
        return a_grad

    @staticmethod
    @custom_bwd(device_type="cuda")
    def backward(*args):
        raise NotImplementedError

class _Axis_angle_to_matrix_fw(Function):
    @staticmethod
    @custom_fwd(cast_inputs=torch.half, device_type="cuda")
    def forward(ctx, a: torch.Tensor):
        # a: (..., 3), float
        # out: (..., 3, 3), float
        a = a.contiguous()

        a_shape = a.shape
        assert a_shape[-1] == 3, a_shape
        assert a.is_floating_point(), a.dtype
        a = a.view(-1, a_shape[-1])

        B = a.shape[0]
        alloc = torch.zeros(B * 9, dtype=a.dtype, device=a.device)
        out = alloc[B * 0 : B * 9].view(B, 9)

        (_dqtorch_cuda if a.is_cuda else _dqtorch_cpu).axis_angle_to_matrix_fw(a, out, B)
        a = a.view(a_shape)
        out = out.view(a_shape[:-1] + (3, 3))
        ctx.save_for_backward(a)
        return out

    @staticmethod
    @custom_bwd(device_type="cuda")
    def backward(ctx, out_grad):
        a, = ctx.saved_tensors
        return _Axis_angle_to_matrix_bw.apply(out_grad, a)

axis_angle_to_matrix = _Axis_angle_to_matrix_fw.apply


# ===== Matrix to Axis Angle

class _Matrix_to_axis_angle_bw(Function):
    @staticmethod
    @custom_fwd(cast_inputs=torch.half, device_type="cuda")
    def forward(ctx, out_grad: torch.Tensor, a: torch.Tensor):
        # out_grad: (..., 3), float
        # a: (..., 3, 3), float
        # a_grad: (..., 3, 3), float
        out_grad = out_grad.contiguous()
        a = a.contiguous()

        out_grad_shape = out_grad.shape
        a_shape = a.shape
        assert out_grad_shape[-1] == 3, out_grad_shape
        assert a_shape[-2] == 3 and a_shape[-1] == 3, a_shape
        assert out_grad_shape[:-1] == a_shape[:-2], (out_grad_shape, a_shape)
        assert a.is_floating_point() and out_grad.dtype == a.dtype, (out_grad.dtype, a.dtype)
        assert out_grad.device == a.device, (out_grad.device, a.device)
        out_grad = out_grad.view(-1, out_grad_shape[-1])
        a = a.view(-1, a_shape[-2] * a_shape[-1])
        
        B = out_grad.shape[0]
        alloc = torch.zeros(B * 9, dtype=a.dtype, device=a.device)
        a_grad = alloc[B * 0 : B * 9].view(B, 3, 3)

        (_dqtorch_cuda if a.is_cuda else _dqtorch_cpu).matrix_to_axis_angle_bw(out_grad, a, a_grad, B)
        a_grad = a_grad.view(a_shape)
        return a_grad

    @staticmethod
    @custom_bwd(device_type="cuda")
    def backward(*args):
        raise NotImplementedError

class _Matrix_to_axis_angle_fw(Function):
    @staticmethod
    @custom_fwd(cast_inputs=torch.half, device_type="cuda")
    def forward(ctx, a: torch.Tensor):
        # a: (..., 3, 3), float
        # out: (..., 3), float
        a = a.contiguous()

        a_shape = a.shape
        assert a_shape[-2] == 3 and a_shape[-1] == 3, a_shape
        assert a.is_floating_point(), a.dtype
        a = a.view(-1, a_shape[-2] * a_shape[-1])

        B = a.shape[0]
        alloc = torch.zeros(B * 3, dtype=a.dtype, device=a.device)
        out = alloc[B * 0 : B * 3].view(B, 3)

        (_dqtorch_cuda if a.is_cuda else _dqtorch_cpu).matrix_to_axis_angle_fw(a, out, B)
        a = a.view(a_shape)
        out = out.view(a_shape[:-2] + (3,))
        ctx.save_for_backward(a)
        return out

    @staticmethod
    @custom_bwd(device_type="cuda")
    def backward(ctx, out_grad):
        a, = ctx.saved_tensors
        return _Matrix_to_axis_angle_bw.apply(out_grad, a)

matrix_to_axis_angle = _Matrix_to_axis_angle_fw.apply


# ===== Quaternion Translation Multiply

class _Quaternion_translation_mul_bw(Function):
    @staticmethod
    @custom_fwd(cast_inputs=torch.half, device_type="cuda")
    def forward(ctx, out_q_grad: torch.Tensor, out_t_grad: torch.Tensor, a_q: torch.Tensor, a_t: torch.Tensor, b_q: torch.Tensor, b_t: torch.Tensor):
        # out_q_grad: (..., 4), float
        # out_t_grad: (..., 3), float
        # a_q: (..., 4), float
        # a_t: (..., 3), float
        # b_q: (..., 4), float
        # b_t: (..., 3), float
        out_q_grad = out_q_grad.contiguous()
        out_t_grad = out_t_grad.contiguous()
        a_q = a_q.contiguous()
        a_t = a_t.contiguous()
        b_q = b_q.contiguous()
        b_t = b_t.contiguous()

        out_q_grad_shape = out_q_grad.shape
        out_t_grad_shape = out_t_grad.shape
        a_q_shape = a_q.shape
        a_t_shape = a_t.shape
        b_q_shape = b_q.shape
        b_t_shape = b_t.shape
        assert out_q_grad_shape[-1] == 4, out_q_grad_shape
        assert out_t_grad_shape[-1] == 3, out_t_grad_shape
        assert a_q_shape[-1] == 4, a_q_shape
        assert a_t_shape[-1] == 3, a_t_shape
        assert b_q_shape[-1] == 4, b_q_shape
        assert b_t_shape[-1] == 3, b_t_shape
        assert out_q_grad_shape[:-1] == out_t_grad_shape[:-1] == a_q_shape[:-1] == a_t_shape[:-1] == b_q_shape[:-1] == b_t_shape[:-1], (out_q_grad_shape, out_t_grad_shape, a_q_shape, a_t_shape, b_q_shape, b_t_shape)
        assert a_q.is_floating_point() and out_q_grad.dtype == out_t_grad.dtype == a_q.dtype == a_t.dtype == b_q.dtype == b_t.dtype, (out_q_grad.dtype, out_t_grad.dtype, a_q.dtype, a_t.dtype, b_q.dtype, b_t.dtype)
        assert out_q_grad.device == out_t_grad.device == a_q.device == a_t.device == b_q.device == b_t.device, (out_q_grad.device, out_t_grad.device, a_q.device, a_t.device, b_q.device, b_t.device)
        out_q_grad = out_q_grad.view(-1, out_q_grad_shape[-1])
        out_t_grad = out_t_grad.view(-1, out_t_grad_shape[-1])
        a_q = a_q.view(-1, a_q_shape[-1])
        a_t = a_t.view(-1, a_t_shape[-1])
        b_q = b_q.view(-1, b_q_shape[-1])
        b_t = b_t.view(-1, b_t_shape[-1])

        B = a_q.shape[0]
        alloc = torch.zeros(B * 14, dtype=a_q.dtype, device=a_q.device)
        a_q_grad = alloc[B * 0 : B * 4].view(B, 4)
        a_t_grad = alloc[B * 4 : B * 7].view(B, 3)
        b_q_grad = alloc[B * 7 : B * 11].view(B, 4)
        b_t_grad = alloc[B * 11 : B * 14].view(B, 3)

        (_dqtorch_cuda if a_q.is_cuda else _dqtorch_cpu).quaternion_translation_mul_bw(out_q_grad, out_t_grad, a_q, a_t, b_q, b_t, a_q_grad, a_t_grad, b_q_grad, b_t_grad, B)
        a_q_grad = a_q_grad.view(a_q_shape)
        a_t_grad = a_t_grad.view(a_t_shape)
        b_q_grad = b_q_grad.view(b_q_shape)
        b_t_grad = b_t_grad.view(b_t_shape)
        return a_q_grad, a_t_grad, b_q_grad, b_t_grad

    @staticmethod
    @custom_bwd(device_type="cuda")
    def backward(*args):
        raise NotImplementedError

class _Quaternion_translation_mul_fw(Function):
    @staticmethod
    @custom_fwd(cast_inputs=torch.half, device_type="cuda")
    def forward(ctx, a_q: torch.Tensor, a_t: torch.Tensor, b_q: torch.Tensor, b_t: torch.Tensor):
        # a_q: (..., 4), float
        # a_t: (..., 3), float
        # b_q: (..., 4), float
        # b_t: (..., 3), float
        a_q = a_q.contiguous()
        a_t = a_t.contiguous()
        b_q = b_q.contiguous()
        b_t = b_t.contiguous()

        a_q_shape = a_q.shape
        a_t_shape = a_t.shape
        b_q_shape = b_q.shape
        b_t_shape = b_t.shape
        assert a_q_shape[-1] == 4, a_q_shape
        assert a_t_shape[-1] == 3, a_t_shape
        assert b_q_shape[-1] == 4, b_q_shape
        assert b_t_shape[-1] == 3, b_t_shape
        assert a_q_shape[:-1] == a_t_shape[:-1] == b_q_shape[:-1] == b_t_shape[:-1], (a_q_shape, a_t_shape, b_q_shape, b_t_shape)
        assert a_q.is_floating_point() and a_q.dtype == a_t.dtype == b_q.dtype == b_t.dtype, (a_q.dtype, a_t.dtype, b_q.dtype, b_t.dtype)
        assert a_q.device == a_t.device == b_q.device == b_t.device, (a_q.device, a_t.device, b_q.device, b_t.device)
        a_q = a_q.view(-1, a_q_shape[-1])
        a_t = a_t.view(-1, a_t_shape[-1])
        b_q = b_q.view(-1, b_q_shape[-1])
        b_t = b_t.view(-1, b_t_shape[-1])

        B = a_q.shape[0]
        alloc = torch.zeros(B * 7, dtype=a_q.dtype, device=a_q.device)
        out_q = alloc[B * 0 : B * 4].view(B, 4)
        out_t = alloc[B * 4 : B * 7].view(B, 3)

        (_dqtorch_cuda if a_q.is_cuda else _dqtorch_cpu).quaternion_translation_mul_fw(a_q, a_t, b_q, b_t, out_q, out_t, B);
        a_q = a_q.view(a_q_shape)
        a_t = a_t.view(a_t_shape)
        b_q = b_q.view(b_q_shape)
        b_t = b_t.view(b_t_shape)
        out_q = out_q.view(a_q_shape)
        out_t = out_t.view(a_t_shape)
        ctx.save_for_backward(a_q, a_t, b_q, b_t)
        return out_q, out_t

    @staticmethod
    @custom_bwd(device_type="cuda")
    def backward(ctx, out_q_grad, out_t_grad):
        a_q, a_t, b_q, b_t = ctx.saved_tensors
        return _Quaternion_translation_mul_bw.apply(out_q_grad, out_t_grad, a_q, a_t, b_q, b_t)

def quaternion_translation_mul(a: QuaternionTranslation, b: QuaternionTranslation) -> QuaternionTranslation:
    return _Quaternion_translation_mul_fw.apply(a[0], a[1], b[0], b[1])


# ===== Quaternion Translation Apply

class _Quaternion_translation_apply_bw(Function):
    @staticmethod
    @custom_fwd(cast_inputs=torch.half, device_type="cuda")
    def forward(ctx, out_grad: torch.Tensor, a_q: torch.Tensor, a_t: torch.Tensor, b: torch.Tensor):
        # out_grad: (..., 4), float
        # a_q: (..., 4), float
        # a_t: (..., 3), float
        # b: (..., 3), float
        out_grad = out_grad.contiguous()
        a_q = a_q.contiguous()
        a_t = a_t.contiguous()
        b = b.contiguous()

        out_grad_shape = out_grad.shape
        a_q_shape = a_q.shape
        a_t_shape = a_t.shape
        b_shape = b.shape
        assert out_grad_shape[-1] == 3, out_grad_shape
        assert a_q_shape[-1] == 4, a_q_shape
        assert a_t_shape[-1] == 3, a_t_shape
        assert b_shape[-1] == 3, b_shape
        assert out_grad_shape[:-1] == a_q_shape[:-1] == a_t_shape[:-1] == b_shape[:-1], (out_grad_shape, a_q_shape, a_t_shape, b_shape)
        assert a_q.is_floating_point() and out_grad.dtype == a_q.dtype == a_t.dtype == b.dtype, (out_grad.dtype, a_q.dtype, a_t.dtype, b.dtype)
        assert out_grad.device == a_q.device == a_t.device == b.device, (out_grad.device, a_q.device, a_t.device, b.device)
        out_grad = out_grad.view(-1, out_grad_shape[-1])
        a_q = a_q.view(-1, a_q_shape[-1])
        a_t = a_t.view(-1, a_t_shape[-1])
        b = b.view(-1, b_shape[-1])

        B = a_q.shape[0]
        alloc = torch.zeros(B * 10, dtype=a_q.dtype, device=a_q.device)
        a_q_grad = alloc[B * 0 : B * 4].view(B, 4)
        a_t_grad = alloc[B * 4 : B * 7].view(B, 3)
        b_grad = alloc[B * 7 : B * 10].view(B, 3)

        (_dqtorch_cuda if a_q.is_cuda else _dqtorch_cpu).quaternion_translation_apply_bw(out_grad, a_q, a_t, b, a_q_grad, a_t_grad, b_grad, B)
        a_q_grad = a_q_grad.view(a_q_shape)
        a_t_grad = a_t_grad.view(a_t_shape)
        b_grad = b_grad.view(b_shape)
        return a_q_grad, a_t_grad, b_grad

    @staticmethod
    @custom_bwd(device_type="cuda")
    def backward(*args):
        raise NotImplementedError

class _Quaternion_translation_apply_fw(Function):
    @staticmethod
    @custom_fwd(cast_inputs=torch.half, device_type="cuda")
    def forward(ctx, a_q: torch.Tensor, a_t: torch.Tensor, b: torch.Tensor):
        # a_q: (..., 4), float
        # a_t: (..., 3), float
        # b: (..., 3), float
        a_q = a_q.contiguous()
        a_t = a_t.contiguous()
        b = b.contiguous()

        a_q_shape = a_q.shape
        a_t_shape = a_t.shape
        b_shape = b.shape
        assert a_q_shape[-1] == 4, a_q_shape
        assert a_t_shape[-1] == 3, a_t_shape
        assert b_shape[-1] == 3, b_shape
        assert a_q_shape[:-1] == a_t_shape[:-1] == b_shape[:-1], (a_q_shape, a_t_shape, b_shape)
        assert a_q.is_floating_point() and a_q.dtype == a_t.dtype == b.dtype, (a_q.dtype, a_t.dtype, b.dtype)
        assert a_q.device == a_t.device == b.device, (a_q.device, a_t.device, b.device)
        a_q = a_q.view(-1, a_q_shape[-1])
        a_t = a_t.view(-1, a_t_shape[-1])
        b = b.view(-1, b_shape[-1])

        B = a_q.shape[0]
        alloc = torch.zeros(B * 3, dtype=a_q.dtype, device=a_q.device)
        out = alloc[B * 0 : B * 3].view(B, 3)

        (_dqtorch_cuda if a_q.is_cuda else _dqtorch_cpu).quaternion_translation_apply_fw(a_q, a_t, b, out, B)
        a_q = a_q.view(a_q_shape)
        a_t = a_t.view(a_t_shape)
        b = b.view(b_shape)
        out = out.view(b_shape)
        ctx.save_for_backward(a_q, a_t, b)
        return out

    @staticmethod
    @custom_bwd(device_type="cuda")
    def backward(ctx, out_grad):
        a_q, a_t, b = ctx.saved_tensors
        return _Quaternion_translation_apply_bw.apply(out_grad, a_q, a_t, b)

quaternion_translation_apply = _Quaternion_translation_apply_fw.apply


# ===== Quaternion Translation Inverse

class _Quaternion_translation_inverse_bw(Function):
    @staticmethod
    @custom_fwd(cast_inputs=torch.half, device_type="cuda")
    def forward(ctx, out_q_grad: torch.Tensor, out_t_grad: torch.Tensor, a_q: torch.Tensor, a_t: torch.Tensor):
        # out_q_grad: (..., 4), float
        # out_t_grad: (..., 3), float
        # a_q: (..., 4), float
        # a_t: (..., 3), float
        out_q_grad = out_q_grad.contiguous()
        out_t_grad = out_t_grad.contiguous()
        a_q = a_q.contiguous()
        a_t = a_t.contiguous()

        out_q_grad_shape = out_q_grad.shape
        out_t_grad_shape = out_t_grad.shape
        a_q_shape = a_q.shape
        a_t_shape = a_t.shape
        assert out_q_grad_shape[-1] == 4, out_q_grad_shape
        assert out_t_grad_shape[-1] == 3, out_t_grad_shape
        assert a_q_shape[-1] == 4, a_q_shape
        assert a_t_shape[-1] == 3, a_t_shape
        assert out_q_grad_shape[:-1] == out_t_grad_shape[:-1] == a_q_shape[:-1] == a_t_shape[:-1], (out_q_grad_shape, out_t_grad_shape, a_q_shape, a_t_shape)
        assert a_q.is_floating_point() and out_q_grad.dtype == out_t_grad.dtype == a_q.dtype == a_t.dtype, (out_q_grad.dtype, out_t_grad.dtype, a_q.dtype, a_t.dtype)
        assert out_q_grad.device == out_t_grad.device == a_q.device == a_t.device, (out_q_grad.device, out_t_grad.device, a_q.device, a_t.device)
        out_q_grad = out_q_grad.view(-1, out_q_grad_shape[-1])
        out_t_grad = out_t_grad.view(-1, out_t_grad_shape[-1])
        a_q = a_q.view(-1, a_q_shape[-1])
        a_t = a_t.view(-1, a_t_shape[-1])

        B = a_q.shape[0]
        alloc = torch.zeros(B * 7, dtype=a_q.dtype, device=a_q.device)
        a_q_grad = alloc[B * 0 : B * 4].view(B, 4)
        a_t_grad = alloc[B * 4 : B * 7].view(B, 3)
        
        (_dqtorch_cuda if a_q.is_cuda else _dqtorch_cpu).quaternion_translation_inverse_bw(out_q_grad, out_t_grad, a_q, a_t, a_q_grad, a_t_grad, B)
        a_q_grad = a_q_grad.view(a_q_shape)
        a_t_grad = a_t_grad.view(a_t_shape)
        return a_q_grad, a_t_grad

    @staticmethod
    @custom_bwd(device_type="cuda")
    def backward(*args):
        raise NotImplementedError

class _Quaternion_translation_inverse_fw(Function):
    @staticmethod
    @custom_fwd(cast_inputs=torch.half, device_type="cuda")
    def forward(ctx, a_q: torch.Tensor, a_t: torch.Tensor):
        # a_q: (..., 4), float
        # a_t: (..., 3), float
        a_q = a_q.contiguous()
        a_t = a_t.contiguous()

        a_q_shape = a_q.shape
        a_t_shape = a_t.shape
        assert a_q_shape[-1] == 4, a_q_shape
        assert a_t_shape[-1] == 3, a_t_shape
        assert a_q_shape[:-1] == a_t_shape[:-1], (a_q_shape, a_t_shape)
        assert a_q.is_floating_point() and a_q.dtype == a_t.dtype, (a_q.dtype, a_t.dtype)
        assert a_q.device == a_t.device, (a_q.device, a_t.device)
        a_q = a_q.view(-1, a_q_shape[-1])
        a_t = a_t.view(-1, a_t_shape[-1])

        B = a_q.shape[0]
        alloc = torch.zeros(B * 7, dtype=a_q.dtype, device=a_q.device)
        out_q = alloc[B * 0 : B * 4].view(B, 4)
        out_t = alloc[B * 4 : B * 7].view(B, 3)

        (_dqtorch_cuda if a_q.is_cuda else _dqtorch_cpu).quaternion_translation_inverse_fw(a_q, a_t, out_q, out_t, B)
        a_q = a_q.view(a_q_shape)
        a_t = a_t.view(a_t_shape)
        out_q = out_q.view(a_q_shape)
        out_t = out_t.view(a_t_shape)
        ctx.save_for_backward(a_q, a_t)
        return out_q, out_t

    @staticmethod
    @custom_bwd(device_type="cuda")
    def backward(ctx, out_q_grad, out_t_grad):
        a_q, a_t = ctx.saved_tensors
        return _Quaternion_translation_inverse_bw.apply(out_q_grad, out_t_grad, a_q, a_t)

quaternion_translation_inverse = _Quaternion_translation_inverse_fw.apply


# ===== Dual Quaternion Multiply

class _Dual_quaternion_mul_bw(Function):
    @staticmethod
    @custom_fwd(cast_inputs=torch.half, device_type="cuda")
    def forward(ctx, out_r_grad: torch.Tensor, out_d_grad: torch.Tensor, a_r: torch.Tensor, a_d: torch.Tensor, b_r: torch.Tensor, b_d: torch.Tensor):
        # out_r_grad: (..., 4), float
        # out_d_grad: (..., 4), float
        # a_r: (..., 4), float
        # a_d: (..., 4), float
        # b_r: (..., 4), float
        # b_d: (..., 4), float
        out_r_grad = out_r_grad.contiguous()
        out_d_grad = out_d_grad.contiguous()
        a_r = a_r.contiguous()
        a_d = a_d.contiguous()
        b_r = b_r.contiguous()
        b_d = b_d.contiguous()

        out_r_grad_shape = out_r_grad.shape
        out_d_grad_shape = out_d_grad.shape
        a_r_shape = a_r.shape
        a_d_shape = a_d.shape
        b_r_shape = b_r.shape
        b_d_shape = b_d.shape
        assert out_r_grad_shape[-1] == 4, out_r_grad_shape
        assert out_d_grad_shape[-1] == 4, out_d_grad_shape
        assert a_r_shape[-1] == 4, a_r_shape
        assert a_d_shape[-1] == 4, a_d_shape
        assert b_r_shape[-1] == 4, b_r_shape
        assert b_d_shape[-1] == 4, b_d_shape
        assert out_r_grad_shape[:-1] == out_d_grad_shape[:-1] == a_r_shape[:-1] == a_d_shape[:-1] == b_r_shape[:-1] == b_d_shape[:-1], (out_r_grad_shape, out_d_grad_shape, a_r_shape, a_d_shape, b_r_shape, b_d_shape)
        assert a_r.is_floating_point() and out_r_grad.dtype == out_d_grad.dtype == a_r.dtype == a_d.dtype == b_r.dtype == b_d.dtype, (out_r_grad.dtype, out_d_grad.dtype, a_r.dtype, a_d.dtype, b_r.dtype, b_d.dtype)
        assert out_r_grad.device == out_d_grad.device == a_r.device == a_d.device == b_r.device == b_d.device, (out_r_grad.device, out_d_grad.device, a_r.device, a_d.device, b_r.device, b_d.device)
        out_r_grad = out_r_grad.view(-1, out_r_grad_shape[-1])
        out_d_grad = out_d_grad.view(-1, out_d_grad_shape[-1])
        a_r = a_r.view(-1, a_r_shape[-1])
        a_d = a_d.view(-1, a_d_shape[-1])
        b_r = b_r.view(-1, b_r_shape[-1])
        b_d = b_d.view(-1, b_d_shape[-1])

        B = out_r_grad.shape[0]
        alloc = torch.zeros(B *16, dtype=a_r.dtype, device=a_r.device)
        a_r_grad = alloc[B * 0 : B * 4].view(B, 4)
        a_d_grad = alloc[B * 4 : B * 8].view(B, 4)
        b_r_grad = alloc[B * 8 : B * 12].view(B, 4)
        b_d_grad = alloc[B * 12 : B * 16].view(B, 4)

        (_dqtorch_cuda if a_r.is_cuda else _dqtorch_cpu).dual_quaternion_mul_bw(out_r_grad, out_d_grad, a_r, a_d, b_r, b_d, a_r_grad, a_d_grad, b_r_grad, b_d_grad, B)
        a_r_grad = a_r_grad.view(a_r_shape)
        a_d_grad = a_d_grad.view(a_d_shape)
        b_r_grad = b_r_grad.view(b_r_shape)
        b_d_grad = b_d_grad.view(b_d_shape)
        return a_r_grad, a_d_grad, b_r_grad, b_d_grad

    @staticmethod
    @custom_bwd(device_type="cuda")
    def backward(*args):
        raise NotImplementedError

class _Dual_quaternion_mul_fw(Function):
    @staticmethod
    @custom_fwd(cast_inputs=torch.half, device_type="cuda")
    def forward(ctx, a_r: torch.Tensor, a_d: torch.Tensor, b_r: torch.Tensor, b_d: torch.Tensor):
        # a_r: (..., 4), float
        # a_d: (..., 4), float
        # b_r: (..., 4), float
        # b_d: (..., 4), float
        a_r = a_r.contiguous()
        a_d = a_d.contiguous()
        b_r = b_r.contiguous()
        b_d = b_d.contiguous()

        a_r_shape = a_r.shape
        a_d_shape = a_d.shape
        b_r_shape = b_r.shape
        b_d_shape = b_d.shape
        assert a_r_shape[-1] == 4, a_r_shape
        assert a_d_shape[-1] == 4, a_d_shape
        assert b_r_shape[-1] == 4, b_r_shape
        assert b_d_shape[-1] == 4, b_d_shape
        assert a_r_shape[:-1] == a_d_shape[:-1] == b_r_shape[:-1] == b_d_shape[:-1], (a_r_shape, a_d_shape, b_r_shape, b_d_shape)
        assert a_r.is_floating_point() and a_r.dtype == a_d.dtype == b_r.dtype == b_d.dtype, (a_r.dtype, a_d.dtype, b_r.dtype, b_d.dtype)
        assert a_r.device == a_d.device == b_r.device == b_d.device, (a_r.device, a_d.device, b_r.device, b_d.device)
        a_r = a_r.view(-1, a_r_shape[-1])
        a_d = a_d.view(-1, a_d_shape[-1])
        b_r = b_r.view(-1, b_r_shape[-1])
        b_d = b_d.view(-1, b_d_shape[-1])

        B = a_r.shape[0]
        alloc = torch.zeros(B * 8, dtype=a_r.dtype, device=a_r.device)
        out_r = alloc[B * 0 : B * 4].view(B, 4)
        out_d = alloc[B * 4 : B * 8].view(B, 4)

        (_dqtorch_cuda if a_r.is_cuda else _dqtorch_cpu).dual_quaternion_mul_fw(a_r, a_d, b_r, b_d, out_r, out_d, B);
        a_r = a_r.view(a_r_shape)
        a_d = a_d.view(a_d_shape)
        b_r = b_r.view(b_r_shape)
        b_d = b_d.view(b_d_shape)
        out_r = out_r.view(a_r_shape)
        out_d = out_d.view(a_d_shape)
        ctx.save_for_backward(a_r, a_d, b_r, b_d)
        return out_r, out_d

    @staticmethod
    @custom_bwd(device_type="cuda")
    def backward(ctx, out_r_grad, out_d_grad):
        a_r, a_d, b_r, b_d = ctx.saved_tensors
        return _Dual_quaternion_mul_bw.apply(out_r_grad, out_d_grad, a_r, a_d, b_r, b_d)

def dual_quaternion_mul(a: DualQuaternions, b: DualQuaternions) -> DualQuaternions:
    return _Dual_quaternion_mul_fw.apply(a[0], a[1], b[0], b[1])


# ===== Dual Quaternion Apply

class _Dual_quaternion_apply_bw(Function):
    @staticmethod
    @custom_fwd(cast_inputs=torch.half, device_type="cuda")
    def forward(ctx, out_grad: torch.Tensor, a_r: torch.Tensor, a_d: torch.Tensor, b: torch.Tensor):
        # out_grad: (..., 4), float
        # a_r: (..., 4), float
        # a_d: (..., 4), float
        # b: (..., 3), float
        out_grad = out_grad.contiguous()
        a_r = a_r.contiguous()
        a_d = a_d.contiguous()
        b = b.contiguous()

        out_grad_shape = out_grad.shape
        a_r_shape = a_r.shape
        a_d_shape = a_d.shape
        b_shape = b.shape
        assert out_grad_shape[-1] == 3, out_grad_shape
        assert a_r_shape[-1] == 4, a_r_shape
        assert a_d_shape[-1] == 4, a_d_shape
        assert b_shape[-1] == 3, b_shape
        assert out_grad_shape[:-1] == a_r_shape[:-1] == a_d_shape[:-1] == b_shape[:-1], (out_grad_shape, a_r_shape, a_d_shape, b_shape)
        assert a_r.is_floating_point() and out_grad.dtype == a_r.dtype == a_d.dtype == b.dtype, (out_grad.dtype, a_r.dtype, a_d.dtype, b.dtype)
        assert out_grad.device == a_r.device == a_d.device == b.device, (out_grad.device, a_r.device, a_d.device, b.device)
        out_grad = out_grad.view(-1, out_grad_shape[-1])
        a_r = a_r.view(-1, a_r_shape[-1])
        a_d = a_d.view(-1, a_d_shape[-1])
        b = b.view(-1, b_shape[-1])

        B = a_r.shape[0]
        alloc = torch.zeros(B * 11, dtype=a_r.dtype, device=a_r.device)
        a_r_grad = alloc[B * 0 : B * 4].view(B, 4)
        a_d_grad = alloc[B * 4 : B * 8].view(B, 4)
        b_grad = alloc[B * 8 : B * 11].view(B, 3)

        (_dqtorch_cuda if a_r.is_cuda else _dqtorch_cpu).dual_quaternion_apply_bw(out_grad, a_r, a_d, b, a_r_grad, a_d_grad, b_grad, B)
        a_r_grad = a_r_grad.view(a_r_shape)
        a_d_grad = a_d_grad.view(a_d_shape)
        b_grad = b_grad.view(b_shape)
        return a_r_grad, a_d_grad, b_grad

    @staticmethod
    @custom_bwd(device_type="cuda")
    def backward(*args):
        raise NotImplementedError

class _Dual_quaternion_apply_fw(Function):
    @staticmethod
    @custom_fwd(cast_inputs=torch.half, device_type="cuda")
    def forward(ctx, a_r: torch.Tensor, a_d: torch.Tensor, b: torch.Tensor):
        # a_r: (..., 4), float
        # a_d: (..., 4), float
        # b: (..., 3), float
        a_r = a_r.contiguous()
        a_d = a_d.contiguous()
        b = b.contiguous()

        a_r_shape = a_r.shape
        a_d_shape = a_d.shape
        b_shape = b.shape
        assert a_r_shape[-1] == 4, a_r_shape
        assert a_d_shape[-1] == 4, a_d_shape
        assert b_shape[-1] == 3, b_shape
        assert a_r_shape[:-1] == a_d_shape[:-1] == b_shape[:-1], (a_r_shape, a_d_shape, b_shape)
        assert a_r.is_floating_point() and a_r.dtype == a_d.dtype == b.dtype, (a_r.dtype, a_d.dtype, b.dtype)
        assert a_r.device == a_d.device == b.device, (a_r.device, a_d.device, b.device)
        a_r = a_r.view(-1, a_r_shape[-1])
        a_d = a_d.view(-1, a_d_shape[-1])
        b = b.view(-1, b_shape[-1])

        B = a_r.shape[0]
        alloc = torch.zeros(B * 3, dtype=b.dtype, device=b.device)
        out = alloc[B * 0 : B * 3].view(B, 3)

        (_dqtorch_cuda if a_r.is_cuda else _dqtorch_cpu).dual_quaternion_apply_fw(a_r, a_d, b, out, B)
        a_r = a_r.view(a_r_shape)
        a_d = a_d.view(a_d_shape)
        b = b.view(b_shape)
        out = out.view(b_shape)
        ctx.save_for_backward(a_r, a_d, b)
        return out

    @staticmethod
    @custom_bwd(device_type="cuda")
    def backward(ctx, out_grad):
        a_r, a_d, b = ctx.saved_tensors
        return _Dual_quaternion_apply_bw.apply(out_grad, a_r, a_d, b)

def dual_quaternion_apply(a: DualQuaternions, point: torch.Tensor) -> torch.Tensor:
    return _Dual_quaternion_apply_fw.apply(a[0], a[1], point)


# ===== Dual Quaternion Q Conjugate
# Note: Python version here is faster than custom CUDA autograd function

def dual_quaternion_q_conjugate(a: DualQuaternions) -> DualQuaternions:
    return quaternion_conjugate(a[0]), quaternion_conjugate(a[1])

dual_quaternion_inverse = dual_quaternion_q_conjugate


# ===== Dual Quaternion D Conjugate
# Note: Python version here is faster than custom CUDA autograd function

def dual_quaternion_d_conjugate(a: DualQuaternions) -> DualQuaternions:
    return a[0], -a[1]


# ===== Dual Quaternion 3rd Conjugate
# Note: Python version here is faster than custom CUDA autograd function

def dual_quaternion_3rd_conjugate(a: DualQuaternions) -> DualQuaternions:
    return dual_quaternion_d_conjugate(dual_quaternion_q_conjugate(a))


# ===== Dual Quaternion Norm

class _Dual_quaternion_norm_bw(Function):
    @staticmethod
    @custom_fwd(cast_inputs=torch.half, device_type="cuda")
    def forward(ctx, out_r_grad: torch.Tensor, out_d_grad: torch.Tensor, a_r: torch.Tensor, a_d: torch.Tensor):
        # out_r_grad: (..., 4), float
        # out_d_grad: (..., 4), float
        # a_r: (..., 4), float
        # a_d: (..., 4), float
        out_r_grad = out_r_grad.contiguous()
        out_d_grad = out_d_grad.contiguous()
        a_r = a_r.contiguous()
        a_d = a_d.contiguous()

        out_r_grad_shape = out_r_grad.shape
        out_d_grad_shape = out_d_grad.shape
        a_r_shape = a_r.shape
        a_d_shape = a_d.shape
        assert out_r_grad_shape[-1] == 4, out_r_grad_shape
        assert out_d_grad_shape[-1] == 4, out_d_grad_shape
        assert a_r_shape[-1] == 4, a_r_shape
        assert a_d_shape[-1] == 4, a_d_shape
        assert out_r_grad_shape[:-1] == out_d_grad_shape[:-1] == a_r_shape[:-1] == a_d_shape[:-1], (out_r_grad_shape, out_d_grad_shape, a_r_shape, a_d_shape)
        assert a_r.is_floating_point() and out_r_grad.dtype == out_d_grad.dtype == a_r.dtype == a_d.dtype, (out_r_grad.dtype, out_d_grad.dtype, a_r.dtype, a_d.dtype)
        assert out_r_grad.device == out_d_grad.device == a_r.device == a_d.device, (out_r_grad.device, out_d_grad.device, a_r.device, a_d.device)
        out_r_grad = out_r_grad.view(-1, out_r_grad_shape[-1])
        out_d_grad = out_d_grad.view(-1, out_d_grad_shape[-1])
        a_r = a_r.view(-1, a_r_shape[-1])
        a_d = a_d.view(-1, a_d_shape[-1])

        B = a_r.shape[0]
        alloc = torch.zeros(B * 8, dtype=a_r.dtype, device=a_r.device)
        a_r_grad = alloc[B * 0 : B * 4].view(B, 4)
        a_d_grad = alloc[B * 4 : B * 8].view(B, 4)
        
        (_dqtorch_cuda if a_r.is_cuda else _dqtorch_cpu).dual_quaternion_norm_bw(out_r_grad, out_d_grad, a_r, a_d, a_r_grad, a_d_grad, B)
        a_r_grad = a_r_grad.view(a_r_shape)
        a_d_grad = a_d_grad.view(a_d_shape)
        return a_r_grad, a_d_grad

    @staticmethod
    @custom_bwd(device_type="cuda")
    def backward(*args):
        raise NotImplementedError

class _Dual_quaternion_norm_fw(Function):
    @staticmethod
    @custom_fwd(cast_inputs=torch.half, device_type="cuda")
    def forward(ctx, a_r: torch.Tensor, a_d: torch.Tensor):
        # a_r: (..., 4), float
        # a_d: (..., 4), float
        a_r = a_r.contiguous()
        a_d = a_d.contiguous()

        a_r_shape = a_r.shape
        a_d_shape = a_d.shape
        assert a_r_shape[-1] == 4, a_r_shape
        assert a_d_shape[-1] == 4, a_d_shape
        assert a_r_shape[:-1] == a_d_shape[:-1], (a_r_shape, a_d_shape)
        assert a_r.is_floating_point() and a_r.dtype == a_d.dtype, (a_r.dtype, a_d.dtype)
        assert a_r.device == a_d.device, (a_r.device, a_d.device)
        a_r = a_r.view(-1, a_r_shape[-1])
        a_d = a_d.view(-1, a_d_shape[-1])

        B = a_r.shape[0]
        alloc = torch.zeros(B * 8, dtype=a_r.dtype, device=a_r.device)
        out_r = alloc[B * 0 : B * 4].view(B, 4)
        out_d = alloc[B * 4 : B * 8].view(B, 4)

        (_dqtorch_cuda if a_r.is_cuda else _dqtorch_cpu).dual_quaternion_norm_fw(a_r, a_d, out_r, out_d, B)
        a_r = a_r.view(a_r_shape)
        a_d = a_d.view(a_d_shape)
        out_r = out_r.view(a_r_shape)
        out_d = out_d.view(a_d_shape)
        ctx.save_for_backward(a_r, a_d)
        return out_r, out_d

    @staticmethod
    @custom_bwd(device_type="cuda")
    def backward(ctx, out_r_grad, out_d_grad):
        a_r, a_d = ctx.saved_tensors
        return _Dual_quaternion_norm_bw.apply(out_r_grad, out_d_grad, a_r, a_d)

def dual_quaternion_norm(a: DualQuaternions) -> DualQuaternions:
    return _Dual_quaternion_norm_fw.apply(a[0], a[1])    


# ===== Quaternion Translation to Dual Quaternion

class _Quaternion_translation_to_dual_quaternion_bw(Function):
    @staticmethod
    @custom_fwd(cast_inputs=torch.half, device_type="cuda")
    def forward(ctx, out_r_grad: torch.Tensor, out_d_grad: torch.Tensor, a_q: torch.Tensor, a_t: torch.Tensor):
        # out_r_grad: (..., 4), float
        # out_d_grad: (..., 4), float
        # a_q: (..., 4), float
        # a_t: (..., 3), float
        out_r_grad = out_r_grad.contiguous()
        out_d_grad = out_d_grad.contiguous()
        a_q = a_q.contiguous()
        a_t = a_t.contiguous()

        out_r_grad_shape = out_r_grad.shape
        out_d_grad_shape = out_d_grad.shape
        a_q_shape = a_q.shape
        a_t_shape = a_t.shape
        assert out_r_grad_shape[-1] == 4, out_r_grad_shape
        assert out_d_grad_shape[-1] == 4, out_d_grad_shape
        assert a_q_shape[-1] == 4, a_q_shape
        assert a_t_shape[-1] == 3, a_t_shape
        assert out_r_grad_shape[:-1] == out_d_grad_shape[:-1] == a_q_shape[:-1] == a_t_shape[:-1], (out_r_grad_shape, out_d_grad_shape, a_q_shape, a_t_shape)
        assert a_q.is_floating_point() and out_r_grad.dtype == out_d_grad.dtype == a_q.dtype == a_t.dtype, (out_r_grad.dtype, out_d_grad.dtype, a_q.dtype, a_t.dtype)
        assert out_r_grad.device == out_d_grad.device == a_q.device == a_t.device, (out_r_grad.device, out_d_grad.device, a_q.device, a_t.device)
        out_r_grad = out_r_grad.view(-1, out_r_grad_shape[-1])
        out_d_grad = out_d_grad.view(-1, out_d_grad_shape[-1])
        a_q = a_q.view(-1, a_q_shape[-1])
        a_t = a_t.view(-1, a_t_shape[-1])

        B = a_q.shape[0]
        alloc = torch.zeros(B * 7, dtype=a_q.dtype, device=a_q.device)
        a_q_grad = alloc[B * 0 : B * 4].view(B, 4)
        a_t_grad = alloc[B * 4 : B * 7].view(B, 3)
        
        (_dqtorch_cuda if a_q.is_cuda else _dqtorch_cpu).quaternion_translation_to_dual_quaternion_bw(out_r_grad, out_d_grad, a_q, a_t, a_q_grad, a_t_grad, B)
        a_q_grad = a_q_grad.view(a_q_shape)
        a_t_grad = a_t_grad.view(a_t_shape)
        return a_q_grad, a_t_grad

    @staticmethod
    @custom_bwd(device_type="cuda")
    def backward(*args):
        raise NotImplementedError

class _Quaternion_translation_to_dual_quaternion_fw(Function):
    @staticmethod
    @custom_fwd(cast_inputs=torch.half, device_type="cuda")
    def forward(ctx, a_q: torch.Tensor, a_t: torch.Tensor):
        # a_q: (..., 4), float
        # a_t: (..., 3), float
        a_q = a_q.contiguous()
        a_t = a_t.contiguous()

        a_q_shape = a_q.shape
        a_t_shape = a_t.shape
        assert a_q_shape[-1] == 4, a_q_shape
        assert a_t_shape[-1] == 3, a_t_shape
        assert a_q_shape[:-1] == a_t_shape[:-1], (a_q_shape, a_t_shape)
        assert a_q.is_floating_point() and a_q.dtype == a_t.dtype, (a_q.dtype, a_t.dtype)
        assert a_q.device == a_t.device, (a_q.device, a_t.device)
        a_q = a_q.view(-1, a_q_shape[-1])
        a_t = a_t.view(-1, a_t_shape[-1])

        B = a_q.shape[0]
        alloc = torch.zeros(B * 8, dtype=a_q.dtype, device=a_q.device)
        out_r = alloc[B * 0 : B * 4].view(B, 4)
        out_d = alloc[B * 4 : B * 8].view(B, 4)

        (_dqtorch_cuda if a_q.is_cuda else _dqtorch_cpu).quaternion_translation_to_dual_quaternion_fw(a_q, a_t, out_r, out_d, B)
        a_q = a_q.view(a_q_shape)
        a_t = a_t.view(a_t_shape)
        out_r = out_r.view(a_q_shape)
        out_d = out_d.view(a_q_shape)
        ctx.save_for_backward(a_q, a_t)
        return out_r, out_d

    @staticmethod
    @custom_bwd(device_type="cuda")
    def backward(ctx, out_r_grad, out_d_grad):
        a_q, a_t = ctx.saved_tensors
        return _Quaternion_translation_to_dual_quaternion_bw.apply(out_r_grad, out_d_grad, a_q, a_t)

quaternion_translation_to_dual_quaternion = _Quaternion_translation_to_dual_quaternion_fw.apply


# ===== Dual Quaternion to Quaternion Translation

class _Dual_quaternion_to_quaternion_translation_bw(Function):
    @staticmethod
    @custom_fwd(cast_inputs=torch.half, device_type="cuda")
    def forward(ctx, out_q_grad: torch.Tensor, out_t_grad: torch.Tensor, a_r: torch.Tensor, a_d: torch.Tensor):
        # out_q_grad: (..., 4), float
        # out_t_grad: (..., 3), float
        # a_r: (..., 4), float
        # a_d: (..., 4), float
        out_q_grad = out_q_grad.contiguous()
        out_t_grad = out_t_grad.contiguous()
        a_r = a_r.contiguous()
        a_d = a_d.contiguous()

        out_q_grad_shape = out_q_grad.shape
        out_t_grad_shape = out_t_grad.shape
        a_r_shape = a_r.shape
        a_d_shape = a_d.shape
        assert out_q_grad_shape[-1] == 4, out_q_grad_shape
        assert out_t_grad_shape[-1] == 3, out_t_grad_shape
        assert a_r_shape[-1] == 4, a_r_shape
        assert a_d_shape[-1] == 4, a_d_shape
        assert out_q_grad_shape[:-1] == out_t_grad_shape[:-1] == a_r_shape[:-1] == a_d_shape[:-1], (out_q_grad_shape, out_t_grad_shape, a_r_shape, a_d_shape)
        assert a_r.is_floating_point() and out_q_grad.dtype == out_t_grad.dtype == a_r.dtype == a_d.dtype, (out_q_grad.dtype, out_t_grad.dtype, a_r.dtype, a_d.dtype)
        assert out_q_grad.device == out_t_grad.device == a_r.device == a_d.device, (out_q_grad.device, out_t_grad.device, a_r.device, a_d.device)
        out_q_grad = out_q_grad.view(-1, out_q_grad_shape[-1])
        out_t_grad = out_t_grad.view(-1, out_t_grad_shape[-1])
        a_r = a_r.view(-1, a_r_shape[-1])
        a_d = a_d.view(-1, a_d_shape[-1])

        B = a_r.shape[0]
        alloc = torch.zeros(B * 8, dtype=a_r.dtype, device=a_r.device)
        a_r_grad = alloc[B * 0 : B * 4].view(B, 4)
        a_d_grad = alloc[B * 4 : B * 8].view(B, 4)
        
        (_dqtorch_cuda if a_r.is_cuda else _dqtorch_cpu).dual_quaternion_to_quaternion_translation_bw(out_q_grad, out_t_grad, a_r, a_d, a_r_grad, a_d_grad, B)
        a_r_grad = a_r_grad.view(a_r_shape)
        a_d_grad = a_d_grad.view(a_d_shape)
        return a_r_grad, a_d_grad

    @staticmethod
    @custom_bwd(device_type="cuda")
    def backward(*args):
        raise NotImplementedError

class _Dual_quaternion_to_quaternion_translation_fw(Function):
    @staticmethod
    @custom_fwd(cast_inputs=torch.half, device_type="cuda")
    def forward(ctx, a_r: torch.Tensor, a_d: torch.Tensor):
        # a_r: (..., 4), float
        # a_d: (..., 4), float
        a_r = a_r.contiguous()
        a_d = a_d.contiguous()

        a_r_shape = a_r.shape
        a_d_shape = a_d.shape
        assert a_r_shape[-1] == 4, a_r_shape
        assert a_d_shape[-1] == 4, a_d_shape
        assert a_r_shape[:-1] == a_d_shape[:-1], (a_r_shape, a_d_shape)
        assert a_r.is_floating_point() and a_r.dtype == a_d.dtype, (a_r.dtype, a_d.dtype)
        assert a_r.device == a_d.device, (a_r.device, a_d.device)
        a_r = a_r.view(-1, a_r_shape[-1])
        a_d = a_d.view(-1, a_d_shape[-1])

        B = a_r.shape[0]
        alloc = torch.zeros(B * 7, dtype=a_r.dtype, device=a_r.device)
        out_q = alloc[B * 0 : B * 4].view(B, 4)
        out_t = alloc[B * 4 : B * 7].view(B, 3)

        (_dqtorch_cuda if a_r.is_cuda else _dqtorch_cpu).dual_quaternion_to_quaternion_translation_fw(a_r, a_d, out_q, out_t, B)
        a_r = a_r.view(a_r_shape)
        a_d = a_d.view(a_d_shape)
        out_q = out_q.view(a_r_shape)
        out_t = out_t.view(a_r_shape[:-1] + (3,))
        ctx.save_for_backward(a_r, a_d)
        return out_q, out_t

    @staticmethod
    @custom_bwd(device_type="cuda")
    def backward(ctx, out_q_grad, out_t_grad):
        a_r, a_d = ctx.saved_tensors
        return _Dual_quaternion_to_quaternion_translation_bw.apply(out_q_grad, out_t_grad, a_r, a_d)

def dual_quaternion_to_quaternion_translation(a: DualQuaternions) -> QuaternionTranslation:
    return _Dual_quaternion_to_quaternion_translation_fw.apply(a[0], a[1])    


# ===== Quaternion Translation to SE3

class _Quaternion_translation_to_se3_bw(Function):
    @staticmethod
    @custom_fwd(cast_inputs=torch.half, device_type="cuda")
    def forward(ctx, out_grad: torch.Tensor, a_q: torch.Tensor, a_t: torch.Tensor):
        # out_grad: (..., 4, 4), float
        # a_q: (..., 4), float
        # a_t: (..., 3), float
        out_grad = out_grad.contiguous()
        a_q = a_q.contiguous()
        a_t = a_t.contiguous()

        out_grad_shape = out_grad.shape
        a_q_shape = a_q.shape
        a_t_shape = a_t.shape
        assert out_grad_shape[-1] == 4 and out_grad_shape[-2] == 4, out_grad_shape
        assert a_q_shape[-1] == 4, a_q_shape
        assert a_t_shape[-1] == 3, a_t_shape
        assert out_grad_shape[:-2] == a_q_shape[:-1] == a_t_shape[:-1], (out_grad_shape, a_q_shape, a_t_shape)
        assert a_q.is_floating_point() and out_grad.dtype == a_q.dtype == a_t.dtype, (out_grad.dtype, a_q.dtype, a_t.dtype)
        assert out_grad.device == a_q.device == a_t.device, (out_grad.device, a_q.device, a_t.device)
        out_grad = out_grad.view(-1, out_grad_shape[-2] * out_grad_shape[-1])
        a_q = a_q.view(-1, a_q_shape[-1])
        a_t = a_t.view(-1, a_t_shape[-1])

        B = a_q.shape[0]
        alloc = torch.zeros(B * 7, dtype=a_q.dtype, device=a_q.device)
        a_q_grad = alloc[B * 0 : B * 4].view(B, 4)
        a_t_grad = alloc[B * 4 : B * 7].view(B, 3)
        
        (_dqtorch_cuda if a_q.is_cuda else _dqtorch_cpu).quaternion_translation_to_se3_bw(out_grad, a_q, a_t, a_q_grad, a_t_grad, B)
        a_q_grad = a_q_grad.view(a_q_shape)
        a_t_grad = a_t_grad.view(a_t_shape)
        return a_q_grad, a_t_grad

    @staticmethod
    @custom_bwd(device_type="cuda")
    def backward(*args):
        raise NotImplementedError

class _Quaternion_translation_to_se3_fw(Function):
    @staticmethod
    @custom_fwd(cast_inputs=torch.half, device_type="cuda")
    def forward(ctx, a_q: torch.Tensor, a_t: torch.Tensor):
        # a_q: (..., 4), float
        # a_t: (..., 3), float
        a_q = a_q.contiguous()
        a_t = a_t.contiguous()

        a_q_shape = a_q.shape
        a_t_shape = a_t.shape
        assert a_q_shape[-1] == 4, a_q_shape
        assert a_t_shape[-1] == 3, a_t_shape
        assert a_q_shape[:-1] == a_t_shape[:-1], (a_q_shape, a_t_shape)
        assert a_q.is_floating_point() and a_q.dtype == a_t.dtype, (a_q.dtype, a_t.dtype)
        assert a_q.device == a_t.device, (a_q.device, a_t.device)
        a_q = a_q.view(-1, a_q_shape[-1])
        a_t = a_t.view(-1, a_t_shape[-1])

        B = a_q.shape[0]
        alloc = torch.zeros(B * 16, dtype=a_q.dtype, device=a_q.device)
        out = alloc[B * 0 : B * 16].view(B, 4, 4)

        (_dqtorch_cuda if a_q.is_cuda else _dqtorch_cpu).quaternion_translation_to_se3_fw(a_q, a_t, out, B)
        a_q = a_q.view(a_q_shape)
        a_t = a_t.view(a_t_shape)
        out = out.view(a_q_shape[:-1] + (4, 4))
        ctx.save_for_backward(a_q, a_t)
        return out

    @staticmethod
    @custom_bwd(device_type="cuda")
    def backward(ctx, out_grad):
        a_q, a_t = ctx.saved_tensors
        return _Quaternion_translation_to_se3_bw.apply(out_grad, a_q, a_t)

quaternion_translation_to_se3 = _Quaternion_translation_to_se3_fw.apply


# ===== SE3 to Quaternion Translation

class _SE3_to_quaternion_translation_bw(Function):
    @staticmethod
    @custom_fwd(cast_inputs=torch.half, device_type="cuda")
    def forward(ctx, out_q_grad: torch.Tensor, out_t_grad: torch.Tensor, a: torch.Tensor):
        # out_q_grad: (..., 4), float
        # out_t_grad: (..., 3), float
        # a: (..., 4, 4), float
        out_q_grad = out_q_grad.contiguous()
        out_t_grad = out_t_grad.contiguous()
        a = a.contiguous()

        out_q_grad_shape = out_q_grad.shape
        out_t_grad_shape = out_t_grad.shape
        a_shape = a.shape
        assert out_q_grad_shape[-1] == 4, out_q_grad_shape
        assert out_t_grad_shape[-1] == 3, out_t_grad_shape
        assert a_shape[-2] == 4 and a_shape[-1] == 4, a_shape
        assert out_q_grad_shape[:-1] == out_t_grad_shape[:-1] == a_shape[:-2], (out_q_grad_shape, out_t_grad_shape, a_shape)
        assert a.is_floating_point() and out_q_grad.dtype == out_t_grad.dtype == a.dtype, (out_q_grad.dtype, out_t_grad.dtype, a.dtype)
        assert out_q_grad.device == out_t_grad.device == a.device, (out_q_grad.device, out_t_grad.device, a.device)
        out_q_grad = out_q_grad.view(-1, out_q_grad_shape[-1])
        out_t_grad = out_t_grad.view(-1, out_t_grad_shape[-1])
        a = a.view(-1, a_shape[-2] * a_shape[-1])

        B = a.shape[0]
        alloc = torch.zeros(B * 16, dtype=a.dtype, device=a.device)
        a_grad = alloc[B * 0 : B * 16].view(B, 4, 4)
        
        (_dqtorch_cuda if a.is_cuda else _dqtorch_cpu).se3_to_quaternion_translation_bw(out_q_grad, out_t_grad, a, a_grad, B)
        a_grad = a_grad.view(a_shape)
        return a_grad

    @staticmethod
    @custom_bwd(device_type="cuda")
    def backward(*args):
        raise NotImplementedError

class _SE3_to_quaternion_translation_fw(Function):
    @staticmethod
    @custom_fwd(cast_inputs=torch.half, device_type="cuda")
    def forward(ctx, a: torch.Tensor):
        # a: (..., 4, 4), float
        a = a.contiguous()

        a_shape = a.shape
        assert a_shape[-2] == 4 and a_shape[-1] == 4, a_shape
        assert a.is_floating_point(), a.dtype
        a = a.view(-1, a_shape[-2] * a_shape[-1])

        B = a.shape[0]
        alloc = torch.zeros(B * 7, dtype=a.dtype, device=a.device)
        out_q = alloc[B * 0 : B * 4].view(B, 4)
        out_t = alloc[B * 4 : B * 7].view(B, 3)

        (_dqtorch_cuda if a.is_cuda else _dqtorch_cpu).se3_to_quaternion_translation_fw(a, out_q, out_t, B)
        a = a.view(a_shape)
        out_q = out_q.view(a_shape[:-2] + (4,))
        out_t = out_t.view(a_shape[:-2] + (3,))
        ctx.save_for_backward(a)
        return out_q, out_t

    @staticmethod
    @custom_bwd(device_type="cuda")
    def backward(ctx, out_q_grad, out_t_grad):
        a, = ctx.saved_tensors
        return _SE3_to_quaternion_translation_bw.apply(out_q_grad, out_t_grad, a)

se3_to_quaternion_translation = _SE3_to_quaternion_translation_fw.apply


# ===== Dual Quaternion to SE3

class _Dual_quaternion_to_se3_bw(Function):
    @staticmethod
    @custom_fwd(cast_inputs=torch.half, device_type="cuda")
    def forward(ctx, out_grad: torch.Tensor, a_r: torch.Tensor, a_d: torch.Tensor):
        # out_grad: (..., 4, 4), float
        # a_r: (..., 4), float
        # a_d: (..., 4), float
        out_grad = out_grad.contiguous()
        a_r = a_r.contiguous()
        a_d = a_d.contiguous()

        out_grad_shape = out_grad.shape
        a_r_shape = a_r.shape
        a_d_shape = a_d.shape
        assert out_grad_shape[-1] == 4 and out_grad_shape[-2] == 4, out_grad_shape
        assert a_r_shape[-1] == 4, a_r_shape
        assert a_d_shape[-1] == 4, a_d_shape
        assert out_grad_shape[:-2] == a_r_shape[:-1] == a_d_shape[:-1], (out_grad_shape, a_r_shape, a_d_shape)
        assert a_r.is_floating_point() and out_grad.dtype == a_r.dtype == a_d.dtype, (out_grad.dtype, a_r.dtype, a_d.dtype)
        assert out_grad.device == a_r.device == a_d.device, (out_grad.device, a_r.device, a_d.device)
        out_grad = out_grad.view(-1, out_grad_shape[-2] * out_grad_shape[-1])
        a_r = a_r.view(-1, a_r_shape[-1])
        a_d = a_d.view(-1, a_d_shape[-1])

        B = a_r.shape[0]
        alloc = torch.zeros(B * 8, dtype=a_r.dtype, device=a_r.device)
        a_r_grad = alloc[B * 0 : B * 4].view(B, 4)
        a_d_grad = alloc[B * 4 : B * 8].view(B, 4)
        
        (_dqtorch_cuda if a_r.is_cuda else _dqtorch_cpu).dual_quaternion_to_se3_bw(out_grad, a_r, a_d, a_r_grad, a_d_grad, B)
        a_r_grad = a_r_grad.view(a_r_shape)
        a_d_grad = a_d_grad.view(a_d_shape)
        return a_r_grad, a_d_grad

    @staticmethod
    @custom_bwd(device_type="cuda")
    def backward(*args):
        raise NotImplementedError

class _Dual_quaternion_to_se3_fw(Function):
    @staticmethod
    @custom_fwd(cast_inputs=torch.half, device_type="cuda")
    def forward(ctx, a_r: torch.Tensor, a_d: torch.Tensor):
        # a_r: (..., 4), float
        # a_d: (..., 4), float
        a_r = a_r.contiguous()
        a_d = a_d.contiguous()

        a_r_shape = a_r.shape
        a_d_shape = a_d.shape
        assert a_r_shape[-1] == 4, a_r_shape
        assert a_d_shape[-1] == 4, a_d_shape
        assert a_r_shape[:-1] == a_d_shape[:-1], (a_r_shape, a_d_shape)
        assert a_r.is_floating_point() and a_r.dtype == a_d.dtype, (a_r.dtype, a_d.dtype)
        assert a_r.device == a_d.device, (a_r.device, a_d.device)
        a_r = a_r.view(-1, a_r_shape[-1])
        a_d = a_d.view(-1, a_d_shape[-1])

        B = a_r.shape[0]
        alloc = torch.zeros(B * 16, dtype=a_r.dtype, device=a_r.device)
        out = alloc[B * 0 : B * 16].view(B, 4, 4)

        (_dqtorch_cuda if a_r.is_cuda else _dqtorch_cpu).dual_quaternion_to_se3_fw(a_r, a_d, out, B)
        a_r = a_r.view(a_r_shape)
        a_d = a_d.view(a_d_shape)
        out = out.view(a_r_shape[:-1] + (4, 4))
        ctx.save_for_backward(a_r, a_d)
        return out

    @staticmethod
    @custom_bwd(device_type="cuda")
    def backward(ctx, out_grad):
        a_r, a_d = ctx.saved_tensors
        return _Dual_quaternion_to_se3_bw.apply(out_grad, a_r, a_d)

def dual_quaternion_to_se3(a: DualQuaternions) -> torch.Tensor:
    return _Dual_quaternion_to_se3_fw.apply(a[0], a[1])


# ===== SE3 to Dual Quaternion

class _SE3_to_dual_quaternion_bw(Function):
    @staticmethod
    @custom_fwd(cast_inputs=torch.half, device_type="cuda")
    def forward(ctx, out_r_grad: torch.Tensor, out_d_grad: torch.Tensor, a: torch.Tensor):
        # out_r_grad: (..., 4), float
        # out_d_grad: (..., 4), float
        # a: (..., 4, 4), float
        out_r_grad = out_r_grad.contiguous()
        out_d_grad = out_d_grad.contiguous()
        a = a.contiguous()

        out_r_grad_shape = out_r_grad.shape
        out_d_grad_shape = out_d_grad.shape
        a_shape = a.shape
        assert out_r_grad_shape[-1] == 4, out_r_grad_shape
        assert out_d_grad_shape[-1] == 4, out_d_grad_shape
        assert a_shape[-2] == 4 and a_shape[-1] == 4, a_shape
        assert out_r_grad_shape[:-1] == out_d_grad_shape[:-1] == a_shape[:-2], (out_r_grad_shape, out_d_grad_shape, a_shape)
        assert a.is_floating_point() and out_r_grad.dtype == out_d_grad.dtype == a.dtype, (out_r_grad.dtype, out_d_grad.dtype, a.dtype)
        assert out_r_grad.device == out_d_grad.device == a.device, (out_r_grad.device, out_d_grad.device, a.device)
        out_r_grad = out_r_grad.view(-1, out_r_grad_shape[-1])
        out_d_grad = out_d_grad.view(-1, out_d_grad_shape[-1])
        a = a.view(-1, a_shape[-2] * a_shape[-1])

        B = a.shape[0]
        alloc = torch.zeros(B * 16, dtype=a.dtype, device=a.device)
        a_grad = alloc[B * 0 : B * 16].view(B, 4, 4)
        
        (_dqtorch_cuda if a.is_cuda else _dqtorch_cpu).se3_to_dual_quaternion_bw(out_r_grad, out_d_grad, a, a_grad, B)
        a_grad = a_grad.view(a_shape)
        return a_grad

    @staticmethod
    @custom_bwd(device_type="cuda")
    def backward(*args):
        raise NotImplementedError

class _SE3_to_dual_quaternion_fw(Function):
    @staticmethod
    @custom_fwd(cast_inputs=torch.half, device_type="cuda")
    def forward(ctx, a: torch.Tensor):
        # a: (..., 4, 4), float
        a = a.contiguous()

        a_shape = a.shape
        assert a_shape[-2] == 4 and a_shape[-1] == 4, a_shape
        assert a.is_floating_point(), a.dtype
        a = a.view(-1, a_shape[-2] * a_shape[-1])

        B = a.shape[0]
        alloc = torch.zeros(B * 8, dtype=a.dtype, device=a.device)
        out_r = alloc[B * 0 : B * 4].view(B, 4)
        out_d = alloc[B * 4 : B * 8].view(B, 4)

        (_dqtorch_cuda if a.is_cuda else _dqtorch_cpu).se3_to_dual_quaternion_fw(a, out_r, out_d, B)
        a = a.view(a_shape)
        out_r = out_r.view(a_shape[:-2] + (4,))
        out_d = out_d.view(a_shape[:-2] + (4,))
        ctx.save_for_backward(a)
        return out_r, out_d

    @staticmethod
    @custom_bwd(device_type="cuda")
    def backward(ctx, out_r_grad, out_d_grad):
        a, = ctx.saved_tensors
        return _SE3_to_dual_quaternion_bw.apply(out_r_grad, out_d_grad, a)

se3_to_dual_quaternion = _SE3_to_dual_quaternion_fw.apply


# ===== Dual Quaternion Library

def dual_quaternion_linear_blend(w: torch.Tensor, dq_basis: DualQuaternions) -> DualQuaternions:
    dq_r, dq_d = dq_basis
    blended_dq_r = torch.einsum("nk,ktd->ntd", w, dq_r)
    blended_dq_d = torch.einsum("nk,ktd->ntd", w, dq_d)
    q_r_mag_inv = blended_dq_r.norm(p=2, dim=-1, keepdim=True).reciprocal()
    blended_dq_r = blended_dq_r * q_r_mag_inv
    blended_dq_d = blended_dq_d * q_r_mag_inv
    return blended_dq_r, blended_dq_d


def dual_quaternion_linear_blend_batch(w: torch.Tensor, dq_basis: DualQuaternions) -> DualQuaternions:
    dq_r, dq_d = dq_basis
    blended_dq_r = torch.einsum("bnk,bktd->bntd", w, dq_r)
    blended_dq_d = torch.einsum("bnk,bktd->bntd", w, dq_d)
    q_r_mag_inv = blended_dq_r.norm(p=2, dim=-1, keepdim=True).reciprocal()
    blended_dq_r = blended_dq_r * q_r_mag_inv
    blended_dq_d = blended_dq_d * q_r_mag_inv
    return blended_dq_r, blended_dq_d


