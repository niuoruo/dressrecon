import os
import sys
import time
import torch

sys.path.insert(0, os.path.dirname(__file__))
import rotation_conversions

from typing import Tuple
DualQuaternions = Tuple[torch.Tensor, torch.Tensor]
QuaternionTranslation = Tuple[torch.Tensor, torch.Tensor]

from check_func import check_func


# ===== Updated PyTorch3D Quaternion Library

def pytorch3d_quaternion_raw_multiply(a, b):
    if a.shape[-1] == 3:
        a = torch.cat([torch.zeros_like(a[..., :1]), a], dim=-1)
    if b.shape[-1] == 3:
        b = torch.cat([torch.zeros_like(b[..., :1]), b], dim=-1)
    return rotation_conversions.quaternion_raw_multiply(a, b)


def pytorch3d_quaternion_multiply(a, b):
    if a.shape[-1] == 3:
        a = torch.cat([torch.zeros_like(a[..., :1]), a], dim=-1)
    if b.shape[-1] == 3:
        b = torch.cat([torch.zeros_like(b[..., :1]), b], dim=-1)
    return rotation_conversions.quaternion_multiply(a, b)


# ===== PyTorch Dual Quaternion Library

def quaternion_translation_apply(q: torch.Tensor, t: torch.Tensor, point: torch.Tensor) -> torch.Tensor:
    p = rotation_conversions.quaternion_apply(q, point)
    return p + t


def quaternion_translation_inverse(q: torch.Tensor, t: torch.Tensor) -> QuaternionTranslation:
    q_inv = rotation_conversions.quaternion_invert(q)
    t_inv = rotation_conversions.quaternion_apply(q_inv, -t)
    return q_inv, t_inv


def quaternion_translation_to_dual_quaternion(q: torch.Tensor, t: torch.Tensor) -> DualQuaternions:
    q_d = 0.5 * pytorch3d_quaternion_raw_multiply(t, q)
    return q, q_d


def dual_quaternion_to_quaternion_translation(dq: DualQuaternions) -> QuaternionTranslation:
    q_r, q_d = dq
    q_r_inv = rotation_conversions.quaternion_invert(q_r)
    t = 2 * pytorch3d_quaternion_raw_multiply(q_d, q_r_inv)[..., 1:]
    return q_r, t


def dual_quaternion_to_se3(dq: DualQuaternions) -> torch.Tensor:
    q_r, t = dual_quaternion_to_quaternion_translation(dq)
    return quaternion_translation_to_se3(q_r, t)


def se3_to_dual_quaternion(se3: torch.Tensor) -> DualQuaternions:
    q, t = se3_to_quaternion_translation(se3)
    return quaternion_translation_to_dual_quaternion(q, t)


def quaternion_translation_to_se3(q: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
    rmat = rotation_conversions.quaternion_to_matrix(q)
    rt4x4 = torch.cat([rmat, t[..., None]], dim=-1)  # ..., 3, 4
    rt4x4 = torch.cat([rt4x4, torch.zeros_like(rt4x4[..., :1, :])], dim=-2)  # ..., 4, 4
    rt4x4[..., 3, 3] = 1
    return rt4x4


def se3_to_quaternion_translation(se3: torch.Tensor) -> QuaternionTranslation:
    q = rotation_conversions.matrix_to_quaternion(se3[..., :3, :3])
    t = se3[..., :3, 3]
    return q, t


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


def dual_quaternion_apply(dq: DualQuaternions, point: torch.Tensor) -> torch.Tensor:
    q, t = dual_quaternion_to_quaternion_translation(dq)
    return quaternion_translation_apply(q, t, point)


def quaternion_translation_mul(qt1: QuaternionTranslation, qt2: QuaternionTranslation) -> QuaternionTranslation:
    q1, t1 = qt1
    q2, t2 = qt2
    q = pytorch3d_quaternion_raw_multiply(q1, q2)
    t = rotation_conversions.quaternion_apply(q1, t2) + t1
    return q, t


def dual_quaternion_mul(dq1: DualQuaternions, dq2: DualQuaternions) -> DualQuaternions:
    q_r1, q_d1 = dq1
    q_r2, q_d2 = dq2
    r_r = pytorch3d_quaternion_raw_multiply(q_r1, q_r2)
    r_d = pytorch3d_quaternion_raw_multiply(q_r1, q_d2) + pytorch3d_quaternion_raw_multiply(q_d1, q_r2)
    return r_r, r_d


def dual_quaternion_q_conjugate(dq: DualQuaternions) -> DualQuaternions:
    r = rotation_conversions.quaternion_invert(dq[0])
    d = rotation_conversions.quaternion_invert(dq[1])
    return r, d


def dual_quaternion_d_conjugate(dq: DualQuaternions) -> DualQuaternions:
    return (dq[0], -dq[1])


def dual_quaternion_3rd_conjugate(dq: DualQuaternions) -> DualQuaternions:
    return dual_quaternion_d_conjugate(dual_quaternion_q_conjugate(dq))


def dual_quaternion_norm(dq: DualQuaternions) -> DualQuaternions:
    dq_qd = dual_quaternion_q_conjugate(dq)
    return dual_quaternion_mul(dq, dq_qd)


def dual_quaternion_inverse(dq: DualQuaternions) -> DualQuaternions:
    return dual_quaternion_q_conjugate(dq)


if __name__ == "__main__":
    import dqtorch

    torch.random.manual_seed(0)
    test_settings = {
        "float64_cpu": (torch.float64, "cpu"),
        "float64_cuda": (torch.float64, "cuda"),
        "float32_cpu": (torch.float32, "cpu"),
        "float32_cuda": (torch.float32, "cuda"),
    }
    for label, (dtype, device) in test_settings.items():
        print(f"Testing {label}")
        s = 128 if device == "cuda" else 32

        # Create input quaternions with small angles and zero-angles
        w = torch.randn(s, s, s, 4, dtype=dtype, device=device)
        w_small = torch.tensor([1, 0, 0, 0], dtype=dtype, device=device)[None, None, None] + w[:s // 4] / 1e10
        w_small[-1, -1, :, 1:] = 0
        w = torch.cat([w, w_small], dim=0)
        w /= torch.norm(w, dim=-1, keepdim=True)
        w = rotation_conversions.standardize_quaternion(w)
        w /= torch.norm(w, dim=-1, keepdim=True)

        x = torch.randn(s, s, s, 4, dtype=dtype, device=device)
        x_small = torch.tensor([1, 0, 0, 0], dtype=dtype, device=device)[None, None, None] + x[:s // 4] / 1e10
        x_small[-1, -1, :, 1:] = 0
        x = torch.cat([x, x_small], dim=0)
        x /= torch.norm(x, dim=-1, keepdim=True)
        x = rotation_conversions.standardize_quaternion(x)
        x /= torch.norm(x, dim=-1, keepdim=True)

        y = torch.randn(s, s, s, 4, dtype=dtype, device=device)
        y_small = torch.tensor([1, 0, 0, 0], dtype=dtype, device=device)[None, None, None] + y[:s // 4] / 1e10
        y_small[-1, :, -1, 1:] = 0
        y = torch.cat([y, y_small], dim=0)
        y /= torch.norm(y, dim=-1, keepdim=True)
        y = rotation_conversions.standardize_quaternion(y)
        y /= torch.norm(y, dim=-1, keepdim=True)

        z = torch.randn(s, s, s, 4, dtype=dtype, device=device)
        z_small = torch.tensor([1, 0, 0, 0], dtype=dtype, device=device)[None, None, None] + z[:s // 4] / 1e10
        z_small[-1, :, -1, 1:] = 0
        z = torch.cat([z, z_small], dim=0)
        z /= torch.norm(z, dim=-1, keepdim=True)
        z = rotation_conversions.standardize_quaternion(z)
        z /= torch.norm(z, dim=-1, keepdim=True)

        x_mat = rotation_conversions.quaternion_to_matrix(x)
        x_aa = rotation_conversions.quaternion_to_axis_angle(x)

        x_dq = (w, x)
        x_qt = (w, x[..., 1:])
        x_se3 = dual_quaternion_to_se3(x_dq)

        y_dq = (y, z)
        y_qt = (y, z[..., 1:])
        y_se3 = dual_quaternion_to_se3(y_dq)

        # ===== Quaternion Library
        check_func(
            "standardize_quaternion", x, dqtorch.standardize_quaternion,
            rotation_conversions.standardize_quaternion
        )
        check_func(
            "quaternion_raw_multiply", (x, y), dqtorch.quaternion_raw_multiply,
            rotation_conversions.quaternion_raw_multiply
        )
        check_func(
            "quaternion_multiply", (x, y), dqtorch.quaternion_multiply,
            rotation_conversions.quaternion_multiply
        )
        check_func(
            "quaternion_multiply_same", (x, x), dqtorch.quaternion_multiply,
            rotation_conversions.quaternion_multiply
        )
        check_func(
            "quaternion_multiply_4x3", (x, y[..., 1:]), dqtorch.quaternion_multiply,
            pytorch3d_quaternion_multiply
        )
        check_func(
            "quaternion_multiply_3x4", (x[..., 1:], y), dqtorch.quaternion_multiply,
            pytorch3d_quaternion_multiply
        )
        check_func(
            "quaternion_multiply_3x3", (x[..., 1:], y[..., 1:]), dqtorch.quaternion_multiply,
            pytorch3d_quaternion_multiply
        )
        check_func(
            "quaternion_conjugate", x, dqtorch.quaternion_conjugate,
            rotation_conversions.quaternion_invert
        )
        check_func(
            "quaternion_apply", (x, y[..., 1:]), dqtorch.quaternion_apply,
            rotation_conversions.quaternion_apply
        )
        check_func(
            "quaternion_to_matrix", x, dqtorch.quaternion_to_matrix,
            rotation_conversions.quaternion_to_matrix
        )
        check_func(
            "matrix_to_quaternion", x_mat, dqtorch.matrix_to_quaternion,
            rotation_conversions.matrix_to_quaternion
        )
        check_func(
            "axis_angle_to_quaternion", x_aa, dqtorch.axis_angle_to_quaternion,
            rotation_conversions.axis_angle_to_quaternion
        )
        check_func(
            "quaternion_to_axis_angle", x, dqtorch.quaternion_to_axis_angle,
            rotation_conversions.quaternion_to_axis_angle
        )
        check_func(
            "axis_angle_to_matrix", x_aa, dqtorch.axis_angle_to_matrix,
            rotation_conversions.axis_angle_to_matrix
        )
        check_func(
            "matrix_to_axis_angle", x_mat, dqtorch.matrix_to_axis_angle,
            rotation_conversions.matrix_to_axis_angle
        )

        # ===== Dual Quaternion Library
        check_func(
            "quaternion_translation_mul", (x_qt, y_qt), dqtorch.quaternion_translation_mul,
            quaternion_translation_mul
        )
        check_func(
            "quaternion_translation_apply", (x_qt[0], x_qt[1], y_qt[1]), dqtorch.quaternion_translation_apply,
            quaternion_translation_apply
        )
        check_func(
            "quaternion_translation_inverse", (x_qt[0], x_qt[1]), dqtorch.quaternion_translation_inverse,
            quaternion_translation_inverse
        )
        check_func(
            "dual_quaternion_mul", (x_dq, y_dq), dqtorch.dual_quaternion_mul,
            dual_quaternion_mul
        )
        check_func(
            "dual_quaternion_apply", (x_dq, y_qt[1]), dqtorch.dual_quaternion_apply,
            dual_quaternion_apply
        )
        check_func(
            "dual_quaternion_q_conjugate", (x_dq,), dqtorch.dual_quaternion_q_conjugate,
            dual_quaternion_q_conjugate
        )
        check_func(
            "dual_quaternion_d_conjugate", (x_dq,), dqtorch.dual_quaternion_d_conjugate,
            dual_quaternion_d_conjugate
        )
        check_func(
            "dual_quaternion_3rd_conjugate", (x_dq,), dqtorch.dual_quaternion_3rd_conjugate,
            dual_quaternion_3rd_conjugate
        )
        check_func(
            "dual_quaternion_norm", (x_dq,), dqtorch.dual_quaternion_norm,
            dual_quaternion_norm
        )
        check_func(
            "dual_quaternion_inverse", (x_dq,), dqtorch.dual_quaternion_inverse,
            dual_quaternion_inverse
        )
        check_func(
            "quaternion_translation_to_dual_quaternion", (x_qt[0], x_qt[1]),
            dqtorch.quaternion_translation_to_dual_quaternion,
            quaternion_translation_to_dual_quaternion
        )
        check_func(
            "dual_quaternion_to_quaternion_translation", (x_dq,),
            dqtorch.dual_quaternion_to_quaternion_translation,
            dual_quaternion_to_quaternion_translation
        )
        check_func(
            "quaternion_translation_to_se3", (x_qt[0], x_qt[1]), dqtorch.quaternion_translation_to_se3,
            quaternion_translation_to_se3
        )
        check_func(
            "se3_to_quaternion_translation", x_se3, dqtorch.se3_to_quaternion_translation,
            se3_to_quaternion_translation
        )
        check_func(
            "dual_quaternion_to_se3", (x_dq,), dqtorch.dual_quaternion_to_se3,
            dual_quaternion_to_se3
        )
        check_func(
            "se3_to_dual_quaternion", x_se3, dqtorch.se3_to_dual_quaternion,
            se3_to_dual_quaternion
        )
