#include "dqtorch.h"
#include <iostream>

// #define gpuErrorCheck(ans) { gpuAssert((ans), __FILE__, __LINE__); }
#define gpuErrorCheck(ans)
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
   if (code != cudaSuccess) 
   {
      fprintf(stderr, "GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort) exit(code);
   }
}

// ===== Device: Standardize Quaternion Forward

template <typename scalar_t>
__device__ void _standardize_quaternion_fw(const scalar_t *__restrict__ a, scalar_t *__restrict__ out) {
    if (a[0] < 0) {
        out[0] = -a[0];
        out[1] = -a[1];
        out[2] = -a[2];
        out[3] = -a[3];
    } else {
        out[0] = a[0];
        out[1] = a[1];
        out[2] = a[2];
        out[3] = a[3];
    }
}

// ===== Device: Standardize Quaternion Backward

template <typename scalar_t>
__device__ void _standardize_quaternion_bw(const scalar_t *__restrict__ out_grad, const scalar_t *__restrict__ a,
                                           scalar_t *__restrict__ a_grad) {
    if (a[0] < 0) {
        a_grad[0] += -out_grad[0];
        a_grad[1] += -out_grad[1];
        a_grad[2] += -out_grad[2];
        a_grad[3] += -out_grad[3];
    } else {
        a_grad[0] += out_grad[0];
        a_grad[1] += out_grad[1];
        a_grad[2] += out_grad[2];
        a_grad[3] += out_grad[3];
    }
}

// ===== Device: Quaternion Raw Multiply Forward

template <typename scalar_t>
__device__ void _quaternion_raw_multiply_fw(const scalar_t *__restrict__ a, const scalar_t *__restrict__ b,
                                            scalar_t *__restrict__ out, const uint32_t Da, const uint32_t Db, const uint32_t Do) {
    scalar_t aw, ax, ay, az;
    if (Da == 3) {
        aw = 0.;
        ax = a[0];
        ay = a[1];
        az = a[2];
    } else {
        aw = a[0];
        ax = a[1];
        ay = a[2];
        az = a[3];
    }

    scalar_t bw, bx, by, bz;
    if (Db == 3) {
        bw = 0.;
        bx = b[0];
        by = b[1];
        bz = b[2];
    } else {
        bw = b[0];
        bx = b[1];
        by = b[2];
        bz = b[3];
    }

    if (Do == 3) {
        out[0] = aw * bx + ax * bw + ay * bz - az * by;
        out[1] = aw * by - ax * bz + ay * bw + az * bx;
        out[2] = aw * bz + ax * by - ay * bx + az * bw;
    } else {
        out[0] = aw * bw - ax * bx - ay * by - az * bz;
        out[1] = aw * bx + ax * bw + ay * bz - az * by;
        out[2] = aw * by - ax * bz + ay * bw + az * bx;
        out[3] = aw * bz + ax * by - ay * bx + az * bw;
    }
}

// ===== Device: Quaternion Raw Multiply Backward

template <typename scalar_t>
__device__ void _quaternion_raw_multiply_bw(const scalar_t *__restrict__ out_grad, const scalar_t *__restrict__ a,
                                            const scalar_t *__restrict__ b, scalar_t *__restrict__ a_grad,
                                            scalar_t *__restrict__ b_grad, const uint32_t Da,
                                            const uint32_t Db, const uint32_t Do) {
    scalar_t aw, ax, ay, az;
    if (Da == 3) {
        aw = 0.;
        ax = a[0];
        ay = a[1];
        az = a[2];
    } else {
        aw = a[0];
        ax = a[1];
        ay = a[2];
        az = a[3];
    }

    scalar_t bw, bx, by, bz;
    if (Db == 3) {
        bw = 0.;
        bx = b[0];
        by = b[1];
        bz = b[2];
    } else {
        bw = b[0];
        bx = b[1];
        by = b[2];
        bz = b[3];
    }

    scalar_t dw, dx, dy, dz;
    if (Do == 3) {
        dw = 0.;
        dx = out_grad[0];
        dy = out_grad[1];
        dz = out_grad[2];
    } else {
        dw = out_grad[0];
        dx = out_grad[1];
        dy = out_grad[2];
        dz = out_grad[3];
    }

    if (Da == 3) {
        a_grad[0] += -bx * dw + bw * dx - bz * dy + by * dz;
        a_grad[1] += -by * dw + bz * dx + bw * dy - bx * dz;
        a_grad[2] += -bz * dw - by * dx + bx * dy + bw * dz;
    } else {
        a_grad[0] += bw * dw + bx * dx + by * dy + bz * dz;
        a_grad[1] += -bx * dw + bw * dx - bz * dy + by * dz;
        a_grad[2] += -by * dw + bz * dx + bw * dy - bx * dz;
        a_grad[3] += -bz * dw - by * dx + bx * dy + bw * dz;
    }

    if (Db == 3) {
        b_grad[0] += -ax * dw + aw * dx + az * dy - ay * dz;
        b_grad[1] += -ay * dw - az * dx + aw * dy + ax * dz;
        b_grad[2] += -az * dw + ay * dx - ax * dy + aw * dz;
    } else {
        b_grad[0] += aw * dw + ax * dx + ay * dy + az * dz;
        b_grad[1] += -ax * dw + aw * dx + az * dy - ay * dz;
        b_grad[2] += -ay * dw - az * dx + aw * dy + ax * dz;
        b_grad[3] += -az * dw + ay * dx - ax * dy + aw * dz;
    }
}

// ===== Device: Quaternion Multiply Forward

template <typename scalar_t>
__device__ void _quaternion_multiply_fw(const scalar_t *__restrict__ a, const scalar_t *__restrict__ b, scalar_t *__restrict__ out, const uint32_t Da, const uint32_t Db) {
    scalar_t raw[4] = {};
    _quaternion_raw_multiply_fw<scalar_t>(a, b, raw, Da, Db, 4);
    _standardize_quaternion_fw<scalar_t>(raw, out);
}

// ===== Device: Quaternion Multiply Backward

template <typename scalar_t>
__device__ void _quaternion_multiply_bw(const scalar_t *__restrict__ out_grad, const scalar_t *__restrict__ a, const scalar_t *__restrict__ b, scalar_t *__restrict__ a_grad, scalar_t *__restrict__ b_grad, const uint32_t Da, const uint32_t Db) {
    scalar_t raw[4] = {};
    _quaternion_raw_multiply_fw<scalar_t>(a, b, raw, Da, Db, 4);

    scalar_t raw_grad[4] = {};
    _standardize_quaternion_bw<scalar_t>(out_grad, raw, raw_grad);
    _quaternion_raw_multiply_bw<scalar_t>(raw_grad, a, b, a_grad, b_grad, Da, Db, 4);
}

// ===== Device: Quaternion Conjugate Forward

template <typename scalar_t>
__device__ void _quaternion_conjugate_fw(const scalar_t *__restrict__ a, scalar_t *__restrict__ out) {
    out[0] = a[0];
    out[1] = -a[1];
    out[2] = -a[2];
    out[3] = -a[3];
}

// ===== Device: Quaternion Conjugate Backward

template <typename scalar_t>
__device__ void _quaternion_conjugate_bw(const scalar_t *__restrict__ out_grad, const scalar_t *__restrict__ a, scalar_t *__restrict__ a_grad) {
    a_grad[0] += out_grad[0];
    a_grad[1] += -out_grad[1];
    a_grad[2] += -out_grad[2];
    a_grad[3] += -out_grad[3];
}

// ===== Device: Quaternion Apply Forward

template <typename scalar_t>
__device__ void _quaternion_apply_fw(const scalar_t *__restrict__ a, const scalar_t *__restrict__ b, scalar_t *__restrict__ out) {
    scalar_t raw[4] = {};
    scalar_t a_inv[4] = {};
    _quaternion_raw_multiply_fw<scalar_t>(a, b, raw, 4, 3, 4);
    _quaternion_conjugate_fw<scalar_t>(a, a_inv);
    _quaternion_raw_multiply_fw<scalar_t>(raw, a_inv, out, 4, 4, 3);
}

// ===== Device: Quaternion Apply Backward

template <typename scalar_t>
__device__ void _quaternion_apply_bw(const scalar_t *__restrict__ out_grad, const scalar_t *__restrict__ a, const scalar_t *__restrict__ b, scalar_t *__restrict__ a_grad, scalar_t *__restrict__ b_grad) {
    scalar_t raw[4] = {};
    scalar_t a_inv[4] = {};
    _quaternion_raw_multiply_fw<scalar_t>(a, b, raw, 4, 3, 4);
    _quaternion_conjugate_fw<scalar_t>(a, a_inv);

    scalar_t raw_grad[4] = {};
    scalar_t a_inv_grad[4] = {};
    _quaternion_raw_multiply_bw<scalar_t>(out_grad, raw, a_inv, raw_grad, a_inv_grad, 4, 4, 3);
    _quaternion_conjugate_bw<scalar_t>(a_inv_grad, a, a_grad);
    _quaternion_raw_multiply_bw<scalar_t>(raw_grad, a, b, a_grad, b_grad, 4, 3, 4);
}

// ===== Device: Quaternion to Matrix Forward

template <typename scalar_t>
__device__ void _quaternion_to_matrix_fw(const scalar_t *__restrict__ a, scalar_t *__restrict__ out) {
    scalar_t w = a[0];
    scalar_t x = a[1];
    scalar_t y = a[2];
    scalar_t z = a[3];

    scalar_t ww = w * w;
    scalar_t xx = x * x;
    scalar_t yy = y * y;
    scalar_t zz = z * z;
    scalar_t s = 2. / (ww + xx + yy + zz);

    scalar_t xy = x * y;
    scalar_t xz = x * z;
    scalar_t yz = y * z;
    scalar_t xw = x * w;
    scalar_t yw = y * w;
    scalar_t zw = z * w;

    out[0] = 1. - s * (yy + zz);
    out[1] = s * (xy - zw);
    out[2] = s * (xz + yw);
    out[3] = s * (xy + zw);
    out[4] = 1. - s * (xx + zz);
    out[5] = s * (yz - xw);
    out[6] = s * (xz - yw);
    out[7] = s * (yz + xw);
    out[8] = 1. - s * (xx + yy);
}

// ===== Device: Quaternion to Matrix Backward

template <typename scalar_t>
__device__ void _quaternion_to_matrix_bw(const scalar_t *__restrict__ out_grad, const scalar_t *__restrict__ a,
                                         scalar_t *__restrict__ a_grad) {
    scalar_t w = a[0];
    scalar_t x = a[1];
    scalar_t y = a[2];
    scalar_t z = a[3];

    scalar_t ww = w * w;
    scalar_t xx = x * x;
    scalar_t yy = y * y;
    scalar_t zz = z * z;
    scalar_t s = 2. / (ww + xx + yy + zz);

    scalar_t ss = s * s;
    scalar_t ds_dw = -w * ss;
    scalar_t ds_dx = -x * ss;
    scalar_t ds_dy = -y * ss;
    scalar_t ds_dz = -z * ss;

    scalar_t xy = x * y;
    scalar_t xz = x * z;
    scalar_t yz = y * z;
    scalar_t xw = x * w;
    scalar_t yw = y * w;
    scalar_t zw = z * w;

    scalar_t sw = s * w;
    scalar_t sx = s * x;
    scalar_t sy = s * y;
    scalar_t sz = s * z;

    scalar_t d00 = out_grad[0];
    scalar_t d01 = out_grad[1];
    scalar_t d02 = out_grad[2];
    scalar_t d10 = out_grad[3];
    scalar_t d11 = out_grad[4];
    scalar_t d12 = out_grad[5];
    scalar_t d20 = out_grad[6];
    scalar_t d21 = out_grad[7];
    scalar_t d22 = out_grad[8];

    scalar_t tmp = -(yy + zz) * d00 + (xy - zw) * d01 + (xz + yw) * d02 + (xy + zw) * d10 - (xx + zz) * d11 +
                   (yz - xw) * d12 + (xz - yw) * d20 + (yz + xw) * d21 - (xx + yy) * d22;

    a_grad[0] += ds_dw * tmp + sx * (d21 - d12) + sy * (d02 - d20) + sz * (d10 - d01);
    a_grad[1] += ds_dx * tmp + sw * (d21 - d12) - 2.f * sx * (d11 + d22) + sy * (d10 + d01) + sz * (d20 + d02);
    a_grad[2] += ds_dy * tmp + sw * (d02 - d20) + sx * (d10 + d01) - 2.f * sy * (d00 + d22) + sz * (d21 + d12);
    a_grad[3] += ds_dz * tmp + sw * (d10 - d01) + sx * (d20 + d02) + sy * (d21 + d12) - 2.f * sz * (d00 + d11);
}

// ===== Device: Matrix to Raw Quaternion Forward

template <typename scalar_t>
__device__ void _matrix_to_raw_quaternion_fw(const scalar_t *__restrict__ a, scalar_t *__restrict__ out) {
    scalar_t a00 = a[0];
    scalar_t a01 = a[1];
    scalar_t a02 = a[2];
    scalar_t a10 = a[3];
    scalar_t a11 = a[4];
    scalar_t a12 = a[5];
    scalar_t a20 = a[6];
    scalar_t a21 = a[7];
    scalar_t a22 = a[8];

    scalar_t two_sww = 1. + a00 + a11 + a22;
    scalar_t two_sxx = 1. + a00 - a11 - a22;
    scalar_t two_syy = 1. - a00 + a11 - a22;
    scalar_t two_szz = 1. - a00 - a11 + a22;

    bool compare_wx = two_sww > two_sxx;
    scalar_t best_wx = compare_wx ? two_sww : two_sxx;
    bool compare_yz = two_syy > two_szz;
    scalar_t best_yz = compare_yz ? two_syy : two_szz;

    // Use the quaternion candidate with largest denominator for numerical stability
    if (best_wx > best_yz) {
        if (compare_wx) {
            // two_sww has largest denominator
            scalar_t four_w = 2. * sqrt(two_sww);
            scalar_t inv_four_w = 1. / four_w;
            out[0] = 0.25 * four_w;
            scalar_t two_sxw = a21 - a12;
            out[1] = two_sxw * inv_four_w;
            scalar_t two_syw = a02 - a20;
            out[2] = two_syw * inv_four_w;
            scalar_t two_szw = a10 - a01;
            out[3] = two_szw * inv_four_w;
        } else {
            // two_sxx has largest denominator
            scalar_t four_x = 2. * sqrt(two_sxx);
            scalar_t inv_four_x = 1. / four_x;
            out[1] = 0.25 * four_x;
            scalar_t two_swx = a21 - a12;
            out[0] = two_swx * inv_four_x;
            scalar_t two_syx = a10 + a01;
            out[2] = two_syx * inv_four_x;
            scalar_t two_szx = a02 + a20;
            out[3] = two_szx * inv_four_x;
        }
    } else {
        if (compare_yz) {
            // two_syy has largest denominator
            scalar_t four_y = 2. * sqrt(two_syy);
            scalar_t inv_four_y = 1. / four_y;
            out[2] = 0.25 * four_y;
            scalar_t two_swy = a02 - a20;
            out[0] = two_swy * inv_four_y;
            scalar_t two_sxy = a10 + a01;
            out[1] = two_sxy * inv_four_y;
            scalar_t two_szy = a12 + a21;
            out[3] = two_szy * inv_four_y;
        } else {
            // two_szz has largest denominator
            scalar_t four_z = 2. * sqrt(two_szz);
            scalar_t inv_four_z = 1. / four_z;
            out[3] = 0.25 * four_z;
            scalar_t two_swz = a10 - a01;
            out[0] = two_swz * inv_four_z;
            scalar_t two_sxz = a02 + a20;
            out[1] = two_sxz * inv_four_z;
            scalar_t two_syz = a12 + a21;
            out[2] = two_syz * inv_four_z;
        }
    }
}

// ===== Device: Matrix to Raw Quaternion Backward

template <typename scalar_t>
__device__ void _matrix_to_raw_quaternion_bw(const scalar_t *__restrict__ out_grad, const scalar_t *__restrict__ a,
                                         scalar_t *__restrict__ a_grad) {
    scalar_t dw = out_grad[0];
    scalar_t dx = out_grad[1];
    scalar_t dy = out_grad[2];
    scalar_t dz = out_grad[3];

    scalar_t a00 = a[0];
    scalar_t a01 = a[1];
    scalar_t a02 = a[2];
    scalar_t a10 = a[3];
    scalar_t a11 = a[4];
    scalar_t a12 = a[5];
    scalar_t a20 = a[6];
    scalar_t a21 = a[7];
    scalar_t a22 = a[8];

    scalar_t two_sww = 1. + a00 + a11 + a22;
    scalar_t two_sxx = 1. + a00 - a11 - a22;
    scalar_t two_syy = 1. - a00 + a11 - a22;
    scalar_t two_szz = 1. - a00 - a11 + a22;

    bool compare_wx = two_sww > two_sxx;
    scalar_t best_wx = compare_wx ? two_sww : two_sxx;
    bool compare_yz = two_syy > two_szz;
    scalar_t best_yz = compare_yz ? two_syy : two_szz;

    // Use the quaternion candidate with largest denominator for numerical stability
    if (best_wx > best_yz) {
        if (compare_wx) {
            // two_sww has largest denominator
            scalar_t four_w = 2. * sqrt(two_sww);
            scalar_t inv_four_w = 1. / four_w;
            scalar_t inv_four_w_cubed = inv_four_w * inv_four_w * inv_four_w;
            scalar_t diag = 0.5 * inv_four_w * dw - 2. * inv_four_w_cubed *
                            ((a21 - a12) * dx + (a02 - a20) * dy + (a10 - a01) * dz);

            a_grad[0] += diag;
            a_grad[1] += -inv_four_w * dz;
            a_grad[2] += inv_four_w * dy;
            a_grad[3] += inv_four_w * dz;
            a_grad[4] += diag;
            a_grad[5] += -inv_four_w * dx;
            a_grad[6] += -inv_four_w * dy;
            a_grad[7] += inv_four_w * dx;
            a_grad[8] += diag;
        } else {
            // two_sxx has largest denominator
            scalar_t four_x = 2. * sqrt(two_sxx);
            scalar_t inv_four_x = 1. / four_x;
            scalar_t inv_four_x_cubed = inv_four_x * inv_four_x * inv_four_x;
            scalar_t diag = 0.5 * inv_four_x * dx - 2. * inv_four_x_cubed *
                            ((a21 - a12) * dw + (a10 + a01) * dy + (a02 + a20) * dz);

            a_grad[0] += diag;
            a_grad[1] += inv_four_x * dy;
            a_grad[2] += inv_four_x * dz;
            a_grad[3] += inv_four_x * dy;
            a_grad[4] += -diag;
            a_grad[5] += -inv_four_x * dw;
            a_grad[6] += inv_four_x * dz;
            a_grad[7] += inv_four_x * dw;
            a_grad[8] += -diag;
        }
    } else {
        if (compare_yz) {
            // two_syy has largest denominator
            scalar_t four_y = 2. * sqrt(two_syy);
            scalar_t inv_four_y = 1. / four_y;
            scalar_t inv_four_y_cubed = inv_four_y * inv_four_y * inv_four_y;
            scalar_t diag = 0.5 * inv_four_y * dy - 2. * inv_four_y_cubed *
                            ((a02 - a20) * dw + (a10 + a01) * dx + (a12 + a21) * dz);

            a_grad[0] += -diag;
            a_grad[1] += inv_four_y * dx;
            a_grad[2] += inv_four_y * dw;
            a_grad[3] += inv_four_y * dx;
            a_grad[4] += diag;
            a_grad[5] += inv_four_y * dz;
            a_grad[6] += -inv_four_y * dw;
            a_grad[7] += inv_four_y * dz;
            a_grad[8] += -diag;
        } else {
            // two_szz has largest denominator
            scalar_t four_z = 2. * sqrt(two_szz);
            scalar_t inv_four_z = 1. / four_z;
            scalar_t inv_four_z_cubed = inv_four_z * inv_four_z * inv_four_z;
            scalar_t diag = 0.5 * inv_four_z * dz - 2. * inv_four_z_cubed *
                            ((a10 - a01) * dw + (a02 + a20) * dx + (a12 + a21) * dy);

            a_grad[0] += -diag;
            a_grad[1] += -inv_four_z * dw;
            a_grad[2] += inv_four_z * dx;
            a_grad[3] += inv_four_z * dw;
            a_grad[4] += -diag;
            a_grad[5] += inv_four_z * dy;
            a_grad[6] += inv_four_z * dx;
            a_grad[7] += inv_four_z * dy;
            a_grad[8] += diag;
        }
    }
}

// ===== Device: Matrix to Quaternion Forward

template <typename scalar_t>
__device__ void _matrix_to_quaternion_fw(const scalar_t *__restrict__ a, scalar_t *__restrict__ out) {
    scalar_t raw[4] = {};
    _matrix_to_raw_quaternion_fw<scalar_t>(a, raw);
    _standardize_quaternion_fw<scalar_t>(raw, out);
}

// ===== Device: Matrix to Quaternion Backward

template <typename scalar_t>
__device__ void _matrix_to_quaternion_bw(const scalar_t *__restrict__ out_grad, const scalar_t *__restrict__ a, scalar_t *__restrict__ a_grad) {
    scalar_t raw[4] = {};
    _matrix_to_raw_quaternion_fw<scalar_t>(a, raw);

    scalar_t raw_grad[4] = {};
    _standardize_quaternion_bw<scalar_t>(out_grad, raw, raw_grad);
    _matrix_to_raw_quaternion_bw<scalar_t>(raw_grad, a, a_grad);
}

// ===== Device: Axis Angle to Quaternion Forward

template <typename scalar_t>
__device__ void _axis_angle_to_quaternion_fw(const scalar_t *__restrict__ a, scalar_t *__restrict__ out) {
    scalar_t x = a[0];
    scalar_t y = a[1];
    scalar_t z = a[2];

    scalar_t angle_sq = x * x + y * y + z * z;
    scalar_t angle = sqrt(angle_sq);
    scalar_t half_angle = 0.5 * angle;
    scalar_t sin_half_angle_over_angle;
    if (abs(angle) < 1e-6) {
        // For x small, sin(x/2) is about x/2 - (x/2)^3/6
        // So sin(x/2)/x is about 1/2 - (x*x)/48
        sin_half_angle_over_angle = 0.5 - angle_sq / 48;
    } else {
        sin_half_angle_over_angle = sin(half_angle) / angle;
    }

    out[0] = cos(half_angle);
    out[1] = x * sin_half_angle_over_angle;
    out[2] = y * sin_half_angle_over_angle;
    out[3] = z * sin_half_angle_over_angle;
}

// ===== Device: Axis Angle to Quaternion Backward

template <typename scalar_t>
__device__ void _axis_angle_to_quaternion_bw(const scalar_t *__restrict__ out_grad, const scalar_t *__restrict__ a, scalar_t *__restrict__ a_grad) {
    scalar_t dw = out_grad[0];
    scalar_t dx = out_grad[1];
    scalar_t dy = out_grad[2];
    scalar_t dz = out_grad[3];

    scalar_t x = a[0];
    scalar_t y = a[1];
    scalar_t z = a[2];

    scalar_t angle_sq = x * x + y * y + z * z;
    scalar_t angle = sqrt(angle_sq);
    scalar_t sin_half_angle_over_angle;
    scalar_t tmp;
    if (abs(angle) < 1e-6) {
        // For x small, sin(x/2) is about x/2 - (x/2)^3/6
        // So sin(x/2)/x is about 1/2 - (x*x)/48
        sin_half_angle_over_angle = 0.5 - angle_sq / 48.;
        tmp = -1. / 24.;
    } else {
        scalar_t half_angle = 0.5 * angle;
        scalar_t inv_angle = 1. / angle;
        sin_half_angle_over_angle = sin(half_angle) * inv_angle;
        tmp = (0.5 * cos(half_angle) - sin_half_angle_over_angle) * inv_angle * inv_angle;
    }
    scalar_t tmp_ad = tmp * (x * dx + y * dy + z * dz);
    a_grad[0] += sin_half_angle_over_angle * (-0.5 * x * dw + dx) + x * tmp_ad;
    a_grad[1] += sin_half_angle_over_angle * (-0.5 * y * dw + dy) + y * tmp_ad;
    a_grad[2] += sin_half_angle_over_angle * (-0.5 * z * dw + dz) + z * tmp_ad;
}

// ===== Device: Quaternion to Axis Angle Forward

template <typename scalar_t>
__device__ void _quaternion_to_axis_angle_fw(const scalar_t *__restrict__ a, scalar_t *__restrict__ out) {
    scalar_t w = a[0];
    scalar_t x = a[1];
    scalar_t y = a[2];
    scalar_t z = a[3];

    scalar_t n = sqrt(x * x + y * y + z * z);
    scalar_t half_angle = atan2(n, w);
    scalar_t angle = 2. * half_angle;
    scalar_t angle_over_sin_half_angle;
    if (abs(angle) < 1e-6) {
        // For x small, sin(x/2) is about x/2 - (x/2)^3/6
        // So sin(x/2)/x is about 1/2 - (x*x)/48
        angle_over_sin_half_angle = 48. / (24. - angle * angle);
    } else {
        angle_over_sin_half_angle = angle / sin(half_angle);
    }
    out[0] = x * angle_over_sin_half_angle;
    out[1] = y * angle_over_sin_half_angle;
    out[2] = z * angle_over_sin_half_angle;
}

// ===== Device: Quaternion to Axis Angle Backward

template <typename scalar_t>
__device__ void _quaternion_to_axis_angle_bw(const scalar_t *__restrict__ out_grad, const scalar_t *__restrict__ a, scalar_t *__restrict__ a_grad) {
    scalar_t dx = out_grad[0];
    scalar_t dy = out_grad[1];
    scalar_t dz = out_grad[2];

    scalar_t w = a[0];
    scalar_t x = a[1];
    scalar_t y = a[2];
    scalar_t z = a[3];

    scalar_t n = sqrt(x * x + y * y + z * z);
    scalar_t ad = x * dx + y * dy + z * dz;
    scalar_t half_angle = atan2(n, w);
    scalar_t angle = 2. * half_angle;
    scalar_t angle_over_sin_half_angle;
    scalar_t tmp1;
    if (abs(angle) < 1e-6) {
        angle_over_sin_half_angle = 48. / (24. - angle * angle);
        tmp1 = angle * ad * angle_over_sin_half_angle * angle_over_sin_half_angle / 12.;
    } else {
        scalar_t sin_half_angle = sin(half_angle);
        angle_over_sin_half_angle = angle / sin_half_angle;
        tmp1 = (2. * sin_half_angle - angle * cos(half_angle)) * ad / (sin_half_angle * sin_half_angle);
    }
    a_grad[0] += -tmp1 * n;
    scalar_t tmp2 = (n == static_cast<scalar_t>(0.)) ? static_cast<scalar_t>(0.) : (tmp1 * w / n);
    a_grad[1] += angle_over_sin_half_angle * dx + x * tmp2;
    a_grad[2] += angle_over_sin_half_angle * dy + y * tmp2;
    a_grad[3] += angle_over_sin_half_angle * dz + z * tmp2;
}

// ===== Device: Axis Angle to Matrix Forward

template <typename scalar_t>
__device__ void _axis_angle_to_matrix_fw(const scalar_t *__restrict__ a, scalar_t *__restrict__ out) {
    scalar_t raw[4] = {};
    _axis_angle_to_quaternion_fw<scalar_t>(a, raw);
    _quaternion_to_matrix_fw<scalar_t>(raw, out);
}

// ===== Device: Axis Angle to Matrix Backward

template <typename scalar_t>
__device__ void _axis_angle_to_matrix_bw(const scalar_t *__restrict__ out_grad, const scalar_t *__restrict__ a, scalar_t *__restrict__ a_grad) {
    scalar_t raw[4] = {};
    _axis_angle_to_quaternion_fw<scalar_t>(a, raw);

    scalar_t raw_grad[4] = {};
    _quaternion_to_matrix_bw<scalar_t>(out_grad, raw, raw_grad);
    _axis_angle_to_quaternion_bw<scalar_t>(raw_grad, a, a_grad);
}

// ===== Device: Matrix to Axis Angle Forward

template <typename scalar_t>
__device__ void _matrix_to_axis_angle_fw(const scalar_t *__restrict__ a, scalar_t *__restrict__ out) {
    scalar_t raw[4] = {};
    _matrix_to_quaternion_fw<scalar_t>(a, raw);
    _quaternion_to_axis_angle_fw<scalar_t>(raw, out);
}

// ===== Device: Matrix to Axis Angle Backward

template <typename scalar_t>
__device__ void _matrix_to_axis_angle_bw(const scalar_t *__restrict__ out_grad, const scalar_t *__restrict__ a, scalar_t *__restrict__ a_grad) {
    scalar_t raw[4] = {};
    _matrix_to_quaternion_fw<scalar_t>(a, raw);

    scalar_t raw_grad[4] = {};
    _quaternion_to_axis_angle_bw<scalar_t>(out_grad, raw, raw_grad);
    _matrix_to_quaternion_bw<scalar_t>(raw_grad, a, a_grad);
}

// ===== Device: Quaternion Translation Multiply Forward

template <typename scalar_t>
__device__ void _quaternion_translation_mul_fw(const scalar_t *__restrict__ a_q, const scalar_t *__restrict__ a_t, const scalar_t *__restrict__ b_q, const scalar_t *__restrict__ b_t, scalar_t *__restrict__ out_q, scalar_t *__restrict__ out_t) {
    _quaternion_raw_multiply_fw<scalar_t>(a_q, b_q, out_q, 4, 4, 4);
    _quaternion_apply_fw<scalar_t>(a_q, b_t, out_t);
    out_t[0] += a_t[0];
    out_t[1] += a_t[1];
    out_t[2] += a_t[2];
}

// ===== Device: Quaternion Translation Multiply Backward

template <typename scalar_t>
__device__ void _quaternion_translation_mul_bw(const scalar_t *__restrict__ out_q_grad, const scalar_t *__restrict__ out_t_grad, const scalar_t *__restrict__ a_q, const scalar_t *__restrict__ a_t, const scalar_t *__restrict__ b_q, const scalar_t *__restrict__ b_t, scalar_t *__restrict__ a_q_grad, scalar_t *__restrict__ a_t_grad, scalar_t *__restrict__ b_q_grad, scalar_t *__restrict__ b_t_grad) {
    a_t_grad[0] += out_t_grad[0];
    a_t_grad[1] += out_t_grad[1];
    a_t_grad[2] += out_t_grad[2];
    _quaternion_apply_bw<scalar_t>(out_t_grad, a_q, b_t, a_q_grad, b_t_grad);
    _quaternion_raw_multiply_bw<scalar_t>(out_q_grad, a_q, b_q, a_q_grad, b_q_grad, 4, 4, 4);
}

// ===== Device: Quaternion Translation Apply Forward

template <typename scalar_t>
__device__ void _quaternion_translation_apply_fw(const scalar_t *__restrict__ a_q, const scalar_t *__restrict__ a_t, const scalar_t *__restrict__ b, scalar_t *__restrict__ out) {
    _quaternion_apply_fw<scalar_t>(a_q, b, out);
    out[0] += a_t[0];
    out[1] += a_t[1];
    out[2] += a_t[2];
}

// ===== Device: Quaternion Translation Apply Backward

template <typename scalar_t>
__device__ void _quaternion_translation_apply_bw(const scalar_t *__restrict__ out_grad, const scalar_t *__restrict__ a_q, const scalar_t *__restrict__ a_t, const scalar_t *__restrict__ b, scalar_t *__restrict__ a_q_grad, scalar_t *__restrict__ a_t_grad, scalar_t *__restrict__ b_grad) {
    a_t_grad[0] += out_grad[0];
    a_t_grad[1] += out_grad[1];
    a_t_grad[2] += out_grad[2];
    _quaternion_apply_bw<scalar_t>(out_grad, a_q, b, a_q_grad, b_grad);
}

// ===== Device: Quaternion Translation Inverse Forward

template <typename scalar_t>
__device__ void _quaternion_translation_inverse_fw(const scalar_t *__restrict__ a_q, const scalar_t *__restrict__ a_t, scalar_t *__restrict__ out_q, scalar_t *__restrict__ out_t) {
    scalar_t a_t_neg[3] = {};
    a_t_neg[0] = -a_t[0];
    a_t_neg[1] = -a_t[1];
    a_t_neg[2] = -a_t[2];
    _quaternion_conjugate_fw<scalar_t>(a_q, out_q);
    _quaternion_apply_fw<scalar_t>(out_q, a_t_neg, out_t);
}

// ===== Device: Quaternion Translation Inverse Backward

template <typename scalar_t>
__device__ void _quaternion_translation_inverse_bw(const scalar_t *__restrict__ out_q_grad, const scalar_t *__restrict__ out_t_grad, const scalar_t *__restrict__ a_q, const scalar_t *__restrict__ a_t, scalar_t *__restrict__ a_q_grad, scalar_t *__restrict__ a_t_grad) {
    scalar_t out_q[4] = {};
    scalar_t a_t_neg[3] = {};
    _quaternion_conjugate_fw<scalar_t>(a_q, out_q);
    a_t_neg[0] = -a_t[0];
    a_t_neg[1] = -a_t[1];
    a_t_neg[2] = -a_t[2];

    scalar_t out_q_grad_[4] = {};
    scalar_t a_t_neg_grad[3] = {};
    out_q_grad_[0] += out_q_grad[0];
    out_q_grad_[1] += out_q_grad[1];
    out_q_grad_[2] += out_q_grad[2];
    out_q_grad_[3] += out_q_grad[3];
    _quaternion_apply_bw<scalar_t>(out_t_grad, out_q, a_t_neg, out_q_grad_, a_t_neg_grad);
    _quaternion_conjugate_bw<scalar_t>(out_q_grad_, a_q, a_q_grad);
    a_t_grad[0] += -a_t_neg_grad[0];
    a_t_grad[1] += -a_t_neg_grad[1];
    a_t_grad[2] += -a_t_neg_grad[2];
}

// ===== Device: Dual Quaternion Multiply Forward

template <typename scalar_t>
__device__ void _dual_quaternion_mul_fw(const scalar_t *__restrict__ a_r, const scalar_t *__restrict__ a_d, const scalar_t *__restrict__ b_r, const scalar_t *__restrict__ b_d, scalar_t *__restrict__ out_r, scalar_t *__restrict__ out_d) {
    scalar_t out_d2[4] = {};
    _quaternion_raw_multiply_fw<scalar_t>(a_r, b_r, out_r, 4, 4, 4);
    _quaternion_raw_multiply_fw<scalar_t>(a_r, b_d, out_d, 4, 4, 4);
    _quaternion_raw_multiply_fw<scalar_t>(a_d, b_r, out_d2, 4, 4, 4);
    out_d[0] += out_d2[0];
    out_d[1] += out_d2[1];
    out_d[2] += out_d2[2];
    out_d[3] += out_d2[3];
}

// ===== Device: Dual Quaternion Multiply Backward

template <typename scalar_t>
__device__ void _dual_quaternion_mul_bw(const scalar_t *__restrict__ out_r_grad, const scalar_t *__restrict__ out_d_grad, const scalar_t *__restrict__ a_r, const scalar_t *__restrict__ a_d, const scalar_t *__restrict__ b_r, const scalar_t *__restrict__ b_d, scalar_t *__restrict__ a_r_grad, scalar_t *__restrict__ a_d_grad, scalar_t *__restrict__ b_r_grad, scalar_t *__restrict__ b_d_grad) {
    _quaternion_raw_multiply_bw<scalar_t>(out_d_grad, a_r, b_d, a_r_grad, b_d_grad, 4, 4, 4);
    _quaternion_raw_multiply_bw<scalar_t>(out_d_grad, a_d, b_r, a_d_grad, b_r_grad, 4, 4, 4);
    _quaternion_raw_multiply_bw<scalar_t>(out_r_grad, a_r, b_r, a_r_grad, b_r_grad, 4, 4, 4);
}

// ===== Device: Dual Quaternion Apply Forward

template <typename scalar_t>
__device__ void _dual_quaternion_to_quaternion_translation_fw(const scalar_t *__restrict__ a_r, const scalar_t *__restrict__ a_d, scalar_t *__restrict__ out_q, scalar_t *__restrict__ out_t);

template <typename scalar_t>
__device__ void _dual_quaternion_apply_fw(const scalar_t *__restrict__ a_r, const scalar_t *__restrict__ a_d, const scalar_t *__restrict__ b, scalar_t *__restrict__ out) {
    scalar_t a_q[4] = {};
    scalar_t a_t[3] = {};
    _dual_quaternion_to_quaternion_translation_fw<scalar_t>(a_r, a_d, a_q, a_t);
    _quaternion_translation_apply_fw<scalar_t>(a_q, a_t, b, out);
}

// ===== Device: Dual Quaternion Apply Backward

template <typename scalar_t>
__device__ void _dual_quaternion_to_quaternion_translation_bw(const scalar_t *__restrict__ out_q_grad, const scalar_t *__restrict__ out_t_grad, const scalar_t *__restrict__ a_r, const scalar_t *__restrict__ a_d, scalar_t *__restrict__ a_r_grad, scalar_t *__restrict__ a_d_grad);

template <typename scalar_t>
__device__ void _dual_quaternion_apply_bw(const scalar_t *__restrict__ out_grad, const scalar_t *__restrict__ a_r, const scalar_t *__restrict__ a_d, const scalar_t *__restrict__ b, scalar_t *__restrict__ a_r_grad, scalar_t *__restrict__ a_d_grad, scalar_t *__restrict__ b_grad) {
    scalar_t a_q[4] = {};
    scalar_t a_t[3] = {};
    _dual_quaternion_to_quaternion_translation_fw<scalar_t>(a_r, a_d, a_q, a_t);

    scalar_t a_q_grad[4] = {};
    scalar_t a_t_grad[3] = {};
    _quaternion_translation_apply_bw<scalar_t>(out_grad, a_q, a_t, b, a_q_grad, a_t_grad, b_grad);
    _dual_quaternion_to_quaternion_translation_bw<scalar_t>(a_q_grad, a_t_grad, a_r, a_d, a_r_grad, a_d_grad);
}

// ===== Device: Dual Quaternion Q Conjugate Forward

template <typename scalar_t>
__device__ void _dual_quaternion_q_conjugate_fw(const scalar_t *__restrict__ a_r, const scalar_t *__restrict__ a_d, scalar_t *__restrict__ out_r, scalar_t *__restrict__ out_d) {
    _quaternion_conjugate_fw<scalar_t>(a_r, out_r);
    _quaternion_conjugate_fw<scalar_t>(a_d, out_d);
}

// ===== Device: Dual Quaternion Q Conjugate Backward

template <typename scalar_t>
__device__ void _dual_quaternion_q_conjugate_bw(const scalar_t *__restrict__ out_r_grad, const scalar_t *__restrict__ out_d_grad, const scalar_t *__restrict__ a_r, const scalar_t *__restrict__ a_d, scalar_t *__restrict__ a_r_grad, scalar_t *__restrict__ a_d_grad) {
    _quaternion_conjugate_bw<scalar_t>(out_r_grad, a_r, a_r_grad);
    _quaternion_conjugate_bw<scalar_t>(out_d_grad, a_d, a_d_grad);
}

// ===== Device: Dual Quaternion D Conjugate Forward

template <typename scalar_t>
__device__ void _dual_quaternion_d_conjugate_fw(const scalar_t *__restrict__ a_r, const scalar_t *__restrict__ a_d, scalar_t *__restrict__ out_r, scalar_t *__restrict__ out_d) {
    out_r[0] = a_r[0];
    out_r[1] = a_r[1];
    out_r[2] = a_r[2];
    out_r[3] = a_r[3];
    out_d[0] = -a_d[0];
    out_d[1] = -a_d[1];
    out_d[2] = -a_d[2];
    out_d[3] = -a_d[3];
}

// ===== Device: Dual Quaternion D Conjugate Backward

template <typename scalar_t>
__device__ void _dual_quaternion_d_conjugate_bw(const scalar_t *__restrict__ out_r_grad, const scalar_t *__restrict__ out_d_grad, const scalar_t *__restrict__ a_r, const scalar_t *__restrict__ a_d, scalar_t *__restrict__ a_r_grad, scalar_t *__restrict__ a_d_grad) {
    a_r_grad[0] += out_r_grad[0];
    a_r_grad[1] += out_r_grad[1];
    a_r_grad[2] += out_r_grad[2];
    a_r_grad[3] += out_r_grad[3];
    a_d_grad[0] += -out_d_grad[0];
    a_d_grad[1] += -out_d_grad[1];
    a_d_grad[2] += -out_d_grad[2];
    a_d_grad[3] += -out_d_grad[3];
}

// ===== Device: Dual Quaternion 3rd Conjugate Forward

template <typename scalar_t>
__device__ void _dual_quaternion_3rd_conjugate_fw(const scalar_t *__restrict__ a_r, const scalar_t *__restrict__ a_d, scalar_t *__restrict__ out_r, scalar_t *__restrict__ out_d) {
    scalar_t raw_r[4] = {};
    scalar_t raw_d[4] = {};
    _dual_quaternion_q_conjugate_fw<scalar_t>(a_r, a_d, raw_r, raw_d);
    _dual_quaternion_d_conjugate_fw<scalar_t>(raw_r, raw_d, out_r, out_d);
}

// ===== Device: Dual Quaternion 3rd Conjugate Backward

template <typename scalar_t>
__device__ void _dual_quaternion_3rd_conjugate_bw(const scalar_t *__restrict__ out_r_grad, const scalar_t *__restrict__ out_d_grad, const scalar_t *__restrict__ a_r, const scalar_t *__restrict__ a_d, scalar_t *__restrict__ a_r_grad, scalar_t *__restrict__ a_d_grad) {
    scalar_t raw_r[4] = {};
    scalar_t raw_d[4] = {};
    _dual_quaternion_q_conjugate_fw<scalar_t>(a_r, a_d, raw_r, raw_d);

    scalar_t raw_r_grad[4] = {};
    scalar_t raw_d_grad[4] = {};
    _dual_quaternion_d_conjugate_bw<scalar_t>(out_r_grad, out_d_grad, raw_r, raw_d, raw_r_grad, raw_d_grad);
    _dual_quaternion_q_conjugate_bw<scalar_t>(raw_r_grad, raw_d_grad, a_r, a_d, a_r_grad, a_d_grad);
}

// ===== Device: Dual Quaternion Norm Forward

template <typename scalar_t>
__device__ void _dual_quaternion_norm_fw(const scalar_t *__restrict__ a_r, const scalar_t *__restrict__ a_d, scalar_t *__restrict__ out_r, scalar_t *__restrict__ out_d) {
    scalar_t a_inv_r[4] = {};
    scalar_t a_inv_d[4] = {};
    _dual_quaternion_q_conjugate_fw<scalar_t>(a_r, a_d, a_inv_r, a_inv_d);
    _dual_quaternion_mul_fw<scalar_t>(a_r, a_d, a_inv_r, a_inv_d, out_r, out_d);
}

// ===== Device: Dual Quaternion Norm Backward

template <typename scalar_t>
__device__ void _dual_quaternion_norm_bw(const scalar_t *__restrict__ out_r_grad, const scalar_t *__restrict__ out_d_grad, const scalar_t *__restrict__ a_r, const scalar_t *__restrict__ a_d, scalar_t *__restrict__ a_r_grad, scalar_t *__restrict__ a_d_grad) {
    scalar_t a_inv_r[4] = {};
    scalar_t a_inv_d[4] = {};
    _dual_quaternion_q_conjugate_fw<scalar_t>(a_r, a_d, a_inv_r, a_inv_d);

    scalar_t a_inv_r_grad[4] = {};
    scalar_t a_inv_d_grad[4] = {};
    _dual_quaternion_mul_bw<scalar_t>(out_r_grad, out_d_grad, a_r, a_d, a_inv_r, a_inv_d, a_r_grad, a_d_grad, a_inv_r_grad, a_inv_d_grad);
    _dual_quaternion_q_conjugate_bw<scalar_t>(a_inv_r_grad, a_inv_d_grad, a_r, a_d, a_r_grad, a_d_grad);
}

// ===== Device: Quaternion Translation to Dual Quaternion Forward

template <typename scalar_t>
__device__ void _quaternion_translation_to_dual_quaternion_fw(const scalar_t *__restrict__ a_q, const scalar_t *__restrict__ a_t, scalar_t *__restrict__ out_r, scalar_t *__restrict__ out_d) {
    scalar_t raw_d[4] = {};
    out_r[0] = a_q[0];
    out_r[1] = a_q[1];
    out_r[2] = a_q[2];
    out_r[3] = a_q[3];
    _quaternion_raw_multiply_fw<scalar_t>(a_t, a_q, raw_d, 3, 4, 4);
    out_d[0] = 0.5 * raw_d[0];
    out_d[1] = 0.5 * raw_d[1];
    out_d[2] = 0.5 * raw_d[2];
    out_d[3] = 0.5 * raw_d[3];
}

// ===== Device: Quaternion Translation to Dual Quaternion Backward

template <typename scalar_t>
__device__ void _quaternion_translation_to_dual_quaternion_bw(const scalar_t *__restrict__ out_r_grad, const scalar_t *__restrict__ out_d_grad, const scalar_t *__restrict__ a_q, const scalar_t *__restrict__ a_t, scalar_t *__restrict__ a_q_grad, scalar_t *__restrict__ a_t_grad) {
    scalar_t raw_d_grad[4] = {};
    a_q_grad[0] += out_r_grad[0];
    a_q_grad[1] += out_r_grad[1];
    a_q_grad[2] += out_r_grad[2];
    a_q_grad[3] += out_r_grad[3];
    raw_d_grad[0] += 0.5 * out_d_grad[0];
    raw_d_grad[1] += 0.5 * out_d_grad[1];
    raw_d_grad[2] += 0.5 * out_d_grad[2];
    raw_d_grad[3] += 0.5 * out_d_grad[3];
    _quaternion_raw_multiply_bw<scalar_t>(raw_d_grad, a_t, a_q, a_t_grad, a_q_grad, 3, 4, 4);
}

// ===== Device: Dual Quaternion to Quaternion Translation Forward

template <typename scalar_t>
__device__ void _dual_quaternion_to_quaternion_translation_fw(const scalar_t *__restrict__ a_r, const scalar_t *__restrict__ a_d, scalar_t *__restrict__ out_q, scalar_t *__restrict__ out_t) {
    scalar_t a_r_conj[4] = {};
    scalar_t raw_t[3] = {};
    out_q[0] = a_r[0];
    out_q[1] = a_r[1];
    out_q[2] = a_r[2];
    out_q[3] = a_r[3];
    _quaternion_conjugate_fw<scalar_t>(a_r, a_r_conj);
    _quaternion_raw_multiply_fw<scalar_t>(a_d, a_r_conj, raw_t, 4, 4, 3);
    out_t[0] = 2.0 * raw_t[0];
    out_t[1] = 2.0 * raw_t[1];
    out_t[2] = 2.0 * raw_t[2];
}

// ===== Device: Dual Quaternion to Quaternion Translation Backward

template <typename scalar_t>
__device__ void _dual_quaternion_to_quaternion_translation_bw(const scalar_t *__restrict__ out_q_grad, const scalar_t *__restrict__ out_t_grad, const scalar_t *__restrict__ a_r, const scalar_t *__restrict__ a_d, scalar_t *__restrict__ a_r_grad, scalar_t *__restrict__ a_d_grad) {
    scalar_t a_r_conj[4] = {};
    _quaternion_conjugate_fw<scalar_t>(a_r, a_r_conj);

    scalar_t raw_t_grad[3] = {};
    scalar_t a_r_conj_grad[4] = {};
    a_r_grad[0] += out_q_grad[0];
    a_r_grad[1] += out_q_grad[1];
    a_r_grad[2] += out_q_grad[2];
    a_r_grad[3] += out_q_grad[3];
    raw_t_grad[0] += 2.0 * out_t_grad[0];
    raw_t_grad[1] += 2.0 * out_t_grad[1];
    raw_t_grad[2] += 2.0 * out_t_grad[2];
    _quaternion_raw_multiply_bw<scalar_t>(raw_t_grad, a_d, a_r_conj, a_d_grad, a_r_conj_grad, 4, 4, 3);
    _quaternion_conjugate_bw<scalar_t>(a_r_conj_grad, a_r, a_r_grad);
}

// ===== Device: Quaternion Translation to SE3 Forward

template <typename scalar_t>
__device__ void _quaternion_translation_to_se3_fw(const scalar_t *__restrict__ a_q, const scalar_t *__restrict__ a_t, scalar_t *__restrict__ out) {
    scalar_t rmat[9] = {};
    _quaternion_to_matrix_fw<scalar_t>(a_q, rmat);
    out[0] = rmat[0];
    out[1] = rmat[1];
    out[2] = rmat[2];
    out[3] = a_t[0];
    out[4] = rmat[3];
    out[5] = rmat[4];
    out[6] = rmat[5];
    out[7] = a_t[1];
    out[8] = rmat[6];
    out[9] = rmat[7];
    out[10] = rmat[8];
    out[11] = a_t[2];
    out[12] = 0.;
    out[13] = 0.;
    out[14] = 0.;
    out[15] = 1.;
}

// ===== Device: Quaternion Translation to SE3 Backward

template <typename scalar_t>
__device__ void _quaternion_translation_to_se3_bw(const scalar_t *__restrict__ out_grad, const scalar_t *__restrict__ a_q, const scalar_t *__restrict__ a_t, scalar_t *__restrict__ a_q_grad, scalar_t *__restrict__ a_t_grad) {
    scalar_t rmat_grad[9] = {};
    rmat_grad[0] += out_grad[0];
    rmat_grad[1] += out_grad[1];
    rmat_grad[2] += out_grad[2];
    rmat_grad[3] += out_grad[4];
    rmat_grad[4] += out_grad[5];
    rmat_grad[5] += out_grad[6];
    rmat_grad[6] += out_grad[8];
    rmat_grad[7] += out_grad[9];
    rmat_grad[8] += out_grad[10];
    a_t_grad[0] += out_grad[3];
    a_t_grad[1] += out_grad[7];
    a_t_grad[2] += out_grad[11];
    _quaternion_to_matrix_bw<scalar_t>(rmat_grad, a_q, a_q_grad);
}

// ===== Device: SE3 to Quaternion Translation Forward

template <typename scalar_t>
__device__ void _se3_to_quaternion_translation_fw(const scalar_t *__restrict__ a, scalar_t *__restrict__ out_q, scalar_t *__restrict__ out_t) {
    scalar_t rmat[9] = {};
    rmat[0] = a[0];
    rmat[1] = a[1];
    rmat[2] = a[2];
    rmat[3] = a[4];
    rmat[4] = a[5];
    rmat[5] = a[6];
    rmat[6] = a[8];
    rmat[7] = a[9];
    rmat[8] = a[10];
    out_t[0] = a[3];
    out_t[1] = a[7];
    out_t[2] = a[11];
    _matrix_to_quaternion_fw<scalar_t>(rmat, out_q);
}

// ===== Device: SE3 to Quaternion Translation Backward

template <typename scalar_t>
__device__ void _se3_to_quaternion_translation_bw(const scalar_t *__restrict__ out_q_grad, const scalar_t *__restrict__ out_t_grad, const scalar_t *__restrict__ a, scalar_t *__restrict__ a_grad) {
    scalar_t rmat[9] = {};
    rmat[0] = a[0];
    rmat[1] = a[1];
    rmat[2] = a[2];
    rmat[3] = a[4];
    rmat[4] = a[5];
    rmat[5] = a[6];
    rmat[6] = a[8];
    rmat[7] = a[9];
    rmat[8] = a[10];

    scalar_t rmat_grad[9] = {};
    _matrix_to_quaternion_bw<scalar_t>(out_q_grad, rmat, rmat_grad);
    a_grad[0] += rmat_grad[0];
    a_grad[1] += rmat_grad[1];
    a_grad[2] += rmat_grad[2];
    a_grad[3] += out_t_grad[0];
    a_grad[4] += rmat_grad[3];
    a_grad[5] += rmat_grad[4];
    a_grad[6] += rmat_grad[5];
    a_grad[7] += out_t_grad[1];
    a_grad[8] += rmat_grad[6];
    a_grad[9] += rmat_grad[7];
    a_grad[10] += rmat_grad[8];
    a_grad[11] += out_t_grad[2];
}

// ===== Device: Dual Quaternion to SE3 Forward

template <typename scalar_t>
__device__ void _dual_quaternion_to_se3_fw(const scalar_t *__restrict__ a_r, const scalar_t *__restrict__ a_d, scalar_t *__restrict__ out) {
    scalar_t raw_r[4] = {};
    scalar_t raw_d[3] = {};
    _dual_quaternion_to_quaternion_translation_fw<scalar_t>(a_r, a_d, raw_r, raw_d);
    _quaternion_translation_to_se3_fw<scalar_t>(raw_r, raw_d, out);
}

// ===== Device: Dual Quaternion to SE3 Backward

template <typename scalar_t>
__device__ void _dual_quaternion_to_se3_bw(const scalar_t *__restrict__ out_grad, const scalar_t *__restrict__ a_r, const scalar_t *__restrict__ a_d, scalar_t *__restrict__ a_r_grad, scalar_t *__restrict__ a_d_grad) {
    scalar_t a_q[4] = {};
    scalar_t a_t[3] = {};
    _dual_quaternion_to_quaternion_translation_fw<scalar_t>(a_r, a_d, a_q, a_t);

    scalar_t a_q_grad[4] = {};
    scalar_t a_t_grad[3] = {};
    _quaternion_translation_to_se3_bw<scalar_t>(out_grad, a_q, a_t, a_q_grad, a_t_grad);
    _dual_quaternion_to_quaternion_translation_bw<scalar_t>(a_q_grad, a_t_grad, a_r, a_d, a_r_grad, a_d_grad);
}

// ===== Device: SE3 to Dual Quaternion Forward

template <typename scalar_t>
__device__ void _se3_to_dual_quaternion_fw(const scalar_t *__restrict__ a, scalar_t *__restrict__ out_r, scalar_t *__restrict__ out_d) {
    scalar_t out_q[4] = {};
    scalar_t out_t[3] = {};
    _se3_to_quaternion_translation_fw<scalar_t>(a, out_q, out_t);
    _quaternion_translation_to_dual_quaternion_fw<scalar_t>(out_q, out_t, out_r, out_d);
}

// ===== Device: SE3 to Dual Quaternion Backward

template <typename scalar_t>
__device__ void _se3_to_dual_quaternion_bw(const scalar_t *__restrict__ out_r_grad, const scalar_t *__restrict__ out_d_grad, const scalar_t *__restrict__ a, scalar_t *__restrict__ a_grad) {
    scalar_t out_q[4] = {};
    scalar_t out_t[3] = {};
    _se3_to_quaternion_translation_fw<scalar_t>(a, out_q, out_t);

    scalar_t out_q_grad[4] = {};
    scalar_t out_t_grad[3] = {};
    _quaternion_translation_to_dual_quaternion_bw<scalar_t>(out_r_grad, out_d_grad, out_q, out_t, out_q_grad, out_t_grad);
    _se3_to_quaternion_translation_bw<scalar_t>(out_q_grad, out_t_grad, a, a_grad);
}

// =============================================================================

// ===== Kernel+Host: Standardize Quaternion Forward

template <typename scalar_t>
__global__ void kernel_standardize_quaternion_fw(const scalar_t *__restrict__ a, scalar_t *__restrict__ out,
                                                 const uint32_t B) {
    const uint32_t i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i >= B) {
        return;
    }
    _standardize_quaternion_fw<scalar_t>(a + i * 4, out + i * 4);
}

void standardize_quaternion_fw(at::Tensor a, at::Tensor out, const uint32_t B) {
    CHECK_IS_CONTIGUOUS_FLOAT_CUDA(a);
    CHECK_IS_CONTIGUOUS_FLOAT_CUDA(out);

    AT_DISPATCH_FLOATING_TYPES_AND_HALF(
        a.scalar_type(), "standardize_quaternion_fw", ([&] { 
            kernel_standardize_quaternion_fw<scalar_t><<<DIV_ROUND_UP(B, N_THREADS), N_THREADS>>>(a.data_ptr<scalar_t>(), out.data_ptr<scalar_t>(), B);
            gpuErrorCheck(cudaDeviceSynchronize());
        }));
}

// ===== Kernel+Host: Standardize Quaternion Backward

template <typename scalar_t>
__global__ void kernel_standardize_quaternion_bw(const scalar_t *__restrict__ out_grad, const scalar_t *__restrict__ a,
                                                 scalar_t *__restrict__ a_grad, const uint32_t B) {
    const uint32_t i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i >= B) {
        return;
    }
    _standardize_quaternion_bw<scalar_t>(out_grad + i * 4, a + i * 4, a_grad + i * 4);
}

void standardize_quaternion_bw(at::Tensor out_grad, at::Tensor a, at::Tensor a_grad, const uint32_t B) {
    CHECK_IS_CONTIGUOUS_FLOAT_CUDA(out_grad);
    CHECK_IS_CONTIGUOUS_FLOAT_CUDA(a);
    CHECK_IS_CONTIGUOUS_FLOAT_CUDA(a_grad);

    AT_DISPATCH_FLOATING_TYPES_AND_HALF(a.scalar_type(), "standardize_quaternion_bw", ([&] {
        kernel_standardize_quaternion_bw<scalar_t><<<DIV_ROUND_UP(B, N_THREADS), N_THREADS>>>(out_grad.data_ptr<scalar_t>(), a.data_ptr<scalar_t>(), a_grad.data_ptr<scalar_t>(), B);
        gpuErrorCheck(cudaDeviceSynchronize());
    }));
}

// ===== Kernel+Host: Quaternion Raw Multiply Forward

template <typename scalar_t>
__global__ void kernel_quaternion_raw_multiply_fw(const scalar_t *__restrict__ a, const scalar_t *__restrict__ b,
                                                  scalar_t *__restrict__ out, const uint32_t B, const uint32_t Da,
                                                  const uint32_t Db) {
    const uint32_t i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i >= B) {
        return;
    }
    _quaternion_raw_multiply_fw<scalar_t>(a + i * Da, b + i * Db, out + i * 4, Da, Db, 4);
}

void quaternion_raw_multiply_fw(at::Tensor a, at::Tensor b, at::Tensor out, const uint32_t B, const uint32_t Da,
                                const uint32_t Db) {
    CHECK_IS_CONTIGUOUS_FLOAT_CUDA(a);
    CHECK_IS_CONTIGUOUS_FLOAT_CUDA(b);
    CHECK_IS_CONTIGUOUS_FLOAT_CUDA(out);

    AT_DISPATCH_FLOATING_TYPES_AND_HALF(a.scalar_type(), "quaternion_raw_multiply_fw", ([&] {
        kernel_quaternion_raw_multiply_fw<scalar_t><<<DIV_ROUND_UP(B, N_THREADS), N_THREADS>>>(a.data_ptr<scalar_t>(), b.data_ptr<scalar_t>(), out.data_ptr<scalar_t>(), B, Da, Db);
        gpuErrorCheck(cudaDeviceSynchronize());
    }));
}

// ===== Kernel+Host: Quaternion Raw Multiply Backward

template <typename scalar_t>
__global__ void kernel_quaternion_raw_multiply_bw(const scalar_t *__restrict__ out_grad,
                                                  const scalar_t *__restrict__ a, const scalar_t *__restrict__ b,
                                                  scalar_t *__restrict__ a_grad,
                                                  scalar_t *__restrict__ b_grad, const uint32_t B,
                                                  const uint32_t Da, const uint32_t Db) {
    const uint32_t i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i >= B) {
        return;
    }
    _quaternion_raw_multiply_bw<scalar_t>(out_grad + i * 4, a + i * Da, b + i * Db, a_grad + i * Da,
                                          b_grad + i * Db, Da, Db, 4);
}

void quaternion_raw_multiply_bw(at::Tensor out_grad, at::Tensor a, at::Tensor b, at::Tensor a_grad,
                                at::Tensor b_grad, const uint32_t B, const uint32_t Da, const uint32_t Db) {
    CHECK_IS_CONTIGUOUS_FLOAT_CUDA(out_grad);
    CHECK_IS_CONTIGUOUS_FLOAT_CUDA(a);
    CHECK_IS_CONTIGUOUS_FLOAT_CUDA(b);
    CHECK_IS_CONTIGUOUS_FLOAT_CUDA(a_grad);
    CHECK_IS_CONTIGUOUS_FLOAT_CUDA(b_grad);

    AT_DISPATCH_FLOATING_TYPES_AND_HALF(a.scalar_type(), "quaternion_raw_multiply_bw", ([&] {
        kernel_quaternion_raw_multiply_bw<scalar_t><<<DIV_ROUND_UP(B, N_THREADS), N_THREADS>>>(out_grad.data_ptr<scalar_t>(), a.data_ptr<scalar_t>(), b.data_ptr<scalar_t>(), a_grad.data_ptr<scalar_t>(), b_grad.data_ptr<scalar_t>(), B, Da, Db);
        gpuErrorCheck(cudaDeviceSynchronize());
    }));
}

// ===== Kernel+Host: Quaternion Multiply Forward

template <typename scalar_t>
__global__ void kernel_quaternion_multiply_fw(const scalar_t *__restrict__ a, const scalar_t *__restrict__ b, scalar_t *__restrict__ out,
                                                 const uint32_t B, const uint32_t Da, const uint32_t Db) {
    const uint32_t i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i >= B) {
        return;
    }
    _quaternion_multiply_fw<scalar_t>(a + i * Da, b + i * Db, out + i * 4, Da, Db);
}

void quaternion_multiply_fw(at::Tensor a, at::Tensor b, at::Tensor out, const uint32_t B, const uint32_t Da, const uint32_t Db) {
    CHECK_IS_CONTIGUOUS_FLOAT_CUDA(a);
    CHECK_IS_CONTIGUOUS_FLOAT_CUDA(b);
    CHECK_IS_CONTIGUOUS_FLOAT_CUDA(out);

    AT_DISPATCH_FLOATING_TYPES_AND_HALF(a.scalar_type(), "quaternion_multiply_fw", ([&] {
        kernel_quaternion_multiply_fw<scalar_t><<<DIV_ROUND_UP(B, N_THREADS), N_THREADS>>>(a.data_ptr<scalar_t>(), b.data_ptr<scalar_t>(), out.data_ptr<scalar_t>(), B, Da, Db);
        gpuErrorCheck(cudaDeviceSynchronize());
    }));
}

// ===== Kernel+Host: Quaternion Multiply Backward

template <typename scalar_t>
__global__ void kernel_quaternion_multiply_bw(const scalar_t *__restrict__ out_grad, const scalar_t *__restrict__ a, const scalar_t *__restrict__ b,
                                                 scalar_t *__restrict__ a_grad, scalar_t *__restrict__ b_grad, const uint32_t B, const uint32_t Da, const uint32_t Db) {
    const uint32_t i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i >= B) {
        return;
    }
    _quaternion_multiply_bw<scalar_t>(out_grad + i * 4, a + i * Da, b + i * Db, a_grad + i * Da, b_grad + i * Db, Da, Db);
}

void quaternion_multiply_bw(at::Tensor out_grad, at::Tensor a, at::Tensor b, at::Tensor a_grad, at::Tensor b_grad, const uint32_t B, const uint32_t Da, const uint32_t Db) {
    CHECK_IS_CONTIGUOUS_FLOAT_CUDA(out_grad);
    CHECK_IS_CONTIGUOUS_FLOAT_CUDA(a);
    CHECK_IS_CONTIGUOUS_FLOAT_CUDA(b);
    CHECK_IS_CONTIGUOUS_FLOAT_CUDA(a_grad);
    CHECK_IS_CONTIGUOUS_FLOAT_CUDA(b_grad);

    AT_DISPATCH_FLOATING_TYPES_AND_HALF(a.scalar_type(), "quaternion_multiply_bw", ([&] {
        kernel_quaternion_multiply_bw<scalar_t><<<DIV_ROUND_UP(B, N_THREADS), N_THREADS>>>(out_grad.data_ptr<scalar_t>(), a.data_ptr<scalar_t>(), b.data_ptr<scalar_t>(), a_grad.data_ptr<scalar_t>(), b_grad.data_ptr<scalar_t>(), B, Da, Db);
        gpuErrorCheck(cudaDeviceSynchronize());
    }));
}

// ===== Kernel+Host: Quaternion Conjugate Forward

template <typename scalar_t>
__global__ void kernel_quaternion_conjugate_fw(const scalar_t *__restrict__ a, scalar_t *__restrict__ out,
                                                 const uint32_t B) {
    const uint32_t i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i >= B) {
        return;
    }
    _quaternion_conjugate_fw<scalar_t>(a + i * 4, out + i * 4);
}

void quaternion_conjugate_fw(at::Tensor a, at::Tensor out, const uint32_t B) {
    CHECK_IS_CONTIGUOUS_FLOAT_CUDA(a);
    CHECK_IS_CONTIGUOUS_FLOAT_CUDA(out);

    AT_DISPATCH_FLOATING_TYPES_AND_HALF(a.scalar_type(), "quaternion_conjugate_fw", ([&] {
        kernel_quaternion_conjugate_fw<scalar_t><<<DIV_ROUND_UP(B, N_THREADS), N_THREADS>>>(a.data_ptr<scalar_t>(), out.data_ptr<scalar_t>(), B);
        gpuErrorCheck(cudaDeviceSynchronize());
    }));
}

// ===== Kernel+Host: Quaternion Conjugate Backward

template <typename scalar_t>
__global__ void kernel_quaternion_conjugate_bw(const scalar_t *__restrict__ out_grad, const scalar_t *__restrict__ a,
                                                 scalar_t *__restrict__ a_grad, const uint32_t B) {
    const uint32_t i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i >= B) {
        return;
    }
    _quaternion_conjugate_bw<scalar_t>(out_grad + i * 4, a + i * 4, a_grad + i * 4);
}

void quaternion_conjugate_bw(at::Tensor out_grad, at::Tensor a, at::Tensor a_grad, const uint32_t B) {
    CHECK_IS_CONTIGUOUS_FLOAT_CUDA(out_grad);
    CHECK_IS_CONTIGUOUS_FLOAT_CUDA(a);
    CHECK_IS_CONTIGUOUS_FLOAT_CUDA(a_grad);

    AT_DISPATCH_FLOATING_TYPES_AND_HALF(a.scalar_type(), "quaternion_conjugate_bw", ([&] {
        kernel_quaternion_conjugate_bw<scalar_t><<<DIV_ROUND_UP(B, N_THREADS), N_THREADS>>>(out_grad.data_ptr<scalar_t>(), a.data_ptr<scalar_t>(), a_grad.data_ptr<scalar_t>(), B);
        gpuErrorCheck(cudaDeviceSynchronize());
    }));
}

// ===== Kernel+Host: Quaternion Apply Forward

template <typename scalar_t>
__global__ void kernel_quaternion_apply_fw(const scalar_t *__restrict__ a, const scalar_t *__restrict__ b, scalar_t *__restrict__ out,
                                                 const uint32_t B) {
    const uint32_t i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i >= B) {
        return;
    }
    _quaternion_apply_fw<scalar_t>(a + i * 4, b + i * 3, out + i * 3);
}

void quaternion_apply_fw(at::Tensor a, at::Tensor b, at::Tensor out, const uint32_t B) {
    CHECK_IS_CONTIGUOUS_FLOAT_CUDA(a);
    CHECK_IS_CONTIGUOUS_FLOAT_CUDA(b);
    CHECK_IS_CONTIGUOUS_FLOAT_CUDA(out);

    AT_DISPATCH_FLOATING_TYPES_AND_HALF(a.scalar_type(), "quaternion_apply_fw", ([&] {
        kernel_quaternion_apply_fw<scalar_t><<<DIV_ROUND_UP(B, N_THREADS), N_THREADS>>>(a.data_ptr<scalar_t>(), b.data_ptr<scalar_t>(), out.data_ptr<scalar_t>(), B);
        gpuErrorCheck(cudaDeviceSynchronize());
    }));
}

// ===== Kernel+Host: Quaternion Apply Backward

template <typename scalar_t>
__global__ void kernel_quaternion_apply_bw(const scalar_t *__restrict__ out_grad, const scalar_t *__restrict__ a, const scalar_t *__restrict__ b,
                                                 scalar_t *__restrict__ a_grad, scalar_t *__restrict__ b_grad, const uint32_t B) {
    const uint32_t i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i >= B) {
        return;
    }
    _quaternion_apply_bw(out_grad + i * 3, a + i * 4, b + i * 3, a_grad + i * 4, b_grad + i * 3);
}

void quaternion_apply_bw(at::Tensor out_grad, at::Tensor a, at::Tensor b, at::Tensor a_grad, at::Tensor b_grad, const uint32_t B) {
    CHECK_IS_CONTIGUOUS_FLOAT_CUDA(out_grad);
    CHECK_IS_CONTIGUOUS_FLOAT_CUDA(a);
    CHECK_IS_CONTIGUOUS_FLOAT_CUDA(b);
    CHECK_IS_CONTIGUOUS_FLOAT_CUDA(a_grad);
    CHECK_IS_CONTIGUOUS_FLOAT_CUDA(b_grad);

    AT_DISPATCH_FLOATING_TYPES_AND_HALF(a.scalar_type(), "quaternion_apply_bw", ([&] {
        kernel_quaternion_apply_bw<scalar_t><<<DIV_ROUND_UP(B, N_THREADS), N_THREADS>>>(out_grad.data_ptr<scalar_t>(), a.data_ptr<scalar_t>(), b.data_ptr<scalar_t>(), a_grad.data_ptr<scalar_t>(), b_grad.data_ptr<scalar_t>(), B);
        gpuErrorCheck(cudaDeviceSynchronize());
    }));
}

// ===== Kernel+Host: Quaternion to Matrix Forward

template <typename scalar_t>
__global__ void kernel_quaternion_to_matrix_fw(const scalar_t *__restrict__ a, scalar_t *__restrict__ out,
                                                 const uint32_t B) {
    const uint32_t i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i >= B) {
        return;
    }
    _quaternion_to_matrix_fw<scalar_t>(a + i * 4, out + i * 9);
}

void quaternion_to_matrix_fw(at::Tensor a, at::Tensor out, const uint32_t B) {
    CHECK_IS_CONTIGUOUS_FLOAT_CUDA(a);
    CHECK_IS_CONTIGUOUS_FLOAT_CUDA(out);

    AT_DISPATCH_FLOATING_TYPES_AND_HALF(a.scalar_type(), "quaternion_to_matrix_fw", ([&] {
        kernel_quaternion_to_matrix_fw<scalar_t><<<DIV_ROUND_UP(B, N_THREADS), N_THREADS>>>(a.data_ptr<scalar_t>(), out.data_ptr<scalar_t>(), B);
        gpuErrorCheck(cudaDeviceSynchronize());
    }));
}

// ===== Kernel+Host: Quaternion to Matrix Backward

template <typename scalar_t>
__global__ void kernel_quaternion_to_matrix_bw(const scalar_t *__restrict__ out_grad, const scalar_t *__restrict__ a,
                                                 scalar_t *__restrict__ a_grad, const uint32_t B) {
    const uint32_t i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i >= B) {
        return;
    }
    _quaternion_to_matrix_bw<scalar_t>(out_grad + i * 9, a + i * 4, a_grad + i * 4);
}

void quaternion_to_matrix_bw(at::Tensor out_grad, at::Tensor a, at::Tensor a_grad, const uint32_t B) {
    CHECK_IS_CONTIGUOUS_FLOAT_CUDA(out_grad);
    CHECK_IS_CONTIGUOUS_FLOAT_CUDA(a);
    CHECK_IS_CONTIGUOUS_FLOAT_CUDA(a_grad);

    AT_DISPATCH_FLOATING_TYPES_AND_HALF(a.scalar_type(), "quaternion_to_matrix_bw", ([&] {
        kernel_quaternion_to_matrix_bw<scalar_t><<<DIV_ROUND_UP(B, N_THREADS), N_THREADS>>>(out_grad.data_ptr<scalar_t>(), a.data_ptr<scalar_t>(), a_grad.data_ptr<scalar_t>(), B);
        gpuErrorCheck(cudaDeviceSynchronize());
    }));
}

// ===== Kernel+Host: Matrix to Raw Quaternion Forward

template <typename scalar_t>
__global__ void kernel_matrix_to_raw_quaternion_fw(const scalar_t *__restrict__ a, scalar_t *__restrict__ out,
                                                 const uint32_t B) {
    const uint32_t i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i >= B) {
        return;
    }
    _matrix_to_raw_quaternion_fw<scalar_t>(a + i * 9, out + i * 4);
}

void matrix_to_raw_quaternion_fw(at::Tensor a, at::Tensor out, const uint32_t B) {
    CHECK_IS_CONTIGUOUS_FLOAT_CUDA(a);
    CHECK_IS_CONTIGUOUS_FLOAT_CUDA(out);

    AT_DISPATCH_FLOATING_TYPES_AND_HALF(a.scalar_type(), "matrix_to_raw_quaternion_fw", ([&] {
        kernel_matrix_to_raw_quaternion_fw<scalar_t><<<DIV_ROUND_UP(B, N_THREADS), N_THREADS>>>(a.data_ptr<scalar_t>(), out.data_ptr<scalar_t>(), B);
        gpuErrorCheck(cudaDeviceSynchronize());
    }));
}

// ===== Kernel+Host: Matrix to Raw Quaternion Backward

template <typename scalar_t>
__global__ void kernel_matrix_to_raw_quaternion_bw(const scalar_t *__restrict__ out_grad, const scalar_t *__restrict__ a,
                                                 scalar_t *__restrict__ a_grad, const uint32_t B) {
    const uint32_t i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i >= B) {
        return;
    }
    _matrix_to_raw_quaternion_bw<scalar_t>(out_grad + i * 4, a + i * 9, a_grad + i * 9);
}

void matrix_to_raw_quaternion_bw(at::Tensor out_grad, at::Tensor a, at::Tensor a_grad, const uint32_t B) {
    CHECK_IS_CONTIGUOUS_FLOAT_CUDA(out_grad);
    CHECK_IS_CONTIGUOUS_FLOAT_CUDA(a);
    CHECK_IS_CONTIGUOUS_FLOAT_CUDA(a_grad);

    AT_DISPATCH_FLOATING_TYPES_AND_HALF(a.scalar_type(), "matrix_to_raw_quaternion_bw", ([&] {
        kernel_matrix_to_raw_quaternion_bw<scalar_t><<<DIV_ROUND_UP(B, N_THREADS), N_THREADS>>>(out_grad.data_ptr<scalar_t>(), a.data_ptr<scalar_t>(), a_grad.data_ptr<scalar_t>(), B);
        gpuErrorCheck(cudaDeviceSynchronize());
    }));
}

// ===== Kernel+Host: Matrix to Quaternion Forward

template <typename scalar_t>
__global__ void kernel_matrix_to_quaternion_fw(const scalar_t *__restrict__ a, scalar_t *__restrict__ out,
                                                 const uint32_t B) {
    const uint32_t i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i >= B) {
        return;
    }
    _matrix_to_quaternion_fw<scalar_t>(a + i * 9, out + i * 4);
}

void matrix_to_quaternion_fw(at::Tensor a, at::Tensor out, const uint32_t B) {
    CHECK_IS_CONTIGUOUS_FLOAT_CUDA(a);
    CHECK_IS_CONTIGUOUS_FLOAT_CUDA(out);

    AT_DISPATCH_FLOATING_TYPES_AND_HALF(a.scalar_type(), "matrix_to_quaternion_fw", ([&] {
        kernel_matrix_to_quaternion_fw<scalar_t><<<DIV_ROUND_UP(B, N_THREADS), N_THREADS>>>(a.data_ptr<scalar_t>(), out.data_ptr<scalar_t>(), B);
        gpuErrorCheck(cudaDeviceSynchronize());
    }));
}

// ===== Kernel+Host: Matrix to Quaternion Backward

template <typename scalar_t>
__global__ void kernel_matrix_to_quaternion_bw(const scalar_t *__restrict__ out_grad, const scalar_t *__restrict__ a,
                                                 scalar_t *__restrict__ a_grad, const uint32_t B) {
    const uint32_t i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i >= B) {
        return;
    }
    _matrix_to_quaternion_bw<scalar_t>(out_grad + i * 4, a + i * 9, a_grad + i * 9);
}

void matrix_to_quaternion_bw(at::Tensor out_grad, at::Tensor a, at::Tensor a_grad, const uint32_t B) {
    CHECK_IS_CONTIGUOUS_FLOAT_CUDA(out_grad);
    CHECK_IS_CONTIGUOUS_FLOAT_CUDA(a);
    CHECK_IS_CONTIGUOUS_FLOAT_CUDA(a_grad);

    AT_DISPATCH_FLOATING_TYPES_AND_HALF(a.scalar_type(), "matrix_to_quaternion_bw", ([&] {
        kernel_matrix_to_quaternion_bw<scalar_t><<<DIV_ROUND_UP(B, N_THREADS), N_THREADS>>>(out_grad.data_ptr<scalar_t>(), a.data_ptr<scalar_t>(), a_grad.data_ptr<scalar_t>(), B);
        gpuErrorCheck(cudaDeviceSynchronize());
    }));
}

// ===== Kernel+Host: Axis Angle to Quaternion Forward

template <typename scalar_t>
__global__ void kernel_axis_angle_to_quaternion_fw(const scalar_t *__restrict__ a, scalar_t *__restrict__ out,
                                                 const uint32_t B) {
    const uint32_t i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i >= B) {
        return;
    }
    _axis_angle_to_quaternion_fw<scalar_t>(a + i * 3, out + i * 4);
}

void axis_angle_to_quaternion_fw(at::Tensor a, at::Tensor out, const uint32_t B) {
    CHECK_IS_CONTIGUOUS_FLOAT_CUDA(a);
    CHECK_IS_CONTIGUOUS_FLOAT_CUDA(out);

    AT_DISPATCH_FLOATING_TYPES_AND_HALF(a.scalar_type(), "axis_angle_to_quaternion_fw", ([&] {
        kernel_axis_angle_to_quaternion_fw<scalar_t><<<DIV_ROUND_UP(B, N_THREADS), N_THREADS>>>(a.data_ptr<scalar_t>(), out.data_ptr<scalar_t>(), B);
        gpuErrorCheck(cudaDeviceSynchronize());
    }));
}

// ===== Kernel+Host: Axis Angle to Quaternion Backward

template <typename scalar_t>
__global__ void kernel_axis_angle_to_quaternion_bw(const scalar_t *__restrict__ out_grad, const scalar_t *__restrict__ a,
                                                 scalar_t *__restrict__ a_grad, const uint32_t B) {
    const uint32_t i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i >= B) {
        return;
    }
    _axis_angle_to_quaternion_bw<scalar_t>(out_grad + i * 4, a + i * 3, a_grad + i * 3);
}

void axis_angle_to_quaternion_bw(at::Tensor out_grad, at::Tensor a, at::Tensor a_grad, const uint32_t B) {
    CHECK_IS_CONTIGUOUS_FLOAT_CUDA(out_grad);
    CHECK_IS_CONTIGUOUS_FLOAT_CUDA(a);
    CHECK_IS_CONTIGUOUS_FLOAT_CUDA(a_grad);

    AT_DISPATCH_FLOATING_TYPES_AND_HALF(a.scalar_type(), "axis_angle_to_quaternion_bw", ([&] {
        kernel_axis_angle_to_quaternion_bw<scalar_t><<<DIV_ROUND_UP(B, N_THREADS), N_THREADS>>>(out_grad.data_ptr<scalar_t>(), a.data_ptr<scalar_t>(), a_grad.data_ptr<scalar_t>(), B);
        gpuErrorCheck(cudaDeviceSynchronize());
    }));
}

// ===== Kernel+Host: Quaternion to Axis Angle Forward

template <typename scalar_t>
__global__ void kernel_quaternion_to_axis_angle_fw(const scalar_t *__restrict__ a, scalar_t *__restrict__ out,
                                                 const uint32_t B) {
    const uint32_t i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i >= B) {
        return;
    }
    _quaternion_to_axis_angle_fw<scalar_t>(a + i * 4, out + i * 3);
}

void quaternion_to_axis_angle_fw(at::Tensor a, at::Tensor out, const uint32_t B) {
    CHECK_IS_CONTIGUOUS_FLOAT_CUDA(a);
    CHECK_IS_CONTIGUOUS_FLOAT_CUDA(out);

    AT_DISPATCH_FLOATING_TYPES_AND_HALF(a.scalar_type(), "quaternion_to_axis_angle_fw", ([&] {
        kernel_quaternion_to_axis_angle_fw<scalar_t><<<DIV_ROUND_UP(B, N_THREADS), N_THREADS>>>(a.data_ptr<scalar_t>(), out.data_ptr<scalar_t>(), B);
        gpuErrorCheck(cudaDeviceSynchronize());
    }));
}

// ===== Kernel+Host: Quaternion to Axis Angle Backward

template <typename scalar_t>
__global__ void kernel_quaternion_to_axis_angle_bw(const scalar_t *__restrict__ out_grad, const scalar_t *__restrict__ a,
                                                 scalar_t *__restrict__ a_grad, const uint32_t B) {
    const uint32_t i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i >= B) {
        return;
    }
    _quaternion_to_axis_angle_bw<scalar_t>(out_grad + i * 3, a + i * 4, a_grad + i * 4);
}

void quaternion_to_axis_angle_bw(at::Tensor out_grad, at::Tensor a, at::Tensor a_grad, const uint32_t B) {
    CHECK_IS_CONTIGUOUS_FLOAT_CUDA(out_grad);
    CHECK_IS_CONTIGUOUS_FLOAT_CUDA(a);
    CHECK_IS_CONTIGUOUS_FLOAT_CUDA(a_grad);

    AT_DISPATCH_FLOATING_TYPES_AND_HALF(a.scalar_type(), "quaternion_to_axis_angle_bw", ([&] {
        kernel_quaternion_to_axis_angle_bw<scalar_t><<<DIV_ROUND_UP(B, N_THREADS), N_THREADS>>>(out_grad.data_ptr<scalar_t>(), a.data_ptr<scalar_t>(), a_grad.data_ptr<scalar_t>(), B);
        gpuErrorCheck(cudaDeviceSynchronize());
    }));
}

// ===== Kernel+Host: Axis Angle to Matrix Forward

template <typename scalar_t>
__global__ void kernel_axis_angle_to_matrix_fw(const scalar_t *__restrict__ a, scalar_t *__restrict__ out,
                                                 const uint32_t B) {
    const uint32_t i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i >= B) {
        return;
    }
    _axis_angle_to_matrix_fw<scalar_t>(a + i * 3, out + i * 9);
}

void axis_angle_to_matrix_fw(at::Tensor a, at::Tensor out, const uint32_t B) {
    CHECK_IS_CONTIGUOUS_FLOAT_CUDA(a);
    CHECK_IS_CONTIGUOUS_FLOAT_CUDA(out);

    AT_DISPATCH_FLOATING_TYPES_AND_HALF(a.scalar_type(), "axis_angle_to_matrix_fw", ([&] {
        kernel_axis_angle_to_matrix_fw<scalar_t><<<DIV_ROUND_UP(B, N_THREADS), N_THREADS>>>(a.data_ptr<scalar_t>(), out.data_ptr<scalar_t>(), B);
        gpuErrorCheck(cudaDeviceSynchronize());
    }));
}

// ===== Kernel+Host: Axis Angle to Matrix Backward

template <typename scalar_t>
__global__ void kernel_axis_angle_to_matrix_bw(const scalar_t *__restrict__ out_grad, const scalar_t *__restrict__ a,
                                                 scalar_t *__restrict__ a_grad, const uint32_t B) {
    const uint32_t i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i >= B) {
        return;
    }
    _axis_angle_to_matrix_bw(out_grad + i * 9, a + i * 3, a_grad + i * 3);
}

void axis_angle_to_matrix_bw(at::Tensor out_grad, at::Tensor a, at::Tensor a_grad, const uint32_t B) {
    CHECK_IS_CONTIGUOUS_FLOAT_CUDA(out_grad);
    CHECK_IS_CONTIGUOUS_FLOAT_CUDA(a);
    CHECK_IS_CONTIGUOUS_FLOAT_CUDA(a_grad);

    AT_DISPATCH_FLOATING_TYPES_AND_HALF(a.scalar_type(), "axis_angle_to_matrix_bw", ([&] {
        kernel_axis_angle_to_matrix_bw<scalar_t><<<DIV_ROUND_UP(B, N_THREADS), N_THREADS>>>(out_grad.data_ptr<scalar_t>(), a.data_ptr<scalar_t>(), a_grad.data_ptr<scalar_t>(), B);
        gpuErrorCheck(cudaDeviceSynchronize());
    }));
}

// ===== Kernel+Host: Matrix to Axis Angle Forward

template <typename scalar_t>
__global__ void kernel_matrix_to_axis_angle_fw(const scalar_t *__restrict__ a, scalar_t *__restrict__ out,
                                                 const uint32_t B) {
    const uint32_t i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i >= B) {
        return;
    }
    _matrix_to_axis_angle_fw<scalar_t>(a + i * 9, out + i * 3);
}

void matrix_to_axis_angle_fw(at::Tensor a, at::Tensor out, const uint32_t B) {
    CHECK_IS_CONTIGUOUS_FLOAT_CUDA(a);
    CHECK_IS_CONTIGUOUS_FLOAT_CUDA(out);

    AT_DISPATCH_FLOATING_TYPES_AND_HALF(a.scalar_type(), "matrix_to_axis_angle_fw", ([&] {
        kernel_matrix_to_axis_angle_fw<scalar_t><<<DIV_ROUND_UP(B, N_THREADS), N_THREADS>>>(a.data_ptr<scalar_t>(), out.data_ptr<scalar_t>(), B);
        gpuErrorCheck(cudaDeviceSynchronize());
    }));
}

// ===== Kernel+Host: Matrix to Axis Angle Backward

template <typename scalar_t>
__global__ void kernel_matrix_to_axis_angle_bw(const scalar_t *__restrict__ out_grad, const scalar_t *__restrict__ a,
                                                 scalar_t *__restrict__ a_grad, const uint32_t B) {
    const uint32_t i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i >= B) {
        return;
    }
    _matrix_to_axis_angle_bw<scalar_t>(out_grad + i * 3, a + i * 9, a_grad + i * 9);
}

void matrix_to_axis_angle_bw(at::Tensor out_grad, at::Tensor a, at::Tensor a_grad, const uint32_t B) {
    CHECK_IS_CONTIGUOUS_FLOAT_CUDA(out_grad);
    CHECK_IS_CONTIGUOUS_FLOAT_CUDA(a);
    CHECK_IS_CONTIGUOUS_FLOAT_CUDA(a_grad);

    AT_DISPATCH_FLOATING_TYPES_AND_HALF(a.scalar_type(), "matrix_to_axis_angle_bw", ([&] {
        kernel_matrix_to_axis_angle_bw<scalar_t><<<DIV_ROUND_UP(B, N_THREADS), N_THREADS>>>(out_grad.data_ptr<scalar_t>(), a.data_ptr<scalar_t>(), a_grad.data_ptr<scalar_t>(), B);
        gpuErrorCheck(cudaDeviceSynchronize());
    }));
}

// ===== Kernel+Host: Quaternion Translation Multiply Forward

template <typename scalar_t>
__global__ void kernel_quaternion_translation_mul_fw(const scalar_t *__restrict__ a_q, const scalar_t *__restrict__ a_t, const scalar_t *__restrict__ b_q, const scalar_t *__restrict__ b_t, scalar_t *__restrict__ out_q, scalar_t *__restrict__ out_t, const uint32_t B) {
    const uint32_t i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i >= B) {
        return;
    }
    _quaternion_translation_mul_fw<scalar_t>(a_q + i * 4, a_t + i * 3, b_q + i * 4, b_t + i * 3, out_q + i * 4, out_t + i * 3);
}

void quaternion_translation_mul_fw(at::Tensor a_q, at::Tensor a_t, at::Tensor b_q, at::Tensor b_t, at::Tensor out_q, at::Tensor out_t, const uint32_t B) {
    CHECK_IS_CONTIGUOUS_FLOAT_CUDA(a_q);
    CHECK_IS_CONTIGUOUS_FLOAT_CUDA(a_t);
    CHECK_IS_CONTIGUOUS_FLOAT_CUDA(b_q);
    CHECK_IS_CONTIGUOUS_FLOAT_CUDA(b_t);
    CHECK_IS_CONTIGUOUS_FLOAT_CUDA(out_q);
    CHECK_IS_CONTIGUOUS_FLOAT_CUDA(out_t);

    AT_DISPATCH_FLOATING_TYPES_AND_HALF(a_q.scalar_type(), "quaternion_translation_mul_fw", ([&] {
        kernel_quaternion_translation_mul_fw<scalar_t><<<DIV_ROUND_UP(B, N_THREADS), N_THREADS>>>(a_q.data_ptr<scalar_t>(), a_t.data_ptr<scalar_t>(), b_q.data_ptr<scalar_t>(), b_t.data_ptr<scalar_t>(), out_q.data_ptr<scalar_t>(), out_t.data_ptr<scalar_t>(), B);
        gpuErrorCheck(cudaDeviceSynchronize());
    }));
}

// ===== Kernel+Host: Quaternion Translation Multiply Backward

template <typename scalar_t>
__global__ void kernel_quaternion_translation_mul_bw(const scalar_t *__restrict__ out_q_grad, const scalar_t *__restrict__ out_t_grad, const scalar_t *__restrict__ a_q, const scalar_t *__restrict__ a_t, const scalar_t *__restrict__ b_q, const scalar_t *__restrict__ b_t, scalar_t *__restrict__ a_q_grad, scalar_t *__restrict__ a_t_grad, scalar_t *__restrict__ b_q_grad, scalar_t *__restrict__ b_t_grad, const uint32_t B) {
    const uint32_t i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i >= B) {
        return;
    }
    _quaternion_translation_mul_bw<scalar_t>(out_q_grad + i * 4, out_t_grad + i * 3, a_q + i * 4, a_t + i * 3, b_q + i * 4, b_t + i * 3, a_q_grad + i * 4, a_t_grad + i * 3, b_q_grad + i * 4, b_t_grad + i * 3);
}

void quaternion_translation_mul_bw(at::Tensor out_q_grad, at::Tensor out_t_grad, at::Tensor a_q, at::Tensor a_t, at::Tensor b_q, at::Tensor b_t, at::Tensor a_q_grad, at::Tensor a_t_grad, at::Tensor b_q_grad, at::Tensor b_t_grad, const uint32_t B) {
    CHECK_IS_CONTIGUOUS_FLOAT_CUDA(out_q_grad);
    CHECK_IS_CONTIGUOUS_FLOAT_CUDA(out_t_grad);
    CHECK_IS_CONTIGUOUS_FLOAT_CUDA(a_q);
    CHECK_IS_CONTIGUOUS_FLOAT_CUDA(a_t);
    CHECK_IS_CONTIGUOUS_FLOAT_CUDA(b_q);
    CHECK_IS_CONTIGUOUS_FLOAT_CUDA(b_t);
    CHECK_IS_CONTIGUOUS_FLOAT_CUDA(a_q_grad);
    CHECK_IS_CONTIGUOUS_FLOAT_CUDA(a_t_grad);
    CHECK_IS_CONTIGUOUS_FLOAT_CUDA(b_q_grad);
    CHECK_IS_CONTIGUOUS_FLOAT_CUDA(b_t_grad);

    AT_DISPATCH_FLOATING_TYPES_AND_HALF(a_q.scalar_type(), "quaternion_translation_mul_bw", ([&] {
        kernel_quaternion_translation_mul_bw<scalar_t><<<DIV_ROUND_UP(B, N_THREADS), N_THREADS>>>(out_q_grad.data_ptr<scalar_t>(), out_t_grad.data_ptr<scalar_t>(), a_q.data_ptr<scalar_t>(), a_t.data_ptr<scalar_t>(), b_q.data_ptr<scalar_t>(), b_t.data_ptr<scalar_t>(), a_q_grad.data_ptr<scalar_t>(), a_t_grad.data_ptr<scalar_t>(), b_q_grad.data_ptr<scalar_t>(), b_t_grad.data_ptr<scalar_t>(), B);
        gpuErrorCheck(cudaDeviceSynchronize());
    }));
}

// ===== Kernel+Host: Quaternion Translation Apply Forward

template <typename scalar_t>
__global__ void kernel_quaternion_translation_apply_fw(const scalar_t *__restrict__ a_q, const scalar_t *__restrict__ a_t, const scalar_t *__restrict__ b, scalar_t *__restrict__ out, const uint32_t B) {
    const uint32_t i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i >= B) {
        return;
    }
    _quaternion_translation_apply_fw<scalar_t>(a_q + i * 4, a_t + i * 3, b + i * 3, out + i * 3);
}

void quaternion_translation_apply_fw(at::Tensor a_q, at::Tensor a_t, at::Tensor b, at::Tensor out, const uint32_t B) {
    CHECK_IS_CONTIGUOUS_FLOAT_CUDA(a_q);
    CHECK_IS_CONTIGUOUS_FLOAT_CUDA(a_t);
    CHECK_IS_CONTIGUOUS_FLOAT_CUDA(b);
    CHECK_IS_CONTIGUOUS_FLOAT_CUDA(out);

    AT_DISPATCH_FLOATING_TYPES_AND_HALF(a_q.scalar_type(), "quaternion_translation_apply_fw", ([&] {
        kernel_quaternion_translation_apply_fw<scalar_t><<<DIV_ROUND_UP(B, N_THREADS), N_THREADS>>>(a_q.data_ptr<scalar_t>(), a_t.data_ptr<scalar_t>(), b.data_ptr<scalar_t>(), out.data_ptr<scalar_t>(), B);
        gpuErrorCheck(cudaDeviceSynchronize());
    }));
}

// ===== Kernel+Host: Quaternion Translation Apply Backward

template <typename scalar_t>
__global__ void kernel_quaternion_translation_apply_bw(const scalar_t *__restrict__ out_grad, const scalar_t *__restrict__ a_q, const scalar_t *__restrict__ a_t, const scalar_t *__restrict__ b, scalar_t *__restrict__ a_q_grad, scalar_t *__restrict__ a_t_grad, scalar_t *__restrict__ b_grad, const uint32_t B) {
    const uint32_t i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i >= B) {
        return;
    }
    _quaternion_translation_apply_bw<scalar_t>(out_grad + i * 3, a_q + i * 4, a_t + i * 3, b + i * 3, a_q_grad + i * 4, a_t_grad + i * 3, b_grad + i * 3);
}

void quaternion_translation_apply_bw(at::Tensor out_grad, at::Tensor a_q, at::Tensor a_t, at::Tensor b, at::Tensor a_q_grad, at::Tensor a_t_grad, at::Tensor b_grad, const uint32_t B) {
    CHECK_IS_CONTIGUOUS_FLOAT_CUDA(out_grad);
    CHECK_IS_CONTIGUOUS_FLOAT_CUDA(a_q);
    CHECK_IS_CONTIGUOUS_FLOAT_CUDA(a_t);
    CHECK_IS_CONTIGUOUS_FLOAT_CUDA(b);
    CHECK_IS_CONTIGUOUS_FLOAT_CUDA(a_q_grad);
    CHECK_IS_CONTIGUOUS_FLOAT_CUDA(a_t_grad);
    CHECK_IS_CONTIGUOUS_FLOAT_CUDA(b_grad);

    AT_DISPATCH_FLOATING_TYPES_AND_HALF(a_q.scalar_type(), "quaternion_translation_apply_bw", ([&] {
        kernel_quaternion_translation_apply_bw<scalar_t><<<DIV_ROUND_UP(B, N_THREADS), N_THREADS>>>(out_grad.data_ptr<scalar_t>(), a_q.data_ptr<scalar_t>(), a_t.data_ptr<scalar_t>(), b.data_ptr<scalar_t>(), a_q_grad.data_ptr<scalar_t>(), a_t_grad.data_ptr<scalar_t>(), b_grad.data_ptr<scalar_t>(), B);
        gpuErrorCheck(cudaDeviceSynchronize());
    }));
}

// ===== Kernel+Host: Quaternion Translation Inverse Forward

template <typename scalar_t>
__global__ void kernel_quaternion_translation_inverse_fw(const scalar_t *__restrict__ a_q, const scalar_t *__restrict__ a_t, scalar_t *__restrict__ out_q, scalar_t *__restrict__ out_t, const uint32_t B) {
    const uint32_t i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i >= B) {
        return;
    }
    _quaternion_translation_inverse_fw<scalar_t>(a_q + i * 4, a_t + i * 3, out_q + i * 4, out_t + i * 3);
}

void quaternion_translation_inverse_fw(at::Tensor a_q, at::Tensor a_t, at::Tensor out_q, at::Tensor out_t, const uint32_t B) {
    CHECK_IS_CONTIGUOUS_FLOAT_CUDA(a_q);
    CHECK_IS_CONTIGUOUS_FLOAT_CUDA(a_t);
    CHECK_IS_CONTIGUOUS_FLOAT_CUDA(out_q);
    CHECK_IS_CONTIGUOUS_FLOAT_CUDA(out_t);

    AT_DISPATCH_FLOATING_TYPES_AND_HALF(a_q.scalar_type(), "quaternion_translation_inverse_fw", ([&] {
        kernel_quaternion_translation_inverse_fw<scalar_t><<<DIV_ROUND_UP(B, N_THREADS), N_THREADS>>>(a_q.data_ptr<scalar_t>(), a_t.data_ptr<scalar_t>(), out_q.data_ptr<scalar_t>(), out_t.data_ptr<scalar_t>(), B);
        gpuErrorCheck(cudaDeviceSynchronize());
    }));
}

// ===== Kernel+Host: Quaternion Translation Inverse Backward

template <typename scalar_t>
__global__ void kernel_quaternion_translation_inverse_bw(const scalar_t *__restrict__ out_q_grad, const scalar_t *__restrict__ out_t_grad, const scalar_t *__restrict__ a_q, const scalar_t *__restrict__ a_t, scalar_t *__restrict__ a_q_grad, scalar_t *__restrict__ a_t_grad, const uint32_t B) {
    const uint32_t i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i >= B) {
        return;
    }
    _quaternion_translation_inverse_bw<scalar_t>(out_q_grad + i * 4, out_t_grad + i * 3, a_q + i * 4, a_t + i * 3, a_q_grad + i * 4, a_t_grad + i * 3);
}

void quaternion_translation_inverse_bw(at::Tensor out_q_grad, at::Tensor out_t_grad, at::Tensor a_q, at::Tensor a_t, at::Tensor a_q_grad, at::Tensor a_t_grad, const uint32_t B) {
    CHECK_IS_CONTIGUOUS_FLOAT_CUDA(out_q_grad);
    CHECK_IS_CONTIGUOUS_FLOAT_CUDA(out_t_grad);
    CHECK_IS_CONTIGUOUS_FLOAT_CUDA(a_q);
    CHECK_IS_CONTIGUOUS_FLOAT_CUDA(a_t);
    CHECK_IS_CONTIGUOUS_FLOAT_CUDA(a_q_grad);
    CHECK_IS_CONTIGUOUS_FLOAT_CUDA(a_t_grad);

    AT_DISPATCH_FLOATING_TYPES_AND_HALF(a_q.scalar_type(), "quaternion_translation_inverse_bw", ([&] {
        kernel_quaternion_translation_inverse_bw<scalar_t><<<DIV_ROUND_UP(B, N_THREADS), N_THREADS>>>(out_q_grad.data_ptr<scalar_t>(), out_t_grad.data_ptr<scalar_t>(), a_q.data_ptr<scalar_t>(), a_t.data_ptr<scalar_t>(), a_q_grad.data_ptr<scalar_t>(), a_t_grad.data_ptr<scalar_t>(), B);
        gpuErrorCheck(cudaDeviceSynchronize());
    }));
}

// ===== Kernel+Host: Dual Quaternion Multiply Forward

template <typename scalar_t>
__global__ void kernel_dual_quaternion_mul_fw(const scalar_t *__restrict__ a_r, const scalar_t *__restrict__ a_d, const scalar_t *__restrict__ b_r, const scalar_t *__restrict__ b_d, scalar_t *__restrict__ out_r, scalar_t *__restrict__ out_d, const uint32_t B) {
    const uint32_t i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i >= B) {
        return;
    }
    _dual_quaternion_mul_fw<scalar_t>(a_r + i * 4, a_d + i * 4, b_r + i * 4, b_d + i * 4, out_r + i * 4, out_d + i * 4);
}

void dual_quaternion_mul_fw(at::Tensor a_r, at::Tensor a_d, at::Tensor b_r, at::Tensor b_d, at::Tensor out_r, at::Tensor out_d, const uint32_t B) {
    CHECK_IS_CONTIGUOUS_FLOAT_CUDA(a_r);
    CHECK_IS_CONTIGUOUS_FLOAT_CUDA(a_d);
    CHECK_IS_CONTIGUOUS_FLOAT_CUDA(b_r);
    CHECK_IS_CONTIGUOUS_FLOAT_CUDA(b_d);
    CHECK_IS_CONTIGUOUS_FLOAT_CUDA(out_r);
    CHECK_IS_CONTIGUOUS_FLOAT_CUDA(out_d);

    AT_DISPATCH_FLOATING_TYPES_AND_HALF(a_r.scalar_type(), "dual_quaternion_mul_fw", ([&] {
        kernel_dual_quaternion_mul_fw<scalar_t><<<DIV_ROUND_UP(B, N_THREADS), N_THREADS>>>(a_r.data_ptr<scalar_t>(), a_d.data_ptr<scalar_t>(), b_r.data_ptr<scalar_t>(), b_d.data_ptr<scalar_t>(), out_r.data_ptr<scalar_t>(), out_d.data_ptr<scalar_t>(), B);
        gpuErrorCheck(cudaDeviceSynchronize());
    }));
}

// ===== Kernel+Host: Dual Quaternion Multiply Backward

template <typename scalar_t>
__global__ void kernel_dual_quaternion_mul_bw(const scalar_t *__restrict__ out_r_grad, const scalar_t *__restrict__ out_d_grad, const scalar_t *__restrict__ a_r, const scalar_t *__restrict__ a_d, const scalar_t *__restrict__ b_r, const scalar_t *__restrict__ b_d, scalar_t *__restrict__ a_r_grad, scalar_t *__restrict__ a_d_grad, scalar_t *__restrict__ b_r_grad, scalar_t *__restrict__ b_d_grad, const uint32_t B) {
    const uint32_t i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i >= B) {
        return;
    }
    _dual_quaternion_mul_bw<scalar_t>(out_r_grad + i * 4, out_d_grad + i * 4, a_r + i * 4, a_d + i * 4, b_r + i * 4, b_d + i * 4, a_r_grad + i * 4, a_d_grad + i * 4, b_r_grad + i * 4, b_d_grad + i * 4);
}

void dual_quaternion_mul_bw(at::Tensor out_r_grad, at::Tensor out_d_grad, at::Tensor a_r, at::Tensor a_d, at::Tensor b_r, at::Tensor b_d, at::Tensor a_r_grad, at::Tensor a_d_grad, at::Tensor b_r_grad, at::Tensor b_d_grad, const uint32_t B) {
    CHECK_IS_CONTIGUOUS_FLOAT_CUDA(out_r_grad);
    CHECK_IS_CONTIGUOUS_FLOAT_CUDA(out_d_grad);
    CHECK_IS_CONTIGUOUS_FLOAT_CUDA(a_r);
    CHECK_IS_CONTIGUOUS_FLOAT_CUDA(a_d);
    CHECK_IS_CONTIGUOUS_FLOAT_CUDA(b_r);
    CHECK_IS_CONTIGUOUS_FLOAT_CUDA(b_d);
    CHECK_IS_CONTIGUOUS_FLOAT_CUDA(a_r_grad);
    CHECK_IS_CONTIGUOUS_FLOAT_CUDA(a_d_grad);
    CHECK_IS_CONTIGUOUS_FLOAT_CUDA(b_r_grad);
    CHECK_IS_CONTIGUOUS_FLOAT_CUDA(b_d_grad);

    AT_DISPATCH_FLOATING_TYPES_AND_HALF(a_r.scalar_type(), "dual_quaternion_mul_bw", ([&] {
        kernel_dual_quaternion_mul_bw<scalar_t><<<DIV_ROUND_UP(B, N_THREADS), N_THREADS>>>(out_r_grad.data_ptr<scalar_t>(), out_d_grad.data_ptr<scalar_t>(), a_r.data_ptr<scalar_t>(), a_d.data_ptr<scalar_t>(), b_r.data_ptr<scalar_t>(), b_d.data_ptr<scalar_t>(), a_r_grad.data_ptr<scalar_t>(), a_d_grad.data_ptr<scalar_t>(), b_r_grad.data_ptr<scalar_t>(), b_d_grad.data_ptr<scalar_t>(), B);
        gpuErrorCheck(cudaDeviceSynchronize());
    }));
}

// ===== Kernel+Host: Dual Quaternion Apply Forward

template <typename scalar_t>
__global__ void kernel_dual_quaternion_apply_fw(const scalar_t *__restrict__ a_r, const scalar_t *__restrict__ a_d, const scalar_t *__restrict__ b, scalar_t *__restrict__ out, const uint32_t B) {
    const uint32_t i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i >= B) {
        return;
    }
    _dual_quaternion_apply_fw<scalar_t>(a_r + i * 4, a_d + i * 4, b + i * 3, out + i * 3);
}

void dual_quaternion_apply_fw(at::Tensor a_r, at::Tensor a_d, at::Tensor b, at::Tensor out, const uint32_t B) {
    CHECK_IS_CONTIGUOUS_FLOAT_CUDA(a_r);
    CHECK_IS_CONTIGUOUS_FLOAT_CUDA(a_d);
    CHECK_IS_CONTIGUOUS_FLOAT_CUDA(b);
    CHECK_IS_CONTIGUOUS_FLOAT_CUDA(out);

    AT_DISPATCH_FLOATING_TYPES_AND_HALF(a_r.scalar_type(), "dual_quaternion_apply_fw", ([&] {
        kernel_dual_quaternion_apply_fw<scalar_t><<<DIV_ROUND_UP(B, N_THREADS), N_THREADS>>>(a_r.data_ptr<scalar_t>(), a_d.data_ptr<scalar_t>(), b.data_ptr<scalar_t>(), out.data_ptr<scalar_t>(), B);
        gpuErrorCheck(cudaDeviceSynchronize());
    }));
}

// ===== Kernel+Host: Dual Quaternion Apply Backward

template <typename scalar_t>
__global__ void kernel_dual_quaternion_apply_bw(const scalar_t *__restrict__ out_grad, const scalar_t *__restrict__ a_r, const scalar_t *__restrict__ a_d, const scalar_t *__restrict__ b, scalar_t *__restrict__ a_r_grad, scalar_t *__restrict__ a_d_grad, scalar_t *__restrict__ b_grad, const uint32_t B) {
    const uint32_t i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i >= B) {
        return;
    }
    _dual_quaternion_apply_bw<scalar_t>(out_grad + i * 3, a_r + i * 4, a_d + i * 4, b + i * 3, a_r_grad + i * 4, a_d_grad + i * 4, b_grad + i * 3);
}

void dual_quaternion_apply_bw(at::Tensor out_grad, at::Tensor a_r, at::Tensor a_d, at::Tensor b, at::Tensor a_r_grad, at::Tensor a_d_grad, at::Tensor b_grad, const uint32_t B) {
    CHECK_IS_CONTIGUOUS_FLOAT_CUDA(out_grad);
    CHECK_IS_CONTIGUOUS_FLOAT_CUDA(a_r);
    CHECK_IS_CONTIGUOUS_FLOAT_CUDA(a_d);
    CHECK_IS_CONTIGUOUS_FLOAT_CUDA(b);
    CHECK_IS_CONTIGUOUS_FLOAT_CUDA(a_r_grad);
    CHECK_IS_CONTIGUOUS_FLOAT_CUDA(a_d_grad);
    CHECK_IS_CONTIGUOUS_FLOAT_CUDA(b_grad);

    AT_DISPATCH_FLOATING_TYPES_AND_HALF(a_r.scalar_type(), "dual_quaternion_apply_bw", ([&] {
        kernel_dual_quaternion_apply_bw<scalar_t><<<DIV_ROUND_UP(B, N_THREADS), N_THREADS>>>(out_grad.data_ptr<scalar_t>(), a_r.data_ptr<scalar_t>(), a_d.data_ptr<scalar_t>(), b.data_ptr<scalar_t>(), a_r_grad.data_ptr<scalar_t>(), a_d_grad.data_ptr<scalar_t>(), b_grad.data_ptr<scalar_t>(), B);
        gpuErrorCheck(cudaDeviceSynchronize());
    }));
}

// ===== Kernel+Host: Dual Quaternion Q Conjugate Forward

template <typename scalar_t>
__global__ void kernel_dual_quaternion_q_conjugate_fw(const scalar_t *__restrict__ a_r, const scalar_t *__restrict__ a_d, scalar_t *__restrict__ out_r, scalar_t *__restrict__ out_d, const uint32_t B) {
    const uint32_t i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i >= B) {
        return;
    }
    _dual_quaternion_q_conjugate_fw<scalar_t>(a_r + i * 4, a_d + i * 4, out_r + i * 4, out_d + i * 4);
}

void dual_quaternion_q_conjugate_fw(at::Tensor a_r, at::Tensor a_d, at::Tensor out_r, at::Tensor out_d, const uint32_t B) {
    CHECK_IS_CONTIGUOUS_FLOAT_CUDA(a_r);
    CHECK_IS_CONTIGUOUS_FLOAT_CUDA(a_d);
    CHECK_IS_CONTIGUOUS_FLOAT_CUDA(out_r);
    CHECK_IS_CONTIGUOUS_FLOAT_CUDA(out_d);

    AT_DISPATCH_FLOATING_TYPES_AND_HALF(a_r.scalar_type(), "dual_quaternion_q_conjugate_fw", ([&] {
        kernel_dual_quaternion_q_conjugate_fw<scalar_t><<<DIV_ROUND_UP(B, N_THREADS), N_THREADS>>>(a_r.data_ptr<scalar_t>(), a_d.data_ptr<scalar_t>(), out_r.data_ptr<scalar_t>(), out_d.data_ptr<scalar_t>(), B);
        gpuErrorCheck(cudaDeviceSynchronize());
    }));
}

// ===== Kernel+Host: Dual Quaternion Q Conjugate Backward

template <typename scalar_t>
__global__ void kernel_dual_quaternion_q_conjugate_bw(const scalar_t *__restrict__ out_r_grad, const scalar_t *__restrict__ out_d_grad, const scalar_t *__restrict__ a_r, const scalar_t *__restrict__ a_d, scalar_t *__restrict__ a_r_grad, scalar_t *__restrict__ a_d_grad, const uint32_t B) {
    const uint32_t i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i >= B) {
        return;
    }
    _dual_quaternion_q_conjugate_bw<scalar_t>(out_r_grad + i * 4, out_d_grad + i * 4, a_r + i * 4, a_d + i * 4, a_r_grad + i * 4, a_d_grad + i * 4);
}

void dual_quaternion_q_conjugate_bw(at::Tensor out_r_grad, at::Tensor out_d_grad, at::Tensor a_r, at::Tensor a_d, at::Tensor a_r_grad, at::Tensor a_d_grad, const uint32_t B) {
    CHECK_IS_CONTIGUOUS_FLOAT_CUDA(out_r_grad);
    CHECK_IS_CONTIGUOUS_FLOAT_CUDA(out_d_grad);
    CHECK_IS_CONTIGUOUS_FLOAT_CUDA(a_r);
    CHECK_IS_CONTIGUOUS_FLOAT_CUDA(a_d);
    CHECK_IS_CONTIGUOUS_FLOAT_CUDA(a_r_grad);
    CHECK_IS_CONTIGUOUS_FLOAT_CUDA(a_d_grad);

    AT_DISPATCH_FLOATING_TYPES_AND_HALF(a_r.scalar_type(), "dual_quaternion_q_conjugate_bw", ([&] {
        kernel_dual_quaternion_q_conjugate_bw<scalar_t><<<DIV_ROUND_UP(B, N_THREADS), N_THREADS>>>(out_r_grad.data_ptr<scalar_t>(), out_d_grad.data_ptr<scalar_t>(), a_r.data_ptr<scalar_t>(), a_d.data_ptr<scalar_t>(), a_r_grad.data_ptr<scalar_t>(), a_d_grad.data_ptr<scalar_t>(), B);
        gpuErrorCheck(cudaDeviceSynchronize());
    }));
}

// ===== Kernel+Host: Dual Quaternion D Conjugate Forward

template <typename scalar_t>
__global__ void kernel_dual_quaternion_d_conjugate_fw(const scalar_t *__restrict__ a_r, const scalar_t *__restrict__ a_d, scalar_t *__restrict__ out_r, scalar_t *__restrict__ out_d, const uint32_t B) {
    const uint32_t i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i >= B) {
        return;
    }
    _dual_quaternion_d_conjugate_fw<scalar_t>(a_r + i * 4, a_d + i * 4, out_r + i * 4, out_d + i * 4);
}

void dual_quaternion_d_conjugate_fw(at::Tensor a_r, at::Tensor a_d, at::Tensor out_r, at::Tensor out_d, const uint32_t B) {
    CHECK_IS_CONTIGUOUS_FLOAT_CUDA(a_r);
    CHECK_IS_CONTIGUOUS_FLOAT_CUDA(a_d);
    CHECK_IS_CONTIGUOUS_FLOAT_CUDA(out_r);
    CHECK_IS_CONTIGUOUS_FLOAT_CUDA(out_d);

    AT_DISPATCH_FLOATING_TYPES_AND_HALF(a_r.scalar_type(), "dual_quaternion_d_conjugate_fw", ([&] {
        kernel_dual_quaternion_d_conjugate_fw<scalar_t><<<DIV_ROUND_UP(B, N_THREADS), N_THREADS>>>(a_r.data_ptr<scalar_t>(), a_d.data_ptr<scalar_t>(), out_r.data_ptr<scalar_t>(), out_d.data_ptr<scalar_t>(), B);
        gpuErrorCheck(cudaDeviceSynchronize());
    }));
}

// ===== Kernel+Host: Dual Quaternion D Conjugate Backward

template <typename scalar_t>
__global__ void kernel_dual_quaternion_d_conjugate_bw(const scalar_t *__restrict__ out_r_grad, const scalar_t *__restrict__ out_d_grad, const scalar_t *__restrict__ a_r, const scalar_t *__restrict__ a_d, scalar_t *__restrict__ a_r_grad, scalar_t *__restrict__ a_d_grad, const uint32_t B) {
    const uint32_t i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i >= B) {
        return;
    }
    _dual_quaternion_d_conjugate_bw<scalar_t>(out_r_grad + i * 4, out_d_grad + i * 4, a_r + i * 4, a_d + i * 4, a_r_grad + i * 4, a_d_grad + i * 4);
}

void dual_quaternion_d_conjugate_bw(at::Tensor out_r_grad, at::Tensor out_d_grad, at::Tensor a_r, at::Tensor a_d, at::Tensor a_r_grad, at::Tensor a_d_grad, const uint32_t B) {
    CHECK_IS_CONTIGUOUS_FLOAT_CUDA(out_r_grad);
    CHECK_IS_CONTIGUOUS_FLOAT_CUDA(out_d_grad);
    CHECK_IS_CONTIGUOUS_FLOAT_CUDA(a_r);
    CHECK_IS_CONTIGUOUS_FLOAT_CUDA(a_d);
    CHECK_IS_CONTIGUOUS_FLOAT_CUDA(a_r_grad);
    CHECK_IS_CONTIGUOUS_FLOAT_CUDA(a_d_grad);

    AT_DISPATCH_FLOATING_TYPES_AND_HALF(a_r.scalar_type(), "dual_quaternion_d_conjugate_bw", ([&] {
        kernel_dual_quaternion_d_conjugate_bw<scalar_t><<<DIV_ROUND_UP(B, N_THREADS), N_THREADS>>>(out_r_grad.data_ptr<scalar_t>(), out_d_grad.data_ptr<scalar_t>(), a_r.data_ptr<scalar_t>(), a_d.data_ptr<scalar_t>(), a_r_grad.data_ptr<scalar_t>(), a_d_grad.data_ptr<scalar_t>(), B);
        gpuErrorCheck(cudaDeviceSynchronize());
    }));
}

// ===== Kernel+Host: Dual Quaternion 3rd Conjugate Forward

template <typename scalar_t>
__global__ void kernel_dual_quaternion_3rd_conjugate_fw(const scalar_t *__restrict__ a_r, const scalar_t *__restrict__ a_d, scalar_t *__restrict__ out_r, scalar_t *__restrict__ out_d, const uint32_t B) {
    const uint32_t i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i >= B) {
        return;
    }
    _dual_quaternion_3rd_conjugate_fw<scalar_t>(a_r + i * 4, a_d + i * 4, out_r + i * 4, out_d + i * 4);
}

void dual_quaternion_3rd_conjugate_fw(at::Tensor a_r, at::Tensor a_d, at::Tensor out_r, at::Tensor out_d, const uint32_t B) {
    CHECK_IS_CONTIGUOUS_FLOAT_CUDA(a_r);
    CHECK_IS_CONTIGUOUS_FLOAT_CUDA(a_d);
    CHECK_IS_CONTIGUOUS_FLOAT_CUDA(out_r);
    CHECK_IS_CONTIGUOUS_FLOAT_CUDA(out_d);

    AT_DISPATCH_FLOATING_TYPES_AND_HALF(a_r.scalar_type(), "dual_quaternion_3rd_conjugate_fw", ([&] {
        kernel_dual_quaternion_3rd_conjugate_fw<scalar_t><<<DIV_ROUND_UP(B, N_THREADS), N_THREADS>>>(a_r.data_ptr<scalar_t>(), a_d.data_ptr<scalar_t>(), out_r.data_ptr<scalar_t>(), out_d.data_ptr<scalar_t>(), B);
        gpuErrorCheck(cudaDeviceSynchronize());
    }));
}

// ===== Kernel+Host: Dual Quaternion 3rd Conjugate Backward

template <typename scalar_t>
__global__ void kernel_dual_quaternion_3rd_conjugate_bw(const scalar_t *__restrict__ out_r_grad, const scalar_t *__restrict__ out_d_grad, const scalar_t *__restrict__ a_r, const scalar_t *__restrict__ a_d, scalar_t *__restrict__ a_r_grad, scalar_t *__restrict__ a_d_grad, const uint32_t B) {
    const uint32_t i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i >= B) {
        return;
    }
    _dual_quaternion_3rd_conjugate_bw<scalar_t>(out_r_grad + i * 4, out_d_grad + i * 4, a_r + i * 4, a_d + i * 4, a_r_grad + i * 4, a_d_grad + i * 4);
}

void dual_quaternion_3rd_conjugate_bw(at::Tensor out_r_grad, at::Tensor out_d_grad, at::Tensor a_r, at::Tensor a_d, at::Tensor a_r_grad, at::Tensor a_d_grad, const uint32_t B) {
    CHECK_IS_CONTIGUOUS_FLOAT_CUDA(out_r_grad);
    CHECK_IS_CONTIGUOUS_FLOAT_CUDA(out_d_grad);
    CHECK_IS_CONTIGUOUS_FLOAT_CUDA(a_r);
    CHECK_IS_CONTIGUOUS_FLOAT_CUDA(a_d);
    CHECK_IS_CONTIGUOUS_FLOAT_CUDA(a_r_grad);
    CHECK_IS_CONTIGUOUS_FLOAT_CUDA(a_d_grad);

    AT_DISPATCH_FLOATING_TYPES_AND_HALF(a_r.scalar_type(), "dual_quaternion_3rd_conjugate_bw", ([&] {
        kernel_dual_quaternion_3rd_conjugate_bw<scalar_t><<<DIV_ROUND_UP(B, N_THREADS), N_THREADS>>>(out_r_grad.data_ptr<scalar_t>(), out_d_grad.data_ptr<scalar_t>(), a_r.data_ptr<scalar_t>(), a_d.data_ptr<scalar_t>(), a_r_grad.data_ptr<scalar_t>(), a_d_grad.data_ptr<scalar_t>(), B);
        gpuErrorCheck(cudaDeviceSynchronize());
    }));
}

// ===== Kernel+Host: Dual Quaternion Norm Forward

template <typename scalar_t>
__global__ void kernel_dual_quaternion_norm_fw(const scalar_t *__restrict__ a_r, const scalar_t *__restrict__ a_d, scalar_t *__restrict__ out_r, scalar_t *__restrict__ out_d, const uint32_t B) {
    const uint32_t i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i >= B) {
        return;
    }
    _dual_quaternion_norm_fw<scalar_t>(a_r + i * 4, a_d + i * 4, out_r + i * 4, out_d + i * 4);
}

void dual_quaternion_norm_fw(at::Tensor a_r, at::Tensor a_d, at::Tensor out_r, at::Tensor out_d, const uint32_t B) {
    CHECK_IS_CONTIGUOUS_FLOAT_CUDA(a_r);
    CHECK_IS_CONTIGUOUS_FLOAT_CUDA(a_d);
    CHECK_IS_CONTIGUOUS_FLOAT_CUDA(out_r);
    CHECK_IS_CONTIGUOUS_FLOAT_CUDA(out_d);

    AT_DISPATCH_FLOATING_TYPES_AND_HALF(a_r.scalar_type(), "dual_quaternion_norm_fw", ([&] {
        kernel_dual_quaternion_norm_fw<scalar_t><<<DIV_ROUND_UP(B, N_THREADS), N_THREADS>>>(a_r.data_ptr<scalar_t>(), a_d.data_ptr<scalar_t>(), out_r.data_ptr<scalar_t>(), out_d.data_ptr<scalar_t>(), B);
        gpuErrorCheck(cudaDeviceSynchronize());
    }));
}

// ===== Kernel+Host: Dual Quaternion Norm Backward

template <typename scalar_t>
__global__ void kernel_dual_quaternion_norm_bw(const scalar_t *__restrict__ out_r_grad, const scalar_t *__restrict__ out_d_grad, const scalar_t *__restrict__ a_r, const scalar_t *__restrict__ a_d, scalar_t *__restrict__ a_r_grad, scalar_t *__restrict__ a_d_grad, const uint32_t B) {
    const uint32_t i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i >= B) {
        return;
    }
    _dual_quaternion_norm_bw<scalar_t>(out_r_grad + i * 4, out_d_grad + i * 4, a_r + i * 4, a_d + i * 4, a_r_grad + i * 4, a_d_grad + i * 4);
}

void dual_quaternion_norm_bw(at::Tensor out_r_grad, at::Tensor out_d_grad, at::Tensor a_r, at::Tensor a_d, at::Tensor a_r_grad, at::Tensor a_d_grad, const uint32_t B) {
    CHECK_IS_CONTIGUOUS_FLOAT_CUDA(out_r_grad);
    CHECK_IS_CONTIGUOUS_FLOAT_CUDA(out_d_grad);
    CHECK_IS_CONTIGUOUS_FLOAT_CUDA(a_r);
    CHECK_IS_CONTIGUOUS_FLOAT_CUDA(a_d);
    CHECK_IS_CONTIGUOUS_FLOAT_CUDA(a_r_grad);
    CHECK_IS_CONTIGUOUS_FLOAT_CUDA(a_d_grad);

    AT_DISPATCH_FLOATING_TYPES_AND_HALF(a_r.scalar_type(), "dual_quaternion_norm_bw", ([&] {
        kernel_dual_quaternion_norm_bw<scalar_t><<<DIV_ROUND_UP(B, N_THREADS), N_THREADS>>>(out_r_grad.data_ptr<scalar_t>(), out_d_grad.data_ptr<scalar_t>(), a_r.data_ptr<scalar_t>(), a_d.data_ptr<scalar_t>(), a_r_grad.data_ptr<scalar_t>(), a_d_grad.data_ptr<scalar_t>(), B);
        gpuErrorCheck(cudaDeviceSynchronize());
    }));
}

// ===== Kernel+Host: Quaternion Translation to Dual Quaternion Forward

template <typename scalar_t>
__global__ void kernel_quaternion_translation_to_dual_quaternion_fw(const scalar_t *__restrict__ a_q, const scalar_t *__restrict__ a_t, scalar_t *__restrict__ out_r, scalar_t *__restrict__ out_d, const uint32_t B) {
    const uint32_t i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i >= B) {
        return;
    }
    _quaternion_translation_to_dual_quaternion_fw<scalar_t>(a_q + i * 4, a_t + i * 3, out_r + i * 4, out_d + i * 4);
}

void quaternion_translation_to_dual_quaternion_fw(at::Tensor a_q, at::Tensor a_t, at::Tensor out_r, at::Tensor out_d, const uint32_t B) {
    CHECK_IS_CONTIGUOUS_FLOAT_CUDA(a_q);
    CHECK_IS_CONTIGUOUS_FLOAT_CUDA(a_t);
    CHECK_IS_CONTIGUOUS_FLOAT_CUDA(out_r);
    CHECK_IS_CONTIGUOUS_FLOAT_CUDA(out_d);

    AT_DISPATCH_FLOATING_TYPES_AND_HALF(a_q.scalar_type(), "quaternion_translation_to_dual_quaternion_fw", ([&] {
        kernel_quaternion_translation_to_dual_quaternion_fw<scalar_t><<<DIV_ROUND_UP(B, N_THREADS), N_THREADS>>>(a_q.data_ptr<scalar_t>(), a_t.data_ptr<scalar_t>(), out_r.data_ptr<scalar_t>(), out_d.data_ptr<scalar_t>(), B);
        gpuErrorCheck(cudaDeviceSynchronize());
    }));
}

// ===== Kernel+Host: Quaternion Translation to Dual Quaternion Backward

template <typename scalar_t>
__global__ void kernel_quaternion_translation_to_dual_quaternion_bw(const scalar_t *__restrict__ out_r_grad, const scalar_t *__restrict__ out_d_grad, const scalar_t *__restrict__ a_q, const scalar_t *__restrict__ a_t, scalar_t *__restrict__ a_q_grad, scalar_t *__restrict__ a_t_grad, const uint32_t B) {
    const uint32_t i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i >= B) {
        return;
    }
    _quaternion_translation_to_dual_quaternion_bw<scalar_t>(out_r_grad + i * 4, out_d_grad + i * 4, a_q + i * 4, a_t + i * 3, a_q_grad + i * 4, a_t_grad + i * 3);
}

void quaternion_translation_to_dual_quaternion_bw(at::Tensor out_r_grad, at::Tensor out_d_grad, at::Tensor a_q, at::Tensor a_t, at::Tensor a_q_grad, at::Tensor a_t_grad, const uint32_t B) {
    CHECK_IS_CONTIGUOUS_FLOAT_CUDA(out_r_grad);
    CHECK_IS_CONTIGUOUS_FLOAT_CUDA(out_d_grad);
    CHECK_IS_CONTIGUOUS_FLOAT_CUDA(a_q);
    CHECK_IS_CONTIGUOUS_FLOAT_CUDA(a_t);
    CHECK_IS_CONTIGUOUS_FLOAT_CUDA(a_q_grad);
    CHECK_IS_CONTIGUOUS_FLOAT_CUDA(a_t_grad);

    AT_DISPATCH_FLOATING_TYPES_AND_HALF(a_q.scalar_type(), "quaternion_translation_to_dual_quaternion_bw", ([&] {
        kernel_quaternion_translation_to_dual_quaternion_bw<scalar_t><<<DIV_ROUND_UP(B, N_THREADS), N_THREADS>>>(out_r_grad.data_ptr<scalar_t>(), out_d_grad.data_ptr<scalar_t>(), a_q.data_ptr<scalar_t>(), a_t.data_ptr<scalar_t>(), a_q_grad.data_ptr<scalar_t>(), a_t_grad.data_ptr<scalar_t>(), B);
        gpuErrorCheck(cudaDeviceSynchronize());
    }));
}

// ===== Kernel+Host: Dual Quaternion to Quaternion Translation Forward

template <typename scalar_t>
__global__ void kernel_dual_quaternion_to_quaternion_translation_fw(const scalar_t *__restrict__ a_r, const scalar_t *__restrict__ a_d, scalar_t *__restrict__ out_q, scalar_t *__restrict__ out_t, const uint32_t B) {
    const uint32_t i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i >= B) {
        return;
    }
    _dual_quaternion_to_quaternion_translation_fw<scalar_t>(a_r + i * 4, a_d + i * 4, out_q + i * 4, out_t + i * 3);
}

void dual_quaternion_to_quaternion_translation_fw(at::Tensor a_r, at::Tensor a_d, at::Tensor out_q, at::Tensor out_t, const uint32_t B) {
    CHECK_IS_CONTIGUOUS_FLOAT_CUDA(a_r);
    CHECK_IS_CONTIGUOUS_FLOAT_CUDA(a_d);
    CHECK_IS_CONTIGUOUS_FLOAT_CUDA(out_q);
    CHECK_IS_CONTIGUOUS_FLOAT_CUDA(out_t);

    AT_DISPATCH_FLOATING_TYPES_AND_HALF(a_r.scalar_type(), "dual_quaternion_to_quaternion_translation_fw", ([&] {
        kernel_dual_quaternion_to_quaternion_translation_fw<scalar_t><<<DIV_ROUND_UP(B, N_THREADS), N_THREADS>>>(a_r.data_ptr<scalar_t>(), a_d.data_ptr<scalar_t>(), out_q.data_ptr<scalar_t>(), out_t.data_ptr<scalar_t>(), B);
        gpuErrorCheck(cudaDeviceSynchronize());
    }));
}

// ===== Kernel+Host: Dual Quaternion to Quaternion Translation Backward

template <typename scalar_t>
__global__ void kernel_dual_quaternion_to_quaternion_translation_bw(const scalar_t *__restrict__ out_q_grad, const scalar_t *__restrict__ out_t_grad, const scalar_t *__restrict__ a_r, const scalar_t *__restrict__ a_d, scalar_t *__restrict__ a_r_grad, scalar_t *__restrict__ a_d_grad, const uint32_t B) {
    const uint32_t i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i >= B) {
        return;
    }
    _dual_quaternion_to_quaternion_translation_bw<scalar_t>(out_q_grad + i * 4, out_t_grad + i * 3, a_r + i * 4, a_d + i * 4, a_r_grad + i * 4, a_d_grad + i * 4);
}

void dual_quaternion_to_quaternion_translation_bw(at::Tensor out_q_grad, at::Tensor out_t_grad, at::Tensor a_r, at::Tensor a_d, at::Tensor a_r_grad, at::Tensor a_d_grad, const uint32_t B) {
    CHECK_IS_CONTIGUOUS_FLOAT_CUDA(out_q_grad);
    CHECK_IS_CONTIGUOUS_FLOAT_CUDA(out_t_grad);
    CHECK_IS_CONTIGUOUS_FLOAT_CUDA(a_r);
    CHECK_IS_CONTIGUOUS_FLOAT_CUDA(a_d);
    CHECK_IS_CONTIGUOUS_FLOAT_CUDA(a_r_grad);
    CHECK_IS_CONTIGUOUS_FLOAT_CUDA(a_d_grad);

    AT_DISPATCH_FLOATING_TYPES_AND_HALF(a_r.scalar_type(), "dual_quaternion_to_quaternion_translation_bw", ([&] {
        kernel_dual_quaternion_to_quaternion_translation_bw<scalar_t><<<DIV_ROUND_UP(B, N_THREADS), N_THREADS>>>(out_q_grad.data_ptr<scalar_t>(), out_t_grad.data_ptr<scalar_t>(), a_r.data_ptr<scalar_t>(), a_d.data_ptr<scalar_t>(), a_r_grad.data_ptr<scalar_t>(), a_d_grad.data_ptr<scalar_t>(), B);
        gpuErrorCheck(cudaDeviceSynchronize());
    }));
}

// ===== Kernel+Host: Quaternion Translation to SE3 Forward

template <typename scalar_t>
__global__ void kernel_quaternion_translation_to_se3_fw(const scalar_t *__restrict__ a_q, const scalar_t *__restrict__ a_t, scalar_t *__restrict__ out, const uint32_t B) {
    const uint32_t i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i >= B) {
        return;
    }
    _quaternion_translation_to_se3_fw<scalar_t>(a_q + i * 4, a_t + i * 3, out + i * 16);
}

void quaternion_translation_to_se3_fw(at::Tensor a_q, at::Tensor a_t, at::Tensor out, const uint32_t B) {
    CHECK_IS_CONTIGUOUS_FLOAT_CUDA(a_q);
    CHECK_IS_CONTIGUOUS_FLOAT_CUDA(a_t);
    CHECK_IS_CONTIGUOUS_FLOAT_CUDA(out);

    AT_DISPATCH_FLOATING_TYPES_AND_HALF(a_q.scalar_type(), "quaternion_translation_to_se3_fw", ([&] {
        kernel_quaternion_translation_to_se3_fw<scalar_t><<<DIV_ROUND_UP(B, N_THREADS), N_THREADS>>>(a_q.data_ptr<scalar_t>(), a_t.data_ptr<scalar_t>(), out.data_ptr<scalar_t>(), B);
        gpuErrorCheck(cudaDeviceSynchronize());
    }));
}

// ===== Kernel+Host: Quaternion Translation to SE3 Backward

template <typename scalar_t>
__global__ void kernel_quaternion_translation_to_se3_bw(const scalar_t *__restrict__ out_grad, const scalar_t *__restrict__ a_q, const scalar_t *__restrict__ a_t, scalar_t *__restrict__ a_q_grad, scalar_t *__restrict__ a_t_grad, const uint32_t B) {
    const uint32_t i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i >= B) {
        return;
    }
    _quaternion_translation_to_se3_bw<scalar_t>(out_grad + i * 16, a_q + i * 4, a_t + i * 3, a_q_grad + i * 4, a_t_grad + i * 3);
}

void quaternion_translation_to_se3_bw(at::Tensor out_grad, at::Tensor a_q, at::Tensor a_t, at::Tensor a_q_grad, at::Tensor a_t_grad, const uint32_t B) {
    CHECK_IS_CONTIGUOUS_FLOAT_CUDA(out_grad);
    CHECK_IS_CONTIGUOUS_FLOAT_CUDA(a_q);
    CHECK_IS_CONTIGUOUS_FLOAT_CUDA(a_t);
    CHECK_IS_CONTIGUOUS_FLOAT_CUDA(a_q_grad);
    CHECK_IS_CONTIGUOUS_FLOAT_CUDA(a_t_grad);

    AT_DISPATCH_FLOATING_TYPES_AND_HALF(a_q.scalar_type(), "quaternion_translation_to_se3_bw", ([&] {
        kernel_quaternion_translation_to_se3_bw<scalar_t><<<DIV_ROUND_UP(B, N_THREADS), N_THREADS>>>(out_grad.data_ptr<scalar_t>(), a_q.data_ptr<scalar_t>(), a_t.data_ptr<scalar_t>(), a_q_grad.data_ptr<scalar_t>(), a_t_grad.data_ptr<scalar_t>(), B);
        gpuErrorCheck(cudaDeviceSynchronize());
    }));
}

// ===== Kernel+Host: SE3 to Quaternion Translation Forward

template <typename scalar_t>
__global__ void kernel_se3_to_quaternion_translation_fw(const scalar_t *__restrict__ a, scalar_t *__restrict__ out_q, scalar_t *__restrict__ out_t, const uint32_t B) {
    const uint32_t i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i >= B) {
        return;
    }
    _se3_to_quaternion_translation_fw<scalar_t>(a + i * 16, out_q + i * 4, out_t + i * 3);
}

void se3_to_quaternion_translation_fw(at::Tensor a, at::Tensor out_q, at::Tensor out_t, const uint32_t B) {
    CHECK_IS_CONTIGUOUS_FLOAT_CUDA(a);
    CHECK_IS_CONTIGUOUS_FLOAT_CUDA(out_q);
    CHECK_IS_CONTIGUOUS_FLOAT_CUDA(out_t);

    AT_DISPATCH_FLOATING_TYPES_AND_HALF(a.scalar_type(), "se3_to_quaternion_translation_fw", ([&] {
        kernel_se3_to_quaternion_translation_fw<scalar_t><<<DIV_ROUND_UP(B, N_THREADS), N_THREADS>>>(a.data_ptr<scalar_t>(), out_q.data_ptr<scalar_t>(), out_t.data_ptr<scalar_t>(), B);
        gpuErrorCheck(cudaDeviceSynchronize());
    }));
}

// ===== Kernel+Host: SE3 to Quaternion Translation Backward

template <typename scalar_t>
__global__ void kernel_se3_to_quaternion_translation_bw(const scalar_t *__restrict__ out_q_grad, const scalar_t *__restrict__ out_t_grad, const scalar_t *__restrict__ a, scalar_t *__restrict__ a_grad, const uint32_t B) {
    const uint32_t i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i >= B) {
        return;
    }
    _se3_to_quaternion_translation_bw<scalar_t>(out_q_grad + i * 4, out_t_grad + i * 3, a + i * 16, a_grad + i * 16);
}

void se3_to_quaternion_translation_bw(at::Tensor out_q_grad, at::Tensor out_t_grad, at::Tensor a, at::Tensor a_grad, const uint32_t B) {
    CHECK_IS_CONTIGUOUS_FLOAT_CUDA(out_q_grad);
    CHECK_IS_CONTIGUOUS_FLOAT_CUDA(out_t_grad);
    CHECK_IS_CONTIGUOUS_FLOAT_CUDA(a);
    CHECK_IS_CONTIGUOUS_FLOAT_CUDA(a_grad);

    AT_DISPATCH_FLOATING_TYPES_AND_HALF(a.scalar_type(), "se3_to_quaternion_translation_bw", ([&] {
        kernel_se3_to_quaternion_translation_bw<scalar_t><<<DIV_ROUND_UP(B, N_THREADS), N_THREADS>>>(out_q_grad.data_ptr<scalar_t>(), out_t_grad.data_ptr<scalar_t>(), a.data_ptr<scalar_t>(), a_grad.data_ptr<scalar_t>(), B);
        gpuErrorCheck(cudaDeviceSynchronize());
    }));
}

// ===== Kernel+Host: Dual Quaternion to SE3 Forward

template <typename scalar_t>
__global__ void kernel_dual_quaternion_to_se3_fw(const scalar_t *__restrict__ a_r, const scalar_t *__restrict__ a_d, scalar_t *__restrict__ out, const uint32_t B) {
    const uint32_t i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i >= B) {
        return;
    }
    _dual_quaternion_to_se3_fw<scalar_t>(a_r + i * 4, a_d + i * 4, out + i * 16);
}

void dual_quaternion_to_se3_fw(at::Tensor a_r, at::Tensor a_d, at::Tensor out, const uint32_t B) {
    CHECK_IS_CONTIGUOUS_FLOAT_CUDA(a_r);
    CHECK_IS_CONTIGUOUS_FLOAT_CUDA(a_d);
    CHECK_IS_CONTIGUOUS_FLOAT_CUDA(out);

    AT_DISPATCH_FLOATING_TYPES_AND_HALF(a_r.scalar_type(), "dual_quaternion_to_se3_fw", ([&] {
        kernel_dual_quaternion_to_se3_fw<scalar_t><<<DIV_ROUND_UP(B, N_THREADS), N_THREADS>>>(a_r.data_ptr<scalar_t>(), a_d.data_ptr<scalar_t>(), out.data_ptr<scalar_t>(), B);
        gpuErrorCheck(cudaDeviceSynchronize());
    }));
}

// ===== Kernel+Host: Dual Quaternion to SE3 Backward

template <typename scalar_t>
__global__ void kernel_dual_quaternion_to_se3_bw(const scalar_t *__restrict__ out_grad, const scalar_t *__restrict__ a_r, const scalar_t *__restrict__ a_d, scalar_t *__restrict__ a_r_grad, scalar_t *__restrict__ a_d_grad, const uint32_t B) {
    const uint32_t i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i >= B) {
        return;
    }
    _dual_quaternion_to_se3_bw<scalar_t>(out_grad + i * 16, a_r + i * 4, a_d + i * 4, a_r_grad + i * 4, a_d_grad + i * 4);
}

void dual_quaternion_to_se3_bw(at::Tensor out_grad, at::Tensor a_r, at::Tensor a_d, at::Tensor a_r_grad, at::Tensor a_d_grad, const uint32_t B) {
    CHECK_IS_CONTIGUOUS_FLOAT_CUDA(out_grad);
    CHECK_IS_CONTIGUOUS_FLOAT_CUDA(a_r);
    CHECK_IS_CONTIGUOUS_FLOAT_CUDA(a_d);
    CHECK_IS_CONTIGUOUS_FLOAT_CUDA(a_r_grad);
    CHECK_IS_CONTIGUOUS_FLOAT_CUDA(a_d_grad);

    AT_DISPATCH_FLOATING_TYPES_AND_HALF(a_r.scalar_type(), "dual_quaternion_to_se3_bw", ([&] {
        kernel_dual_quaternion_to_se3_bw<scalar_t><<<DIV_ROUND_UP(B, N_THREADS), N_THREADS>>>(out_grad.data_ptr<scalar_t>(), a_r.data_ptr<scalar_t>(), a_d.data_ptr<scalar_t>(), a_r_grad.data_ptr<scalar_t>(), a_d_grad.data_ptr<scalar_t>(), B);
        gpuErrorCheck(cudaDeviceSynchronize());
    }));
}

// ===== Kernel+Host: SE3 to Dual Quaternion Forward

template <typename scalar_t>
__global__ void kernel_se3_to_dual_quaternion_fw(const scalar_t *__restrict__ a, scalar_t *__restrict__ out_r, scalar_t *__restrict__ out_d, const uint32_t B) {
    const uint32_t i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i >= B) {
        return;
    }
    _se3_to_dual_quaternion_fw<scalar_t>(a + i * 16, out_r + i * 4, out_d + i * 4);
}

void se3_to_dual_quaternion_fw(at::Tensor a, at::Tensor out_r, at::Tensor out_d, const uint32_t B) {
    CHECK_IS_CONTIGUOUS_FLOAT_CUDA(a);
    CHECK_IS_CONTIGUOUS_FLOAT_CUDA(out_r);
    CHECK_IS_CONTIGUOUS_FLOAT_CUDA(out_d);

    AT_DISPATCH_FLOATING_TYPES_AND_HALF(a.scalar_type(), "se3_to_dual_quaternion_fw", ([&] {
        kernel_se3_to_dual_quaternion_fw<scalar_t><<<DIV_ROUND_UP(B, N_THREADS), N_THREADS>>>(a.data_ptr<scalar_t>(), out_r.data_ptr<scalar_t>(), out_d.data_ptr<scalar_t>(), B);
        gpuErrorCheck(cudaDeviceSynchronize());
    }));
}

// ===== Kernel+Host: SE3 to Dual Quaternion Backward

template <typename scalar_t>
__global__ void kernel_se3_to_dual_quaternion_bw(const scalar_t *__restrict__ out_r_grad, const scalar_t *__restrict__ out_d_grad, const scalar_t *__restrict__ a, scalar_t *__restrict__ a_grad, const uint32_t B) {
    const uint32_t i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i >= B) {
        return;
    }
    _se3_to_dual_quaternion_bw<scalar_t>(out_r_grad + i * 4, out_d_grad + i * 4, a + i * 16, a_grad + i * 16);
}

void se3_to_dual_quaternion_bw(at::Tensor out_r_grad, at::Tensor out_d_grad, at::Tensor a, at::Tensor a_grad, const uint32_t B) {
    CHECK_IS_CONTIGUOUS_FLOAT_CUDA(out_r_grad);
    CHECK_IS_CONTIGUOUS_FLOAT_CUDA(out_d_grad);
    CHECK_IS_CONTIGUOUS_FLOAT_CUDA(a);
    CHECK_IS_CONTIGUOUS_FLOAT_CUDA(a_grad);

    AT_DISPATCH_FLOATING_TYPES_AND_HALF(a.scalar_type(), "se3_to_dual_quaternion_bw", ([&] {
        kernel_se3_to_dual_quaternion_bw<scalar_t><<<DIV_ROUND_UP(B, N_THREADS), N_THREADS>>>(out_r_grad.data_ptr<scalar_t>(), out_d_grad.data_ptr<scalar_t>(), a.data_ptr<scalar_t>(), a_grad.data_ptr<scalar_t>(), B);
        gpuErrorCheck(cudaDeviceSynchronize());
    }));
}
