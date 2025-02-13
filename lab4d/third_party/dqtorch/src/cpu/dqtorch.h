#pragma once

#include <stdint.h>
#include <torch/torch.h>

#define CHECK_CPU(x) TORCH_CHECK(x.device().is_cpu(), #x " must be a CPU tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be a contiguous tensor")
#define CHECK_IS_INT(x) TORCH_CHECK(x.scalar_type() == at::ScalarType : Int, #x " must be an int tensor")
#define CHECK_IS_FLOATING(x)                                                                                      \
    TORCH_CHECK(x.scalar_type() == at::ScalarType::Float || x.scalar_type() == at::ScalarType::Double ||          \
                    x.scalar_type() == at::ScalarType::Half,                                                      \
                #x " must be a floating tensor")
#define CHECK_IS_CONTIGUOUS_FLOAT_CPU(x)                                                                         \
    CHECK_CPU(x);                                                                                                \
    CHECK_CONTIGUOUS(x);                                                                                          \
    CHECK_IS_FLOATING(x)

#define N_THREADS 256

#define DIV_ROUND_UP(val, divisor) ((val + divisor - 1) / divisor)

// Standardize Quaternion
void standardize_quaternion_fw(at::Tensor a, at::Tensor out, const uint32_t B);
void standardize_quaternion_bw(at::Tensor out_grad, at::Tensor a, at::Tensor a_grad, const uint32_t B);

// Quaternion Raw Multiply
void quaternion_raw_multiply_fw(at::Tensor a, at::Tensor b, at::Tensor out, const uint32_t B, const uint32_t Da,
                                const uint32_t Db);
void quaternion_raw_multiply_bw(at::Tensor out_grad, at::Tensor a, at::Tensor b, at::Tensor a_grad,
                                at::Tensor b_grad, const uint32_t B, const uint32_t Da, const uint32_t Db);

// Quaternion Multiply
void quaternion_multiply_fw(at::Tensor a, at::Tensor b, at::Tensor out, const uint32_t B, const uint32_t Da, const uint32_t Db);
void quaternion_multiply_bw(at::Tensor out_grad, at::Tensor a, at::Tensor b, at::Tensor a_grad, at::Tensor b_grad, const uint32_t B, const uint32_t Da, const uint32_t Db);

// Quaternion Conjugate
void quaternion_conjugate_fw(at::Tensor a, at::Tensor out, const uint32_t B);
void quaternion_conjugate_bw(at::Tensor out_grad, at::Tensor a, at::Tensor a_grad, const uint32_t B);

// Quaternion Apply
void quaternion_apply_fw(at::Tensor a, at::Tensor b, at::Tensor out, const uint32_t B);
void quaternion_apply_bw(at::Tensor out_grad, at::Tensor a, at::Tensor b, at::Tensor a_grad, at::Tensor b_grad, const uint32_t B);

// Quaternion to Matrix
void quaternion_to_matrix_fw(at::Tensor a, at::Tensor out, const uint32_t B);
void quaternion_to_matrix_bw(at::Tensor out_grad, at::Tensor a, at::Tensor a_grad, const uint32_t B);

// Matrix to Raw Quaternion
void matrix_to_raw_quaternion_fw(at::Tensor a, at::Tensor out, const uint32_t B);
void matrix_to_raw_quaternion_bw(at::Tensor out_grad, at::Tensor a, at::Tensor a_grad, const uint32_t B);

// Matrix to Quaternion
void matrix_to_quaternion_fw(at::Tensor a, at::Tensor out, const uint32_t B);
void matrix_to_quaternion_bw(at::Tensor out_grad, at::Tensor a, at::Tensor a_grad, const uint32_t B);

// Axis Angle to Quaternion
void axis_angle_to_quaternion_fw(at::Tensor a, at::Tensor out, const uint32_t B);
void axis_angle_to_quaternion_bw(at::Tensor out_grad, at::Tensor a, at::Tensor a_grad, const uint32_t B);

// Quaternion to Axis Angle
void quaternion_to_axis_angle_fw(at::Tensor a, at::Tensor out, const uint32_t B);
void quaternion_to_axis_angle_bw(at::Tensor out_grad, at::Tensor a, at::Tensor a_grad, const uint32_t B);

// Axis Angle to Matrix
void axis_angle_to_matrix_fw(at::Tensor a, at::Tensor out, const uint32_t B);
void axis_angle_to_matrix_bw(at::Tensor out_grad, at::Tensor a, at::Tensor a_grad, const uint32_t B);

// Matrix to Axis Angle
void matrix_to_axis_angle_fw(at::Tensor a, at::Tensor out, const uint32_t B);
void matrix_to_axis_angle_bw(at::Tensor out_grad, at::Tensor a, at::Tensor a_grad, const uint32_t B);

// Quaternion Translation Multiply
void quaternion_translation_mul_fw(at::Tensor a_q, at::Tensor a_t, at::Tensor b_q, at::Tensor b_t, at::Tensor out_q, at::Tensor out_t, const uint32_t B);
void quaternion_translation_mul_bw(at::Tensor out_q_grad, at::Tensor out_t_grad, at::Tensor a_q, at::Tensor a_t, at::Tensor b_q, at::Tensor b_t, at::Tensor a_q_grad, at::Tensor a_t_grad, at::Tensor b_q_grad, at::Tensor b_t_grad, const uint32_t B);

// Quaternion Translation Apply
void quaternion_translation_apply_fw(at::Tensor a_q, at::Tensor a_t, at::Tensor b, at::Tensor out, const uint32_t B);
void quaternion_translation_apply_bw(at::Tensor out_grad, at::Tensor a_q, at::Tensor a_t, at::Tensor b, at::Tensor a_q_grad, at::Tensor a_t_grad, at::Tensor b_grad, const uint32_t B);

// Quaternion Translation Inverse
void quaternion_translation_inverse_fw(at::Tensor a_q, at::Tensor a_t, at::Tensor out_q, at::Tensor out_t, const uint32_t B);
void quaternion_translation_inverse_bw(at::Tensor out_q_grad, at::Tensor out_t_grad, at::Tensor a_q, at::Tensor a_t, at::Tensor a_q_grad, at::Tensor a_t_grad, const uint32_t B);

// Dual Quaternion Multiply
void dual_quaternion_mul_fw(at::Tensor a_r, at::Tensor a_d, at::Tensor b_r, at::Tensor b_d, at::Tensor out_r, at::Tensor out_d, const uint32_t B);
void dual_quaternion_mul_bw(at::Tensor out_r_grad, at::Tensor out_d_grad, at::Tensor a_r, at::Tensor a_d, at::Tensor b_r, at::Tensor b_d, at::Tensor a_r_grad, at::Tensor a_d_grad, at::Tensor b_r_grad, at::Tensor b_d_grad, const uint32_t B);

// Dual Quaternion Apply
void dual_quaternion_apply_fw(at::Tensor a_r, at::Tensor a_d, at::Tensor b, at::Tensor out, const uint32_t B);
void dual_quaternion_apply_bw(at::Tensor out_grad, at::Tensor a_r, at::Tensor a_d, at::Tensor b, at::Tensor a_r_grad, at::Tensor a_d_grad, at::Tensor b_grad, const uint32_t B);

// Dual Quaternion Q Conjugate
void dual_quaternion_q_conjugate_fw(at::Tensor a_r, at::Tensor a_d, at::Tensor out_r, at::Tensor out_d, const uint32_t B);
void dual_quaternion_q_conjugate_bw(at::Tensor out_r_grad, at::Tensor out_d_grad, at::Tensor a_r, at::Tensor a_d, at::Tensor a_r_grad, at::Tensor a_d_grad, const uint32_t B);

// Dual Quaternion D Conjugate
void dual_quaternion_d_conjugate_fw(at::Tensor a_r, at::Tensor a_d, at::Tensor out_r, at::Tensor out_d, const uint32_t B);
void dual_quaternion_d_conjugate_bw(at::Tensor out_r_grad, at::Tensor out_d_grad, at::Tensor a_r, at::Tensor a_d, at::Tensor a_r_grad, at::Tensor a_d_grad, const uint32_t B);

// Dual Quaternion 3rd Conjugate
void dual_quaternion_3rd_conjugate_fw(at::Tensor a_r, at::Tensor a_d, at::Tensor out_r, at::Tensor out_d, const uint32_t B);
void dual_quaternion_3rd_conjugate_bw(at::Tensor out_r_grad, at::Tensor out_d_grad, at::Tensor a_r, at::Tensor a_d, at::Tensor a_r_grad, at::Tensor a_d_grad, const uint32_t B);

// Dual Quaternion Norm
void dual_quaternion_norm_fw(at::Tensor a_r, at::Tensor a_d, at::Tensor out_r, at::Tensor out_d, const uint32_t B);
void dual_quaternion_norm_bw(at::Tensor out_r_grad, at::Tensor out_d_grad, at::Tensor a_r, at::Tensor a_d, at::Tensor a_r_grad, at::Tensor a_d_grad, const uint32_t B);

// Quaternion Translation to Dual Quaternion
void quaternion_translation_to_dual_quaternion_fw(at::Tensor a_q, at::Tensor a_t, at::Tensor out_r, at::Tensor out_d, const uint32_t B);
void quaternion_translation_to_dual_quaternion_bw(at::Tensor out_r_grad, at::Tensor out_d_grad, at::Tensor a_q, at::Tensor a_t, at::Tensor a_q_grad, at::Tensor a_t_grad, const uint32_t B);

// Dual Quaternion to Quaternion Translation
void dual_quaternion_to_quaternion_translation_fw(at::Tensor a_r, at::Tensor a_d, at::Tensor out_q, at::Tensor out_t, const uint32_t B);
void dual_quaternion_to_quaternion_translation_bw(at::Tensor out_q_grad, at::Tensor out_t_grad, at::Tensor a_r, at::Tensor a_d, at::Tensor a_r_grad, at::Tensor a_d_grad, const uint32_t B);

// Quaternion Translation to SE3
void quaternion_translation_to_se3_fw(at::Tensor a_q, at::Tensor a_t, at::Tensor out, const uint32_t B);
void quaternion_translation_to_se3_bw(at::Tensor out_grad, at::Tensor a_q, at::Tensor a_t, at::Tensor a_q_grad, at::Tensor a_t_grad, const uint32_t B);

// SE3 to Quaternion Translation
void se3_to_quaternion_translation_fw(at::Tensor a, at::Tensor out_q, at::Tensor out_t, const uint32_t B);
void se3_to_quaternion_translation_bw(at::Tensor out_q_grad, at::Tensor out_t_grad, at::Tensor a, at::Tensor a_grad, const uint32_t B);

// Dual Quaternion to SE3
void dual_quaternion_to_se3_fw(at::Tensor a_r, at::Tensor a_d, at::Tensor out, const uint32_t B);
void dual_quaternion_to_se3_bw(at::Tensor out_grad, at::Tensor a_r, at::Tensor a_d, at::Tensor a_r_grad, at::Tensor a_d_grad, const uint32_t B);

// SE3 to Dual Quaternion
void se3_to_dual_quaternion_fw(at::Tensor a, at::Tensor out_r, at::Tensor out_d, const uint32_t B);
void se3_to_dual_quaternion_bw(at::Tensor out_r_grad, at::Tensor out_d_grad, at::Tensor a, at::Tensor a_grad, const uint32_t B);
