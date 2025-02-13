import os
import sys

import torch

sys.path.insert(0, f"{os.path.join(os.path.dirname(__file__))}/../third_party")

try:
    from dqtorch import (
        quaternion_conjugate,
        standardize_quaternion,
        quaternion_raw_multiply as quaternion_mul,
        axis_angle_to_quaternion,
        quaternion_to_axis_angle,
        quaternion_to_matrix,
        matrix_to_quaternion,
        quaternion_apply,
    )

    from dqtorch import (
        dual_quaternion_apply,
        dual_quaternion_inverse,
        dual_quaternion_mul,
        dual_quaternion_norm,
        dual_quaternion_to_quaternion_translation,
        dual_quaternion_to_se3,
        quaternion_translation_apply,
        quaternion_translation_inverse,
        quaternion_translation_mul,
        quaternion_translation_to_dual_quaternion,
        quaternion_translation_to_se3,
        se3_to_dual_quaternion,
        se3_to_quaternion_translation,
    )
except:
    print("Could not load dqtorch library, falling back to a slower PyTorch implementation")

    from dqtorch.rotation_conversions import (
        quaternion_invert as quaternion_conjugate,
        standardize_quaternion,
        quaternion_raw_multiply as quaternion_mul,
        axis_angle_to_quaternion,
        quaternion_to_axis_angle,
        quaternion_to_matrix,
        matrix_to_quaternion,
        quaternion_apply,
    )

    from dqtorch.test import (
        dual_quaternion_apply,
        dual_quaternion_inverse,
        dual_quaternion_mul,
        dual_quaternion_norm,
        dual_quaternion_to_quaternion_translation,
        dual_quaternion_to_se3,
        quaternion_translation_apply,
        quaternion_translation_inverse,
        quaternion_translation_mul,
        quaternion_translation_to_dual_quaternion,
        quaternion_translation_to_se3,
        se3_to_dual_quaternion,
        se3_to_quaternion_translation,
    )
