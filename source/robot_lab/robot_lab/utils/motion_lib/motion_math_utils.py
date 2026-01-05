import torch
from robot_lab.utils.isaacgym_torch_utils import *
from robot_lab.utils.torch_utils import *


def compute_so3_derivative(rotations: torch.Tensor, dt: float) -> torch.Tensor:
    """Computes the derivative of a sequence of SO3 rotations using central differences.

    Args:
        rotations: Quaternion rotations with shape (T, 4).
        dt: Time step.
    Returns:
        Angular velocities with shape (T, 3).
    """
    if rotations.shape[0] < 3:
        # For very short sequences, fall back to forward differences
        root_drot = quat_diff(rotations[:-1], rotations[1:])
        omega = quat_to_exp_map(root_drot) / dt
        omega = torch.cat([omega, omega[-1:]], dim=0)  # Repeat last
        return omega

    # Use central differences for interior points
    q_prev, q_next = rotations[:-2], rotations[2:]
    q_rel = quat_mul(q_next, quat_conjugate(q_prev))
    omega_interior = quat_to_exp_map(q_rel) / (2.0 * dt)

    # Handle boundaries with forward/backward differences
    q_start_rel = quat_mul(rotations[1], quat_conjugate(rotations[0]))
    omega_start = quat_to_exp_map(q_start_rel) / dt

    q_end_rel = quat_mul(rotations[-1], quat_conjugate(rotations[-2]))
    omega_end = quat_to_exp_map(q_end_rel) / dt

    # Combine all parts
    omega = torch.cat(
        [omega_start.unsqueeze(0), omega_interior, omega_end.unsqueeze(0)], dim=0
    )
    return omega
