import torch
import torch.nn.functional as F

import isaaclab.utils.math as math_utils
from typing import Union


def quat_to_6d_rotation(quat: torch.Tensor) -> torch.Tensor:
    """
    Converts a quaternion to a 6D rotation representation.

    The 6D representation consists of the first two rows of the rotation matrix.

    Args:
        quat: A tensor of shape (..., 4) representing the quaternion.

    Returns:
        A tensor of shape (..., 6) representing the 6D rotation.
    """
    rot_mat = math_utils.matrix_from_quat(quat)
    # The 6D rotation is the first two rows of the rotation matrix, flattened.
    # Using [..., 0, :] is more robust to different batch dimensions.
    return torch.cat([rot_mat[..., 0, :], rot_mat[..., 1, :]], dim=-1)

def sixd_to_rotmat(rot_6d: torch.Tensor) -> torch.Tensor:
    """
    Convert 6D rotation representation to a 3x3 rotation matrix using Gram-Schmidt orthogonalization.

    Args:
        rot_6d (torch.Tensor): [N, 6] or [..., 6] 6D rotation representation.

    Returns:
        torch.Tensor: Rotation matrices with shape [N, 3, 3] or [..., 3, 3].
    """
    # Split into two 3D vectors
    a1 = rot_6d[..., 0:3]  # First vector
    a2 = rot_6d[..., 3:6]  # Second vector

    # Normalize the first vector
    b1 = F.normalize(a1, dim=-1)

    # Make the second vector orthogonal to the first
    a2_proj_b1 = (b1 * (a2 * b1).sum(dim=-1, keepdim=True))
    b2 = F.normalize(a2 - a2_proj_b1, dim=-1)

    # Compute the third vector as cross product
    b3 = torch.cross(b1, b2, dim=-1)

    # Stack to form rotation matrix
    rot_mat = torch.stack((b1, b2, b3), dim=-1)  # [..., 3, 3]
    return rot_mat

def quat_from_6d_rotation(rot_6d: torch.Tensor) -> torch.Tensor:
    """
    Converts a 6D rotation representation back to a quaternion.

    The 6D representation consists of the first two rows of the rotation matrix.
    This function reconstructs the full rotation matrix using Gram-Schmidt
    orthonormalization and then converts it to a quaternion.

    Args:
        rot_6d: A tensor of shape (..., 6) representing the 6D rotation.

    Returns:
        A tensor of shape (..., 4) representing the quaternion.
    """
    # Ensure the input has the correct last dimension
    assert rot_6d.shape[-1] == 6, "Input tensor must have 6 dimensions in the last axis."

    # Extract the first two rows of the rotation matrix
    r1_raw = rot_6d[..., 0:3]
    r2_raw = rot_6d[..., 3:6]

    # Orthonormalize the rows using the Gram-Schmidt process
    r1 = torch.nn.functional.normalize(r1_raw, p=2, dim=-1)
    dot_product = torch.sum(r1 * r2_raw, dim=-1, keepdim=True)
    r2_orthogonal = r2_raw - dot_product * r1
    r2 = torch.nn.functional.normalize(r2_orthogonal, p=2, dim=-1)
    r3 = torch.cross(r1, r2, dim=-1)
    rot_mat = torch.stack((r1, r2, r3), dim=-2)
    quat = math_utils.quat_from_matrix(rot_mat)

    return quat


def triangle_vertices_to_SE3(vertices: torch.Tensor, axis_len: float = 0.1) -> torch.Tensor:
    """
    Converts three vertices of an equilateral triangle back to a SE(3) pose.

    This function reconstructs the SE(3) transformation from the three vertices
    of the transformed triangle.

    Args:
        vertices: A tensor of shape (..., 3, 3) containing the coordinates of the three vertices.
        axis_len: Distance from the origin to each vertex of the equilateral triangle.
                  This must match the value used in the forward conversion. Default is 0.1.

    Returns:
        A tensor of shape (..., 4, 4) representing the reconstructed SE(3) transformation matrix.
    """
    # Get device and dtype from input tensor
    device = vertices.device
    dtype = vertices.dtype
    batch_shape = vertices.shape[:-2]

    translation = torch.mean(vertices, dim=-2)
    centered_vertices = vertices - translation.unsqueeze(-2)
    x_new = torch.nn.functional.normalize(centered_vertices[..., 0, :], p=2, dim=-1)
    edge1 = centered_vertices[..., 1, :] - centered_vertices[..., 0, :]
    edge2 = centered_vertices[..., 2, :] - centered_vertices[..., 0, :]
    z_new = torch.nn.functional.normalize(torch.cross(edge1, edge2, dim=-1), p=2, dim=-1)
    y_new = torch.cross(z_new, x_new, dim=-1)
    rotation = torch.stack([x_new, y_new, z_new], dim=-1)
    se3 = torch.eye(4, device=device, dtype=dtype).expand(*batch_shape, 4, 4).clone()
    se3[..., :3, :3] = rotation
    se3[..., :3, 3] = translation

    return se3


def triangle_vertices_from_SE3(se3: torch.Tensor, axis_len: float = 0.15) -> torch.Tensor:
    """
    Converts a SE(3) pose to a 3D representation using three vertices of an equilateral triangle.

    This function transforms a canonical equilateral triangle from the identity pose using the
    provided SE(3) transformation. The three vertices of the transformed triangle fully
    encode the SE(3) transformation (both rotation and translation).

    Args:
        se3: A tensor of shape (..., 4, 4) representing the SE(3) transformation matrix.
        axis_len: Distance from the origin to each vertex of the equilateral triangle
                  in the identity pose. Default is 0.1.

    Returns:
        A tensor of shape (..., 3, 3) containing the coordinates of the three vertices
        of the transformed triangle. Each row is a vertex [x, y, z].
    """
    # Get device and dtype from input tensor
    device = se3.device
    dtype = se3.dtype
    batch_shape = se3.shape[:-2]

    # Define canonical equilateral triangle vertices in the x-y plane
    angle1 = torch.tensor(2 * torch.pi / 3, device=device, dtype=dtype)
    angle2 = torch.tensor(4 * torch.pi / 3, device=device, dtype=dtype)
    p1 = torch.tensor([axis_len, 0, 0], device=device, dtype=dtype)
    p2 = torch.tensor(
        [axis_len * torch.cos(angle1), axis_len * torch.sin(angle1), 0],
        device=device,
        dtype=dtype,
    )
    p3 = torch.tensor(
        [axis_len * torch.cos(angle2), axis_len * torch.sin(angle2), 0],
        device=device,
        dtype=dtype,
    )

    # Stack into a (3, 3) tensor and add homogeneous coordinate
    canonical_vertices = torch.stack([p1, p2, p3], dim=0)
    canonical_vertices_homo = torch.cat([canonical_vertices, torch.ones(3, 1, device=device, dtype=dtype)], dim=1)

    # Expand canonical vertices to match batch shape of se3
    canonical_vertices_homo = canonical_vertices_homo.expand(*batch_shape, 3, 4)

    # Apply the SE(3) transformation
    # se3 is (..., 4, 4), canonical_vertices_homo.transpose is (..., 4, 3)
    # result is (..., 4, 3)
    transformed_vertices_homo = se3 @ canonical_vertices_homo.transpose(-1, -2)

    # Convert back to 3D coordinates and return in shape (..., 3, 3)
    return transformed_vertices_homo[..., :3, :].transpose(-1, -2)


def pos_quat_to_SE3(pos: torch.Tensor, quat: torch.Tensor) -> torch.Tensor:
    """
    Converts a position and quaternion to a SE(3) pose.

    Args:
        pos: A tensor of shape (..., 3) representing the position.
        quat: A tensor of shape (..., 4) representing the quaternion.

    Returns:
        A tensor of shape (..., 4, 4) representing the SE(3) pose.
    """
    se3 = torch.eye(4, device=pos.device, dtype=pos.dtype).expand(*pos.shape[:-1], 4, 4).clone()
    se3[..., :3, 3] = pos
    se3[..., :3, :3] = math_utils.matrix_from_quat(quat)
    return se3

# TODO: need furter test on this function
def quat_slerp(q_start: torch.Tensor, q_end: torch.Tensor, t_param: Union[float, torch.Tensor], DOT_THRESHOLD: float = 0.9995) -> torch.Tensor:
    """
    Spherical linear interpolation (SLERP) between two quaternions with batch support.
    Quaternions are expected in (w, x, y, z) or (x, y, z, w) format consistently.
    The function internally normalizes for dot product calculation, assuming inputs are near-unit.
    Output quaternions will be on the unit hypersphere.

    Args:
        q_start (torch.Tensor): The first quaternion tensor of shape (..., 4).
        q_end (torch.Tensor): The second quaternion tensor of shape (..., 4).
        t_param (Union[float, torch.Tensor]): Interpolation coefficient.
            If float, applied uniformly to all pairs in the batch.
            If tensor, its shape must be broadcastable to (..., 1) 
            (e.g., scalar, (N,), (B, N), where q_start is (N,4) or (B,N,4)).
        DOT_THRESHOLD (float): Threshold for the dot product. If the absolute dot product
            is above this, linear interpolation (LERP) is used for stability and to handle
            nearly collinear quaternions.

    Returns:
        torch.Tensor: The interpolated quaternion tensor of shape (..., 4).
    """
    # Input assertions
    if not (isinstance(q_start, torch.Tensor) and q_start.shape[-1] == 4):
        raise ValueError("q_start must be a torch.Tensor with the last dimension of size 4.")
    if not (isinstance(q_end, torch.Tensor) and q_end.shape[-1] == 4):
        raise ValueError("q_end must be a torch.Tensor with the last dimension of size 4.")
    if q_start.shape != q_end.shape:
        raise ValueError(f"q_start (shape: {q_start.shape}) and q_end (shape: {q_end.shape}) must have the same shape.")

    # Handle t_param: convert to tensor and ensure it's broadcastable to (..., 1)
    if not isinstance(t_param, torch.Tensor):
        t_param = torch.tensor(t_param, device=q_start.device, dtype=q_start.dtype)

    # Determine target shape for t_param: (B1, B2, ..., Bn, 1)
    t_target_shape = list(q_start.shape[:-1]) + [1]

    if t_param.ndim == 0:  # Scalar t_param
        t = t_param.reshape([1] * (q_start.ndim - 1) + [1]).expand(t_target_shape)
    elif t_param.ndim == q_start.ndim - 1:  # t_param is like (B1, ..., Bn)
        t = t_param.unsqueeze(-1)
    elif t_param.shape == t_target_shape: # Already correct shape
        t = t_param
    else:
        try: # Try to broadcast if it's a different shape, e.g. (1,) for (N,4)
            t = t_param.expand(t_target_shape)
        except RuntimeError:
            raise ValueError(
                f"Cannot broadcast t_param of shape {t_param.shape} to target shape {t_target_shape} "
                f"for q_start/q_end of shape {q_start.shape}."
            )
    dot = torch.sum(q_start * q_end, dim=-1)  # Shape: (...,)
    q_end_corrected = torch.where(dot.unsqueeze(-1) < 0.0, -q_end, q_end)
    dot_corrected = torch.where(dot < 0.0, -dot, dot)
    lerp_mask = dot_corrected >= DOT_THRESHOLD  # Shape: (...,)
    angle = torch.acos(torch.clamp(dot_corrected, -1.0, 1.0))  # Shape: (...,)
    sin_angle = torch.sin(angle)  # Shape: (...,)
    angle_unsqueezed = angle.unsqueeze(-1)  # Shape: (..., 1)
    sin_angle_unsqueezed = sin_angle.unsqueeze(-1) # Shape: (..., 1)
    eps = torch.finfo(q_start.dtype).eps
    s0 = torch.sin((1.0 - t) * angle_unsqueezed) / (sin_angle_unsqueezed + eps)
    s1 = torch.sin(t * angle_unsqueezed) / (sin_angle_unsqueezed + eps)
    
    slerp_val = s0 * q_start + s1 * q_end_corrected  # Shape: (..., 4)
    lerp_val_calculated = (1.0 - t) * q_start + t * q_end_corrected
    lerp_val_normalized = lerp_val_calculated / (torch.linalg.norm(lerp_val_calculated, dim=-1, keepdim=True) + eps)
    final_result = torch.where(lerp_mask.unsqueeze(-1), lerp_val_normalized, slerp_val)

    return final_result