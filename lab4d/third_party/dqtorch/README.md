# Conventions

## Quaternions

Quaternions are written as (..., 4) tensors [w, x, y, z] or (..., 3) tensors
[x, y, z], in which case it is assumed that w == 0.

Dual quaternions are written as (..., 8) tensors [rw, rx, ry, rz, tw, tx, ty, tz]
where [rw, rx, ry, rz] is the real part and [tw, tx, ty, tz] is the dual part.

## Rotation matrices

Similar to PyTorch3D, transformation matrices assume the points on which the
transformation will be applied are column vectors. The R matrix is written as

    R = [
            [Rxx, Rxy, Rxz],
            [Ryx, Ryy, Ryz],
            [Rzx, Rzy, Rzz],
        ]  # 3, 3

This matrix can be applied to column vectors by post-multiplication by the
points, e.g.

    points = [[0], [1], [2]]  # 3x1 xyz coordinates of a point
    transformed_points = R @ points

To apply the same matrix to points which are row vectors, the R matrix can be
transposed and pre-multiplied by the points, e.g.

    points = [[0, 1, 2]]  # 1x3 xyz coordinates of a point
    transformed_points = points @ R.transpose(1, 0)
