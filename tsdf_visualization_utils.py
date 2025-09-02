import numpy as np
import matplotlib.pyplot as plt

def generate_dummy_tsdf(shape=(100, 100, 100), trunc=0.1):
    """
    Generates a dummy TSDF volume with a flat surface at z=50.
    """
    tsdf = np.ones(shape, dtype=np.float32) * trunc
    surface_z = shape[2] // 2
    for z in range(shape[2]):
        distance = (z - surface_z) * 0.01
        tsdf[:, :, z] = np.clip(distance, -trunc, trunc)
    return tsdf

def find_zero_crossings(tsdf_volume, axis=2):
    """
    Finds zero-crossings along a given axis to project surface points.
    Returns a 2D image where each pixel indicates the height (index along axis)
    of the first zero-crossing (surface).
    """
    # Move the projection axis to the last dimension
    tsdf = np.moveaxis(tsdf_volume, axis, -1)

    print("Move axis shape: ",tsdf.shape)
    
    # Compute sign changes between consecutive voxels
    sign_changes = np.diff(np.sign(tsdf), axis=-1)
    zero_crossings = np.abs(sign_changes) > 1e-5  # detect sign flips

    # Get first zero-crossing along axis
    crossing_idx = zero_crossings.argmax(axis=-1)
    has_crossing = zero_crossings.any(axis=-1)

    # Fill non-crossing pixels with NaNs or max depth
    projection = np.where(has_crossing, crossing_idx, np.nan)

    return projection

def visualize_projection(projection_2d):
    plt.figure(figsize=(8, 6))
    plt.imshow(projection_2d, cmap='viridis')
    plt.title("TSDF Ground Plane Projection (Zero-Crossings)")
    plt.xlabel("X")
    plt.ylabel("Y")
    cbar = plt.colorbar(label='Surface Height (voxel index)')
    cbar.ax.invert_yaxis()
    plt.gca().invert_yaxis()  # Optional: show closer surfaces on top
    plt.show()

if __name__ == "__main__":
    #tsdf_volume = generate_dummy_tsdf()
    tsdf_volume = np.load('./results/RIG_multi_view_perfect_perception_exp/debug/0/tsdf_volume_3.npy')

    """
    tsdf_volume = np.array(
            [
                [
                    [-1,-1,-1],
                    [-1,-1,-1],
                    [-1,-1,-1]
                ],
                [
                    [-1,-1,0],
                    [-1,0,-1],
                    [0,-1,-1]
                ],
                [
                    [-1,0,1],
                    [-1,1,0],
                    [1,0,0]
                ],
                [
                    [-1,1,1],
                    [0,1,1],
                    [1,1,1]
                ]
            ]
        )

    tsdf_volume = np.array(
            [
                [
                    [-1,-1,-1],
                    [-1,-1,-1],
                    [-1,-1,-1]
                ],
                [
                    [-1,-1,-1],
                    [-1,0,-1],
                    [-1,-1,-1]
                ],
                [
                    [-1,0,0],
                    [-1,1,0],
                    [-1,0,0]
                ],
                [
                    [0,1,1],
                    [0,1,1],
                    [0,1,1]
                ]
            ]
        )

    """

    #tsdf_volume = tsdf_volume.T

    print(tsdf_volume.shape)
    projection = find_zero_crossings(tsdf_volume, axis=0)
    visualize_projection(projection)

    projection = find_zero_crossings(tsdf_volume, axis=1)
    visualize_projection(projection)
    
    projection = find_zero_crossings(tsdf_volume, axis=2)
    visualize_projection(projection)
