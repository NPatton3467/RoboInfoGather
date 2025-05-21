import numpy as np
import torch

def compute_iou_3d(box_a, box_b):
    """
    Computes the Intersection over Union (IoU) between two 3D bounding boxes.

    Args:
        box_a (numpy.ndarray): Coordinates of the first box in the format [x_min, y_min, z_min, x_max, y_max, z_max].
        box_b (numpy.ndarray): Coordinates of the second box in the format [x_min, y_min, z_min, x_max, y_max, z_max].

    Returns:
        float: The IoU between the two boxes.
    """
    # Determine the coordinates of the intersection cuboid
    x_a_max = max(box_a[0], box_b[0])
    y_a_max = max(box_a[1], box_b[1])
    z_a_max = max(box_a[2], box_b[2])
    x_a_min = min(box_a[3], box_b[3])
    y_a_min = min(box_a[4], box_b[4])
    z_a_min = min(box_a[5], box_b[5])

    # Compute the volume of intersection
    inter_area = max(0, x_a_min - x_a_max) * max(0, y_a_min - y_a_max) * max(0, z_a_min - z_a_max)

    # Compute the volume of both bounding boxes
    box_a_volume = (box_a[3] - box_a[0]) * (box_a[4] - box_a[1]) * (box_a[5] - box_a[2])
    box_b_volume = (box_b[3] - box_b[0]) * (box_b[4] - box_b[1]) * (box_b[5] - box_b[2])

    # Compute the IoU
    iou = inter_area / float(box_a_volume + box_b_volume - inter_area)
    return iou


def nms_3d(boxes, scores, iou_threshold):
    """
    Performs non-maximum suppression (NMS) on a set of 3D bounding boxes.

    Args:
        boxes (numpy.ndarray): A numpy array of shape (N, 6) containing the bounding boxes
                              coordinates in the format [x_min, y_min, z_min, x_max, y_max, z_max].
        scores (numpy.ndarray): A numpy array of shape (N,) containing the confidence scores
                                for the bounding boxes.
        iou_threshold (float): The IoU threshold for NMS.

    Returns:
        numpy.ndarray: A numpy array containing the indices of the selected boxes after NMS.
    """
    # Sort the bounding boxes by score in descending order
    sorted_indices = np.argsort(scores)[::-1]
    selected_indices = []

    while len(sorted_indices) > 0:
        # Select the box with the highest score
        best_index = sorted_indices[0]
        selected_indices.append(best_index)

        # Remove the selected box from the sorted indices
        sorted_indices = sorted_indices[1:]

        # Compute IoU between the selected box and remaining boxes
        ious = np.array([compute_iou_3d(boxes[best_index], boxes[index]) for index in sorted_indices])

        # Filter out boxes with IoU greater than the threshold
        sorted_indices = sorted_indices[ious <= iou_threshold]

    return np.array(selected_indices)

def suppress_non_max(obj_bel):
    # Create bounding boxes based on voxels and desired dim from LLM
    # then perform NMS over boxes
    xyzs = torch.argwhere(obj_bel.p > obj_bel.threshold)
    boxes = []
    scores = []
    for (xt, yt, zt) in xyzs:
        # Create box
        x_dim = obj_bel.map_params['dim'][0]
        y_dim = obj_bel.map_params['dim'][1]
        z_dim = obj_bel.map_params['dim'][2]
        x_delta = (obj_bel.map_params['res'] * x_dim) / (2 * obj_bel.map_params['obj_map_res'])
        y_delta = (obj_bel.map_params['res'] * y_dim) / (2 * obj_bel.map_params['obj_map_res'])
        z_delta = (obj_bel.map_params['z_res'] * z_dim) / (2 * obj_bel.map_params['obj_z_res'])
        x_min = int(xt - x_delta)
        x_max = int(xt + x_delta)
        y_min = int(xt - y_delta)
        y_max = int(xt + y_delta)
        z_min = int(xt - z_delta)
        z_max = int(xt + z_delta)

        print("Deltas:\n")
        print(x_delta)
        print(y_delta)
        print(z_delta)

        print(obj_bel.map_params)

        boxes.append([x_min, y_min, z_min, x_max, y_max, z_max])
        scores.append(obj_bel.p[xt, yt, zt].detach().cpu())

    # Compute NMS
    iou_threshold = 0.1
    print("Boxes: ", boxes)
    print("Scores: ", scores)
    indices = nms_3d(np.array(boxes), np.array(scores), iou_threshold)

    # Return xt, yt, zt corresponding to indices
    ret_xyzs = []
    for i in indices:
        ret_xyzs.append(xyzs[i])

    return ret_xyzs
