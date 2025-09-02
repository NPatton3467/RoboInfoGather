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
    x_largest_min = max(box_a[0], box_b[0])
    y_largest_min = max(box_a[1], box_b[1])
    z_largest_min = max(box_a[2], box_b[2])
    x_smallest_max = min(box_a[3], box_b[3])
    y_smallest_max = min(box_a[4], box_b[4])
    z_smallest_max = min(box_a[5], box_b[5])

    # Compute the volume of intersection
    x_dim_inter_vol = max(0, x_smallest_max - x_largest_min)
    y_dim_inter_vol = max(0, y_smallest_max - y_largest_min)
    z_dim_inter_vol = max(0, z_smallest_max - z_largest_min)
    inter_vol = x_dim_inter_vol * y_dim_inter_vol * z_dim_inter_vol

    # Compute the volume of both bounding boxes
    box_a_volume = abs((box_a[3] - box_a[0]) * (box_a[4] - box_a[1]) * (box_a[5] - box_a[2]))
    box_b_volume = abs((box_b[3] - box_b[0]) * (box_b[4] - box_b[1]) * (box_b[5] - box_b[2]))

    # Compute the IoU
    iou = inter_vol / float(box_a_volume + box_b_volume - inter_vol)
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
        y_min = int(yt - y_delta)
        y_max = int(yt + y_delta)
        z_min = int(zt - z_delta)
        z_max = int(zt + z_delta)

        print("Deltas:\n")
        print(x_delta)
        print(y_delta)
        print(z_delta)

        print(obj_bel.map_params)

        boxes.append([x_min, y_min, z_min, x_max, y_max, z_max])
        scores.append(obj_bel.p[xt, yt, zt].detach().cpu())

    # Compute NMS
    iou_threshold = 0.25
    print("Boxes: ", boxes)
    print("Scores: ", scores)
    indices = nms_3d(np.array(boxes), np.array(scores), iou_threshold)

    # Return xt, yt, zt corresponding to indices
    ret_xyzs = []
    for i in indices:
        ret_xyzs.append(xyzs[i])

    return ret_xyzs

def check_for_ambiguity(belief):
    xyzs = np.argwhere(belief.p.detach().cpu() > 0.5)
    xyzs = np.swapaxes(xyzs, 0, 1)

    # Loop through and check nearby (doubly nested)
    for (xt1, yt1, zt1) in xyzs:
        for (xt2, yt2, zt2) in xyzs:
            x1 = int(xt1.detach().cpu())
            y1 = int(yt1.detach().cpu())
            z1 = int(zt1.detach().cpu())
            x2 = int(xt2.detach().cpu())
            y2 = int(yt2.detach().cpu())
            z2 = int(zt2.detach().cpu())

            # Check if nearby
            x_diff = (belief.map_params['res'] * abs(x1 - x2)) / belief.map_params['obj_map_res']
            y_diff = (belief.map_params['res'] * abs(y1 - y2)) / belief.map_params['obj_map_res']
            z_diff = (belief.map_params['z_res'] * abs(z1 - z2)) / belief.map_params['obj_z_res']
            x_dim = belief.map_params['dim'][0]
            y_dim = belief.map_params['dim'][1]
            z_dim = belief.map_params['dim'][2]
            x_diff_max = (belief.map_params['res'] * x_dim) / (2 * belief.map_params['obj_map_res'])
            y_diff_max = (belief.map_params['res'] * y_dim) / (2 * belief.map_params['obj_map_res'])
            z_diff_max = (belief.map_params['z_res'] * z_dim) / (2 * belief.map_params['obj_z_res'])

            if (x_diff > x_diff_max or
                y_diff > y_diff_max or
                z_diff > z_diff_max):
                continue

            # Check if features equal
            for feature in belief.feature_bels:
                if belief.feature_bels[feature]['tp'] != "feature_enum_spatial":
                    continue

                f1 = belief.feature_bels[feature]['vals'][x1, y1, z1]
                f2 = belief.feature_bels[feature]['vals'][x2, y2, z2]

                if f1 != f2:
                    belief.feature_bels[feature]['vals'][x1, y1, z1] = 'ambiguous'

    return belief
