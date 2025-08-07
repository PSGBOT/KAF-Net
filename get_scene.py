import torch
import torch.nn.functional as F
import numpy as np
from scipy.ndimage import maximum_filter

def extract_objects(hmap_img, regs_img, w_h_img, peak_thresh=0.3, nms_kernel=3):
    objects = []
    num_classes, H, W = hmap_img.shape

    # Apply NMS to each category channel
    for c in range(num_classes):
        heatmap = hmap_img[c].cpu().numpy()
        # Find local maxima
        nms = (heatmap == maximum_filter(heatmap, size=nms_kernel))
        peaks = np.where((nms) & (heatmap > peak_thresh))
        for y, x in zip(*peaks):
            # Get offset
            offset_x = regs_img[0, y, x].item()
            offset_y = regs_img[1, y, x].item()
            # center in feature map, times downsample rate to get original image center
            center = (x + offset_x, y + offset_y)
            width = w_h_img[0, y, x].item()
            height = w_h_img[1, y, x].item()
            score = heatmap[y, x]
            objects.append({
                "id": len(objects),
                "category": c,
                "center": center,
                "width": width,
                "height": height,
                "score": score,
                "yx": (y, x)  # Save for relation extraction
            })
    return objects

def integrate_along_line(raf_map, y0, x0, y1, x1, inte_width, debug=False):
    # Bresenham's line algorithm or linear interpolation
    num_points = int(np.hypot(y1 - y0, x1 - x0)) + 1
    if num_points <= 0:
        return 0
    ys = np.linspace(y0, y1, num_points)
    xs = np.linspace(x0, x1, num_points)
    unitv = np.array([y1 - y0, x1 - x0])
    norm = np.linalg.norm(unitv)
    if norm == 0:
        return 0
    unitv = unitv / norm  # shape (2,)
    # Perpendicular unit vector
    perp = np.array([-unitv[1], unitv[0]])

    dot_sum = 0
    count = 0
    visited = set()
    # For each point along the line
    for y, x in zip(ys, xs):
        # For each offset along the perpendicular direction
        for w in np.linspace(-inte_width, inte_width, int(2*inte_width)+1):
            yy = y + perp[0] * w
            xx = x + perp[1] * w
            y_int, x_int = int(round(yy)), int(round(xx))
            if (0 <= y_int < raf_map.shape[1]) and (0 <= x_int < raf_map.shape[2]):
                key = (y_int, x_int)
                if key in visited and debug:
                    print(f"Repeated pixel: {key}")
                    continue
                visited.add(key)
                vec = raf_map[:, y_int, x_int].cpu().numpy()
                dot_sum += np.dot(vec, unitv)
                count += 1
    if count == 0:
        return 0
    return dot_sum / count

def extract_relations(raf_img, objects, rel_thresh=0.3):
    relations = []
    num_rel = raf_img.shape[0] // 2  # 14 relations, each with 2 channels (vector)
    for i, subj in enumerate(objects):
        for j, obj in enumerate(objects):
            if i == j:
                continue
            y0, x0 = subj["yx"]
            y1, x1 = obj["yx"]
            for rel in range(num_rel):
                # Get the 2D vector field for this relation
                raf_vec = raf_img[2*rel:2*rel+2]  # shape [2, H, W]
                inte_width = min(abs(subj["width"]), abs(obj["width"]), abs(subj["height"]), abs(obj['height']))  # Use the minimum width for integration
                conf = integrate_along_line(raf_vec, y0, x0, y1, x1, inte_width, True)*obj["score"]*subj["score"]
                if conf > rel_thresh:
                    relations.append({
                        "subject_id": subj["id"],
                        "object_id": obj["id"],
                        "relation": rel,
                        "confidence": conf
                    })
    return relations


def get_scene_graph(kaf):
    # [hmap(tensor[B,13,H,W]), reg(tensor[B,2,H,W]), w_h_(tensor[B,2,H,W]), raf(tensor[B,28,H,W])]
    hmap = kaf[-1][0]
    regs = kaf[-1][1]
    w_h_ = kaf[-1][2]
    raf = kaf[-1][3]
    objects = []
    relations = []
    for i in range(hmap.shape[0]):
        # Process each image's outputs
        hmap_img = hmap[i]  # [13, H, W]
        regs_img = regs[i]  # [2, H, W]
        w_h_img = w_h_[i]  # [2, H, W]
        raf_img = raf[i]   # [28, H, W]

        # Extract objects and relations from the outputs
        objects = extract_objects(hmap_img, regs_img, w_h_img)
        relations = extract_relations(raf_img, objects)
        scene_graph = {
            "objects": objects,
            "relations": relations
        }
        yield scene_graph

def visualize_scene_graph(scene_graph):


    objects = scene_graph["objects"]
    relations = scene_graph["relations"]

    # Visualize objects
    for obj in objects:
        print(f"Object ID: {obj['id']}, Category: {obj['category']}, Center: {obj['center']}, Width: {obj['width']}, Height: {obj['height']}, Score: {obj['score']}, YX: {obj['yx']}")

    # Visualize relations
    for rel in relations:
        print(f"Relation from Object {rel['subject_id']} to Object {rel['object_id']} of type {rel['relation']}, confidence: {rel['confidence']}")



if __name__ == "__main__":
    # Example usage
    # Assuming `kaf` is the output from the KAF-Net model
    kaf = [[
        torch.zeros(1, 13, 32, 32),  # hmap
        torch.randn(1, 2, 32, 32),    # regs
        torch.randn(1, 2, 32, 32),    # w_h_
        torch.randn(1, 28, 32, 32)     # raf
    ]]
    kaf[-1][0][0][0][23][12] = 1
    kaf[-1][0][0][0][13][20] = 1
    kaf[-1][0][0][3][23][12] = 1
    kaf[-1][0][0][3][13][20] = 1
    print("Extracting scene graphs...")
    
    for scene_graph in get_scene_graph(kaf):
        visualize_scene_graph(scene_graph)
# This code defines a function to extract scene graphs from the KAF-Net outputs
# and visualize them. The actual extraction logic will depend on the specific dataset and task. 


