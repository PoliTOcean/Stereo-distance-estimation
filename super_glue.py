import torch
import time
import cv2
import numpy as np
import sys
import matplotlib.pyplot as plt
from gui import LiveGUI 
import tkinter

sys.path.append('SuperGluePretrainedNetwork')
from SuperGluePretrainedNetwork.models.superpoint import SuperPoint
from SuperGluePretrainedNetwork.models.superglue import SuperGlue

from scene import Scene, States


def mouse_callback(event, x, y, flags, param):
    time.sleep(0.12)
    if event == cv2.EVENT_LBUTTONDOWN:
        print(f"Right-click at ({x}, {y})")
        scene.selected_points.append((x, y))
        scene.draw_point((x, y), 6, (0, 0, 255))
    elif event == cv2.EVENT_RBUTTONDOWN:
        print(f"Deleted point at ({x}, {y})")
        scene.selected_points.pop()
        scene.delete_point()

def load_and_calibrate(left_test_path, right_test_path):
    # Change acquisition method
    img_left_test = cv2.imread(left_test_path)
    img_right_test = cv2.imread(right_test_path)

    data_map = np.load("remap_data.npz")
    map1_left = data_map["map1_left"]
    map1_right = data_map["map1_right"]
    map2_left = data_map["map2_left"]
    map2_right = data_map["map2_right"]

    if img_left_test is None or img_right_test is None:
        print("Error: Could not load test images.")
    else:
        # Rectify images
        img_left_rect = cv2.remap(img_left_test, map1_left, map2_left, cv2.INTER_LINEAR)
        img_right_rect = cv2.remap(img_right_test, map1_right, map2_right, cv2.INTER_LINEAR)
    
    return img_left_rect, img_right_rect

def compute_keypoints():
    img0_norm = scene.left_img/255.0
    img1_norm = scene.right_img/255.0

    img0_t = torch.from_numpy(img0_norm[None,None,...]).float().to(device)
    img1_t = torch.from_numpy(img1_norm[None,None,...]).float().to(device)

    # Compute predictions for keypoints
    pred0 = sp({'image': img0_t})
    pred1 = sp({'image': img1_t})

    kpts0, desc0, scores0 = pred0['keypoints'][0], pred0['descriptors'][0], pred0['scores'][0]
    kpts1, desc1, scores1 = pred1['keypoints'][0], pred1['descriptors'][0], pred1['scores'][0]
    
    #kpts0_np = kpts0.cpu().numpy()
    
    for kpt in kpts0.cpu().numpy():
        scene.keypoints.append(kpt)
        scene.draw_point((int(kpt[0]), int(kpt[1])), 2, (0, 255, 0))

def match_keypoints(input_data, radius, a, b, crop_size=300, slope_limit=0.3):
    all_valid_matches = [] 
    half_crop = crop_size // 2
    for x_click, y_click in scene.selected_points:

        y1 = max(0, y_click - half_crop)
        y2 = min(img0.shape[0], y_click + half_crop)
        x1 = max(0, x_click - half_crop)
        x2 = min(img0.shape[1], x_click + half_crop)
        img0_crop = a[y1:y2, x1:x2]
        img1_crop = b[y1:y2, x1:x2]

        img0_norm = img0_crop/255.0
        img1_norm = img1_crop/255.0

        img0_t = torch.from_numpy(img0_norm[None,None,...]).float().to(device)
        img1_t = torch.from_numpy(img1_norm[None,None,...]).float().to(device)
        pred0 = sp({'image': img0_t})
        pred1 = sp({'image': img1_t})

        kpts0, desc0, scores0 = pred0['keypoints'][0], pred0['descriptors'][0], pred0['scores'][0]
        kpts1, desc1, scores1 = pred1['keypoints'][0], pred1['descriptors'][0], pred1['scores'][0]

        click = np.array([x_click - x1, y_click - y1]) 
        dists = np.linalg.norm(kpts0.cpu().numpy() - click[None,:], axis=1)
        idxs0 = np.where(dists < radius)[0]
        
        # Find the nearest keypoint within the radius
        if len(idxs0) > 0:
            nearest_idx = idxs0[np.argmin(dists[idxs0])]
            idxs0 = np.array([nearest_idx])  # Keep only the nearest keypoint

            # Draw selected keypoint
            scene.draw_point((int(kpts0[idxs0][0][0] + x1), int(kpts0[idxs0][0][1] + y1)), 5, (255, 0, 0))
            scene.nnpoint.append((int(kpts0[idxs0][0][0] + x1), int(kpts0[idxs0][0][1] + y1)))
        else:
            print("NOT FOUND")
            return []
        
        query_kpts = kpts0[idxs0] # Select multiple keypoints
        desc0_sel = desc0[:,idxs0].to(device) # Select corresponding descriptors
        scores0_sel = scores0[idxs0].to(device)

        # --- 6) Prepare full right set for SuperGlue
        kpts1_t = kpts1.float().to(device)
        desc1_t = desc1.to(device)
        scores1_t = scores1.to(device)
        
        input_data = {
                'keypoints0': query_kpts[None, ...],
                'keypoints1': kpts1_t[None, ...],
                'descriptors0': desc0_sel[None,...],
                'descriptors1': desc1_t[None,...],
                'scores0': scores0_sel[None,...],
                'scores1': scores1_t[None,...],
                'image0': img0_t,
                'image1': img1_t
        }

        # Run SuperGlue for keypoints matching
        with torch.no_grad():
            pred = sg(input_data)
            matches0, _ = pred['matches0'], pred['matches1']

        valid_matches = matches0[matches0 > -1]
        if len(valid_matches) > 0:
            print(f"Found {len(valid_matches)} valid matches")
            
            # Store the valid matches
            for i, match_idx in enumerate(valid_matches):
                query_idx = idxs0[i]
                matched_pt = kpts1[match_idx]
                disparity = kpts0[query_idx][0].cpu().numpy() - matched_pt[0].cpu().numpy()
                matched_pt[0] = matched_pt[0] + x1 #back to original coordinates
                matched_pt[1] = matched_pt[1] + y1

                print(f"Keypoint {query_idx}: Matched to {matched_pt}, disparity={disparity}")
                all_valid_matches.append({'click_point': (x_click, y_click), 'query_idx': query_idx, 'matched_pt': matched_pt, 'disparity': disparity})

                pt1 = (int(kpts0[query_idx][0].cpu().numpy()+x1), int(kpts0[query_idx][1].cpu().numpy()+y1))
                pt2 = (int(matched_pt[0].cpu().numpy()), int(matched_pt[1].cpu().numpy()))
                disparity_slope = (pt1[1] - pt2[1]) / (pt1[0] - pt2[0])
                scene.draw_line(pt1, pt2, (255, 0, 0))

                print(f"Disparity slope: {disparity_slope}")
                if abs(disparity_slope) > slope_limit:
                    print("INVALID SLOPE!!!")
                    return []

        else:
            print("No valid match found")
            return []

    return all_valid_matches

def get_extremes(all_valid_matches):
    pixel_coords_left = np.array([[match['matched_pt'][0].cpu().numpy(), match['matched_pt'][1].cpu().numpy()] for match in all_valid_matches], dtype=np.float32).T
    pixel_coords_right = np.array([[match['matched_pt'][0].cpu().numpy() - match['disparity'], match['matched_pt'][1].cpu().numpy()] for match in all_valid_matches], dtype=np.float32).T
    
    # Triangulate points
    points_4d_hom = cv2.triangulatePoints(P1, P2, pixel_coords_left, pixel_coords_right)
    # Convert to Euclidean coordinates
    points_3d = points_4d_hom / points_4d_hom[3]
    
    selection_idx = 0
    selected_points = []
    while len(selected_points) < 2:
        key = cv2.waitKey(20) & 0xFF
        if key == ord("m"):
            print(f"Selecting midpoint between points {selection_idx} and {selection_idx+1}")
            point2_3d = points_3d[:3, selection_idx]
            selection_idx += 1
            point3_3d = points_3d[:3, selection_idx]
            selection_idx += 1
            selected_points.append((point2_3d + point3_3d) / 2)
        elif key == ord("s"):
            print(f"Selecting point {selection_idx}")
            selected_points.append(points_3d[:3, selection_idx])
            selection_idx += 1
    
    return selected_points 

def compute_distance(selected_points):
    point1_3d,  point2_3d = selected_points
    # Calculate the distance between the first point and the midpoint
    distance = np.linalg.norm(point1_3d - point2_3d)

    print(f"3D coordinates of Point 1: {point1_3d}")
    print(f"Midpoint between Point 2 and Point 3: {point2_3d}")
    print(f"Distance between Point 1 and the midpoint: {distance}")

    # Convert 3D points to pixel coordinates in image0
    point1_pixel = cv2.projectPoints(point1_3d.reshape(-1, 1, 3), np.zeros((3, 1)), np.zeros((3, 1)), P1[:3, :3], None)[0].reshape(-1, 2)
    midpoint_pixel = cv2.projectPoints(point2_3d.reshape(-1, 1, 3), np.zeros((3, 1)), np.zeros((3, 1)), P1[:3, :3], None)[0].reshape(-1, 2)

    # Draw circles at the projected pixel locations
    img = cv2.cvtColor(img1, cv2.COLOR_GRAY2BGR) 
    cv2.circle(img, (int(point1_pixel[0][0]), int(point1_pixel[0][1])), 5, (0, 0, 255), -1)
    cv2.circle(img, (int(midpoint_pixel[0][0]), int(midpoint_pixel[0][1])), 5, (0, 255, 0), -1)

    # Draw a line between the points
    cv2.line(img, (int(point1_pixel[0][0]), int(point1_pixel[0][1])), (int(midpoint_pixel[0][0]), int(midpoint_pixel[0][1])), (255, 0, 0), 2)

    # Put distance measure on screen inside a white box
    x1, y1 = 50, 50
    x2, y2 = 800, 220
    cv2.rectangle(img, (x1, y1), (x2, y2), color=(255, 255, 255), thickness=-1)

    # Define text parameters
    rounded = str(round(distance / 10, 2))
    text = f"Estimated Length: {rounded} cm"
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 1
    font_thickness = 2
    text_color = (0, 0, 0)  # Black 

    # Get text size to center it
    (text_width, text_height), baseline = cv2.getTextSize(text, font, font_scale, font_thickness)
    text_x = x1 + (x2 - x1 - text_width) // 2
    text_y = y1 + (y2 - y1 + text_height) // 2

    cv2.putText(img, text, (text_x, text_y), font, font_scale, text_color, font_thickness, cv2.LINE_AA)

    while True:
        cv2.imshow("Distance", img)
        key = cv2.waitKey(20) & 0xFF 

        if key == 27 or key == ord("q"): break

    cv2.destroyAllWindows()

def main():
    cv2.namedWindow('Image')
    cv2.setMouseCallback('Image', mouse_callback)
    scene.status = States.KEYPOINTS

    while scene.status != States.COMPUTE_DISTANCE:
        cv2.imshow('Image', scene.img_vis)

        match scene.status:
            case States.KEYPOINTS:
                packed_data = compute_keypoints()
                scene.status = States.SELECTION

            case States.SELECTION:
                key = cv2.waitKey(20) & 0xFF
                if key == 27 and len(scene.selected_points) >= 2:
                    scene.status = States.MATCHING

                # If not enough points are selected
                elif key == 27 and len(scene.selected_points) < 2:
                    print("[!] Need to select at least 2 points!")
                    for _ in range(len(scene.selected_points)):
                        scene.delete_point()
                    
                    scene.selected_points = []
            
            case States.MATCHING:
                all_valid_matches = match_keypoints(packed_data, 10, img0, img1)
                
                # If match computation failed
                if not all_valid_matches:
                    print("[!] Computation match failed!")
                    scene.status = States.SELECTION
                    for _ in range(len(scene.selected_points) + len(scene.nnpoint)):
                        scene.delete_point()
                    
                    scene.selected_points = []
                    scene.nnpoint= []
                
                else:
                    scene.status = States.ESTIMATION

            case States.ESTIMATION:
                selected_point = get_extremes(all_valid_matches)
                scene.status = States.COMPUTE_DISTANCE

    cv2.destroyAllWindows()
    compute_distance(selected_point)


if __name__ == "__main__":
    # Define device
    if torch.cuda.is_available(): device = 'cuda'
    elif torch.backends.mps.is_available(): device = 'mps'
    else: device = 'cpu'
    print(f"Using device: {device}")
    
    # Load camera parameters
    P1 = np.load('P1.npy')
    P2 = np.load('P2.npy')
    
    img0 = cv2.imread('img_left_rectified.jpg',0)
    img1 = cv2.imread('img_right_rectified.jpg',0)

    scene = Scene(img0, img1)
    gui = LiveGUI()

    # Load models
    sp_config = {'descriptor_dim': 256, 
                'nms_radius': 4, 
                'keypoint_threshold': 0.005, 
                'max_keypoints': -1, 
                'remove_borders': 4
    }
    sg_config = {
                'weights': 'outdoor',          
                'sinkhorn_iterations': 50,
                'match_threshold': 0.005,
                'max_keypoints': 512
    } 

    sp = SuperPoint(sp_config).to(device)  # SuperPoint
    sg = SuperGlue(sg_config).eval().to(device) # SuperGlue

    main()
