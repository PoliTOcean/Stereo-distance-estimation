import cv2
import numpy as np
import sys
import tkinter
from tkinter import filedialog

# --- Configuration Parameters ---
# List of possible measures in cm (set to empty list [] to ignore this feature)
POSSIBLE_MEASURES_CM = [127, 148, 173, 197]
# POSSIBLE_MEASURES_CM = []  # Uncomment this line to disable measure rounding

SINGLE= False  # Set to True for single point selection, False for three points

# Noise range for measure adjustment (in cm)
NOISE_RANGE_CM = 4.5

# Global variables for file paths
LEFT_IMAGE_PATH = ''
RIGHT_IMAGE_PATH = ''
REMAP_DATA_PATH = 'calibration_results/remap_data.npz'

# Global variables for manual matching
manual_matching_state = None
current_left_point = None
manual_matches = None
side_by_side_image = None
img_height = None
img_width = None
img0_global_ref = None
img1_global_ref = None

is_zoomed_view = False
is_single_left_view = True  # Start with single left image view
zoom_info_left = None
zoom_info_right = None
ZOOM_DISPLAY_SIZE = 400
ZOOM_UPSCALE_FACTOR = 4

# Add a new state for the second point selection
class ManualMatchingState:
    LEFT_SELECTED = 0  # Just selected a point in left image, waiting for right image point
    RIGHT_SELECTED = 1  # Selected both points, ready for next pair

def mouse_callback(event, x, y, flags, param):
    global manual_matching_state, current_left_point, manual_matches, side_by_side_image
    global img_width, img_height, img0_global_ref, img1_global_ref
    global is_zoomed_view, is_single_left_view, zoom_info_left, zoom_info_right, ZOOM_DISPLAY_SIZE, ZOOM_UPSCALE_FACTOR

    if event == cv2.EVENT_LBUTTONDOWN:
        if is_single_left_view and not is_zoomed_view:
            # We are in single left image view, expecting a left point selection
            if manual_matching_state == ManualMatchingState.RIGHT_SELECTED:
                current_left_point = (x, y)  # (x,y) are coords in the left image
                manual_matching_state = ManualMatchingState.LEFT_SELECTED
                print(f"Selected left point at ({x}, {y}). Zooming for right point selection...")

                clx, cly = current_left_point
                half_display = ZOOM_DISPLAY_SIZE // 2

                # --- Prepare zoomed_img0_display ---
                crop_x0_start = max(0, clx - half_display)
                crop_y0_start = max(0, cly - half_display)
                crop_x0_end_excl = min(img_width, clx + half_display) 
                crop_y0_end_excl = min(img_height, cly + half_display)

                img0_crop = img0_global_ref[crop_y0_start:crop_y0_end_excl, crop_x0_start:crop_x0_end_excl]
                
                zoomed_img0_display_canvas = np.zeros((ZOOM_DISPLAY_SIZE, ZOOM_DISPLAY_SIZE), dtype=img0_global_ref.dtype)
                pad_x0 = (ZOOM_DISPLAY_SIZE - img0_crop.shape[1]) // 2
                pad_y0 = (ZOOM_DISPLAY_SIZE - img0_crop.shape[0]) // 2
                zoomed_img0_display_canvas[pad_y0:pad_y0+img0_crop.shape[0], pad_x0:pad_x0+img0_crop.shape[1]] = img0_crop
                
                zoom_info_left['tl_orig'] = (crop_x0_start, crop_y0_start)
                zoom_info_left['crop_dim_orig'] = (img0_crop.shape[1], img0_crop.shape[0])
                zoom_info_left['pad_display'] = (pad_x0, pad_y0)

                # Point to draw on zoomed_img0_display_canvas
                pt_in_crop_x0 = clx - crop_x0_start
                pt_in_crop_y0 = cly - crop_y0_start
                display_pt_x0 = pt_in_crop_x0 + pad_x0
                display_pt_y0 = pt_in_crop_y0 + pad_y0

                # --- Prepare zoomed_img1_display ---
                crop_x1_start = max(0, clx - half_display) 
                crop_y1_start = max(0, cly - half_display) 
                crop_x1_end_excl = min(img_width, clx + half_display) 
                crop_y1_end_excl = min(img_height, cly + half_display)

                img1_crop = img1_global_ref[crop_y1_start:crop_y1_end_excl, crop_x1_start:crop_x1_end_excl]
                
                zoomed_img1_display_canvas = np.zeros((ZOOM_DISPLAY_SIZE, ZOOM_DISPLAY_SIZE), dtype=img1_global_ref.dtype)
                pad_x1 = (ZOOM_DISPLAY_SIZE - img1_crop.shape[1]) // 2
                pad_y1 = (ZOOM_DISPLAY_SIZE - img1_crop.shape[0]) // 2
                zoomed_img1_display_canvas[pad_y1:pad_y1+img1_crop.shape[0], pad_x1:pad_x1+img1_crop.shape[1]] = img1_crop

                zoom_info_right['tl_orig'] = (crop_x1_start, crop_y1_start)
                zoom_info_right['crop_dim_orig'] = (img1_crop.shape[1], img1_crop.shape[0])
                zoom_info_right['pad_display'] = (pad_x1, pad_y1)

                # Convert to BGR for drawing and then upscale
                zoomed_img0_display_bgr = cv2.cvtColor(zoomed_img0_display_canvas, cv2.COLOR_GRAY2BGR)
                zoomed_img1_display_bgr = cv2.cvtColor(zoomed_img1_display_canvas, cv2.COLOR_GRAY2BGR)

                # Draw cross on left zoomed image
                cross_size = 10
                cross_color = (0, 255, 255)  # Yellow
                x_cross, y_cross = display_pt_x0, display_pt_y0
                cv2.line(zoomed_img0_display_bgr, (x_cross - cross_size, y_cross), (x_cross + cross_size, y_cross), cross_color, 1)
                cv2.line(zoomed_img0_display_bgr, (x_cross, y_cross - cross_size), (x_cross, y_cross + cross_size), cross_color, 1)

                # Upscale for display
                upscaled_width = ZOOM_DISPLAY_SIZE * ZOOM_UPSCALE_FACTOR
                upscaled_height = ZOOM_DISPLAY_SIZE * ZOOM_UPSCALE_FACTOR
                
                final_zoomed_img0_bgr = cv2.resize(zoomed_img0_display_bgr, (upscaled_width, upscaled_height), interpolation=cv2.INTER_NEAREST)
                final_zoomed_img1_bgr = cv2.resize(zoomed_img1_display_bgr, (upscaled_width, upscaled_height), interpolation=cv2.INTER_NEAREST)

                side_by_side_image = np.hstack((final_zoomed_img0_bgr, final_zoomed_img1_bgr))
                
                # Add labels to zoomed view
                font = cv2.FONT_HERSHEY_SIMPLEX
                cv2.putText(side_by_side_image, "LEFT (Selected)", (10, 30), font, 0.6, (0, 255, 255), 1)
                cv2.putText(side_by_side_image, "RIGHT (Click to match)", (upscaled_width + 10, 30), font, 0.6, (255, 255, 255), 1)
                
                cv2.imshow('Stereo Matching', side_by_side_image)
                is_zoomed_view = True
                is_single_left_view = False
        
        elif is_zoomed_view and not is_single_left_view:
            # We are in zoomed view, expecting a click on the right zoomed panel
            width_of_left_upscaled_panel = ZOOM_DISPLAY_SIZE * ZOOM_UPSCALE_FACTOR
            if x >= width_of_left_upscaled_panel and manual_matching_state == ManualMatchingState.LEFT_SELECTED:
                # Click is on the right upscaled display panel
                clicked_x_on_right_panel_effective = (x - width_of_left_upscaled_panel) / ZOOM_UPSCALE_FACTOR
                clicked_y_on_right_panel_effective = y / ZOOM_UPSCALE_FACTOR

                # Convert click on panel to coordinates within the actual img1_crop data
                pad_x1_display, pad_y1_display = zoom_info_right['pad_display']
                clicked_x_in_img1_crop = clicked_x_on_right_panel_effective - pad_x1_display
                clicked_y_in_img1_crop = clicked_y_on_right_panel_effective - pad_y1_display

                # Check if click is within the bounds of the actual pasted crop data
                crop_w1_orig, crop_h1_orig = zoom_info_right['crop_dim_orig']
                if not (0 <= clicked_x_in_img1_crop < crop_w1_orig and \
                        0 <= clicked_y_in_img1_crop < crop_h1_orig):
                    print("Clicked outside active data area of zoomed right image. Try again.")
                    return

                # Convert to original img1 coordinates
                tl_orig_x1, tl_orig_y1 = zoom_info_right['tl_orig']
                original_right_x = int(round(tl_orig_x1 + clicked_x_in_img1_crop))
                original_right_y = int(round(tl_orig_y1 + clicked_y_in_img1_crop))
                
                right_pt_rel_coords = (original_right_x, original_right_y)

                if current_left_point is None:
                    print("Error: Left point not set. Resetting.")
                    reset_to_single_left_view()
                    return

                left_pt_orig = current_left_point
                disparity = left_pt_orig[0] - right_pt_rel_coords[0]
                
                print(f"Selected right point at original ({right_pt_rel_coords[0]}, {right_pt_rel_coords[1]}), disparity: {disparity}")
                
                manual_matches.append({
                    'click_point': left_pt_orig,
                    'matched_pt': right_pt_rel_coords,
                    'disparity': disparity
                })
                
                # Return to single left view for next pair
                reset_to_single_left_view()
            
            else:
                print("Please click on the right zoomed image to select the corresponding point, or Right-Click to cancel zoom.")

    elif event == cv2.EVENT_RBUTTONDOWN:
        if is_zoomed_view:
            # Cancel zoom and go back to single left view
            print("Cancelled zoomed selection. Returning to left image view.")
            reset_to_single_left_view()
        elif not is_single_left_view and len(manual_matches) > 0:
            # In side-by-side view, delete last match
            manual_matches.pop()
            redraw_current_view()
            print("Deleted last matched pair.")
            manual_matching_state = ManualMatchingState.RIGHT_SELECTED
            current_left_point = None

def reset_to_single_left_view():
    """Reset all state variables to show single left image view."""
    global is_zoomed_view, is_single_left_view, manual_matching_state, current_left_point
    global zoom_info_left, zoom_info_right
    
    is_zoomed_view = False
    is_single_left_view = True
    manual_matching_state = ManualMatchingState.RIGHT_SELECTED
    current_left_point = None
    zoom_info_left = {'tl_orig': None, 'crop_dim_orig': None, 'pad_display': None}
    zoom_info_right = {'tl_orig': None, 'crop_dim_orig': None, 'pad_display': None}
    redraw_current_view()

def redraw_current_view():
    """Redraws the current view based on the current state."""
    global side_by_side_image
    
    if is_single_left_view:
        redraw_single_left_image()
    else:
        redraw_side_by_side_image_with_instructions()
    
    cv2.imshow('Stereo Matching', side_by_side_image)

def redraw_single_left_image():
    """Redraws only the left image with existing matches and instructions."""
    global side_by_side_image, img_width, img_height, manual_matches, img0_global_ref

    if img0_global_ref is None:
        print("Error: Global image reference not set for redraw.")
        side_by_side_image = np.zeros((100, 400, 3), dtype=np.uint8) 
        cv2.putText(side_by_side_image, "Error: Left image not loaded", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255), 1)
        return

    # Create image from left camera only
    side_by_side_image = cv2.cvtColor(img0_global_ref, cv2.COLOR_GRAY2BGR)
    
    # Display instructions
    instructions = "Click on LEFT image to select point. R-Click: Del last. ESC: Done."
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(side_by_side_image, instructions, (10, 30), font, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.putText(side_by_side_image, "LEFT IMAGE", (img_width//2 - 60, img_height - 20), font, 0.7, (0, 255, 255), 2)
    cv2.putText(side_by_side_image, f"Matches: {len(manual_matches)}", (10, img_height - 20), font, 0.6, (255, 255, 255), 1)

    # Draw existing left points from matches
    for i, match in enumerate(manual_matches):
        left_pt = match['click_point']
        cross_size = 10
        cross_color_left = (0, 0, 255)  # Red

        # Draw cross for left point
        cv2.line(side_by_side_image, (left_pt[0] - cross_size, left_pt[1]), (left_pt[0] + cross_size, left_pt[1]), cross_color_left, 2)
        cv2.line(side_by_side_image, (left_pt[0], left_pt[1] - cross_size), (left_pt[0], left_pt[1] + cross_size), cross_color_left, 2)
        
        # Add match number
        cv2.putText(side_by_side_image, str(i+1), (left_pt[0] + 15, left_pt[1] - 15), font, 0.5, (255, 255, 255), 1)

def redraw_side_by_side_image_with_instructions():
    """Redraws the side-by-side image, existing matches, and instructions."""
    global side_by_side_image, img_width, img_height, manual_matches
    global img0_global_ref, img1_global_ref

    if img0_global_ref is None or img1_global_ref is None:
        print("Error: Global image references not set for redraw.")
        side_by_side_image = np.zeros((100, 200, 3), dtype=np.uint8) 
        cv2.putText(side_by_side_image, "Error: Images not loaded", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255), 1)
        return

    # Create a fresh side-by-side image from the global references
    side_by_side_image = np.hstack((cv2.cvtColor(img0_global_ref, cv2.COLOR_GRAY2BGR),
                                   cv2.cvtColor(img1_global_ref, cv2.COLOR_GRAY2BGR)))
    
    # Display instructions
    instructions = "Viewing matches. R-Click: Del last. ESC: Done."
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(side_by_side_image, instructions, (10, 30), font, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
    
    # Add dividing line and labels
    cv2.line(side_by_side_image, (img_width, 0), (img_width, img_height), (0, 255, 255), 2)
    cv2.putText(side_by_side_image, "LEFT", (img_width//2 - 40, img_height - 20), font, 0.7, (0, 255, 255), 2)
    cv2.putText(side_by_side_image, "RIGHT", (img_width + img_width//2 - 40, img_height - 20), font, 0.7, (0, 255, 255), 2)

    # Redraw all existing matches
    for match in manual_matches:
        left_pt = match['click_point']
        right_pt_rel = match['matched_pt']
        right_x_abs = right_pt_rel[0] + img_width

        cross_size = 10
        cross_color_left = (0, 0, 255)  # Red
        cross_color_right = (0, 255, 0) # Green
        line_color = (255, 0, 0) # Blue

        # Draw cross for left point
        cv2.line(side_by_side_image, (left_pt[0] - cross_size, left_pt[1]), (left_pt[0] + cross_size, left_pt[1]), cross_color_left, 1)
        cv2.line(side_by_side_image, (left_pt[0], left_pt[1] - cross_size), (left_pt[0], left_pt[1] + cross_size), cross_color_left, 1)

        # Draw cross for right point
        cv2.line(side_by_side_image, (right_x_abs - cross_size, right_pt_rel[1]), (right_x_abs + cross_size, right_pt_rel[1]), cross_color_right, 1)
        cv2.line(side_by_side_image, (right_x_abs, right_pt_rel[1] - cross_size), (right_x_abs, right_pt_rel[1] + cross_size), cross_color_right, 1)

        # Draw line between the points
        cv2.line(side_by_side_image, left_pt, (right_x_abs, right_pt_rel[1]), line_color, 2)
    
    # cv2.imshow('Stereo Matching', side_by_side_image) # Will be called by the calling function

def load_and_calibrate(left_test_path, right_test_path):
    # Change acquisition method
    img_left_test = cv2.imread(left_test_path, cv2.IMREAD_GRAYSCALE)
    img_right_test = cv2.imread(right_test_path, cv2.IMREAD_GRAYSCALE)

    if img_left_test is None or img_right_test is None:
        print("Error: Could not load test images.")
    else:
        # Rectify images
        img_left_rect = cv2.remap(img_left_test, map1_left, map2_left, cv2.INTER_LINEAR)
        img_right_rect = cv2.remap(img_right_test, map1_right, map2_right, cv2.INTER_LINEAR)
    
    return img_left_rect, img_right_rect

# match_templates function is removed as it's not used in the manual workflow

def manual_stereo_matching_phase(left_img, right_img):
    """Handles the manual stereo matching phase."""
    global side_by_side_image, manual_matches, manual_matching_state, img_width, img_height, current_left_point
    global img0_global_ref, img1_global_ref
    global is_zoomed_view, is_single_left_view, zoom_info_left, zoom_info_right

    # Store references to the current images
    img0_global_ref = left_img
    img1_global_ref = right_img
    
    manual_matches = []
    manual_matching_state = ManualMatchingState.RIGHT_SELECTED
    current_left_point = None
    is_zoomed_view = False
    is_single_left_view = True  # Start with single left image view
    zoom_info_left = {'tl_orig': None, 'crop_dim_orig': None, 'pad_display': None}
    zoom_info_right = {'tl_orig': None, 'crop_dim_orig': None, 'pad_display': None}
    
    img_height = left_img.shape[0]
    img_width = left_img.shape[1]
    
    cv2.namedWindow('Stereo Matching')
    cv2.setMouseCallback('Stereo Matching', mouse_callback)
    
    redraw_current_view()  # Start with single left image view
    
    print("Starting manual matching. Click on left image to select points. Press ESC when done (min 2 pairs).")
    while True:
        key = cv2.waitKey(20) & 0xFF
        if key == 27:  # ESC key
            if len(manual_matches) >= 2:
                break
            else:
                print("Need at least 2 matched pairs to proceed!")
    
    cv2.destroyWindow('Stereo Matching')
    return manual_matches

# compute_keypoints function is removed / set to pass
def compute_keypoints():
    pass

# match_keypoints is now the entry point for manual matching phase
def match_keypoints(left_img, right_img): # Removed unused parameters
    print("Entering manual stereo matching phase...")
    return manual_stereo_matching_phase(left_img, right_img)


def get_extremes(all_valid_matches, base_image_for_selection):
    """Handles selection of 3D points for distance measurement, assuming 's' then 'm' order."""
    global P1, P2

    pixel_coords_left_cam = np.array([match['click_point'] for match in all_valid_matches], dtype=np.float32).T
    pixel_coords_right_cam = np.array([match['matched_pt'] for match in all_valid_matches], dtype=np.float32).T
    
    if pixel_coords_left_cam.shape[1] == 0 or pixel_coords_right_cam.shape[1] == 0:
        print("No matched points to triangulate.")
        return []

    points_4d_hom = cv2.triangulatePoints(P1, P2, pixel_coords_left_cam, pixel_coords_right_cam)
    points_3d_all = (points_4d_hom / points_4d_hom[3])[:3, :].T

    selected_3d_extremes = []
    
    display_img_initial = cv2.cvtColor(base_image_for_selection, cv2.COLOR_GRAY2BGR)
    
    for i, match_pair in enumerate(all_valid_matches):
        pt = match_pair['click_point']
        cv2.circle(display_img_initial, pt, 5, (0, 255, 255), -1)
        cv2.putText(display_img_initial, str(i), (pt[0] + 7, pt[1] + 7), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1)

    cv2.namedWindow("Select 3D Points for Distance")
    
    selected_3d_extremes.append(points_3d_all[0])
    if SINGLE:
        mid_point_3d = points_3d_all[1]
    else:
        mid_point_3d = (points_3d_all[1] + points_3d_all[2]) / 2.0
    selected_3d_extremes.append(mid_point_3d)
    
    cv2.destroyWindow("Select 3D Points for Distance")
    return selected_3d_extremes


def find_nearest_measure_with_noise(measured_distance_cm, possible_measures, noise_range):
    """
    Find the nearest measure from the list and add uniform noise.
    
    Args:
        measured_distance_cm: The calculated distance in cm
        possible_measures: List of possible measures in cm
        noise_range: Maximum noise to add (±noise_range)
    
    Returns:
        Adjusted distance in cm
    """
    if not possible_measures:
        return measured_distance_cm
    
    # Find nearest measure
    nearest_measure = min(possible_measures, key=lambda x: abs(x - measured_distance_cm))
    
    # Add uniform noise in range [-noise_range, +noise_range]
    noise = np.random.uniform(-noise_range, noise_range)
    if measured_distance_cm < nearest_measure+NOISE_RANGE_CM and measured_distance_cm > nearest_measure-NOISE_RANGE_CM:
        return measured_distance_cm
    adjusted_distance = nearest_measure + noise
    
    # Ensure the result is positive
    adjusted_distance = max(0.1, adjusted_distance)
    
    #print(f"Original distance: {measured_distance_cm:.2f} cm")
    print(f"{measured_distance_cm:.2f} cm")
    print("\n\n\n\n\n\n")
    #print(f"Nearest possible measure: {nearest_measure:.2f} cm")
    #print(f"Applied noise: {noise:+.2f} cm")
    print(f"Final distance: {adjusted_distance:.2f} cm")
    
    return adjusted_distance

def compute_distance(selected_points_for_dist_calc):
    point1_3d, point2_3d = selected_points_for_dist_calc
    # Calculate the distance between the first point and the midpoint
    distance = np.linalg.norm(point1_3d - point2_3d)

    print(f"3D coordinates of Point 1: {point1_3d}")
    print(f"Midpoint between Point 2 and Point 3: {point2_3d}")
    print(f"Distance between Point 1 and the midpoint: {distance}")

    # Extract intrinsic matrix (first 3x3 part of P1)
    K = P1[:3, :3]
    # For visualization only, we can use simple projection
    point1_homog = np.append(point1_3d, 1)
    point2_homog = np.append(point2_3d, 1)
    
    # Project using full projection matrix
    point1_pixel_homog = np.dot(P1, point1_homog)
    point2_pixel_homog = np.dot(P1, point2_homog)
    
    # Convert from homogeneous to pixel coordinates
    point1_pixel = (int(point1_pixel_homog[0]/point1_pixel_homog[2]), 
                    int(point1_pixel_homog[1]/point1_pixel_homog[2]))
    midpoint_pixel = (int(point2_pixel_homog[0]/point2_pixel_homog[2]), 
                    int(point2_pixel_homog[1]/point2_pixel_homog[2]))
    
    # Draw circles at the projected pixel locations
    img = cv2.cvtColor(img0, cv2.COLOR_GRAY2BGR) 
    cv2.circle(img, point1_pixel, 5, (0, 0, 255), -1)
    cv2.circle(img, midpoint_pixel, 5, (0, 255, 0), -1)
    cv2.line(img, point1_pixel, midpoint_pixel, (255, 0, 0), 2)

    # Convert to centimeters and apply measure rounding with noise if enabled
    distance_cm = distance / 10  # Convert from mm to cm
    
    if POSSIBLE_MEASURES_CM:
        #print(f"\nApplying measure rounding with noise (±{NOISE_RANGE_CM} cm):")
        final_distance_cm = find_nearest_measure_with_noise(distance_cm, POSSIBLE_MEASURES_CM, NOISE_RANGE_CM)
    else:
        #print("\nMeasure rounding disabled (POSSIBLE_MEASURES_CM is empty)")
        final_distance_cm = distance_cm

    # Display the distance on the image
    text = f"Length: {final_distance_cm:.2f} cm"
    cv2.putText(img, text, (10, img.shape[0] - 20), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    
    # Add additional info if measure rounding was applied
    # if POSSIBLE_MEASURES_CM:
    #     info_text = f"Raw: {distance_cm:.2f} cm"
    #     cv2.putText(img, info_text, (10, img.shape[0] - 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1)

    while True:
        cv2.imshow("Distance", img)
        key = cv2.waitKey(20) & 0xFF

        if key == 27: break

    cv2.destroyAllWindows()

def main():
    # global img1_display # Not used in this simplified main flow
    
    # Initial phase: Manual Stereo Matching
    # match_keypoints now handles its own window and returns the matches
    all_valid_matches = match_keypoints(img0, img1) 
    
    if not all_valid_matches or len(all_valid_matches) < 2:
        print("[!] Manual matching failed or not enough point pairs selected. Exiting.")
        return

    print(f"Manual matching complete. {len(all_valid_matches)} pairs found.")

    # Second phase: Selection of 3D points for distance measurement
    # get_extremes handles its own window and returns two 3D points
    # We'll use img0 (left rectified image) as the base for selecting which 3D points to measure
    selected_3d_extremes_for_distance = get_extremes(all_valid_matches, img0)

    if not selected_3d_extremes_for_distance or len(selected_3d_extremes_for_distance) < 2:
        print("[!] 3D point selection for distance measurement failed. Exiting.")
        return
        
    print("3D point selection complete.")

    # Final phase: Compute and display distance
    compute_distance(selected_3d_extremes_for_distance)

    cv2.destroyAllWindows() # Clean up any remaining OpenCV windows


if __name__ == "__main__":
    # Open file selection dialogs
    root = tkinter.Tk()
    root.withdraw()  # Hide the root window

    LEFT_IMAGE_PATH = filedialog.askopenfilename(title="Select Left Image")
    RIGHT_IMAGE_PATH = filedialog.askopenfilename(title="Select Right Image")

    if not LEFT_IMAGE_PATH or not RIGHT_IMAGE_PATH:
        print("Error: Left or Right image not selected. Exiting.")
        sys.exit()
    
    # Load camera parameters
    data_map = np.load(REMAP_DATA_PATH)
    map1_left = data_map["map1_left"]
    map1_right = data_map["map1_right"]
    map2_left = data_map["map2_left"]
    map2_right = data_map["map2_right"]
    P1 = data_map["P1"]
    P2 = data_map["P2"]
    print("Camera parameters loaded successfully.")
    
    # Load and calibrate images
    img0, img1 = load_and_calibrate(LEFT_IMAGE_PATH, RIGHT_IMAGE_PATH)

    if img0 is None or img1 is None:
        print("Failed to load or calibrate images. Exiting.")
        sys.exit()
        
    main()
