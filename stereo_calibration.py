import numpy as np
import cv2
import glob
import matplotlib.pyplot as plt
import os

# --- Configuration Parameters ---
# Chessboard dimensions (number of inner corners)
CHESSBOARD_SIZE = (10, 7)  # Adjust based on your chessboard
SQUARE_SIZE = 35  # Define the real-world size of a square (e.g., in mm)

# Known baseline (set to None if unknown, or provide value in same units as SQUARE_SIZE)
KNOWN_BASELINE = None  # Replace with your actual baseline distance in mm, or set to None

# Stereo calibration flag presets - choose one or customize
STEREO_CALIB_MODE = "CUSTOM"  # Options: "STRICT", "BALANCED", "FLEXIBLE", "CUSTOM"

# Termination criteria for corner refinement
CRITERIA = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.01)
STEREOCALIB_CRITERIA = (cv2.TERM_CRITERIA_MAX_ITER + cv2.TERM_CRITERIA_EPS, 30, 1e-5)

# Paths to calibration images
LEFT_CALIB_PATH_PATTERN = 'photos/left/calibration8/*.jpg'
RIGHT_CALIB_PATH_PATTERN = 'photos/right/calibration8/*.jpg'

# Paths to test images for rectification display
LEFT_TEST_IMAGE_PATH = 'photos/left/test2/stereo_camera-main_20250612_173429567944.jpg'
RIGHT_TEST_IMAGE_PATH = 'photos/right/test2/stereo_camera-right_20250612_173429567944.jpg'

# Output directory for calibration files
OUTPUT_DIR = "calibration_results"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# --- Helper Functions ---
def prepare_object_points(chessboard_size, square_size_val):
    """Prepares the 3D object points for the chessboard."""
    objp = np.zeros((chessboard_size[0] * chessboard_size[1], 3), np.float32)
    objp[:, :2] = np.mgrid[0:chessboard_size[0], 0:chessboard_size[1]].T.reshape(-1, 2)
    objp = objp * square_size_val
    return objp

def find_corners_in_images(image_files, chessboard_size, criteria, objp_template):
    """Finds chessboard corners in a list of images."""
    objpoints_list = []  # 3D points in real world space
    imgpoints_list = []  # 2D points in image plane
    img_shape = None

    print(f"Processing {len(image_files)} images...")
    for fname in image_files:
        img = cv2.imread(fname)
        if img is None:
            print(f"Warning: Could not read image {fname}")
            continue
        
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        if img_shape is None:
            img_shape = gray.shape[::-1]  # Get (width, height)

        ret, corners = cv2.findChessboardCorners(gray, chessboard_size, None)
        if ret:
            objpoints_list.append(objp_template)
            corners_refined = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
            imgpoints_list.append(corners_refined)
        else:
            print(f"Warning: Chessboard not found in {fname}")
            
    if img_shape is None and image_files:
        # Attempt to get shape from the first image if all failed corner detection
        try:
            first_img = cv2.imread(image_files[0])
            if first_img is not None:
                img_shape = cv2.cvtColor(first_img, cv2.COLOR_BGR2GRAY).shape[::-1]
        except Exception as e:
            print(f"Could not determine image shape: {e}")

    return objpoints_list, imgpoints_list, img_shape

def get_stereo_calibration_flags(mode="BALANCED"):
    """Get stereo calibration flags based on selected mode."""
    
    if mode == "STRICT":
        # Most restrictive - use when you trust individual camera calibrations
        flags = cv2.CALIB_FIX_INTRINSIC
        flags |= cv2.CALIB_FIX_PRINCIPAL_POINT
        flags |= cv2.CALIB_SAME_FOCAL_LENGTH
        flags |= cv2.CALIB_RATIONAL_MODEL
        print("Using STRICT mode: Fixed intrinsics, same focal length")
        
    elif mode == "BALANCED":
        # Recommended for most cases - allows some intrinsic refinement
        flags = cv2.CALIB_USE_INTRINSIC_GUESS
        flags |= cv2.CALIB_RATIONAL_MODEL
        flags |= cv2.CALIB_FIX_FOCAL_LENGTH  # Keep focal lengths from individual calibration
        flags |= cv2.CALIB_FIX_PRINCIPAL_POINT
        print("Using BALANCED mode: Some intrinsic refinement allowed")
        
    elif mode == "FLEXIBLE":
        # Most flexible - good when individual calibrations might be suboptimal
        flags = cv2.CALIB_USE_INTRINSIC_GUESS
        flags |= cv2.CALIB_RATIONAL_MODEL
        print("Using FLEXIBLE mode: Maximum parameter optimization")
        
    elif mode == "CUSTOM":
        # Define your own flags here
        flags = cv2.CALIB_FIX_INTRINSIC
        flags |= cv2.CALIB_FIX_PRINCIPAL_POINT
        flags |= cv2.CALIB_FIX_FOCAL_LENGTH
        flags |= cv2.CALIB_RATIONAL_MODEL
        # Add more flags as needed:
        flags |= cv2.CALIB_SAME_FOCAL_LENGTH  # Force same focal length for both cameras
        #flags |= cv2.CALIB_ZERO_TANGENT_DIST  # Assume no tangential distortion
        flags |= cv2.CALIB_FIX_K1  # Fix radial distortion coefficients
        flags |= cv2.CALIB_FIX_K2
        flags |= cv2.CALIB_FIX_K3
        print("Using CUSTOM mode: User-defined flags")
        
    else:
        raise ValueError(f"Unknown stereo calibration mode: {mode}")
    
    return flags

def validate_and_correct_baseline(T_calibrated, known_baseline, tolerance=0.05):
    """Validate calibrated baseline against known value and optionally correct it."""
    calibrated_baseline = abs(T_calibrated[0])
    relative_error = abs(calibrated_baseline - known_baseline) / known_baseline
    
    print(f"\nBaseline Validation:")
    print(f"Known baseline: {known_baseline:.2f} mm")
    print(f"Calibrated baseline: {float(calibrated_baseline):.2f} mm")
    print(f"Absolute difference: {float(abs(calibrated_baseline - known_baseline)):.2f} mm")
    print(f"Relative error: {float(relative_error * 100):.2f}%")
    
    if relative_error > tolerance:
        print(f"WARNING: Large baseline discrepancy (>{tolerance*100:.1f}%).")
        print("Consider:")
        print("1. Using a more flexible stereo calibration mode")
        print("2. Checking image quality and chessboard detection")
        print("3. Verifying the known baseline measurement")
        
        use_known = input("Use known baseline instead of calibrated? (y/n): ").lower().strip()
        if use_known == 'y':
            T_corrected = T_calibrated.copy()
            T_corrected[0] = -known_baseline if T_calibrated[0] < 0 else known_baseline
            print(f"Using known baseline: {known_baseline:.2f} mm")
            return T_corrected, True
        else:
            print("Using calibrated baseline.")
            return T_calibrated, False
    else:
        print("Baseline validation passed. Using calibrated value.")
        return T_calibrated, False

# --- Main Calibration Script ---
def main():
    objp_template = prepare_object_points(CHESSBOARD_SIZE, SQUARE_SIZE)

    left_calib_files = sorted(glob.glob(LEFT_CALIB_PATH_PATTERN))
    right_calib_files = sorted(glob.glob(RIGHT_CALIB_PATH_PATTERN))

    if not left_calib_files or not right_calib_files:
        print("Error: No calibration images found. Check paths.")
        return

    # 1. Shared Intrinsic Calibration (using all images)
    print("--- Step 1: Shared Intrinsic Camera Calibration ---")
    all_calib_files = left_calib_files + right_calib_files
    if not all_calib_files:
        print("Error: No calibration images found for intrinsic calibration.")
        return

    objpoints_combined, imgpoints_combined, img_shape_combined = find_corners_in_images(
        all_calib_files, CHESSBOARD_SIZE, CRITERIA, objp_template
    )

    if not imgpoints_combined:
        print("Error: No chessboard corners found in any image for intrinsic calibration. Exiting.")
        return
    if img_shape_combined is None:
        print("Error: Could not determine image shape for intrinsic calibration. Exiting.")
        return

    print(f"Using {len(imgpoints_combined)} views for shared intrinsic calibration.")
    ret_shared, mtx_shared, dist_shared, rvecs_shared, tvecs_shared = cv2.calibrateCamera(
        objpoints_combined, imgpoints_combined, img_shape_combined, None, None
    )

    if not ret_shared:
        print("Error: Shared intrinsic camera calibration failed.")
        return

    print("Shared Camera Matrix (mtx_shared):", mtx_shared)
    print("Shared Distortion Coefficients (dist_shared):", dist_shared)

    # 2. Prepare points for Stereo Calibration (paired images)
    print("\--- Step 2: Preparing Data for Stereo Calibration ---")
    objpoints_stereo_pairs = []
    imgpoints_left_stereo_pairs = []
    imgpoints_right_stereo_pairs = []
    
    # Ensure we have the same number of left and right images for pairing
    min_paired_images = min(len(left_calib_files), len(right_calib_files))
    if len(left_calib_files) != len(right_calib_files):
        print(f"Warning: Number of left ({len(left_calib_files)}) and right ({len(right_calib_files)}) images differ. Using {min_paired_images} pairs.")

    img_shape_stereo = None # Will be set from the first valid pair

    for i in range(min_paired_images):
        left_img_path = left_calib_files[i]
        right_img_path = right_calib_files[i]

        img_l = cv2.imread(left_img_path)
        img_r = cv2.imread(right_img_path)

        if img_l is None or img_r is None:
            print(f"Warning: Could not read image pair: {left_img_path}, {right_img_path}")
            continue

        gray_l = cv2.cvtColor(img_l, cv2.COLOR_BGR2GRAY)
        gray_r = cv2.cvtColor(img_r, cv2.COLOR_BGR2GRAY)
        
        if img_shape_stereo is None:
            img_shape_stereo = gray_l.shape[::-1]
        elif img_shape_stereo != gray_l.shape[::-1] or img_shape_stereo != gray_r.shape[::-1]:
            print(f"Warning: Image shapes mismatch for pair {left_img_path}, {right_img_path}. Skipping.")
            continue

        ret_l, corners_l = cv2.findChessboardCorners(gray_l, CHESSBOARD_SIZE, None)
        ret_r, corners_r = cv2.findChessboardCorners(gray_r, CHESSBOARD_SIZE, None)

        if ret_l and ret_r:
            objpoints_stereo_pairs.append(objp_template)
            corners_l_refined = cv2.cornerSubPix(gray_l, corners_l, (11, 11), (-1, -1), CRITERIA)
            imgpoints_left_stereo_pairs.append(corners_l_refined)
            corners_r_refined = cv2.cornerSubPix(gray_r, corners_r, (11, 11), (-1, -1), CRITERIA)
            imgpoints_right_stereo_pairs.append(corners_r_refined)
        else:
            print(f"Warning: Chessboard not found in one or both images of pair: {os.path.basename(left_img_path)}, {os.path.basename(right_img_path)}")

    if not imgpoints_left_stereo_pairs or not imgpoints_right_stereo_pairs:
        print("Error: Not enough valid image pairs with detected corners for stereo calibration. Exiting.")
        return
    if img_shape_stereo is None:
        print("Error: Could not determine image shape for stereo calibration. Exiting.")
        return

    print(f"Using {len(imgpoints_left_stereo_pairs)} image pairs for stereo calibration.")

    # 3. Stereo Calibration
    print("--- Step 3: Stereo Calibration ---")
    
    # Get stereo calibration flags based on selected mode
    flags = get_stereo_calibration_flags(STEREO_CALIB_MODE)
    
    print(f"Stereo calibration flags: {flags}")
    print("Flag breakdown:")
    if flags & cv2.CALIB_FIX_INTRINSIC:
        print("  - CALIB_FIX_INTRINSIC: Intrinsic parameters are fixed")
    if flags & cv2.CALIB_USE_INTRINSIC_GUESS:
        print("  - CALIB_USE_INTRINSIC_GUESS: Use provided intrinsics as initial guess")
    if flags & cv2.CALIB_SAME_FOCAL_LENGTH:
        print("  - CALIB_SAME_FOCAL_LENGTH: Force same focal length for both cameras")
    if flags & cv2.CALIB_FIX_FOCAL_LENGTH:
        print("  - CALIB_FIX_FOCAL_LENGTH: Fix focal lengths from individual calibration")
    if flags & cv2.CALIB_FIX_PRINCIPAL_POINT:
        print("  - CALIB_FIX_PRINCIPAL_POINT: Fix principal points")
    if flags & cv2.CALIB_RATIONAL_MODEL:
        print("  - CALIB_RATIONAL_MODEL: Use rational distortion model")

    ret_stereo, mtx_left_stereo, dist_left_stereo, mtx_right_stereo, dist_right_stereo, R, T, E, F = cv2.stereoCalibrate(
        objpoints_stereo_pairs, imgpoints_left_stereo_pairs, imgpoints_right_stereo_pairs,
        mtx_shared, dist_shared,  # Left camera intrinsics
        mtx_shared, dist_shared,  # Right camera intrinsics (same as left)
        img_shape_stereo,
        criteria=STEREOCALIB_CRITERIA,
        flags=flags
    )

    if not ret_stereo:
        print("Error: Stereo calibration failed.")
        print("Try a different STEREO_CALIB_MODE or check your images.")
        return

    print(f"Stereo calibration successful with RMS error: {ret_stereo:.6f}")
    print("Rotation Matrix (R):", R)
    print("Translation Vector (T):", T)
    
    # Print refined intrinsics if they were allowed to change
    if not (flags & cv2.CALIB_FIX_INTRINSIC):
        print("Refined Left Camera Matrix:", mtx_left_stereo)
        print("Refined Right Camera Matrix:", mtx_right_stereo)
        # Use refined matrices for rectification
        mtx_left_final = mtx_left_stereo
        mtx_right_final = mtx_right_stereo
        dist_left_final = dist_left_stereo
        dist_right_final = dist_right_stereo
    else:
        # Use original shared matrices
        mtx_left_final = mtx_shared
        mtx_right_final = mtx_shared
        dist_left_final = dist_shared
        dist_right_final = dist_shared
    
    # Validate and potentially correct the baseline
    if KNOWN_BASELINE is None:
        print("No known baseline provided. Using calibrated baseline.")
    else:
        T_final, baseline_corrected = validate_and_correct_baseline(T, KNOWN_BASELINE)
        if baseline_corrected:
            print("Using corrected baseline for rectification.")
            T = T_final

    # Print baseline
    print(f"Final baseline: {abs(T[0])} mm")

    # 4. Stereo Rectification
    print("--- Step 4: Stereo Rectification ---")
    R1, R2, P1, P2, Q, roi_left, roi_right = cv2.stereoRectify(
        mtx_left_final, dist_left_final, mtx_right_final, dist_right_final,
        img_shape_stereo, R, T,
        flags=cv2.CALIB_ZERO_DISPARITY, alpha=0.9
    )

    print("Rectification Projection Matrix Left (P1):", P1)
    print("Rectification Projection Matrix Right (P2):", P2)
    print("Disparity-to-depth mapping matrix (Q):", Q)

    # Save P1, P2, and Q
    np.save(os.path.join(OUTPUT_DIR, 'P1.npy'), P1)
    np.save(os.path.join(OUTPUT_DIR, 'P2.npy'), P2)
    np.save(os.path.join(OUTPUT_DIR, 'Q.npy'), Q)
    print(f"P1, P2, and Q saved to {OUTPUT_DIR}/")

    # 5. Compute and Save Undistortion Maps
    print("--- Step 5: Computing and Saving Rectification Maps ---")
    map1_left, map2_left = cv2.initUndistortRectifyMap(
        mtx_left_final, dist_left_final, R1, P1, img_shape_stereo, cv2.CV_16SC2
    )
    map1_right, map2_right = cv2.initUndistortRectifyMap(
        mtx_right_final, dist_right_final, R2, P2, img_shape_stereo, cv2.CV_16SC2
    )
    print("Rectification maps computed.")

    # Save calibration results including baseline information
    calibration_data = {
        'map1_left': map1_left, 'map2_left': map2_left,
        'map1_right': map1_right, 'map2_right': map2_right,
        'mtx_left': mtx_left_final, 'mtx_right': mtx_right_final,
        'dist_left': dist_left_final, 'dist_right': dist_right_final,
        'mtx_shared': mtx_shared, 'dist_shared': dist_shared,
        'img_shape': img_shape_stereo, 'Q': Q, 'R': R, 'T': T, 'P1': P1, 'P2': P2,
        'known_baseline': KNOWN_BASELINE,
        'final_baseline': abs(T[0]),
        'calibration_mode': STEREO_CALIB_MODE,
        'rms_error': ret_stereo
    }
    
    np.savez(os.path.join(OUTPUT_DIR, "remap_data.npz"), **calibration_data)
    print(f"Remap data and calibration parameters saved to {os.path.join(OUTPUT_DIR, 'remap_data.npz')}")

    # 6. Load and Rectify Test Images for Display
    print("\--- Step 6: Displaying Rectified Test Images ---")
    img_left_test = cv2.imread(LEFT_TEST_IMAGE_PATH)
    img_right_test = cv2.imread(RIGHT_TEST_IMAGE_PATH)

    if img_left_test is None or img_right_test is None:
        print("Error: Could not load test images. Skipping display.")
        return
    
    if img_left_test.shape[:2][::-1] != img_shape_stereo or img_right_test.shape[:2][::-1] != img_shape_stereo:
        print(f"Warning: Test image dimensions do not match calibration image dimensions.")
        print("Attempting to resize test images for rectification. Results might be suboptimal.")
        img_left_test = cv2.resize(img_left_test, img_shape_stereo)
        img_right_test = cv2.resize(img_right_test, img_shape_stereo)

    img_left_rect = cv2.remap(img_left_test, map1_left, map2_left, cv2.INTER_LINEAR)
    img_right_rect = cv2.remap(img_right_test, map1_right, map2_right, cv2.INTER_LINEAR)

    print("Test images loaded and rectified.")

    # Display rectified images
    fig, axes = plt.subplots(1, 2, figsize=(15, 10))
    axes[0].imshow(cv2.cvtColor(img_left_rect, cv2.COLOR_BGR2RGB))
    axes[0].set_title('Rectified Left Test Image')
    axes[0].axis('off')
    axes[1].imshow(cv2.cvtColor(img_right_rect, cv2.COLOR_BGR2RGB))
    axes[1].set_title('Rectified Right Test Image')
    axes[1].axis('off')
    
    # Draw epipolar lines for verification
    img_left_with_lines = img_left_rect.copy()
    img_right_with_lines = img_right_rect.copy()
    for i in range(0, img_left_rect.shape[0], 30):
        cv2.line(img_left_with_lines, (0, i), (img_left_rect.shape[1], i), (0, 255, 0), 1)
        cv2.line(img_right_with_lines, (0, i), (img_right_rect.shape[1], i), (0, 255, 0), 1)

    fig_with_lines, axes_lines = plt.subplots(1, 2, figsize=(15, 10))
    axes_lines[0].imshow(cv2.cvtColor(img_left_with_lines, cv2.COLOR_BGR2RGB))
    axes_lines[0].set_title('Rectified Left with Epipolar Lines')
    axes_lines[0].axis('off')
    axes_lines[1].imshow(cv2.cvtColor(img_right_with_lines, cv2.COLOR_BGR2RGB))
    axes_lines[1].set_title('Rectified Right with Epipolar Lines')
    axes_lines[1].axis('off')
    
    plt.suptitle("Stereo Rectification Verification")
    plt.show()

    # Save rectified test images
    cv2.imwrite(os.path.join(OUTPUT_DIR, 'img_left_rectified.jpg'), img_left_rect)
    cv2.imwrite(os.path.join(OUTPUT_DIR, 'img_right_rectified.jpg'), img_right_rect)
    print(f"Rectified test images saved to {OUTPUT_DIR}/")

    print("\\nCalibration process completed.")

if __name__ == '__main__':
    main()