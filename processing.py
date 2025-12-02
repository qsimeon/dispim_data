"""
Processing functions for double-diSPIM data transformation pipeline.

This module provides functions for:
- Deskewing 45° sheared slices to rectilinear coordinates
- Aligning cameras within each arm
- Rough alignment of Beta arm to Alpha arm coordinate system
- Fine registration using SimpleITK
- Volume fusion

The pipeline transforms raw double-diSPIM data into a single, isotropic 3D volume.
"""

import numpy as np

# NumPy 2.0 compatibility: Some libraries (xarray, dask, SimpleITK) may use deprecated np.unicode_
# Add compatibility shim immediately after importing numpy, before other imports
if not hasattr(np, 'unicode_'):
    np.unicode_ = np.str_

from scipy import ndimage
from pathlib import Path
import json
import warnings
warnings.filterwarnings('ignore')

try:
    import tifffile
    HAS_TIFFFILE = True
except ImportError:
    HAS_TIFFFILE = False
    warnings.warn("tifffile not available. Save/load functions will not work.")

try:
    import SimpleITK as sitk
    HAS_SITK = True
except (ImportError, AttributeError) as e:
    HAS_SITK = False
    warnings.warn(f"SimpleITK not available. Registration functions will not work. Error: {e}")

try:
    import dask.array as da
    import dask_image.ndfilters as dask_filters
    HAS_DASK = True
except ImportError:
    HAS_DASK = False
    warnings.warn("dask/dask-image not available. Will use scipy for processing.")


# ============================================================================
# Deskewing Functions
# ============================================================================

def calculate_deskew_matrix(pixel_size_um, z_step_um, angle_deg=45.0):
    """
    Calculate the affine transformation matrix for deskewing diSPIM data.
    
    The raw data is sheared because the light sheet is at an angle (typically 45°)
    to the imaging axis. This function calculates the transformation matrix needed
    to convert from sheared "stage-scanning coordinates" to rectilinear Cartesian coordinates.
    
    Parameters:
    -----------
    pixel_size_um : float
        Lateral pixel size in micrometers
    z_step_um : float
        Z-step spacing in micrometers (distance between slices)
    angle_deg : float
        Angle of the light sheet relative to vertical (default: 45°)
        
    Returns:
    --------
    numpy.ndarray : 4x4 affine transformation matrix
        Matrix to transform from sheared to rectilinear coordinates
    """
    angle_rad = np.deg2rad(angle_deg)
    
    # Calculate shear factor
    # The shear depends on the ratio of z-step to pixel size and the angle
    # For diSPIM with 45° angle: shear = (z_step / pixel_size) * tan(45°) = z_step / pixel_size
    shear_factor = (z_step_um / pixel_size_um) * np.tan(angle_rad)
    
    # Create affine transformation matrix
    # The shear is applied along the X-axis (assuming Z is axis 0, Y is axis 1, X is axis 2)
    # For a 3D volume with shape (Z, Y, X), we need to shear in the X direction as we go through Z
    # 
    # Transformation: [x']   [1  0  s  0] [x]
    #                  [y'] = [0  1  0  0] [y]
    #                  [z']   [0  0  1  0] [z]
    #                  [1 ]   [0  0  0  1] [1]
    #
    # Where s is the shear factor
    # But scipy.ndimage.affine_transform uses the inverse transformation matrix
    # and expects it in a specific format
    
    # For scipy.ndimage.affine_transform:
    # - The matrix is 3x3 for 3D arrays
    # - Array shape is (Z, Y, X), so matrix transforms (z, y, x) coordinates
    # - We need to unshear: x' = x - shear_factor * z
    # - Matrix format: [z', y', x'] = matrix @ [z, y, x]
    # - For unshearing: z' = z, y' = y, x' = x - s*z
    # - So matrix[2, 0] = -shear_factor (x component affected by z)
    matrix = np.eye(3)
    matrix[2, 0] = -shear_factor  # X coordinate affected by Z coordinate
    
    # Offset to account for the shear (may need adjustment based on output shape)
    offset = np.array([0.0, 0.0, 0.0])
    
    return matrix, offset


def deskew_stack(stack, pixel_size_um, z_step_um, angle_deg=45.0, 
                 use_dask=False, chunk_size=None):
    """
    Deskew a 3D image stack to remove the 45° shear from light sheet imaging.
    
    Parameters:
    -----------
    stack : numpy.ndarray or dask.array
        3D image stack with shape (slices, height, width) or (Z, Y, X)
    pixel_size_um : float
        Lateral pixel size in micrometers
    z_step_um : float
        Z-step spacing in micrometers
    angle_deg : float
        Light sheet angle in degrees (default: 45°)
    use_dask : bool
        If True and dask is available, use dask for processing (memory-efficient)
    chunk_size : tuple or None
        Chunk size for dask processing (if None, uses default)
        
    Returns:
    --------
    numpy.ndarray : Deskewed 3D stack
        Same dtype as input, but may have different shape due to shear correction
    dict : Transformation information
        Contains 'matrix', 'offset', 'output_shape', 'voxel_spacing'
    """
    if len(stack.shape) != 3:
        raise ValueError(f"Expected 3D stack, got shape {stack.shape}")
    
    # Calculate transformation matrix
    matrix, offset = calculate_deskew_matrix(pixel_size_um, z_step_um, angle_deg)
    
    # Determine output shape
    # The deskewed volume will be wider in X due to the shear correction
    z_size, y_size, x_size = stack.shape
    shear_factor = (z_step_um / pixel_size_um) * np.tan(np.deg2rad(angle_deg))
    output_x_size = int(np.ceil(x_size + abs(shear_factor) * z_size))
    output_shape = (z_size, y_size, output_x_size)
    
    # Choose processing method
    if use_dask and HAS_DASK:
        # Convert to dask array if not already
        if not isinstance(stack, da.Array):
            if chunk_size is None:
                # Default chunk size: process in slices
                chunk_size = (1, y_size, x_size)
            stack_dask = da.from_array(stack, chunks=chunk_size)
        else:
            stack_dask = stack
        
        # Apply transformation using dask
        # Note: dask-image doesn't have direct affine_transform, so we'll use scipy
        # For large datasets, process in chunks
        deskewed = da.map_blocks(
            lambda x: ndimage.affine_transform(
                x, matrix, offset=offset, output_shape=output_shape,
                order=1, mode='constant', cval=0.0, prefilter=False
            ),
            stack_dask,
            dtype=stack.dtype,
            chunks=chunk_size if chunk_size else stack_dask.chunks
        )
        
        # Compute result
        deskewed = deskewed.compute()
    else:
        # Use scipy directly
        deskewed = ndimage.affine_transform(
            stack,
            matrix,
            offset=offset,
            output_shape=output_shape,
            order=1,  # Linear interpolation
            mode='constant',
            cval=0.0,
            prefilter=False
        )
    
    # Calculate new voxel spacing
    # After deskewing, the effective pixel size changes
    # The Z spacing remains the same, but X spacing may need adjustment
    voxel_spacing = {
        'z_um': z_step_um,
        'y_um': pixel_size_um,
        'x_um': pixel_size_um  # X spacing remains the same after deskewing
    }
    
    transform_info = {
        'matrix': matrix,
        'offset': offset,
        'input_shape': stack.shape,
        'output_shape': deskewed.shape,
        'voxel_spacing': voxel_spacing,
        'shear_factor': shear_factor
    }
    
    return deskewed, transform_info


# ============================================================================
# Rough Alignment Functions
# ============================================================================

def flip_z_axis(stack):
    """
    Flip a 3D stack along the Z-axis (first dimension).
    
    Used to invert Beta arm data which views from below.
    
    Parameters:
    -----------
    stack : numpy.ndarray
        3D stack with shape (Z, Y, X)
        
    Returns:
    --------
    numpy.ndarray : Flipped stack
    """
    return np.flip(stack, axis=0)


def rotate_xy_90deg(stack, k=1):
    """
    Rotate a 3D stack 90 degrees in the XY plane (around Z-axis).
    
    Used to align Beta arm which is rotated 90° relative to Alpha arm.
    
    Parameters:
    -----------
    stack : numpy.ndarray
        3D stack with shape (Z, Y, X)
    k : int
        Number of 90° rotations (1 = 90°, 2 = 180°, 3 = 270°, -1 = -90°)
        
    Returns:
    --------
    numpy.ndarray : Rotated stack
    """
    # Rotate in YX plane (axes 1 and 2)
    return np.rot90(stack, k=k, axes=(1, 2))


def rough_align_beta_to_alpha(beta_stack):
    """
    Apply rough alignment transformations to Beta arm data to match Alpha coordinate system.
    
    Transformations:
    1. Flip Z-axis (Beta views from below)
    2. Rotate 90° in XY plane (Beta arm rotated 90° around Z-axis)
    
    Parameters:
    -----------
    beta_stack : numpy.ndarray
        3D Beta arm stack with shape (Z, Y, X)
        
    Returns:
    --------
    numpy.ndarray : Roughly aligned Beta stack
    dict : Transformation information
    """
    # Step 1: Flip Z-axis
    flipped = flip_z_axis(beta_stack)
    
    # Step 2: Rotate 90° in XY plane
    # Rotate clockwise 90° (k=1) to align Beta's YZ plane with Alpha's XZ plane
    aligned = rotate_xy_90deg(flipped, k=1)
    
    transform_info = {
        'z_flipped': True,
        'xy_rotated': True,
        'rotation_k': 1,
        'input_shape': beta_stack.shape,
        'output_shape': aligned.shape
    }
    
    return aligned, transform_info


# ============================================================================
# Registration Functions
# ============================================================================

def align_cameras_within_arm(cam0_stack, cam1_stack, 
                             transform_type='rigid',
                             initial_transform=None,
                             verbose=True,
                             flip_cam1_horizontal=True,
                             max_slices_for_registration=None,
                             use_gpu=False):
    """
    Register two camera stacks within the same arm using SimpleITK.
    
    This aligns the two cameras (e.g., HamCam2 and HamCam1 in Alpha arm)
    that view the same sample from different angles.
    
    IMPORTANT: Because the two objectives in a single arm face each other,
    their cameras capture "mirror images" of the sample in the lateral direction.
    By default, cam1_stack is horizontally flipped before registration to account for this.
    
    NOTE ON GPU: SimpleITK does not support GPU acceleration. If use_gpu=True,
    this function will attempt to use GPU-accelerated alternatives (if available),
    otherwise falls back to CPU-based SimpleITK.
    
    Parameters:
    -----------
    cam0_stack : numpy.ndarray
        First camera stack (fixed image) - shape (Z, Y, X)
    cam1_stack : numpy.ndarray
        Second camera stack (moving image) - shape (Z, Y, X)
        Will be horizontally flipped if flip_cam1_horizontal=True
    transform_type : str
        Type of transformation: 'rigid' (translation + rotation) or 'affine'
    initial_transform : SimpleITK.Transform or None
        Optional initial transformation guess
    verbose : bool
        Print registration progress with detailed logging
    flip_cam1_horizontal : bool
        If True, flip cam1_stack horizontally (along X-axis) before registration
        to account for mirror image relationship between cameras
    max_slices_for_registration : int or None
        If specified, use only the first N slices for registration (faster for testing).
        The transform will be applied to the full stack. Use None for full stack.
    use_gpu : bool
        If True, attempt to use GPU acceleration (currently not supported by SimpleITK,
        but included for future compatibility)
        
    Returns:
    --------
    numpy.ndarray : Transformed cam1_stack aligned to cam0_stack
    SimpleITK.Transform : The transformation that was applied
    dict : Registration metrics and information
    """
    if not HAS_SITK:
        raise ImportError("SimpleITK is required for registration. Install with: pip install SimpleITK")
    
    if use_gpu:
        warnings.warn("GPU acceleration is not currently supported by SimpleITK. Using CPU-based registration.")
    
    import time
    start_time = time.time()
    
    # Optionally use subset of slices for faster registration (for testing)
    original_shape = cam0_stack.shape
    if max_slices_for_registration is not None and max_slices_for_registration < cam0_stack.shape[0]:
        cam0_reg = cam0_stack[:max_slices_for_registration, :, :]
        cam1_reg = cam1_stack[:max_slices_for_registration, :, :]
        if verbose:
            print(f"  Using {max_slices_for_registration}/{cam0_stack.shape[0]} slices for registration (faster)")
    else:
        cam0_reg = cam0_stack
        cam1_reg = cam1_stack
    
    # Apply horizontal flip to cam1 if needed (cameras capture mirror images)
    if flip_cam1_horizontal:
        # Flip along X-axis (axis 2 in ZYX ordering)
        cam1_reg = np.flip(cam1_reg, axis=2)
        if verbose:
            print("  Applied horizontal flip to cam1 to account for mirror image relationship")
    
    # Convert numpy arrays to SimpleITK images
    # SimpleITK expects (X, Y, Z) ordering, so we need to transpose
    # Also, SimpleITK registration doesn't support uint16 directly, so convert to float32
    # Normalize to [0, 1] range for better numerical stability
    if verbose:
        print(f"  Preparing images for registration...")
        print(f"    Input shapes: cam0={cam0_reg.shape}, cam1={cam1_reg.shape}")
    
    cam0_float = cam0_reg.astype(np.float32)
    cam1_float = cam1_reg.astype(np.float32)
    
    # Normalize to [0, 1] range based on data type
    if cam0_reg.dtype == np.uint16:
        cam0_float = cam0_float / 65535.0
    elif cam0_reg.dtype == np.uint8:
        cam0_float = cam0_float / 255.0
    else:
        # For other types, normalize by max value
        cam0_max = cam0_float.max()
        if cam0_max > 0:
            cam0_float = cam0_float / cam0_max
    
    if cam1_reg.dtype == np.uint16:
        cam1_float = cam1_float / 65535.0
    elif cam1_reg.dtype == np.uint8:
        cam1_float = cam1_float / 255.0
    else:
        cam1_max = cam1_float.max()
        if cam1_max > 0:
            cam1_float = cam1_float / cam1_max
    
    fixed_image = sitk.GetImageFromArray(cam0_float.transpose(2, 1, 0))
    moving_image = sitk.GetImageFromArray(cam1_float.transpose(2, 1, 0))
    
    # Set spacing (assuming isotropic in XY, different in Z)
    # We'll use unit spacing for now - can be refined with actual pixel sizes
    fixed_image.SetSpacing([1.0, 1.0, 1.0])
    moving_image.SetSpacing([1.0, 1.0, 1.0])
    
    # Create transform
    if transform_type == 'rigid':
        transform = sitk.Euler3DTransform()
    elif transform_type == 'affine':
        transform = sitk.AffineTransform(3)
    else:
        raise ValueError(f"Unknown transform_type: {transform_type}")
    
    # Set initial transform if provided
    if initial_transform is not None:
        transform.SetParameters(initial_transform.GetParameters())
    
    # Registration method
    registration_method = sitk.ImageRegistrationMethod()
    
    # Similarity metric
    registration_method.SetMetricAsMattesMutualInformation(numberOfHistogramBins=50)
    
    # Optimizer
    if transform_type == 'rigid':
        registration_method.SetOptimizerAsGradientDescent(
            learningRate=1.0,
            numberOfIterations=200,
            convergenceMinimumValue=1e-6,
            convergenceWindowSize=10
        )
    else:
        registration_method.SetOptimizerAsGradientDescent(
            learningRate=0.5,
            numberOfIterations=200,
            convergenceMinimumValue=1e-6,
            convergenceWindowSize=10
        )
    
    # Multi-resolution framework for robustness
    registration_method.SetShrinkFactorsPerLevel(shrinkFactors=[4, 2, 1])
    registration_method.SetSmoothingSigmasPerLevel(smoothingSigmas=[2, 1, 0])
    registration_method.SmoothingSigmasAreSpecifiedInPhysicalUnitsOn()
    
    # Set interpolator
    registration_method.SetInterpolator(sitk.sitkLinear)
    
    # Add progress callback for verbose output
    if verbose:
        def command_iteration(method):
            """Callback function to report registration progress."""
            iteration = method.GetOptimizerIteration()
            metric_value = method.GetMetricValue()
            # Print every 10 iterations to avoid spam
            if iteration % 10 == 0 or iteration < 5:
                print(f"    Iteration {iteration:3d}: Metric = {metric_value:10.6f}")
        
        registration_method.AddCommand(sitk.sitkIterationEvent, 
                                      lambda: command_iteration(registration_method))
        print("  Starting registration...")
        print(f"    This may take several minutes for large volumes...")
    
    # Execute registration
    reg_start_time = time.time()
    final_transform = registration_method.Execute(fixed_image, moving_image)
    reg_time = time.time() - reg_start_time
    
    if verbose:
        print(f"  Registration completed in {reg_time:.1f} seconds")
        print(f"  Optimizer stop condition: {registration_method.GetOptimizerStopConditionDescription()}")
        print(f"  Final metric value: {registration_method.GetMetricValue():.6f}")
        print(f"  Total iterations: {registration_method.GetOptimizerIteration()}")
    
    # Apply transformation
    resampler = sitk.ResampleImageFilter()
    resampler.SetReferenceImage(fixed_image)
    resampler.SetTransform(final_transform)
    resampler.SetInterpolator(sitk.sitkLinear)
    resampler.SetDefaultPixelValue(0)
    
    if verbose:
        print("  Applying transformation to full stack...")
    
    resampled_image = resampler.Execute(moving_image)
    
    # Convert back to numpy array and transpose back to (Z, Y, X)
    aligned_reg = sitk.GetArrayFromImage(resampled_image).transpose(2, 1, 0)
    
    # If we used a subset for registration, apply transform to full stack
    if max_slices_for_registration is not None and max_slices_for_registration < original_shape[0]:
        if verbose:
            print(f"  Applying transform to full stack ({original_shape[0]} slices)...")
        
        # Prepare full cam1 stack (with flip if needed)
        if flip_cam1_horizontal:
            cam1_stack_full = np.flip(cam1_stack, axis=2)
        else:
            cam1_stack_full = cam1_stack
        
        # Convert full stack
        cam1_full_float = cam1_stack_full.astype(np.float32)
        if cam1_stack.dtype == np.uint16:
            cam1_full_float = cam1_full_float / 65535.0
        elif cam1_stack.dtype == np.uint8:
            cam1_full_float = cam1_full_float / 255.0
        else:
            cam1_full_max = cam1_full_float.max()
            if cam1_full_max > 0:
                cam1_full_float = cam1_full_float / cam1_full_max
        
        moving_image_full = sitk.GetImageFromArray(cam1_full_float.transpose(2, 1, 0))
        moving_image_full.SetSpacing([1.0, 1.0, 1.0])
        
        # Create reference image for full stack
        ref_shape_xyz = (original_shape[2], original_shape[1], original_shape[0])
        reference_image_full = sitk.Image(ref_shape_xyz, sitk.sitkFloat32)
        reference_image_full.SetSpacing([1.0, 1.0, 1.0])
        
        # Resample full stack
        resampler_full = sitk.ResampleImageFilter()
        resampler_full.SetReferenceImage(reference_image_full)
        resampler_full.SetTransform(final_transform)
        resampler_full.SetInterpolator(sitk.sitkLinear)
        resampler_full.SetDefaultPixelValue(0)
        
        resampled_full = resampler_full.Execute(moving_image_full)
        aligned_stack = sitk.GetArrayFromImage(resampled_full).transpose(2, 1, 0)
        
        # Convert back to original dtype
        if cam1_stack.dtype == np.uint16:
            aligned_stack = np.clip(aligned_stack * 65535.0, 0, 65535).astype(cam1_stack.dtype)
        elif cam1_stack.dtype == np.uint8:
            aligned_stack = np.clip(aligned_stack * 255.0, 0, 255).astype(cam1_stack.dtype)
        else:
            aligned_stack = aligned_stack.astype(cam1_stack.dtype)
    else:
        # Use the aligned result directly
        aligned_stack = aligned_reg
        # Convert back to original dtype
        if cam1_stack.dtype == np.uint16:
            aligned_stack = np.clip(aligned_stack * 65535.0, 0, 65535).astype(cam1_stack.dtype)
        elif cam1_stack.dtype == np.uint8:
            aligned_stack = np.clip(aligned_stack * 255.0, 0, 255).astype(cam1_stack.dtype)
        else:
            aligned_stack = aligned_stack.astype(cam1_stack.dtype)
    
    total_time = time.time() - start_time
    if verbose:
        print(f"  Total processing time: {total_time:.1f} seconds ({total_time/60:.1f} minutes)")
    
    # Collect metrics
    metrics = {
        'final_metric_value': registration_method.GetMetricValue(),
        'optimizer_iterations': registration_method.GetOptimizerIteration(),
        'optimizer_stop_condition': registration_method.GetOptimizerStopConditionDescription(),
        'registration_time_sec': reg_time,
        'total_time_sec': total_time
    }
    
    return aligned_stack, final_transform, metrics


def register_arms(alpha_fused, beta_fused,
                  transform_type='rigid',
                  initial_transform=None,
                  verbose=True,
                  use_gpu=False):
    """
    Register Beta arm fused volume to Alpha arm fused volume.
    
    This performs fine alignment after rough alignment has been applied.
    
    NOTE ON GPU: SimpleITK does not support GPU acceleration. If use_gpu=True,
    this function will attempt to use GPU-accelerated alternatives (if available),
    otherwise falls back to CPU-based SimpleITK.
    
    Parameters:
    -----------
    alpha_fused : numpy.ndarray
        Fused Alpha arm volume (fixed image) - shape (Z, Y, X)
    beta_fused : numpy.ndarray
        Fused Beta arm volume (moving image) - shape (Z, Y, X)
    transform_type : str
        'rigid' (translation + rotation) or 'affine'
    initial_transform : SimpleITK.Transform or None
        Optional initial transformation guess
    verbose : bool
        Print registration progress with detailed logging
    use_gpu : bool
        If True, attempt to use GPU acceleration (currently not supported by SimpleITK,
        but included for future compatibility)
        
    Returns:
    --------
    numpy.ndarray : Transformed Beta volume aligned to Alpha
    SimpleITK.Transform : The transformation that was applied
    dict : Registration metrics
    """
    if not HAS_SITK:
        raise ImportError("SimpleITK is required for registration. Install with: pip install SimpleITK")
    
    if use_gpu:
        warnings.warn("GPU acceleration is not currently supported by SimpleITK. Using CPU-based registration.")
    
    import time
    start_time = time.time()
    
    if verbose:
        print(f"  Preparing images for registration...")
        print(f"    Input shapes: alpha={alpha_fused.shape}, beta={beta_fused.shape}")
    
    # Convert to SimpleITK images (transpose to X, Y, Z)
    # SimpleITK registration doesn't support uint16 directly, so convert to float32
    # Normalize to [0, 1] range for better numerical stability
    alpha_float = alpha_fused.astype(np.float32)
    beta_float = beta_fused.astype(np.float32)
    
    # Normalize to [0, 1] range based on data type
    if alpha_fused.dtype == np.uint16:
        alpha_float = alpha_float / 65535.0
    elif alpha_fused.dtype == np.uint8:
        alpha_float = alpha_float / 255.0
    else:
        alpha_max = alpha_float.max()
        if alpha_max > 0:
            alpha_float = alpha_float / alpha_max
    
    if beta_fused.dtype == np.uint16:
        beta_float = beta_float / 65535.0
    elif beta_fused.dtype == np.uint8:
        beta_float = beta_float / 255.0
    else:
        beta_max = beta_float.max()
        if beta_max > 0:
            beta_float = beta_float / beta_max
    
    fixed_image = sitk.GetImageFromArray(alpha_float.transpose(2, 1, 0))
    moving_image = sitk.GetImageFromArray(beta_float.transpose(2, 1, 0))
    
    fixed_image.SetSpacing([1.0, 1.0, 1.0])
    moving_image.SetSpacing([1.0, 1.0, 1.0])
    
    # Create transform
    if transform_type == 'rigid':
        transform = sitk.Euler3DTransform()
    elif transform_type == 'affine':
        transform = sitk.AffineTransform(3)
    else:
        raise ValueError(f"Unknown transform_type: {transform_type}")
    
    if initial_transform is not None:
        transform.SetParameters(initial_transform.GetParameters())
    
    # Registration method
    registration_method = sitk.ImageRegistrationMethod()
    
    # Use normalized correlation for inter-arm registration (often works better than MI)
    registration_method.SetMetricAsCorrelation()
    
    # Optimizer
    registration_method.SetOptimizerAsGradientDescent(
        learningRate=1.0,
        numberOfIterations=300,
        convergenceMinimumValue=1e-6,
        convergenceWindowSize=10
    )
    
    # Multi-resolution
    registration_method.SetShrinkFactorsPerLevel(shrinkFactors=[8, 4, 2, 1])
    registration_method.SetSmoothingSigmasPerLevel(smoothingSigmas=[4, 2, 1, 0])
    registration_method.SmoothingSigmasAreSpecifiedInPhysicalUnitsOn()
    
    registration_method.SetInterpolator(sitk.sitkLinear)
    
    # Add progress callback for verbose output
    if verbose:
        def command_iteration(method):
            """Callback function to report registration progress."""
            iteration = method.GetOptimizerIteration()
            metric_value = method.GetMetricValue()
            # Print every 20 iterations to avoid spam (inter-arm registration has more iterations)
            if iteration % 20 == 0 or iteration < 5:
                print(f"    Iteration {iteration:3d}: Metric = {metric_value:10.6f}")
        
        registration_method.AddCommand(sitk.sitkIterationEvent, 
                                      lambda: command_iteration(registration_method))
        print("  Starting inter-arm registration...")
        print(f"    This may take several minutes for large volumes...")
    
    # Execute registration
    reg_start_time = time.time()
    final_transform = registration_method.Execute(fixed_image, moving_image)
    reg_time = time.time() - reg_start_time
    
    if verbose:
        print(f"  Registration completed in {reg_time:.1f} seconds ({reg_time/60:.1f} minutes)")
        print(f"  Optimizer stop condition: {registration_method.GetOptimizerStopConditionDescription()}")
        print(f"  Final metric value: {registration_method.GetMetricValue():.6f}")
        print(f"  Total iterations: {registration_method.GetOptimizerIteration()}")
    
    # Apply transformation
    resampler = sitk.ResampleImageFilter()
    resampler.SetReferenceImage(fixed_image)
    resampler.SetTransform(final_transform)
    resampler.SetInterpolator(sitk.sitkLinear)
    resampler.SetDefaultPixelValue(0)
    
    if verbose:
        print("  Applying transformation...")
    
    resampled_image = resampler.Execute(moving_image)
    
    # Convert back to numpy
    aligned_stack = sitk.GetArrayFromImage(resampled_image).transpose(2, 1, 0)
    
    total_time = time.time() - start_time
    if verbose:
        print(f"  Total processing time: {total_time:.1f} seconds ({total_time/60:.1f} minutes)")
    
    metrics = {
        'final_metric_value': registration_method.GetMetricValue(),
        'optimizer_iterations': registration_method.GetOptimizerIteration(),
        'optimizer_stop_condition': registration_method.GetOptimizerStopConditionDescription(),
        'registration_time_sec': reg_time,
        'total_time_sec': total_time
    }
    
    return aligned_stack, final_transform, metrics


def apply_transform(stack, transform, reference_shape=None, reference_spacing=None):
    """
    Apply a SimpleITK transformation to a numpy array stack.
    
    Parameters:
    -----------
    stack : numpy.ndarray
        3D stack with shape (Z, Y, X)
    transform : SimpleITK.Transform
        Transformation to apply
    reference_shape : tuple or None
        Output shape (Z, Y, X). If None, uses input shape
    reference_spacing : tuple or None
        Spacing for reference image. If None, uses [1, 1, 1]
        
    Returns:
    --------
    numpy.ndarray : Transformed stack
    """
    if not HAS_SITK:
        raise ImportError("SimpleITK is required. Install with: pip install SimpleITK")
    
    # Convert to SimpleITK image
    # Convert to float32 for SimpleITK compatibility
    original_dtype = stack.dtype
    stack_float = stack.astype(np.float32)
    
    # Normalize if needed (preserve relative intensities)
    stack_max = None
    if original_dtype == np.uint16:
        stack_float = stack_float / 65535.0
    elif original_dtype == np.uint8:
        stack_float = stack_float / 255.0
    else:
        stack_max = stack_float.max()
        if stack_max > 1.0:
            stack_float = stack_float / stack_max
    
    moving_image = sitk.GetImageFromArray(stack_float.transpose(2, 1, 0))
    
    if reference_spacing is None:
        reference_spacing = [1.0, 1.0, 1.0]
    moving_image.SetSpacing(reference_spacing)
    
    # Create reference image
    if reference_shape is None:
        reference_shape = stack.shape
    
    # Reference image in (X, Y, Z) order
    ref_shape_xyz = (reference_shape[2], reference_shape[1], reference_shape[0])
    reference_image = sitk.Image(ref_shape_xyz, sitk.sitkFloat32)
    reference_image.SetSpacing(reference_spacing)
    
    # Resample
    resampler = sitk.ResampleImageFilter()
    resampler.SetReferenceImage(reference_image)
    resampler.SetTransform(transform)
    resampler.SetInterpolator(sitk.sitkLinear)
    resampler.SetDefaultPixelValue(0)
    
    resampled_image = resampler.Execute(moving_image)
    
    # Convert back to numpy (Z, Y, X)
    transformed_stack = sitk.GetArrayFromImage(resampled_image).transpose(2, 1, 0)
    
    # Convert back to original dtype
    if original_dtype == np.uint16:
        transformed_stack = np.clip(transformed_stack * 65535.0, 0, 65535).astype(original_dtype)
    elif original_dtype == np.uint8:
        transformed_stack = np.clip(transformed_stack * 255.0, 0, 255).astype(original_dtype)
    else:
        # For other types, scale back if we normalized
        if stack_max is not None and stack_max > 1.0:
            transformed_stack = transformed_stack * stack_max
        transformed_stack = transformed_stack.astype(original_dtype)
    
    return transformed_stack


# ============================================================================
# Fusion Functions
# ============================================================================

def fuse_volumes(*stacks, method='weighted_average', weights=None):
    """
    Fuse multiple registered 3D volumes into a single volume.
    
    Parameters:
    -----------
    *stacks : numpy.ndarray
        Multiple 3D stacks to fuse, all with same shape (Z, Y, X)
    method : str
        Fusion method: 'weighted_average', 'max', 'min', 'median'
    weights : list or None
        Weights for each stack (for weighted_average). If None, equal weights
        
    Returns:
    --------
    numpy.ndarray : Fused volume
    """
    if len(stacks) == 0:
        raise ValueError("At least one stack required")
    
    # Check all stacks have same shape
    shapes = [s.shape for s in stacks]
    if not all(s == shapes[0] for s in shapes):
        raise ValueError(f"All stacks must have the same shape. Got: {shapes}")
    
    # Stack arrays along new axis
    stacked = np.stack(stacks, axis=0)
    
    if method == 'weighted_average':
        if weights is None:
            weights = np.ones(len(stacks)) / len(stacks)
        else:
            weights = np.array(weights)
            weights = weights / weights.sum()  # Normalize
        
        # Weighted average
        fused = np.average(stacked, axis=0, weights=weights)
        
    elif method == 'max':
        fused = np.max(stacked, axis=0)
        
    elif method == 'min':
        fused = np.min(stacked, axis=0)
        
    elif method == 'median':
        fused = np.median(stacked, axis=0)
        
    else:
        raise ValueError(f"Unknown fusion method: {method}")
    
    return fused.astype(stacks[0].dtype)


# ============================================================================
# Visualization Helper Functions
# ============================================================================

def compute_mip(stack, axis=0):
    """
    Compute Maximum Intensity Projection (MIP) along specified axis.
    
    Parameters:
    -----------
    stack : numpy.ndarray
        3D stack with shape (Z, Y, X)
    axis : int
        Axis along which to project (0=Z, 1=Y, 2=X)
        
    Returns:
    --------
    numpy.ndarray : 2D MIP image
    """
    return np.max(stack, axis=axis)


def compute_mip_xyz(stack):
    """
    Compute MIPs along all three axes.
    
    Parameters:
    -----------
    stack : numpy.ndarray
        3D stack with shape (Z, Y, X)
        
    Returns:
    --------
    dict : Dictionary with keys 'xy', 'xz', 'yz' containing 2D MIPs
    """
    return {
        'xy': compute_mip(stack, axis=0),  # Project along Z -> XY view
        'xz': compute_mip(stack, axis=1),  # Project along Y -> XZ view
        'yz': compute_mip(stack, axis=2)    # Project along X -> YZ view
    }


# ============================================================================
# Save/Load Functions for Intermediate Results
# ============================================================================

def save_deskewed_stack(stack, output_path, transform_info=None, metadata=None):
    """
    Save a deskewed stack to disk with associated metadata.
    
    Parameters:
    -----------
    stack : numpy.ndarray
        3D deskewed stack with shape (Z, Y, X)
    output_path : str or Path
        Path where to save the stack (will save as .tif)
    transform_info : dict or None
        Dictionary containing transformation information (from deskew_stack)
    metadata : dict or None
        Additional metadata to save as JSON alongside the stack
        
    Returns:
    --------
    Path : Path to saved file
    Path : Path to metadata JSON file (if metadata provided)
    """
    if not HAS_TIFFFILE:
        raise ImportError("tifffile is required for saving. Install with: pip install tifffile")
    
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Save the stack as TIFF
    tifffile.imwrite(
        str(output_path),
        stack,
        photometric='minisblack',
        metadata={'axes': 'ZYX'}
    )
    
    # Save metadata as JSON if provided
    metadata_path = None
    if transform_info is not None or metadata is not None:
        metadata_path = output_path.with_suffix('.json')
        save_metadata = {}
        if transform_info is not None:
            # Convert numpy arrays to lists for JSON serialization
            transform_info_json = {}
            for key, value in transform_info.items():
                if isinstance(value, np.ndarray):
                    transform_info_json[key] = value.tolist()
                elif isinstance(value, (np.integer, np.floating)):
                    transform_info_json[key] = float(value)
                else:
                    transform_info_json[key] = value
            save_metadata['transform_info'] = transform_info_json
        
        if metadata is not None:
            save_metadata['metadata'] = metadata
        
        with open(metadata_path, 'w') as f:
            json.dump(save_metadata, f, indent=2)
    
    return output_path, metadata_path


def load_deskewed_stack(stack_path, load_metadata=True):
    """
    Load a deskewed stack from disk.
    
    Parameters:
    -----------
    stack_path : str or Path
        Path to the saved stack (.tif file)
    load_metadata : bool
        If True, also load associated metadata JSON file
        
    Returns:
    --------
    numpy.ndarray : The loaded stack
    dict or None : Metadata dictionary if load_metadata=True and file exists
    """
    if not HAS_TIFFFILE:
        raise ImportError("tifffile is required for loading. Install with: pip install tifffile")
    
    stack_path = Path(stack_path)
    
    # Load the stack
    stack = tifffile.imread(str(stack_path))
    
    # Load metadata if requested
    metadata = None
    if load_metadata:
        metadata_path = stack_path.with_suffix('.json')
        if metadata_path.exists():
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
    
    return stack, metadata


def get_deskewed_paths(base_output_dir, acquisition_name, camera_name, arm_name):
    """
    Generate standardized file paths for deskewed stacks.
    
    Parameters:
    -----------
    base_output_dir : str or Path
        Base directory for saving processed results
    acquisition_name : str
        Name of the acquisition (e.g., "Worm1_starved_adult_SWF1188_I")
    camera_name : str
        Name of the camera (e.g., "HamCam2", "HamCam1")
    arm_name : str
        Name of the arm ("alpha" or "beta")
        
    Returns:
    --------
    Path : Path to the deskewed stack file
    Path : Path to the metadata JSON file
    """
    base_output_dir = Path(base_output_dir)
    output_dir = base_output_dir / acquisition_name / 'deskewed'
    
    # Create filename: arm_camera_deskewed.tif
    filename = f"{arm_name}_{camera_name}_deskewed.tif"
    stack_path = output_dir / filename
    metadata_path = stack_path.with_suffix('.json')
    
    return stack_path, metadata_path


def check_deskewed_exists(base_output_dir, acquisition_name, camera_name, arm_name):
    """
    Check if a deskewed stack already exists on disk.
    
    Parameters:
    -----------
    base_output_dir : str or Path
        Base directory for processed results
    acquisition_name : str
        Name of the acquisition
    camera_name : str
        Name of the camera
    arm_name : str
        Name of the arm ("alpha" or "beta")
        
    Returns:
    --------
    bool : True if deskewed stack exists
    Path : Path to the stack file (or None if doesn't exist)
    Path : Path to metadata file (or None if doesn't exist)
    """
    stack_path, metadata_path = get_deskewed_paths(
        base_output_dir, acquisition_name, camera_name, arm_name
    )
    
    exists = stack_path.exists()
    return exists, stack_path if exists else None, metadata_path if exists and metadata_path.exists() else None


# ============================================================================
# Generic Save/Load Functions for All Intermediate Results
# ============================================================================

def save_processed_stack(stack, output_path, metadata=None):
    """
    Generic function to save any processed 3D stack.
    
    Parameters:
    -----------
    stack : numpy.ndarray
        3D stack with shape (Z, Y, X)
    output_path : str or Path
        Path where to save the stack (.tif)
    metadata : dict or None
        Additional metadata to save as JSON
        
    Returns:
    --------
    Path : Path to saved file
    Path or None : Path to metadata JSON file (if metadata provided)
    """
    if not HAS_TIFFFILE:
        raise ImportError("tifffile is required for saving. Install with: pip install tifffile")
    
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Save the stack as TIFF
    tifffile.imwrite(
        str(output_path),
        stack,
        photometric='minisblack',
        metadata={'axes': 'ZYX'}
    )
    
    # Save metadata as JSON if provided
    metadata_path = None
    if metadata is not None:
        metadata_path = output_path.with_suffix('.json')
        # Convert numpy types to Python types for JSON serialization
        metadata_json = {}
        for key, value in metadata.items():
            if isinstance(value, np.ndarray):
                metadata_json[key] = value.tolist()
            elif isinstance(value, (np.integer, np.floating)):
                metadata_json[key] = float(value) if isinstance(value, np.floating) else int(value)
            elif isinstance(value, (list, tuple)):
                metadata_json[key] = [
                    float(v) if isinstance(v, np.floating) else int(v) if isinstance(v, np.integer) else v
                    for v in value
                ]
            else:
                metadata_json[key] = value
        
        with open(metadata_path, 'w') as f:
            json.dump(metadata_json, f, indent=2)
    
    return output_path, metadata_path


def load_processed_stack(stack_path, load_metadata=True):
    """
    Generic function to load any processed 3D stack.
    
    Parameters:
    -----------
    stack_path : str or Path
        Path to the saved stack (.tif file)
    load_metadata : bool
        If True, also load associated metadata JSON file
        
    Returns:
    --------
    numpy.ndarray : The loaded stack
    dict or None : Metadata dictionary if load_metadata=True and file exists
    """
    if not HAS_TIFFFILE:
        raise ImportError("tifffile is required for loading. Install with: pip install tifffile")
    
    stack_path = Path(stack_path)
    
    # Load the stack
    stack = tifffile.imread(str(stack_path))
    
    # Load metadata if requested
    metadata = None
    if load_metadata:
        metadata_path = stack_path.with_suffix('.json')
        if metadata_path.exists():
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
    
    return stack, metadata


def get_processed_paths(base_output_dir, acquisition_name, stage_name, filename_base):
    """
    Generate standardized file paths for processed stacks.
    
    Parameters:
    -----------
    base_output_dir : str or Path
        Base directory for saving processed results
    acquisition_name : str
        Name of the acquisition
    stage_name : str
        Processing stage name (e.g., 'aligned', 'registered', 'fused')
    filename_base : str
        Base filename (e.g., 'alpha_cam1_aligned')
        
    Returns:
    --------
    Path : Path to the stack file
    Path : Path to the metadata JSON file
    """
    base_output_dir = Path(base_output_dir)
    output_dir = base_output_dir / acquisition_name / stage_name
    
    # Create filename
    filename = f"{filename_base}.tif"
    stack_path = output_dir / filename
    metadata_path = stack_path.with_suffix('.json')
    
    return stack_path, metadata_path


def check_processed_exists(base_output_dir, acquisition_name, stage_name, filename_base):
    """
    Check if a processed stack already exists on disk.
    
    Parameters:
    -----------
    base_output_dir : str or Path
        Base directory for processed results
    acquisition_name : str
        Name of the acquisition
    stage_name : str
        Processing stage name
    filename_base : str
        Base filename
        
    Returns:
    --------
    bool : True if processed stack exists
    Path : Path to the stack file (or None if doesn't exist)
    Path : Path to metadata file (or None if doesn't exist)
    """
    stack_path, metadata_path = get_processed_paths(
        base_output_dir, acquisition_name, stage_name, filename_base
    )
    
    exists = stack_path.exists()
    return exists, stack_path if exists else None, metadata_path if exists and metadata_path.exists() else None


# Convenience functions for specific processing stages

def get_aligned_camera_paths(base_output_dir, acquisition_name, camera_name, arm_name):
    """Get paths for intra-arm aligned camera stacks."""
    filename_base = f"{arm_name}_{camera_name}_aligned"
    return get_processed_paths(base_output_dir, acquisition_name, 'aligned', filename_base)


def get_rough_aligned_paths(base_output_dir, acquisition_name, camera_name):
    """Get paths for rough-aligned beta camera stacks."""
    filename_base = f"beta_{camera_name}_rough_aligned"
    return get_processed_paths(base_output_dir, acquisition_name, 'rough_aligned', filename_base)


def get_registered_paths(base_output_dir, acquisition_name, camera_name):
    """Get paths for fine-registered beta camera stacks."""
    filename_base = f"beta_{camera_name}_registered"
    return get_processed_paths(base_output_dir, acquisition_name, 'registered', filename_base)


def get_fused_paths(base_output_dir, acquisition_name):
    """Get paths for fused volume."""
    filename_base = "fused_volume"
    return get_processed_paths(base_output_dir, acquisition_name, 'fused', filename_base)

