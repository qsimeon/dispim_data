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
from scipy import ndimage
import warnings
warnings.filterwarnings('ignore')

try:
    import SimpleITK as sitk
    HAS_SITK = True
except ImportError:
    HAS_SITK = False
    warnings.warn("SimpleITK not available. Registration functions will not work.")

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
                             verbose=True):
    """
    Register two camera stacks within the same arm using SimpleITK.
    
    This aligns the two cameras (e.g., HamCam2 and HamCam1 in Alpha arm)
    that view the same sample from different angles.
    
    Parameters:
    -----------
    cam0_stack : numpy.ndarray
        First camera stack (fixed image) - shape (Z, Y, X)
    cam1_stack : numpy.ndarray
        Second camera stack (moving image) - shape (Z, Y, X)
    transform_type : str
        Type of transformation: 'rigid' (translation + rotation) or 'affine'
    initial_transform : SimpleITK.Transform or None
        Optional initial transformation guess
    verbose : bool
        Print registration progress
        
    Returns:
    --------
    numpy.ndarray : Transformed cam1_stack aligned to cam0_stack
    SimpleITK.Transform : The transformation that was applied
    dict : Registration metrics and information
    """
    if not HAS_SITK:
        raise ImportError("SimpleITK is required for registration. Install with: pip install SimpleITK")
    
    # Convert numpy arrays to SimpleITK images
    # SimpleITK expects (X, Y, Z) ordering, so we need to transpose
    fixed_image = sitk.GetImageFromArray(cam0_stack.transpose(2, 1, 0))
    moving_image = sitk.GetImageFromArray(cam1_stack.transpose(2, 1, 0))
    
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
    
    # Execute registration
    if verbose:
        print("Starting registration...")
    
    final_transform = registration_method.Execute(fixed_image, moving_image)
    
    if verbose:
        print(f"Optimizer stop condition: {registration_method.GetOptimizerStopConditionDescription()}")
        print(f"Final metric value: {registration_method.GetMetricValue():.6f}")
    
    # Apply transformation
    resampler = sitk.ResampleImageFilter()
    resampler.SetReferenceImage(fixed_image)
    resampler.SetTransform(final_transform)
    resampler.SetInterpolator(sitk.sitkLinear)
    resampler.SetDefaultPixelValue(0)
    
    resampled_image = resampler.Execute(moving_image)
    
    # Convert back to numpy array and transpose back to (Z, Y, X)
    aligned_stack = sitk.GetArrayFromImage(resampled_image).transpose(2, 1, 0)
    
    # Collect metrics
    metrics = {
        'final_metric_value': registration_method.GetMetricValue(),
        'optimizer_iterations': registration_method.GetOptimizerIteration(),
        'optimizer_stop_condition': registration_method.GetOptimizerStopConditionDescription()
    }
    
    return aligned_stack, final_transform, metrics


def register_arms(alpha_fused, beta_fused,
                  transform_type='rigid',
                  initial_transform=None,
                  verbose=True):
    """
    Register Beta arm fused volume to Alpha arm fused volume.
    
    This performs fine alignment after rough alignment has been applied.
    
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
        Print registration progress
        
    Returns:
    --------
    numpy.ndarray : Transformed Beta volume aligned to Alpha
    SimpleITK.Transform : The transformation that was applied
    dict : Registration metrics
    """
    if not HAS_SITK:
        raise ImportError("SimpleITK is required for registration. Install with: pip install SimpleITK")
    
    # Convert to SimpleITK images (transpose to X, Y, Z)
    fixed_image = sitk.GetImageFromArray(alpha_fused.transpose(2, 1, 0))
    moving_image = sitk.GetImageFromArray(beta_fused.transpose(2, 1, 0))
    
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
    
    if verbose:
        print("Starting inter-arm registration...")
    
    final_transform = registration_method.Execute(fixed_image, moving_image)
    
    if verbose:
        print(f"Optimizer stop condition: {registration_method.GetOptimizerStopConditionDescription()}")
        print(f"Final metric value: {registration_method.GetMetricValue():.6f}")
    
    # Apply transformation
    resampler = sitk.ResampleImageFilter()
    resampler.SetReferenceImage(fixed_image)
    resampler.SetTransform(final_transform)
    resampler.SetInterpolator(sitk.sitkLinear)
    resampler.SetDefaultPixelValue(0)
    
    resampled_image = resampler.Execute(moving_image)
    
    # Convert back to numpy
    aligned_stack = sitk.GetArrayFromImage(resampled_image).transpose(2, 1, 0)
    
    metrics = {
        'final_metric_value': registration_method.GetMetricValue(),
        'optimizer_iterations': registration_method.GetOptimizerIteration(),
        'optimizer_stop_condition': registration_method.GetOptimizerStopConditionDescription()
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
    moving_image = sitk.GetImageFromArray(stack.transpose(2, 1, 0))
    
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

