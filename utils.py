"""
Utility functions for diSPIM data loading, processing, and visualization.

This module provides functions for:
- Loading and parsing metadata from diSPIM acquisitions
- Loading OME-TIFF image stacks
- Discovering acquisition pairs
- Temporal and spatial alignment analysis
- Image visualization and camera overlay display
"""

import os
import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from datetime import datetime
import tifffile
import imageio
import warnings
warnings.filterwarnings('ignore')


# ============================================================================
# Metadata Parsing
# ============================================================================

def parse_metadata(metadata_path):
    """
    Parse the metadata JSON file and extract key acquisition parameters.
    
    Parameters:
    -----------
    metadata_path : str or Path
        Path to the metadata.txt file
        
    Returns:
    --------
    dict : Dictionary containing parsed metadata parameters
    """
    # Try different encodings to handle files that may not be UTF-8
    encodings = ['utf-8', 'latin-1', 'cp1252', 'iso-8859-1']
    metadata = None
    
    for encoding in encodings:
        try:
            with open(metadata_path, 'r', encoding=encoding, errors='replace') as f:
                metadata = json.load(f)
            break
        except (UnicodeDecodeError, json.JSONDecodeError):
            continue
    
    if metadata is None:
        # Last resort: try with errors='ignore' and latin-1 (can decode any byte)
        with open(metadata_path, 'r', encoding='latin-1', errors='ignore') as f:
            metadata = json.load(f)
    
    summary = metadata.get('Summary', {})
    
    # Parse the nested SPIMAcqSettings JSON string
    spim_settings_str = summary.get('SPIMAcqSettings', '{}')
    try:
        spim_settings = json.loads(spim_settings_str)
    except:
        spim_settings = {}
    
    # Extract key parameters
    parsed = {
        # Image dimensions
        'width': int(summary.get('Width', 0)),
        'height': int(summary.get('Height', 0)),
        'slices': int(summary.get('Slices', 0)),
        'channels': int(summary.get('Channels', 0)),
        'frames': int(summary.get('Frames', 1)),
        
        # Channel information
        'channel_names': summary.get('ChNames', []),
        'slices_first': summary.get('SlicesFirst', 'true').lower() == 'true',
        'time_first': summary.get('TimeFirst', 'false').lower() == 'true',
        
        # Spatial information
        'pixel_size_um': float(summary.get('PixelSize_um', 0)),
        'z_step_um': float(summary.get('z-step_um', 0)),
        'position_x': summary.get('Position_X', '0'),
        'position_y': summary.get('Position_Y', '0'),
        
        # Temporal information
        'start_time': summary.get('StartTime', ''),
        'slice_period_ms': float(summary.get('SlicePeriod_ms', '0 ms').split()[0]),
        'volume_duration_sec': float(summary.get('VolumeDuration', '0 s').split()[0]),
        
        # Acquisition settings
        'delay_before_side': spim_settings.get('delayBeforeSide', 0.25),
        'num_sides': spim_settings.get('numSides', 2),
        'first_side_is_a': spim_settings.get('firstSideIsA', True),
        
        # SPIM mode and camera settings
        'spim_mode': spim_settings.get('spimMode', ''),
        'camera_mode': spim_settings.get('cameraMode', ''),
        'acquire_both_cameras_simultaneously': spim_settings.get('acquireBothCamerasSimultaneously', False),
        
        # Detailed slice timing information
        'slice_timing': spim_settings.get('sliceTiming', {}),
        'scan_delay_ms': spim_settings.get('sliceTiming', {}).get('scanDelay', None),
        'scan_period_ms': spim_settings.get('sliceTiming', {}).get('scanPeriod', None),
        'laser_delay_ms': spim_settings.get('sliceTiming', {}).get('laserDelay', None),
        'laser_duration_ms': spim_settings.get('sliceTiming', {}).get('laserDuration', None),
        'camera_delay_ms': spim_settings.get('sliceTiming', {}).get('cameraDelay', None),
        'camera_exposure_ms': spim_settings.get('sliceTiming', {}).get('cameraExposure', None),
        'slice_duration_ms': spim_settings.get('sliceTiming', {}).get('sliceDuration', None),
        
        # Desired/requested acquisition parameters
        'desired_slice_period_ms': spim_settings.get('desiredSlicePeriod', None),
        'desired_light_exposure_ms': spim_settings.get('desiredLightExposure', None),
        'minimize_slice_period': spim_settings.get('minimizeSlicePeriod', False),
        
        # Duration information
        'duration_slice_ms': spim_settings.get('durationSliceMs', None),
        'duration_volume_ms': spim_settings.get('durationVolumeMs', None),
        'duration_total_sec': spim_settings.get('durationTotalSec', None),
        
        # Other useful info
        'acquisition_name': summary.get('AcquisitionName', ''),
        'date': summary.get('Date', ''),
        'pixel_type': summary.get('PixelType', ''),
        'bit_depth': int(summary.get('BitDepth', 16)),
        'mv_rotations': summary.get('MVRotations', ''),
        'spim_type': summary.get('SPIMtype', ''),
        'laser_exposure_ms': float(summary.get('LaserExposure_ms', 0)) if summary.get('LaserExposure_ms') else None,
        
        # Full metadata for reference
        'raw_summary': summary,
        'raw_spim_settings': spim_settings
    }
    
    return parsed


# ============================================================================
# Data Loading
# ============================================================================

def load_ome_tiff(tiff_path, metadata=None, channel_idx=None, max_slices=None):
    """
    Load an OME-TIFF file and return properly shaped array.
    
    Parameters:
    -----------
    tiff_path : str or Path
        Path to the .ome.tif file
    metadata : dict, optional
        Parsed metadata dictionary. If provided, uses it to determine data organization.
        If None, will try to infer from file.
    channel_idx : int or None
        If specified, return only this channel (0-indexed). If None, return all channels.
    max_slices : int or None
        If specified, load only the first max_slices slices (for memory efficiency)
        
    Returns:
    --------
    numpy.ndarray : Image data
        Shape depends on parameters:
        - If channel_idx specified: (slices, height, width)
        - If channel_idx is None: (slices, channels, height, width) or (channels, slices, height, width)
    """
    print(f"Loading OME-TIFF: {tiff_path}")
    
    # Load the TIFF file
    # tifffile can handle OME-TIFF format and preserve metadata
    with tifffile.TiffFile(tiff_path) as tif:
        # Get the image series (OME-TIFF can have multiple series)
        if len(tif.series) > 0:
            # Get the first (and usually only) series
            series = tif.series[0]
            data = series.asarray()
        else:
            # Fallback: read directly
            data = tifffile.imread(tiff_path)
    
    print(f"Raw data shape: {data.shape}")
    print(f"Data dtype: {data.dtype}")
    
    # Determine data organization from metadata or infer from shape
    if metadata is not None:
        slices_first = metadata.get('slices_first', True)
        num_slices = metadata.get('slices', 200)
        num_channels = metadata.get('channels', 2)
    else:
        # Try to infer: if we have 4D data, assume [slices, channels, height, width]
        # or [channels, slices, height, width]
        if len(data.shape) == 4:
            # Check which dimension is larger (slices vs channels)
            if data.shape[0] > data.shape[1]:
                slices_first = True
                num_slices, num_channels = data.shape[0], data.shape[1]
            else:
                slices_first = False
                num_channels, num_slices = data.shape[0], data.shape[1]
        else:
            # 3D or 2D - assume it's already in the right format
            slices_first = True
            num_slices = data.shape[0] if len(data.shape) >= 3 else 1
            num_channels = 1
    
    # Limit slices if requested (for memory efficiency)
    if max_slices is not None and num_slices > max_slices:
        if slices_first:
            data = data[:max_slices]
        else:
            data = data[:, :max_slices]
        num_slices = max_slices
        print(f"Limited to {max_slices} slices")
    
    # Reshape if needed
    if len(data.shape) == 4:
        if slices_first:
            # Data is [slices, channels, height, width] - this is what we want
            pass
        else:
            # Data is [channels, slices, height, width] - transpose
            data = np.transpose(data, (1, 0, 2, 3))
    elif len(data.shape) == 3:
        # 3D data - need to determine if it's [slices, height, width] or [channels, height, width]
        if slices_first:
            # Assume it's [slices, height, width] with 1 channel
            data = data[:, np.newaxis, :, :]
        else:
            # Assume it's [channels, height, width] with 1 slice
            data = data[np.newaxis, :, :, :]
            data = np.transpose(data, (1, 0, 2, 3))
    elif len(data.shape) == 2:
        # Single 2D image
        data = data[np.newaxis, np.newaxis, :, :]
    
    # Extract specific channel if requested
    if channel_idx is not None:
        if len(data.shape) == 4:
            data = data[:, channel_idx, :, :]
        else:
            print(f"Warning: Cannot extract channel {channel_idx} from data shape {data.shape}")
    
    print(f"Final data shape: {data.shape}")
    return data


def discover_acquisitions(root_dir='.'):
    """
    Discover all alpha/beta acquisition pairs in the directory structure.
    
    Parameters:
    -----------
    root_dir : str or Path
        Root directory to search for acquisitions
        
    Returns:
    --------
    list : List of dictionaries, each containing:
        - 'condition': Top-level folder name (e.g., '1msec_worm')
        - 'run': Second-level folder name (e.g., 'I')
        - 'alpha_path': Path to alpha folder
        - 'beta_path': Path to beta folder
        - 'alpha_metadata': Path to alpha metadata file
        - 'beta_metadata': Path to beta metadata file
        - 'alpha_tiff': Path to alpha OME-TIFF file
        - 'beta_tiff': Path to beta OME-TIFF file
    """
    root_path = Path(root_dir)
    acquisitions = []
    
    # Find all top-level directories (acquisition conditions)
    for condition_dir in sorted(root_path.iterdir()):
        if not condition_dir.is_dir():
            continue
        
        condition_name = condition_dir.name
        
        # Find all second-level directories (runs)
        for run_dir in sorted(condition_dir.iterdir()):
            if not run_dir.is_dir():
                continue
            
            run_name = run_dir.name
            
            # Look for alpha and beta folders
            alpha_folder = None
            beta_folder = None
            
            for subfolder in run_dir.iterdir():
                if not subfolder.is_dir():
                    continue
                
                folder_name = subfolder.name.lower()
                if 'alpha' in folder_name and 'worm' in folder_name:
                    alpha_folder = subfolder
                elif 'beta' in folder_name and 'worm' in folder_name:
                    beta_folder = subfolder
            
            # If we found both alpha and beta, create an acquisition entry
            if alpha_folder and beta_folder:
                # Find metadata and TIFF files
                alpha_metadata = None
                alpha_tiff = None
                beta_metadata = None
                beta_tiff = None
                
                # Look for metadata and TIFF files in alpha folder
                for file in alpha_folder.iterdir():
                    if file.suffix == '.txt' and 'metadata' in file.name:
                        alpha_metadata = file
                    elif file.suffix == '.tif' or file.suffix == '.tiff':
                        alpha_tiff = file
                
                # Look for metadata and TIFF files in beta folder
                for file in beta_folder.iterdir():
                    if file.suffix == '.txt' and 'metadata' in file.name:
                        beta_metadata = file
                    elif file.suffix == '.tif' or file.suffix == '.tiff':
                        beta_tiff = file
                
                # Only add if we have all required files
                if alpha_metadata and alpha_tiff and beta_metadata and beta_tiff:
                    acquisitions.append({
                        'condition': condition_name,
                        'run': run_name,
                        'alpha_path': alpha_folder,
                        'beta_path': beta_folder,
                        'alpha_metadata': alpha_metadata,
                        'beta_metadata': beta_metadata,
                        'alpha_tiff': alpha_tiff,
                        'beta_tiff': beta_tiff
                    })
    
    return acquisitions


# ============================================================================
# Temporal Alignment
# ============================================================================

def parse_start_time(time_str):
    """
    Parse the StartTime string from metadata into a datetime object.
    
    Format: "2025-11-12 17:04:10 -0500"
    """
    try:
        # Try parsing with timezone
        dt = datetime.strptime(time_str, "%Y-%m-%d %H:%M:%S %z")
        return dt
    except:
        try:
            # Try without timezone
            dt = datetime.strptime(time_str, "%Y-%m-%d %H:%M:%S")
            return dt
        except:
            print(f"Warning: Could not parse time string: {time_str}")
            return None


def calculate_temporal_alignment(alpha_meta, beta_meta):
    """
    Calculate temporal alignment between alpha and beta acquisitions.
    
    Parameters:
    -----------
    alpha_meta : dict
        Parsed metadata for alpha arm
    beta_meta : dict
        Parsed metadata for beta arm
        
    Returns:
    --------
    dict : Dictionary containing:
        - 'time_offset_sec': Time difference between start times (beta - alpha)
        - 'alpha_start': Alpha start datetime
        - 'beta_start': Beta start datetime
        - 'slice_period_ms': Average slice period
        - 'delay_before_side': Delay before beta side starts (seconds)
        - 'frame_times_alpha': Array of frame times for alpha (relative to alpha start)
        - 'frame_times_beta': Array of frame times for beta (relative to beta start)
        - 'num_slices': Number of slices
        - 'alpha_slice_duration_ms': Actual slice duration for alpha (if available)
        - 'beta_slice_duration_ms': Actual slice duration for beta (if available)
        - 'alpha_camera_offset_ms': Estimated timing offset between cameras in alpha arm (if sequential)
        - 'beta_camera_offset_ms': Estimated timing offset between cameras in beta arm (if sequential)
        - 'alpha_cameras_simultaneous': Whether alpha cameras acquire simultaneously
        - 'beta_cameras_simultaneous': Whether beta cameras acquire simultaneously
    """
    alpha_start = parse_start_time(alpha_meta['start_time'])
    beta_start = parse_start_time(beta_meta['start_time'])
    
    if alpha_start is None or beta_start is None:
        print("Warning: Could not parse start times. Using default alignment.")
        time_offset_sec = 0.0
    else:
        # Calculate time difference (beta - alpha) in seconds
        time_diff = beta_start - alpha_start
        time_offset_sec = time_diff.total_seconds()
    
    # Get slice periods
    alpha_slice_period = alpha_meta['slice_period_ms'] / 1000.0  # Convert to seconds
    beta_slice_period = beta_meta['slice_period_ms'] / 1000.0
    
    # Use average slice period
    avg_slice_period = (alpha_slice_period + beta_slice_period) / 2.0
    
    # Account for delay before side
    delay_before_side = alpha_meta.get('delay_before_side', 0.25)
    
    # Calculate frame times
    num_slices = min(alpha_meta['slices'], beta_meta['slices'])
    
    # Alpha frame times: starts immediately
    frame_times_alpha = np.arange(num_slices) * alpha_slice_period
    
    # Beta frame times: starts after delay_before_side
    frame_times_beta = np.arange(num_slices) * beta_slice_period + delay_before_side
    
    # Get detailed timing information if available
    alpha_slice_duration = alpha_meta.get('slice_duration_ms')
    beta_slice_duration = beta_meta.get('slice_duration_ms')
    
    # Calculate camera timing offsets if cameras are sequential
    alpha_camera_offset = None
    beta_camera_offset = None
    if not alpha_meta.get('acquire_both_cameras_simultaneously', False):
        # If cameras are sequential, estimate offset from camera delay/exposure
        alpha_timing = alpha_meta.get('slice_timing', {})
        if alpha_timing.get('cameraDelay') is not None and alpha_timing.get('cameraExposure') is not None:
            # Rough estimate: second camera starts after first camera finishes
            alpha_camera_offset = alpha_timing.get('cameraDelay', 0) + alpha_timing.get('cameraExposure', 0)
    
    if not beta_meta.get('acquire_both_cameras_simultaneously', False):
        beta_timing = beta_meta.get('slice_timing', {})
        if beta_timing.get('cameraDelay') is not None and beta_timing.get('cameraExposure') is not None:
            beta_camera_offset = beta_timing.get('cameraDelay', 0) + beta_timing.get('cameraExposure', 0)
    
    return {
        'time_offset_sec': time_offset_sec,
        'alpha_start': alpha_start,
        'beta_start': beta_start,
        'slice_period_ms': avg_slice_period * 1000,
        'delay_before_side': delay_before_side,
        'frame_times_alpha': frame_times_alpha,
        'frame_times_beta': frame_times_beta,
        'num_slices': num_slices,
        # Additional timing information
        'alpha_slice_duration_ms': alpha_slice_duration,
        'beta_slice_duration_ms': beta_slice_duration,
        'alpha_camera_offset_ms': alpha_camera_offset,
        'beta_camera_offset_ms': beta_camera_offset,
        'alpha_cameras_simultaneous': alpha_meta.get('acquire_both_cameras_simultaneously', False),
        'beta_cameras_simultaneous': beta_meta.get('acquire_both_cameras_simultaneously', False)
    }


# ============================================================================
# Spatial Information
# ============================================================================

def extract_spatial_info(alpha_meta, beta_meta):
    """
    Extract spatial calibration and position information.
    
    Parameters:
    -----------
    alpha_meta : dict
        Parsed metadata for alpha arm
    beta_meta : dict
        Parsed metadata for beta arm
        
    Returns:
    --------
    dict : Dictionary containing spatial information
    """
    def parse_position(pos_str):
        """Parse position string like '-0 μm' or '0.1 μm'"""
        try:
            # Extract number from string
            value = float(pos_str.split()[0])
            return value
        except:
            return 0.0
    
    spatial_info = {
        'pixel_size_um': {
            'alpha': alpha_meta['pixel_size_um'],
            'beta': beta_meta['pixel_size_um'],
            'average': (alpha_meta['pixel_size_um'] + beta_meta['pixel_size_um']) / 2.0
        },
        'z_step_um': {
            'alpha': alpha_meta['z_step_um'],
            'beta': beta_meta['z_step_um'],
            'average': (alpha_meta['z_step_um'] + beta_meta['z_step_um']) / 2.0
        },
        'position_x': {
            'alpha': parse_position(alpha_meta['position_x']),
            'beta': parse_position(beta_meta['position_x'])
        },
        'position_y': {
            'alpha': parse_position(alpha_meta['position_y']),
            'beta': parse_position(beta_meta['position_y'])
        },
        'mv_rotations': {
            'alpha': alpha_meta['mv_rotations'],
            'beta': beta_meta['mv_rotations']
        },
        'image_dimensions': {
            'width': alpha_meta['width'],
            'height': alpha_meta['height'],
            'slices': min(alpha_meta['slices'], beta_meta['slices'])
        }
    }
    
    # Calculate physical dimensions
    spatial_info['physical_dimensions_um'] = {
        'xy': {
            'width': spatial_info['image_dimensions']['width'] * spatial_info['pixel_size_um']['average'],
            'height': spatial_info['image_dimensions']['height'] * spatial_info['pixel_size_um']['average']
        },
        'z': {
            'depth': spatial_info['image_dimensions']['slices'] * spatial_info['z_step_um']['average']
        }
    }
    
    return spatial_info


# ============================================================================
# Image Display and Scaling
# ============================================================================

def scale_image_for_display(img, vmin=None, vmax=None):
    """
    Scale image for display without clipping data - preserves full dynamic range.
    Uses linear scaling to 0-1 range for matplotlib display.
    
    Parameters:
    -----------
    img : numpy.ndarray
        Input image (any dtype)
    vmin : float or None
        Minimum value for scaling (None = use image min)
    vmax : float or None
        Maximum value for scaling (None = use image max)
        
    Returns:
    --------
    tuple : (scaled_image, vmin_used, vmax_used)
        scaled_image is float64 in range [0, 1] for matplotlib
    """
    if vmin is None:
        vmin = float(img.min())
    if vmax is None:
        vmax = float(img.max())
    
    # Avoid division by zero
    if vmax == vmin:
        scaled = np.zeros_like(img, dtype=np.float64)
    else:
        scaled = (img.astype(np.float64) - vmin) / (vmax - vmin)
        scaled = np.clip(scaled, 0, 1)
    
    return scaled, vmin, vmax


def create_side_by_side_frame(alpha_slice, beta_slice, normalize=False, vmin_alpha=None, vmax_alpha=None, vmin_beta=None, vmax_beta=None):
    """
    Create a side-by-side frame from alpha and beta slices.
    Returns scaled images ready for matplotlib display (preserves full dynamic range).
    
    Parameters:
    -----------
    alpha_slice : numpy.ndarray
        Single slice from alpha arm (height, width)
    beta_slice : numpy.ndarray
        Single slice from beta arm (height, width)
    normalize : bool
        DEPRECATED - kept for compatibility but not used
    vmin_alpha, vmax_alpha : float or None
        Display range for alpha (None = use full range)
    vmin_beta, vmax_beta : float or None
        Display range for beta (None = use full range)
        
    Returns:
    --------
    tuple : (side_by_side_image, vmin_alpha, vmax_alpha, vmin_beta, vmax_beta)
        Image is float64 in [0,1] range for matplotlib imshow
    """
    # Scale images for display (preserves full dynamic range)
    alpha_scaled, vmin_a, vmax_a = scale_image_for_display(alpha_slice, vmin_alpha, vmax_alpha)
    beta_scaled, vmin_b, vmax_b = scale_image_for_display(beta_slice, vmin_beta, vmax_beta)
    
    # Create side-by-side image
    side_by_side = np.hstack([alpha_scaled, beta_scaled])
    
    return side_by_side, vmin_a, vmax_a, vmin_b, vmax_b


# ============================================================================
# Camera Overlay Visualization
# ============================================================================

def create_camera_overlay(cam1_img, cam2_img, cam1_name='', cam2_name=''):
    """
    Create a red/green overlay of two camera views (similar to MATLAB's imfuse).
    
    This function overlays two camera images from the same arm to visualize
    their spatial relationship. The first camera is shown in red, the second in green.
    Overlapping regions appear yellow.
    
    Parameters:
    -----------
    cam1_img : numpy.ndarray
        First camera image (height, width) - will be displayed in red channel
    cam2_img : numpy.ndarray
        Second camera image (height, width) - will be displayed in green channel
    cam1_name : str
        Name of first camera (for display)
    cam2_name : str
        Name of second camera (for display)
        
    Returns:
    --------
    numpy.ndarray : RGB image (height, width, 3) with values in [0, 1] range
        Red channel = cam1_img (normalized)
        Green channel = cam2_img (normalized)
        Blue channel = zeros
    """
    # Ensure images are the same size
    if cam1_img.shape != cam2_img.shape:
        raise ValueError(f"Camera images must have the same shape. Got {cam1_img.shape} and {cam2_img.shape}")
    
    # Normalize each camera image independently to [0, 1]
    cam1_scaled, _, _ = scale_image_for_display(cam1_img)
    cam2_scaled, _, _ = scale_image_for_display(cam2_img)
    
    # Create RGB overlay: red = cam1, green = cam2, blue = 0
    overlay = np.zeros((cam1_img.shape[0], cam1_img.shape[1], 3), dtype=np.float64)
    overlay[:, :, 0] = cam1_scaled  # Red channel
    overlay[:, :, 1] = cam2_scaled  # Green channel
    # Blue channel stays zero
    
    return overlay


def display_camera_overlays(alpha_data, beta_data, alpha_meta, beta_meta, 
                            slice_indices=None, num_samples=5, figsize=(20, 10)):
    """
    Display camera overlays for multiple slices showing alpha and beta arms side-by-side.
    
    For each slice, shows:
    - Alpha arm: Camera 0 (red) + Camera 1 (green) overlay
    - Beta arm: Camera 0 (red) + Camera 1 (green) overlay
    
    Parameters:
    -----------
    alpha_data : numpy.ndarray
        Alpha image stack (slices, channels, height, width)
    beta_data : numpy.ndarray
        Beta image stack (slices, channels, height, width)
    alpha_meta : dict
        Alpha metadata (must contain 'channel_names')
    beta_meta : dict
        Beta metadata (must contain 'channel_names')
    slice_indices : list of int or None
        Specific slice indices to display. If None, samples evenly across volume.
    num_samples : int
        Number of sample slices to display (if slice_indices is None)
    figsize : tuple
        Figure size (width, height) in inches
        
    Returns:
    --------
    matplotlib.figure.Figure : The created figure
    """
    # Get channel names
    alpha_channels = alpha_meta.get('channel_names', ['Camera 0', 'Camera 1'])
    beta_channels = beta_meta.get('channel_names', ['Camera 0', 'Camera 1'])
    
    # Determine which slices to display
    num_slices = min(alpha_data.shape[0], beta_data.shape[0])
    if slice_indices is None:
        slice_indices = np.linspace(0, num_slices - 1, num_samples, dtype=int)
    else:
        num_samples = len(slice_indices)
    
    # Create figure with subplots: one row per slice, two columns (alpha, beta)
    fig, axes = plt.subplots(num_samples, 2, figsize=figsize)
    if num_samples == 1:
        axes = axes.reshape(1, -1)
    
    for i, slice_idx in enumerate(slice_indices):
        # Extract camera images for this slice
        # Alpha: channels 0 and 1
        alpha_cam0 = alpha_data[slice_idx, 0, :, :]
        alpha_cam1 = alpha_data[slice_idx, 1, :, :]
        
        # Beta: channels 0 and 1
        beta_cam0 = beta_data[slice_idx, 0, :, :]
        beta_cam1 = beta_data[slice_idx, 1, :, :]
        
        # Create overlays
        alpha_overlay = create_camera_overlay(alpha_cam0, alpha_cam1, 
                                             alpha_channels[0], alpha_channels[1])
        beta_overlay = create_camera_overlay(beta_cam0, beta_cam1,
                                            beta_channels[0], beta_channels[1])
        
        # Display alpha overlay (left column)
        axes[i, 0].imshow(alpha_overlay, aspect='equal')
        axes[i, 0].axis('off')
        axes[i, 0].set_title(f'Alpha Arm - Slice {slice_idx}\n'
                            f'{alpha_channels[0]} (red) + {alpha_channels[1]} (green)',
                            fontsize=10)
        
        # Display beta overlay (right column)
        axes[i, 1].imshow(beta_overlay, aspect='equal')
        axes[i, 1].axis('off')
        axes[i, 1].set_title(f'Beta Arm - Slice {slice_idx}\n'
                             f'{beta_channels[0]} (red) + {beta_channels[1]} (green)',
                             fontsize=10)
    
    plt.suptitle('Camera Overlays: Red = Camera 0, Green = Camera 1, Yellow = Overlap',
                 fontsize=14, y=0.995)
    plt.tight_layout()
    
    return fig


# ============================================================================
# Video Creation
# ============================================================================

def create_video_from_stacks(alpha_data, beta_data, output_path, 
                             channel_alpha=0, channel_beta=0,
                             fps=10, max_slices=None, normalize=True):
    """
    Create a video file from alpha and beta image stacks.
    
    Parameters:
    -----------
    alpha_data : numpy.ndarray
        Alpha image stack (slices, channels, height, width) or (slices, height, width)
    beta_data : numpy.ndarray
        Beta image stack (slices, channels, height, width) or (slices, height, width)
    output_path : str or Path
        Output video file path
    channel_alpha : int
        Channel index to use for alpha (if multi-channel)
    channel_beta : int
        Channel index to use for beta (if multi-channel)
    fps : float
        Frames per second for output video
    max_slices : int or None
        Maximum number of slices to include (for testing)
    normalize : bool
        Whether to normalize intensities
        
    Returns:
    --------
    str : Path to created video file
    """
    # Extract channels if needed
    if len(alpha_data.shape) == 4:
        alpha_slices = alpha_data[:, channel_alpha, :, :]
    else:
        alpha_slices = alpha_data
    
    if len(beta_data.shape) == 4:
        beta_slices = beta_data[:, channel_beta, :, :]
    else:
        beta_slices = beta_data
    
    # Limit slices if requested
    num_slices = min(alpha_slices.shape[0], beta_slices.shape[0])
    if max_slices is not None:
        num_slices = min(num_slices, max_slices)
    
    alpha_slices = alpha_slices[:num_slices]
    beta_slices = beta_slices[:num_slices]
    
    print(f"Creating video with {num_slices} frames at {fps} fps...")
    
    # Create frames
    frames = []
    for i in range(num_slices):
        frame, _, _, _, _ = create_side_by_side_frame(alpha_slices[i], beta_slices[i], normalize=normalize)
        # Convert to uint8 for video (0-255 range)
        frame_uint8 = (frame * 255).astype(np.uint8)
        # Convert to RGB for video
        if len(frame_uint8.shape) == 2:
            frame_uint8 = np.stack([frame_uint8, frame_uint8, frame_uint8], axis=-1)
        frames.append(frame_uint8)
        
        if (i + 1) % 50 == 0:
            print(f"  Processed {i + 1}/{num_slices} frames...")
    
    # Save video
    print(f"Saving video to {output_path}...")
    imageio.mimwrite(str(output_path), frames, fps=fps, codec='libx264', quality=8)
    
    print(f"Video saved successfully!")
    return str(output_path)


