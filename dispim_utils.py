"""
Consolidated utility functions for double-diSPIM data processing pipeline.

This module provides functions for:
- Loading and parsing metadata from diSPIM acquisitions
- Loading OME-TIFF image stacks with temporal alignment
- Deskewing 45° sheared slices to rectilinear coordinates
- Saving and loading processed volumes
- Image visualization and camera overlay display

The pipeline processes raw double-diSPIM data through deskewing and prepares
volumes for interactive 3D alignment.
"""

import os
import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from datetime import datetime
import tifffile
import imageio
from scipy import ndimage
from scipy.interpolate import interp1d
import warnings
warnings.filterwarnings('ignore')

# NumPy 2.0 compatibility: Some libraries (xarray, dask, SimpleITK) may use deprecated np.unicode_
# Add compatibility shim immediately after importing numpy, before other imports
if not hasattr(np, 'unicode_'):
    np.unicode_ = np.str_

# Optional dependencies
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

try:
    import cv2
    HAS_OPENCV = True
except ImportError:
    HAS_OPENCV = False
    warnings.warn("OpenCV (cv2) not available. CLAHE preprocessing will not work.")

HAS_TIFFFILE = True  # tifffile is imported above


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
    except Exception:
        # If parsing fails, try to load AcqSettings.txt from the same parent directory
        import os
        spim_settings = {}
        try:
            metadata_dir = os.path.dirname(os.path.abspath(metadata_path))
            acq_settings_path = os.path.join(metadata_dir, "AcqSettings.txt")
            with open(acq_settings_path, 'r', encoding='utf-8', errors='replace') as f:
                acq_settings_str = f.read()
                spim_settings = json.loads(acq_settings_str)
        except Exception:
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

def apply_clahe_to_stack(stack, clip_limit=2.0, tile_grid_size=(8, 8)):
    """
    Apply Contrast Limited Adaptive Histogram Equalization (CLAHE) to a 3D image stack.
    
    This function applies CLAHE to each 2D frame in the stack independently, which is
    appropriate for camera-wise preprocessing where each frame should be enhanced
    based on its own local contrast.
    
    Parameters:
    -----------
    stack : numpy.ndarray
        3D stack with shape (slices, height, width) or 4D with (slices, channels, height, width)
        If 4D, CLAHE is applied per channel independently
    clip_limit : float
        CLAHE clip limit (default: 2.0, same as OpenCV default)
    tile_grid_size : tuple of int
        CLAHE tile grid size (default: (8, 8), same as OpenCV default)
        
    Returns:
    --------
    numpy.ndarray : CLAHE-enhanced stack with same shape and dtype as input
    """
    if not HAS_OPENCV:
        raise ImportError("OpenCV (cv2) is required for CLAHE preprocessing. Install with: pip install opencv-python")
    
    if len(stack.shape) not in [3, 4]:
        raise ValueError(f"Expected 3D or 4D stack, got shape {stack.shape}")
    
    # Create CLAHE object once (reused for all frames)
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
    
    # Handle 3D case: (slices, height, width)
    if len(stack.shape) == 3:
        num_slices, height, width = stack.shape
        enhanced = np.zeros_like(stack)
        
        # Vectorized approach: process all slices
        # CLAHE requires uint8 input, so we need to normalize first
        # Get the data range for proper normalization
        dtype = stack.dtype
        if dtype == np.uint16:
            # Normalize uint16 to uint8 range
            stack_min = stack.min()
            stack_max = stack.max()
            if stack_max > stack_min:
                # Normalize to 0-255 range
                stack_normalized = ((stack.astype(np.float32) - stack_min) / 
                                  (stack_max - stack_min) * 255).astype(np.uint8)
            else:
                stack_normalized = np.zeros_like(stack, dtype=np.uint8)
            
            # Apply CLAHE to each slice
            for i in range(num_slices):
                enhanced_uint8 = clahe.apply(stack_normalized[i])
                # Convert back to uint16 range
                enhanced[i] = ((enhanced_uint8.astype(np.float32) / 255.0) * 
                              (stack_max - stack_min) + stack_min).astype(dtype)
        elif dtype == np.uint8:
            # Direct application for uint8
            for i in range(num_slices):
                enhanced[i] = clahe.apply(stack[i])
        else:
            # For other dtypes, normalize to uint8, apply CLAHE, then convert back
            stack_min = stack.min()
            stack_max = stack.max()
            if stack_max > stack_min:
                stack_normalized = ((stack.astype(np.float32) - stack_min) / 
                                  (stack_max - stack_min) * 255).astype(np.uint8)
                for i in range(num_slices):
                    enhanced_uint8 = clahe.apply(stack_normalized[i])
                    enhanced[i] = ((enhanced_uint8.astype(np.float32) / 255.0) * 
                                  (stack_max - stack_min) + stack_min).astype(dtype)
            else:
                enhanced = stack.copy()
        
        return enhanced
    
    # Handle 4D case: (slices, channels, height, width)
    # Apply CLAHE independently to each channel
    elif len(stack.shape) == 4:
        num_slices, num_channels, height, width = stack.shape
        enhanced = np.zeros_like(stack)
        
        dtype = stack.dtype
        
        # Process each channel independently
        for c in range(num_channels):
            channel_stack = stack[:, c, :, :]  # Extract channel: (slices, height, width)
            enhanced_channel = apply_clahe_to_stack(channel_stack, clip_limit, tile_grid_size)
            enhanced[:, c, :, :] = enhanced_channel
        
        return enhanced
    
    return stack


def load_ome_tiff(tiff_path, metadata=None, channel_idx=None, max_slices=None, apply_clahe=False):
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
    apply_clahe : bool
        If True, apply Contrast Limited Adaptive Histogram Equalization (CLAHE) to each
        channel independently. This enhances local contrast and can improve image quality
        for downstream processing. Default: False
        
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
    
    # Apply CLAHE preprocessing if requested
    if apply_clahe:
        if not HAS_OPENCV:
            warnings.warn("OpenCV not available. Skipping CLAHE preprocessing.")
        else:
            print("Applying CLAHE preprocessing to each channel...")
            # Apply CLAHE before extracting channel (so it processes all channels)
            if len(data.shape) == 4:
                # Apply CLAHE per channel independently
                data = apply_clahe_to_stack(data)
            elif len(data.shape) == 3:
                # Single channel, apply CLAHE directly
                data = apply_clahe_to_stack(data)
            print("CLAHE preprocessing complete.")
    
    # Extract specific channel if requested (after CLAHE if applied)
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
                
                # Only proceed if we have all required files
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
    Calculate temporal alignment between alpha and beta acquisitions using actual start times.
    
    This function handles two acquisition modes:
    1. **Sequential volumes** (`acquire_both_cameras_simultaneously=False`):
       - Path A (Channel 0) acquires full volume first (all N slices)
       - Path B (Channel 1) starts after Path A completes + delay_before_side
       - Total duration = 2*N*slice_period + delay_before_side
    2. **Simultaneous cameras** (`acquire_both_cameras_simultaneously=True`):
       - Path A and Path B alternate per slice (slice A, slice B, slice A, slice B...)
       - Small timing offset between cameras per slice
    
    The function determines which arm started first and which finished first, then calculates
    how many slices need to be discarded from:
    1. The beginning of the early-starting arm (before the other arm started)
    2. The end of the late-finishing arm (after the other arm finished)
    
    This ensures we only keep slices from the overlapping acquisition period.
    
    Parameters:
    -----------
    alpha_meta : dict
        Parsed metadata for alpha arm
    beta_meta : dict
        Parsed metadata for beta arm
        
    Returns:
    --------
    dict : Dictionary containing:
        - 'time_offset_sec': Time difference between start times (beta - alpha) in seconds
        - 'alpha_start': Alpha start datetime
        - 'beta_start': Beta start datetime
        - 'alpha_started_first': Boolean indicating if alpha started before beta
        - 'beta_started_first': Boolean indicating if beta started before alpha
        - 'alpha_finished_first': Boolean indicating if alpha finished before beta
        - 'beta_finished_first': Boolean indicating if beta finished before alpha
        - 'slices_to_discard_alpha_start': Number of slices to discard from start of alpha
        - 'slices_to_discard_alpha_end': Number of slices to discard from end of alpha
        - 'slices_to_discard_beta_start': Number of slices to discard from start of beta
        - 'slices_to_discard_beta_end': Number of slices to discard from end of beta
        - 'alpha_slice_period_sec': Alpha slice period in seconds
        - 'beta_slice_period_sec': Beta slice period in seconds
        - 'slice_period_ms': Average slice period in milliseconds
        - 'num_slices': Number of slices (after discarding)
        - 'alpha_slice_duration_ms': Actual slice duration for alpha (if available)
        - 'beta_slice_duration_ms': Actual slice duration for beta (if available)
        - 'alpha_cam0_offset_ms': Timing offset for alpha cam0 (Path A) in ms (always 0)
        - 'alpha_cam1_offset_ms': Timing offset for alpha cam1 (Path B) in ms
        - 'beta_cam0_offset_ms': Timing offset for beta cam0 (Path A) in ms (always 0)
        - 'beta_cam1_offset_ms': Timing offset for beta cam1 (Path B) in ms
        - 'alpha_cameras_simultaneous': Whether alpha cameras acquire simultaneously
        - 'beta_cameras_simultaneous': Whether beta cameras acquire simultaneously
    """
    alpha_start = parse_start_time(alpha_meta['start_time'])
    beta_start = parse_start_time(beta_meta['start_time'])
    
    # Get slice periods
    alpha_slice_period = alpha_meta['slice_period_ms'] / 1000.0  # Convert to seconds
    beta_slice_period = beta_meta['slice_period_ms'] / 1000.0
    
    # Use average slice period
    avg_slice_period = (alpha_slice_period + beta_slice_period) / 2.0
    
    # Get total number of slices for each arm
    alpha_total_slices = alpha_meta['slices']
    beta_total_slices = beta_meta['slices']
    
    # Check acquisition mode for each arm
    alpha_simultaneous = alpha_meta.get('acquire_both_cameras_simultaneously', False)
    beta_simultaneous = beta_meta.get('acquire_both_cameras_simultaneously', False)
    
    # Get delay_before_side (in seconds) for each arm
    alpha_delay_before_side = alpha_meta.get('delay_before_side', 0.25)  # Default 0.25 seconds
    beta_delay_before_side = beta_meta.get('delay_before_side', 0.25)
    
    # Calculate total acquisition duration for each arm
    # For sequential mode: Path A completes, then Path B starts after delay
    # Total duration = Path A volume + delay + Path B volume
    if not alpha_simultaneous:
        # Sequential: Path A (N slices) + delay + Path B (N slices)
        alpha_duration = (alpha_total_slices * alpha_slice_period + 
                         alpha_delay_before_side + 
                         alpha_total_slices * alpha_slice_period)
    else:
        # Simultaneous: cameras alternate, so duration is just N slices
        alpha_duration = alpha_total_slices * alpha_slice_period
    
    if not beta_simultaneous:
        # Sequential: Path A (N slices) + delay + Path B (N slices)
        beta_duration = (beta_total_slices * beta_slice_period + 
                        beta_delay_before_side + 
                        beta_total_slices * beta_slice_period)
    else:
        # Simultaneous: cameras alternate, so duration is just N slices
        beta_duration = beta_total_slices * beta_slice_period
    
    # Initialize discarding variables
    slices_to_discard_alpha_start = 0
    slices_to_discard_alpha_end = 0
    slices_to_discard_beta_start = 0
    slices_to_discard_beta_end = 0
    alpha_started_first = False
    beta_started_first = False
    alpha_finished_first = False
    beta_finished_first = False
    
    # Calculate time offset and determine which arm started/finished first
    if alpha_start is None or beta_start is None:
        print("Warning: Could not parse start times. Using default alignment (no discarding).")
        time_offset_sec = 0.0
    else:
        # Calculate time difference (beta - alpha) in seconds
        time_diff = beta_start - alpha_start
        time_offset_sec = time_diff.total_seconds()
        
        # Calculate when each arm finishes (relative to their own start)
        alpha_end_time = alpha_duration  # Time when alpha finishes (relative to alpha start)
        beta_end_time = beta_duration     # Time when beta finishes (relative to beta start)
        
        # Determine which arm started first
        if time_offset_sec < 0:
            # Beta started before alpha (negative offset means beta is earlier)
            # Relative to beta start: beta starts at t=0, alpha starts at t=abs(time_offset_sec)
            alpha_started_first = False
            beta_started_first = True
            
            # Discard slices from beta that occurred before alpha started
            slices_to_discard_beta_start = int(np.ceil(abs(time_offset_sec) / beta_slice_period))
            
            # Calculate when each arm finishes (relative to beta start time)
            # Beta finishes at: beta_duration (relative to beta start)
            # Alpha finishes at: abs(time_offset_sec) + alpha_duration (relative to beta start)
            beta_end_time = beta_duration
            alpha_end_time = abs(time_offset_sec) + alpha_duration
            
            if beta_end_time < alpha_end_time:
                # Beta finishes first - discard end slices from alpha
                # Overlap starts when alpha starts (t=abs(time_offset_sec)), ends when beta finishes (t=beta_duration)
                # Overlap duration = beta_duration - abs(time_offset_sec)
                overlap_duration = beta_duration - abs(time_offset_sec)
                alpha_slices_in_overlap = int(np.floor(overlap_duration / alpha_slice_period))
                slices_to_discard_alpha_end = alpha_total_slices - slices_to_discard_alpha_start - alpha_slices_in_overlap
                alpha_finished_first = False
                beta_finished_first = True
            elif alpha_end_time < beta_end_time:
                # Alpha finishes first - discard end slices from beta
                # Overlap starts when alpha starts (t=abs(time_offset_sec)), ends when alpha finishes
                # Overlap duration = alpha_duration
                overlap_duration = alpha_duration
                beta_slices_in_overlap = int(np.floor(overlap_duration / beta_slice_period))
                slices_to_discard_beta_end = beta_total_slices - slices_to_discard_beta_start - beta_slices_in_overlap
                alpha_finished_first = True
                beta_finished_first = False
            else:
                # They finish at the same time
                alpha_finished_first = False
                beta_finished_first = False
                
        elif time_offset_sec > 0:
            # Alpha started before beta (positive offset means alpha is earlier)
            # Relative to alpha start: alpha starts at t=0, beta starts at t=time_offset_sec
            alpha_started_first = True
            beta_started_first = False
            
            # Discard slices from alpha that occurred before beta started
            slices_to_discard_alpha_start = int(np.ceil(time_offset_sec / alpha_slice_period))
            
            # Calculate when each arm finishes (relative to alpha start time)
            # Alpha finishes at: alpha_duration (relative to alpha start)
            # Beta finishes at: time_offset_sec + beta_duration (relative to alpha start)
            alpha_end_time = alpha_duration
            beta_end_time = time_offset_sec + beta_duration
            
            if alpha_end_time < beta_end_time:
                # Alpha finishes first - discard end slices from beta
                # Overlap duration = alpha_duration - time_offset_sec (since beta started after alpha)
                overlap_duration = alpha_duration - time_offset_sec
                beta_slices_in_overlap = int(np.floor(overlap_duration / beta_slice_period))
                slices_to_discard_beta_end = beta_total_slices - slices_to_discard_beta_start - beta_slices_in_overlap
                alpha_finished_first = True
                beta_finished_first = False
            elif beta_end_time < alpha_end_time:
                # Beta finishes first - discard end slices from alpha
                # Overlap duration = beta_duration (since alpha started before beta)
                overlap_duration = beta_duration
                alpha_slices_in_overlap = int(np.floor(overlap_duration / alpha_slice_period))
                slices_to_discard_alpha_end = alpha_total_slices - slices_to_discard_alpha_start - alpha_slices_in_overlap
                alpha_finished_first = False
                beta_finished_first = True
            else:
                # They finish at the same time
                alpha_finished_first = False
                beta_finished_first = False
        else:
            # They started at the same time (or very close)
            alpha_started_first = False
            beta_started_first = False
            
            # Determine which finishes first
            if alpha_duration < beta_duration:
                # Alpha finishes first - discard end slices from beta
                slices_to_discard_beta_end = int(np.floor((beta_duration - alpha_duration) / beta_slice_period))
                alpha_finished_first = True
                beta_finished_first = False
            elif beta_duration < alpha_duration:
                # Beta finishes first - discard end slices from alpha
                slices_to_discard_alpha_end = int(np.floor((alpha_duration - beta_duration) / alpha_slice_period))
                alpha_finished_first = False
                beta_finished_first = True
            else:
                # They finish at the same time
                alpha_finished_first = False
                beta_finished_first = False
    
    # Calculate number of slices after discarding from both start and end
    num_slices_alpha = alpha_total_slices - slices_to_discard_alpha_start - slices_to_discard_alpha_end
    num_slices_beta = beta_total_slices - slices_to_discard_beta_start - slices_to_discard_beta_end
    num_slices = min(num_slices_alpha, num_slices_beta)
    
    # Get detailed timing information if available
    alpha_slice_duration = alpha_meta.get('slice_duration_ms')
    beta_slice_duration = beta_meta.get('slice_duration_ms')
    
    # Calculate camera timing offsets based on acquisition mode
    # For sequential mode: Path A (cam0) acquires full volume, then Path B (cam1) acquires
    # For simultaneous mode: small offset between cameras per slice
    
    # Initialize per-camera offsets (in ms)
    # Path A (cam0) always starts first at t=0
    alpha_cam0_offset = 0.0
    alpha_cam1_offset = 0.0
    beta_cam0_offset = 0.0
    beta_cam1_offset = 0.0
    
    if not alpha_simultaneous:
        # Sequential mode: Path A acquires full volume, then Path B
        # Path B offset = Path A volume duration + delay_before_side (settling time)
        alpha_cam1_offset = (alpha_total_slices * alpha_slice_period + alpha_delay_before_side) * 1000.0
    else:
        # Simultaneous mode: cameras alternate per slice with small timing offset
        alpha_timing = alpha_meta.get('slice_timing', {})
        if alpha_timing.get('cameraDelay') is not None and alpha_timing.get('cameraExposure') is not None:
            alpha_cam1_offset = alpha_timing.get('cameraDelay', 0) + alpha_timing.get('cameraExposure', 0)
    
    if not beta_simultaneous:
        # Sequential mode: Path A acquires full volume, then Path B
        beta_cam1_offset = (beta_total_slices * beta_slice_period + beta_delay_before_side) * 1000.0
    else:
        # Simultaneous mode: cameras alternate per slice with small timing offset
        beta_timing = beta_meta.get('slice_timing', {})
        if beta_timing.get('cameraDelay') is not None and beta_timing.get('cameraExposure') is not None:
            beta_cam1_offset = beta_timing.get('cameraDelay', 0) + beta_timing.get('cameraExposure', 0)
    
    return {
        'time_offset_sec': time_offset_sec,
        'alpha_start': alpha_start,
        'beta_start': beta_start,
        'alpha_started_first': alpha_started_first,
        'beta_started_first': beta_started_first,
        'alpha_finished_first': alpha_finished_first,
        'beta_finished_first': beta_finished_first,
        'slices_to_discard_alpha_start': slices_to_discard_alpha_start,
        'slices_to_discard_alpha_end': slices_to_discard_alpha_end,
        'slices_to_discard_beta_start': slices_to_discard_beta_start,
        'slices_to_discard_beta_end': slices_to_discard_beta_end,
        'alpha_slice_period_sec': alpha_slice_period,
        'beta_slice_period_sec': beta_slice_period,
        'slice_period_ms': avg_slice_period * 1000,
        'num_slices': num_slices,
        # Additional timing information
        'alpha_slice_duration_ms': alpha_slice_duration,
        'beta_slice_duration_ms': beta_slice_duration,
        # Per-camera timing offsets (in ms)
        # Path A (cam0) always starts at 0, Path B (cam1) is offset
        'alpha_cam0_offset_ms': alpha_cam0_offset,
        'alpha_cam1_offset_ms': alpha_cam1_offset,
        'beta_cam0_offset_ms': beta_cam0_offset,
        'beta_cam1_offset_ms': beta_cam1_offset,
        # Acquisition mode flags
        'alpha_cameras_simultaneous': alpha_simultaneous,
        'beta_cameras_simultaneous': beta_simultaneous
    }


def load_temporally_aligned_stacks(alpha_tiff, beta_tiff, alpha_meta, beta_meta, 
                                   interpolation_method='linear', verbose=True, apply_clahe=False):
    """
    Load all 4 camera stacks with proper temporal alignment using interpolation.
    
    This function handles two acquisition modes:
    1. **Sequential volumes** (`acquire_both_cameras_simultaneously=False`):
       - Path A (Channel 0) acquires full volume first, then Path B (Channel 1) starts
       - Path B slices are temporally offset by: N*slice_period + delay_before_side
    2. **Simultaneous cameras** (`acquire_both_cameras_simultaneously=True`):
       - Path A and Path B alternate per slice with small timing offsets
    
    The function:
    1. Loads raw data from all 4 cameras
    2. Calculates exact capture times for each camera/slice based on acquisition mode
    3. Defines a common time grid
    4. Interpolates each camera's data to the common time grid
    
    Parameters:
    -----------
    alpha_tiff : str or Path
        Path to alpha OME-TIFF file
    beta_tiff : str or Path
        Path to beta OME-TIFF file
    alpha_meta : dict
        Parsed metadata for alpha arm
    beta_meta : dict
        Parsed metadata for beta arm
    interpolation_method : str
        Interpolation method: 'linear', 'nearest', 'cubic'
    verbose : bool
        Print progress information
    apply_clahe : bool
        If True, apply CLAHE preprocessing to each channel before temporal alignment.
        This enhances local contrast and can improve image quality. Default: False
        
    Returns:
    --------
    dict : Dictionary containing:
        - 'alpha_cam0': (slices, height, width) - temporally aligned
        - 'alpha_cam1': (slices, height, width) - temporally aligned
        - 'beta_cam0': (slices, height, width) - temporally aligned
        - 'beta_cam1': (slices, height, width) - temporally aligned
        - 'common_times': Array of common time points (seconds)
        - 'alignment_info': Dictionary with temporal alignment details
    """
    if verbose:
        print("=" * 70)
        print("Loading Temporally Aligned Stacks")
        print("=" * 70)
    
    # Load raw data
    if verbose:
        print("\nLoading raw data...")
    alpha_data = load_ome_tiff(alpha_tiff, metadata=alpha_meta, channel_idx=None, apply_clahe=apply_clahe)
    beta_data = load_ome_tiff(beta_tiff, metadata=beta_meta, channel_idx=None, apply_clahe=apply_clahe)
    
    # Extract individual camera stacks
    # Shape is (slices, channels, height, width)
    alpha_cam0_raw = alpha_data[:, 0, :, :]  # Camera 0
    alpha_cam1_raw = alpha_data[:, 1, :, :]  # Camera 1
    beta_cam0_raw = beta_data[:, 0, :, :]    # Camera 0
    beta_cam1_raw = beta_data[:, 1, :, :]    # Camera 1
    
    if verbose:
        print(f"\nRaw data shapes:")
        print(f"  Alpha Cam0: {alpha_cam0_raw.shape}")
        print(f"  Alpha Cam1: {alpha_cam1_raw.shape}")
        print(f"  Beta Cam0: {beta_cam0_raw.shape}")
        print(f"  Beta Cam1: {beta_cam1_raw.shape}")
    
    # Calculate temporal alignment
    temporal_info = calculate_temporal_alignment(alpha_meta, beta_meta)
    
    # Get discarding information
    slices_to_discard_alpha_start = temporal_info['slices_to_discard_alpha_start']
    slices_to_discard_alpha_end = temporal_info['slices_to_discard_alpha_end']
    slices_to_discard_beta_start = temporal_info['slices_to_discard_beta_start']
    slices_to_discard_beta_end = temporal_info['slices_to_discard_beta_end']
    
    if verbose:
        print(f"\nTemporal alignment analysis:")
        print(f"  Time offset (beta - alpha): {temporal_info['time_offset_sec']*1000:.2f} ms")
        if temporal_info['alpha_started_first']:
            print(f"  Alpha started first")
        elif temporal_info['beta_started_first']:
            print(f"  Beta started first")
        else:
            print(f"  Arms started simultaneously (or very close)")
        
        if temporal_info['alpha_finished_first']:
            print(f"  Alpha finished first")
        elif temporal_info['beta_finished_first']:
            print(f"  Beta finished first")
        else:
            print(f"  Arms finished simultaneously (or very close)")
    
    # Discard slices from start of early-starting arm
    if slices_to_discard_alpha_start > 0:
        alpha_cam0_raw = alpha_cam0_raw[slices_to_discard_alpha_start:, :, :]
        alpha_cam1_raw = alpha_cam1_raw[slices_to_discard_alpha_start:, :, :]
        if verbose:
            print(f"  Discarded {slices_to_discard_alpha_start} slices from START of alpha arm")
    
    if slices_to_discard_beta_start > 0:
        beta_cam0_raw = beta_cam0_raw[slices_to_discard_beta_start:, :, :]
        beta_cam1_raw = beta_cam1_raw[slices_to_discard_beta_start:, :, :]
        if verbose:
            print(f"  Discarded {slices_to_discard_beta_start} slices from START of beta arm")
    
    # Discard slices from end of late-finishing arm
    if slices_to_discard_alpha_end > 0:
        alpha_cam0_raw = alpha_cam0_raw[:-slices_to_discard_alpha_end, :, :]
        alpha_cam1_raw = alpha_cam1_raw[:-slices_to_discard_alpha_end, :, :]
        if verbose:
            print(f"  Discarded {slices_to_discard_alpha_end} slices from END of alpha arm")
    
    if slices_to_discard_beta_end > 0:
        beta_cam0_raw = beta_cam0_raw[:-slices_to_discard_beta_end, :, :]
        beta_cam1_raw = beta_cam1_raw[:-slices_to_discard_beta_end, :, :]
        if verbose:
            print(f"  Discarded {slices_to_discard_beta_end} slices from END of beta arm")
    
    # Calculate exact capture times for each camera based on acquisition mode
    alpha_slice_period = temporal_info['alpha_slice_period_sec']  # seconds
    beta_slice_period = temporal_info['beta_slice_period_sec']  # seconds
    
    # Get acquisition mode flags
    alpha_simultaneous = temporal_info['alpha_cameras_simultaneous']
    beta_simultaneous = temporal_info['beta_cameras_simultaneous']
    
    # Calculate capture times for each camera (after discarding)
    num_slices = min(alpha_cam0_raw.shape[0], alpha_cam1_raw.shape[0],
                     beta_cam0_raw.shape[0], beta_cam1_raw.shape[0])
    
    # Get per-camera timing offsets (already accounts for first_side_is_a)
    # These are calculated in calculate_temporal_alignment() based on acquisition mode
    alpha_cam0_offset_sec = temporal_info['alpha_cam0_offset_ms'] / 1000.0
    alpha_cam1_offset_sec = temporal_info['alpha_cam1_offset_ms'] / 1000.0
    beta_cam0_offset_sec = temporal_info['beta_cam0_offset_ms'] / 1000.0
    beta_cam1_offset_sec = temporal_info['beta_cam1_offset_ms'] / 1000.0
    
    # Calculate capture times for each camera
    # Times are relative to the aligned start (time 0, after discarding early slices)
    times_alpha_cam0 = np.arange(num_slices) * alpha_slice_period + alpha_cam0_offset_sec
    times_alpha_cam1 = np.arange(num_slices) * alpha_slice_period + alpha_cam1_offset_sec
    times_beta_cam0 = np.arange(num_slices) * beta_slice_period + beta_cam0_offset_sec
    times_beta_cam1 = np.arange(num_slices) * beta_slice_period + beta_cam1_offset_sec
    
    # Define common time grid
    # Use the union of all time points to find the time range
    all_times = np.concatenate([times_alpha_cam0, times_alpha_cam1, 
                                times_beta_cam0, times_beta_cam1])
    time_min = all_times.min()
    time_max = all_times.max()
    
    # Create common time grid with same number of slices as input
    # Use the average slice period to maintain similar temporal spacing
    avg_period = (alpha_slice_period + beta_slice_period) / 2.0
    # Create common time grid starting from time_min with avg_period spacing
    # This ensures we have the same number of slices but temporally aligned
    common_times = np.arange(time_min, time_min + num_slices * avg_period, avg_period)
    # Ensure we don't exceed the time range
    if len(common_times) > num_slices:
        common_times = common_times[:num_slices]
    elif len(common_times) < num_slices:
        # Extend if needed
        last_time = common_times[-1]
        additional_times = np.arange(last_time + avg_period, 
                                     last_time + (num_slices - len(common_times) + 1) * avg_period,
                                     avg_period)
        common_times = np.concatenate([common_times, additional_times[:num_slices - len(common_times)]])
    num_common_slices = len(common_times)
    
    if verbose:
        print(f"\nTemporal alignment (after discarding early slices):")
        print(f"  Alpha acquisition mode: {'Simultaneous' if alpha_simultaneous else 'Sequential volumes (Path A then Path B)'}")
        print(f"  Beta acquisition mode: {'Simultaneous' if beta_simultaneous else 'Sequential volumes (Path A then Path B)'}")
        print(f"  Alpha Cam0 (Path A) offset: {temporal_info['alpha_cam0_offset_ms']:.2f} ms")
        print(f"  Alpha Cam1 (Path B) offset: {temporal_info['alpha_cam1_offset_ms']:.2f} ms")
        print(f"  Beta Cam0 (Path A) offset: {temporal_info['beta_cam0_offset_ms']:.2f} ms")
        print(f"  Beta Cam1 (Path B) offset: {temporal_info['beta_cam1_offset_ms']:.2f} ms")
        print(f"  Common time grid: {num_common_slices} slices")
        print(f"  Time range: {time_min*1000:.2f} - {time_max*1000:.2f} ms")
    
    # Interpolate each camera to common time grid
    if verbose:
        print(f"\nInterpolating to common time grid...")
    
    def interpolate_stack(stack, times, target_times, method=interpolation_method):
        """Interpolate a 3D stack along the time (first) axis using vectorized operations."""
        # stack shape: (slices, height, width)
        # times: array of time points for each slice
        # target_times: array of target time points
        
        # Reshape stack to (slices, height*width) for vectorized interpolation
        original_shape = stack.shape
        stack_2d = stack.reshape(original_shape[0], -1)  # (slices, height*width)
        
        # Use scipy's interp1d with vectorized input
        # Note: interp1d can handle 2D arrays where each column is interpolated independently
        if method == 'linear':
            interp_func = interp1d(times, stack_2d, kind='linear', axis=0,
                                  bounds_error=False, fill_value='extrapolate')
        elif method == 'nearest':
            interp_func = interp1d(times, stack_2d, kind='nearest', axis=0,
                                  bounds_error=False, fill_value='extrapolate')
        elif method == 'cubic':
            interp_func = interp1d(times, stack_2d, kind='cubic', axis=0,
                                  bounds_error=False, fill_value='extrapolate')
        else:
            raise ValueError(f"Unknown interpolation method: {method}")
        
        # Interpolate all pixels at once
        interpolated_2d = interp_func(target_times)  # (target_slices, height*width)
        
        # Reshape back to (target_slices, height, width)
        interpolated = interpolated_2d.reshape(len(target_times), original_shape[1], original_shape[2])
        
        # Preserve original dtype
        return interpolated.astype(stack.dtype)
    
    # Interpolate all cameras
    if verbose:
        print("  Interpolating Alpha Cam0...")
    alpha_cam0_aligned = interpolate_stack(alpha_cam0_raw, times_alpha_cam0, common_times, interpolation_method)
    
    if verbose:
        print("  Interpolating Alpha Cam1...")
    alpha_cam1_aligned = interpolate_stack(alpha_cam1_raw, times_alpha_cam1, common_times, interpolation_method)
    
    if verbose:
        print("  Interpolating Beta Cam0...")
    beta_cam0_aligned = interpolate_stack(beta_cam0_raw, times_beta_cam0, common_times, interpolation_method)
    
    if verbose:
        print("  Interpolating Beta Cam1...")
    beta_cam1_aligned = interpolate_stack(beta_cam1_raw, times_beta_cam1, common_times, interpolation_method)
    
    if verbose:
        print(f"\nTemporally aligned stacks:")
        print(f"  Alpha Cam0: {alpha_cam0_aligned.shape}")
        print(f"  Alpha Cam1: {alpha_cam1_aligned.shape}")
        print(f"  Beta Cam0: {beta_cam0_aligned.shape}")
        print(f"  Beta Cam1: {beta_cam1_aligned.shape}")
        print("=" * 70)
    
    alignment_info = {
        'temporal_info': temporal_info,
        'times_alpha_cam0': times_alpha_cam0,
        'times_alpha_cam1': times_alpha_cam1,
        'times_beta_cam0': times_beta_cam0,
        'times_beta_cam1': times_beta_cam1,
        'common_times': common_times,
        'interpolation_method': interpolation_method
    }
    
    return {
        'alpha_cam0': alpha_cam0_aligned,
        'alpha_cam1': alpha_cam1_aligned,
        'beta_cam0': beta_cam0_aligned,
        'beta_cam1': beta_cam1_aligned,
        'common_times': common_times,
        'alignment_info': alignment_info
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
    numpy.ndarray : 3x3 transformation matrix
    numpy.ndarray : Offset vector
    """
    angle_rad = np.deg2rad(angle_deg)
    
    # Calculate shear factor
    # The shear depends on the ratio of z-step to pixel size and the angle
    # For diSPIM with 45° angle: shear = (z_step / pixel_size) * tan(45°) = z_step / pixel_size
    shear_factor = (z_step_um / pixel_size_um) * np.tan(angle_rad)
    
    # Create affine transformation matrix
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


def create_camera_overlay(cam1_img, cam2_img, flip_cam2_horizontal=True):
    """
    Create a red/green overlay of two camera views (similar to MATLAB's imfuse).
    
    This function overlays two camera images from the same arm to visualize
    their spatial relationship. The first camera is shown in red, the second in green.
    Overlapping regions appear yellow.
    
    IMPORTANT: Because the two objectives in a single arm face each other,
    their cameras capture "mirror images" of the sample in the lateral direction.
    By default, cam2 is horizontally flipped before overlay to account for this,
    so the overlay shows how they would align after accounting for the mirror relationship.
    
    Parameters:
    -----------
    cam1_img : numpy.ndarray
        First camera image (height, width) - will be displayed in red channel
    cam2_img : numpy.ndarray
        Second camera image (height, width) - will be displayed in green channel
        Will be horizontally flipped if flip_cam2_horizontal=True
    flip_cam2_horizontal : bool
        If True, flip cam2_img horizontally (along X-axis) before overlay
        to account for mirror image relationship between cameras.
        Default is True to show how cameras would align.
        
    Returns:
    --------
    numpy.ndarray : RGB image (height, width, 3) with values in [0, 1] range
        Red channel = cam1_img (normalized)
        Green channel = cam2_img (normalized, optionally flipped)
        Blue channel = zeros
    """
    # Ensure images are the same size
    if cam1_img.shape != cam2_img.shape:
        raise ValueError(f"Camera images must have the same shape. Got {cam1_img.shape} and {cam2_img.shape}")
    
    # Apply horizontal flip to cam2 if needed (cameras capture mirror images)
    if flip_cam2_horizontal:
        cam2_img = np.flip(cam2_img, axis=1)  # Flip along X-axis (axis 1 for 2D image)
    
    # Normalize each camera image independently to [0, 1]
    cam1_scaled, _, _ = scale_image_for_display(cam1_img)
    cam2_scaled, _, _ = scale_image_for_display(cam2_img)
    
    # Create RGB overlay: red = cam1, green = cam2, blue = 0
    overlay = np.zeros((cam1_img.shape[0], cam1_img.shape[1], 3), dtype=np.float64)
    overlay[:, :, 0] = cam1_scaled  # Red channel
    overlay[:, :, 1] = cam2_scaled  # Green channel
    # Blue channel stays zero
    
    return overlay


def create_stitched_camera_view(alpha_slice, beta_slice, camera_index=0):
    """
    Create a stitched side-by-side view of the same camera index from both arms.
    
    This function stitches together the alpha and beta views of the same camera
    to create a full sample view.
    
    - Camera 0: Alpha (left) + Beta (right), no flip
    - Camera 1: Alpha (right) + Beta (left), no flip (reversed order)
    
    The horizontal flip is only needed for intra-arm overlays, not for stitching.
    
    Parameters:
    -----------
    alpha_slice : numpy.ndarray
        Single slice from alpha arm for the specified camera index (height, width)
    beta_slice : numpy.ndarray
        Single slice from beta arm for the same camera index (height, width)
    camera_index : int
        Camera index (0 or 1)
        
    Returns:
    --------
    numpy.ndarray : Stitched image (height, width*2) with values in [0, 1] range
        Camera 0: Alpha (left) + Beta (right)
        Camera 1: Alpha (right) + Beta (left)
    """
    # Scale images for display (preserves full dynamic range)
    alpha_scaled, _, _ = scale_image_for_display(alpha_slice)
    beta_scaled, _, _ = scale_image_for_display(beta_slice)
    
    # Stitch based on camera index
    if camera_index == 0:
        # Camera 0: alpha left, beta right
        stitched = np.hstack([alpha_scaled, beta_scaled])
    else:
        # Camera 1: alpha right, beta left (reversed order)
        stitched = np.hstack([beta_scaled, alpha_scaled])
    
    return stitched


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
# Save/Load Functions for Temporally Aligned Stacks
# ============================================================================

def save_temporally_aligned_stacks(aligned_data, base_output_dir, acquisition_name, 
                                   alpha_meta=None, beta_meta=None):
    """
    Save all 4 temporally aligned camera stacks to disk.
    
    Parameters:
    -----------
    aligned_data : dict
        Dictionary returned from load_temporally_aligned_stacks() containing:
        - 'alpha_cam0', 'alpha_cam1', 'beta_cam0', 'beta_cam1': numpy arrays
        - 'common_times': array of time points
        - 'alignment_info': dict with alignment details
    base_output_dir : str or Path
        Base directory for saving processed results
    acquisition_name : str
        Name of the acquisition
    alpha_meta : dict or None
        Alpha metadata (for camera names)
    beta_meta : dict or None
        Beta metadata (for camera names)
        
    Returns:
    --------
    dict : Dictionary mapping camera names to saved file paths
    """
    base_output_dir = Path(base_output_dir)
    output_dir = base_output_dir / acquisition_name / 'temporally_aligned'
    output_dir.mkdir(parents=True, exist_ok=True)
    
    saved_paths = {}
    
    # Get camera names
    alpha_cam0_name = alpha_meta['channel_names'][0] if alpha_meta else 'Camera0'
    alpha_cam1_name = alpha_meta['channel_names'][1] if alpha_meta else 'Camera1'
    beta_cam0_name = beta_meta['channel_names'][0] if beta_meta else 'Camera0'
    beta_cam1_name = beta_meta['channel_names'][1] if beta_meta else 'Camera1'
    
    # Save each camera stack
    cameras = [
        ('alpha_cam0', aligned_data['alpha_cam0'], 'alpha', alpha_cam0_name),
        ('alpha_cam1', aligned_data['alpha_cam1'], 'alpha', alpha_cam1_name),
        ('beta_cam0', aligned_data['beta_cam0'], 'beta', beta_cam0_name),
        ('beta_cam1', aligned_data['beta_cam1'], 'beta', beta_cam1_name),
    ]
    
    for key, stack, arm, cam_name in cameras:
        filename = f"{arm}_{cam_name}_temporally_aligned.tif"
        stack_path = output_dir / filename
        
        # Save stack
        tifffile.imwrite(
            str(stack_path),
            stack,
            photometric='minisblack',
            metadata={'axes': 'ZYX'}
        )
        saved_paths[key] = stack_path
    
    # Save combined metadata
    metadata_path = output_dir / 'alignment_metadata.json'
    metadata = {
        'acquisition_name': acquisition_name,
        'alignment_info': {},
        'camera_names': {
            'alpha_cam0': alpha_cam0_name,
            'alpha_cam1': alpha_cam1_name,
            'beta_cam0': beta_cam0_name,
            'beta_cam1': beta_cam1_name,
        }
    }
    
    # Convert alignment_info to JSON-serializable format
    if 'alignment_info' in aligned_data:
        align_info = aligned_data['alignment_info']
        metadata['alignment_info'] = {}
        for key, value in align_info.items():
            if isinstance(value, np.ndarray):
                metadata['alignment_info'][key] = value.tolist()
            elif isinstance(value, (np.integer, np.floating)):
                metadata['alignment_info'][key] = float(value)
            elif isinstance(value, dict):
                # Handle nested dicts (like temporal_info)
                metadata['alignment_info'][key] = {}
                for k, v in value.items():
                    if isinstance(v, np.ndarray):
                        metadata['alignment_info'][key][k] = v.tolist()
                    elif isinstance(v, (np.integer, np.floating)):
                        metadata['alignment_info'][key][k] = float(v)
                    elif isinstance(v, (datetime, type(None))):
                        metadata['alignment_info'][key][k] = str(v) if v else None
                    else:
                        metadata['alignment_info'][key][k] = v
            else:
                metadata['alignment_info'][key] = value
    
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    saved_paths['metadata'] = metadata_path
    return saved_paths


def get_temporally_aligned_paths(base_output_dir, acquisition_name, camera_name, arm_name):
    """
    Generate standardized file paths for temporally aligned stacks.
    
    Parameters:
    -----------
    base_output_dir : str or Path
        Base directory for saving processed results
    acquisition_name : str
        Name of the acquisition
    camera_name : str
        Name of the camera
    arm_name : str
        Name of the arm ("alpha" or "beta")
        
    Returns:
    --------
    Path : Path to the temporally aligned stack file
    Path : Path to the metadata JSON file
    """
    base_output_dir = Path(base_output_dir)
    output_dir = base_output_dir / acquisition_name / 'temporally_aligned'
    
    filename = f"{arm_name}_{camera_name}_temporally_aligned.tif"
    stack_path = output_dir / filename
    metadata_path = output_dir / 'alignment_metadata.json'
    
    return stack_path, metadata_path


def check_temporally_aligned_exists(base_output_dir, acquisition_name, alpha_meta, beta_meta):
    """
    Check if all 4 temporally aligned stacks already exist on disk.
    
    Parameters:
    -----------
    base_output_dir : str or Path
        Base directory for processed results
    acquisition_name : str
        Name of the acquisition
    alpha_meta : dict
        Alpha metadata (for camera names)
    beta_meta : dict
        Beta metadata (for camera names)
        
    Returns:
    --------
    bool : True if all stacks exist
    dict : Dictionary mapping camera keys to file paths (if exists)
    Path : Path to metadata file (if exists)
    """
    alpha_cam0_name = alpha_meta['channel_names'][0]
    alpha_cam1_name = alpha_meta['channel_names'][1]
    beta_cam0_name = beta_meta['channel_names'][0]
    beta_cam1_name = beta_meta['channel_names'][1]
    
    paths = {}
    all_exist = True
    
    cameras = [
        ('alpha_cam0', alpha_cam0_name, 'alpha'),
        ('alpha_cam1', alpha_cam1_name, 'alpha'),
        ('beta_cam0', beta_cam0_name, 'beta'),
        ('beta_cam1', beta_cam1_name, 'beta'),
    ]
    
    for key, cam_name, arm in cameras:
        stack_path, metadata_path = get_temporally_aligned_paths(
            base_output_dir, acquisition_name, cam_name, arm
        )
        paths[key] = stack_path
        if not stack_path.exists():
            all_exist = False
    
    # Check metadata file
    metadata_path = Path(base_output_dir) / acquisition_name / 'temporally_aligned' / 'alignment_metadata.json'
    
    return all_exist, paths if all_exist else None, metadata_path if all_exist and metadata_path.exists() else None


def load_temporally_aligned_stacks_from_file(base_output_dir, acquisition_name, 
                                            alpha_meta, beta_meta, verbose=True):
    """
    Load all 4 temporally aligned camera stacks from disk.
    
    Parameters:
    -----------
    base_output_dir : str or Path
        Base directory for processed results
    acquisition_name : str
        Name of the acquisition
    alpha_meta : dict
        Alpha metadata (for camera names)
    beta_meta : dict
        Beta metadata (for camera names)
    verbose : bool
        Print progress information
        
    Returns:
    --------
    dict : Dictionary containing:
        - 'alpha_cam0', 'alpha_cam1', 'beta_cam0', 'beta_cam1': numpy arrays
        - 'common_times': array of time points (if available)
        - 'alignment_info': dict with alignment details (if available)
    """
    if verbose:
        print("=" * 70)
        print("Loading Temporally Aligned Stacks from Disk")
        print("=" * 70)
    
    alpha_cam0_name = alpha_meta['channel_names'][0]
    alpha_cam1_name = alpha_meta['channel_names'][1]
    beta_cam0_name = beta_meta['channel_names'][0]
    beta_cam1_name = beta_meta['channel_names'][1]
    
    # Load each camera stack
    cameras = [
        ('alpha_cam0', alpha_cam0_name, 'alpha'),
        ('alpha_cam1', alpha_cam1_name, 'alpha'),
        ('beta_cam0', beta_cam0_name, 'beta'),
        ('beta_cam1', beta_cam1_name, 'beta'),
    ]
    
    aligned_data = {}
    for key, cam_name, arm in cameras:
        stack_path, _ = get_temporally_aligned_paths(
            base_output_dir, acquisition_name, cam_name, arm
        )
        if verbose:
            print(f"Loading {key} ({arm} {cam_name})...")
        aligned_data[key] = tifffile.imread(str(stack_path))
        if verbose:
            print(f"  Shape: {aligned_data[key].shape}")
    
    # Load metadata if available
    metadata_path = Path(base_output_dir) / acquisition_name / 'temporally_aligned' / 'alignment_metadata.json'
    alignment_info = None
    if metadata_path.exists():
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
            if 'alignment_info' in metadata:
                alignment_info = metadata['alignment_info']
                # Convert lists back to numpy arrays where appropriate
                if 'common_times' in alignment_info:
                    alignment_info['common_times'] = np.array(alignment_info['common_times'])
                if 'times_alpha_cam0' in alignment_info:
                    alignment_info['times_alpha_cam0'] = np.array(alignment_info['times_alpha_cam0'])
                if 'times_alpha_cam1' in alignment_info:
                    alignment_info['times_alpha_cam1'] = np.array(alignment_info['times_alpha_cam1'])
                if 'times_beta_cam0' in alignment_info:
                    alignment_info['times_beta_cam0'] = np.array(alignment_info['times_beta_cam0'])
                if 'times_beta_cam1' in alignment_info:
                    alignment_info['times_beta_cam1'] = np.array(alignment_info['times_beta_cam1'])
    
    aligned_data['alignment_info'] = alignment_info
    if alignment_info and 'common_times' in alignment_info:
        aligned_data['common_times'] = alignment_info['common_times']
    else:
        # Create dummy common_times if not available
        num_slices = aligned_data['alpha_cam0'].shape[0]
        aligned_data['common_times'] = np.arange(num_slices)
    
    if verbose:
        print("=" * 70)
    
    return aligned_data


# ============================================================================
# Generic Save/Load Functions (for any processed stage)
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

