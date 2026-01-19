# Double-diSPIM Processing Pipeline

A complete Python pipeline for processing raw double-diSPIM (dual-view selective plane illumination microscopy) data. This pipeline handles temporal alignment, deskewing, and visualization of multi-camera light sheet microscopy data.

## Overview

This repository provides tools to:
- **Load and parse metadata** from diSPIM acquisitions
- **Temporally align** all 4 camera stacks (accounting for sequential camera timing)
- **Deskew** 45° sheared slices to rectilinear coordinates
- **Visualize** camera combinations and stitched full-sample views
- **Interactively align** volumes in 3D using Napari

## System Geometry

The double-diSPIM system consists of:

- **Alpha Arm (Top)**: Two cameras viewing from above at 45° angles, 90° apart
  - Camera 0: Views XZ plane
  - Camera 1: Views XZ plane (mirror image of Camera 0)
  
- **Beta Arm (Bottom)**: Two cameras viewing from below at 45° angles, 90° apart, rotated 90° around Z-axis
  - Camera 0: Views YZ plane
  - Camera 1: Views YZ plane (mirror image of Camera 0)

- **Raw Data**: Each camera captures 45° sheared slices that need deskewing
- **Temporal Alignment**: Cameras are sequential (not simultaneous), requiring temporal interpolation

## Setup

### 1. Create Conda Environment

```bash
conda env create -f environment.yml
conda activate dispim_visualization
```

### 2. Verify Installation

```bash
python -c "import dispim_utils; print('Import successful!')"
```

## Preprocessing (Optional but Recommended)

Raw OME-TIFF files can be preprocessed to enhance contrast before entering the pipeline. This improves image quality and can lead to better deskewing and alignment results.

### Method 1: Programmatic CLAHE (Recommended)

The pipeline includes built-in Contrast Limited Adaptive Histogram Equalization (CLAHE) preprocessing using OpenCV. This is applied automatically during data loading.

**Usage:**
- Set `APPLY_CLAHE = True` in Section 2 of `dispim_pipeline.ipynb`
- CLAHE is applied to each channel independently during `load_temporally_aligned_stacks()`
- Uses OpenCV's `cv2.createCLAHE()` with default settings (clipLimit=2.0, tileGridSize=(8,8))
- No manual preprocessing required - fully automated

### Method 2: Manual ImageJ/Fiji Preprocessing

Alternatively, you can preprocess files manually in ImageJ/Fiji:

**Quick Summary:**
- Use ImageJ/Fiji to apply "Enhance Contrast" to each channel separately
- Merge channels back together before saving
- Save as `*_enhanced.ome.tif` in the same directory as originals
- The pipeline automatically uses enhanced files when available (see `USE_ENHANCED_DATA` flag in notebook)

**Detailed Instructions:** See the plan file `.cursor/plans/add_contrast_enhancement_preprocessing_c3f04fd5.plan.md` for step-by-step ImageJ/Fiji preprocessing instructions, including batch processing with macros.

## Pipeline Workflow

The main processing pipeline is in `dispim_pipeline.ipynb`. Run cells sequentially:

### Section 1: Setup and Imports
- Import required libraries and utility functions
- Configure matplotlib for visualization

### Section 2: Discover and Select Acquisition
- Automatically discover all alpha/beta acquisition pairs
- **Uses enhanced files if available** (set `USE_ENHANCED_DATA=True` to prefer enhanced versions)
- **Apply CLAHE preprocessing** (set `APPLY_CLAHE=True` for programmatic contrast enhancement)
- Select which acquisition to process (`ACQUISITION_INDEX`)

### Section 3: Load and Parse Metadata
- Parse JSON metadata files
- Extract spatial and temporal parameters
- Calculate pixel sizes, z-steps, and timing offsets

### Section 4: Load Temporally Aligned Stacks
- **Temporal Alignment**: Interpolate all 4 cameras to a common time grid
  - Accounts for sequential camera timing
  - Uses actual start time offsets (not theoretical delays)
  - Saves aligned stacks to avoid recomputation
- **Save/Load**: Automatically saves temporally aligned stacks
  - Set `FORCE_RETEMPORAL_ALIGN=True` to force recomputation

### Section 5: Deskew All Volumes
- **Deskewing**: Remove 45° shear from light sheet imaging
  - Applies 3D affine transformation
  - Corrects for oblique light sheet angle
  - Expands volume in X-direction due to shear correction
- **Save/Load**: Automatically saves all deskewed volumes
  - Set `FORCE_REDESKEW=True` to force recomputation

### Section 6: Visualization
- **Same-Arm Camera Overlays**: Red/green overlays showing camera alignment within each arm
- **Cross-Arm Comparisons**: Side-by-side and overlay views of same camera index across arms

### Section 7: Stitched Camera Views
- **Full Sample Images**: Stitch same camera index from both arms
  - Camera 0: Alpha (left) + Beta (right)
  - Camera 1: Alpha (left) + Beta (right, horizontally flipped)
  - Creates complete sample views combining both arms

## Temporal Alignment

### Why It's Needed

Since cameras are sequential (not simultaneous), each camera captures slices at slightly different times:
- Camera 0 captures slice N at time T
- Camera 1 captures slice N at time T + offset
- Beta arm starts after Alpha arm with a time offset

To create temporally matched stacks, we:
1. Calculate exact capture times for each camera/slice
2. Define a common time grid
3. Interpolate each camera's data to the common grid

### Implementation

The `load_temporally_aligned_stacks()` function:
- Uses actual start time differences (`time_offset_sec`) calculated from metadata
- Accounts for camera offsets within each arm
- Interpolates using linear, nearest, or cubic methods
- Preserves original number of slices

## Deskewing

### What It Does

Raw diSPIM data is sheared because:
- The light sheet is at 45° to the imaging axis
- The stage moves horizontally while capturing slices
- This creates a parallelepiped shape instead of a rectangular volume

Deskewing applies a 3D affine transformation to:
- Remove the 45° shear
- Convert to rectilinear Cartesian coordinates
- Make volumes suitable for alignment and fusion

### Implementation

The `deskew_stack()` function:
- Calculates shear factor: `shear = (z_step / pixel_size) * tan(45°)`
- Applies affine transformation using `scipy.ndimage`
- Expands output volume in X-direction
- Preserves voxel spacing information

## Interactive 3D Alignment with Napari

After deskewing, use the Napari viewer for interactive 3D alignment:

```bash
python view_volumes_napari.py [path_to_deskewed_folder]
```

### Features

- **Load All Volumes**: Automatically loads all 4 deskewed volumes
- **Color-Coded Layers**: Each volume has a distinct color (red, green, blue, yellow)
- **Interactive Transformations**: 
  - Right-click layer → Transform → Set Translate/Rotate/Scale
  - Apply rotations, translations, and scaling per volume
- **Save/Load Transforms**: 
  - Save transformation matrices to JSON
  - Load previously saved transformations
  - Reset all transforms to identity
- **Export Aligned Volumes**: 
  - Export transformed volumes as OME-TIFF files
  - Includes transformation metadata

### Controls

- **Mouse drag**: Rotate 3D view
- **Mouse wheel**: Zoom
- **Right-click drag**: Pan
- **Layer controls**: Adjust opacity, contrast, colormap, blending
- **3D button**: Toggle 3D rendering mode

### Tips

- Use layer opacity sliders to see overlaps
- Use 'additive' blending mode for better overlay visualization
- Toggle layers on/off to compare individual volumes
- Apply transformations interactively, then save matrices for reproducibility

## File Structure

### Input Data

```
datasets/
├── condition_folder/          # e.g., "10msec_worm", "1msec_worm"
│   ├── run_folder/            # e.g., "I", "II", "III"
│   │   ├── beads_alpha_*/     # Alpha arm data
│   │   │   ├── *.ome.tif      # Image stack
│   │   │   └── *_metadata.txt # Metadata JSON
│   │   └── beads_beta_*/      # Beta arm data
│   │       ├── *.ome.tif       # Image stack
│   │       └── *_metadata.txt  # Metadata JSON
```

### Output Data

```
processed_output/
├── {acquisition_name}/
│   ├── temporally_aligned/    # Temporally aligned stacks
│   │   ├── alpha_{camera0}_temporally_aligned.tif
│   │   ├── alpha_{camera1}_temporally_aligned.tif
│   │   ├── beta_{camera0}_temporally_aligned.tif
│   │   ├── beta_{camera1}_temporally_aligned.tif
│   │   └── alignment_metadata.json
│   └── deskewed/              # Deskewed volumes
│       ├── alpha_{camera0}_deskewed.tif
│       ├── alpha_{camera0}_deskewed.json
│       ├── alpha_{camera1}_deskewed.tif
│       ├── alpha_{camera1}_deskewed.json
│       ├── beta_{camera0}_deskewed.tif
│       ├── beta_{camera0}_deskewed.json
│       ├── beta_{camera1}_deskewed.tif
│       └── beta_{camera1}_deskewed.json
```

## Key Functions

### Metadata and Discovery

- `parse_metadata(metadata_path)`: Parse JSON metadata files
- `discover_acquisitions(root_dir)`: Find all alpha/beta acquisition pairs
- `calculate_temporal_alignment(alpha_meta, beta_meta)`: Calculate timing offsets
- `extract_spatial_info(alpha_meta, beta_meta)`: Extract spatial calibration

### Data Loading

- `load_ome_tiff(tiff_path, metadata, channel_idx)`: Load OME-TIFF files
- `load_temporally_aligned_stacks(alpha_tiff, beta_tiff, alpha_meta, beta_meta)`: Load and temporally align all cameras
- `load_temporally_aligned_stacks_from_file(base_dir, acquisition_name, alpha_meta, beta_meta)`: Load saved aligned stacks

### Deskewing

- `deskew_stack(stack, pixel_size_um, z_step_um, angle_deg=45.0)`: Deskew a 3D stack
- `calculate_deskew_matrix(pixel_size_um, z_step_um, angle_deg)`: Calculate transformation matrix

### Save/Load

- `save_temporally_aligned_stacks(aligned_data, base_dir, acquisition_name, alpha_meta, beta_meta)`: Save aligned stacks
- `check_temporally_aligned_exists(base_dir, acquisition_name, alpha_meta, beta_meta)`: Check if aligned stacks exist
- `save_deskewed_stack(stack, output_path, transform_info, metadata)`: Save deskewed volume
- `load_deskewed_stack(stack_path)`: Load deskewed volume
- `check_deskewed_exists(base_dir, acquisition_name, camera_name, arm_name)`: Check if deskewed volume exists

### Visualization

- `create_camera_overlay(cam1_img, cam2_img, flip_cam2_horizontal=True)`: Create red/green overlay
- `create_stitched_camera_view(alpha_slice, beta_slice, camera_index, flip_camera1)`: Create stitched full-sample view
- `create_side_by_side_frame(alpha_slice, beta_slice)`: Create side-by-side comparison
- `compute_mip(stack, axis)`: Compute maximum intensity projection

## Troubleshooting

### Memory Issues

- **Problem**: Out of memory when loading full stacks
- **Solution**: 
  - Process one camera at a time
  - Use `max_slices` parameter to limit slices during exploration
  - Ensure sufficient RAM (full stacks can be 4+ GB per arm)

### Temporal Alignment Issues

- **Problem**: Temporal alignment seems incorrect
- **Solution**:
  - Check that metadata files contain valid start times
  - Verify `time_offset_sec` is calculated correctly (uses actual start times, not theoretical delays)
  - Set `FORCE_RETEMPORAL_ALIGN=True` to recompute

### Deskewing Issues

- **Problem**: Deskewed volumes look incorrect
- **Solution**:
  - Verify pixel size and z-step values from metadata
  - Check that light sheet angle is 45° (standard for diSPIM)
  - Review transformation info in saved JSON metadata
  - Set `FORCE_REDESKEW=True` to recompute

### Napari Viewer Issues

- **Problem**: Viewer doesn't start or volumes don't load
- **Solution**:
  - Ensure Napari is installed: `conda install napari -c conda-forge`
  - Check that deskewed volumes exist in `processed_output/`
- Verify file paths are correct
  - Check console for error messages

### Import Errors

- **Problem**: `ModuleNotFoundError` for `dispim_utils`
- **Solution**:
  - Ensure you're in the correct directory
  - Check that `dispim_utils.py` exists
  - Verify Python path includes current directory

## Dependencies

Key packages:
- `numpy`: Array operations
- `scipy`: Image processing (affine transforms, interpolation)
- `tifffile`: OME-TIFF reading/writing
- `matplotlib`: Visualization
- `napari`: Interactive 3D visualization
- `qtpy`: GUI widgets for Napari

See `environment.yml` for complete list.

## Next Steps

After running the pipeline:

1. **Interactive Alignment**: Use Napari viewer to align volumes manually
2. **Save Transformations**: Save transformation matrices for reproducibility
3. **Export Aligned Volumes**: Export transformed volumes for further analysis
4. **Fusion**: Combine aligned volumes (future work)
5. **Deconvolution**: Apply joint multi-view deconvolution (future work)

## License

This pipeline is provided as-is for research purposes.

## Contact

For questions or issues, please refer to the notebook documentation or contact the development team.
