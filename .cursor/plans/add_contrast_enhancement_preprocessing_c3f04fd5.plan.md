---
name: Add contrast enhancement preprocessing
overview: Add ImageJ/Fiji preprocessing step for contrast enhancement and modify pipeline to optionally use preprocessed data when available.
todos:
  - id: fix_temporal_bug
    content: Fix temporal alignment to use time_offset_sec instead of delay_before_side in load_temporally_aligned_stacks()
    status: completed
  - id: add_temporal_save_load
    content: Add save/load functions for temporally aligned stacks (save_temporally_aligned_stacks, load_temporally_aligned_stacks_from_file, check_temporally_aligned_exists, get_temporally_aligned_paths)
    status: completed
  - id: update_notebook_temporal
    content: Update dispim_pipeline.ipynb to save/load temporally aligned stacks with FORCE_RETEMPORAL_ALIGN flag
    status: completed
  - id: add_stitched_visualization
    content: Add create_stitched_camera_view() function to dispim_utils.py for stitched same-camera-index views
    status: completed
  - id: update_notebook_visualization
    content: Add Section 7 to dispim_pipeline.ipynb showing stitched camera views (Camera 0 and Camera 1)
    status: completed
  - id: enhance_napari_viewer
    content: Enhance view_volumes_napari.py with transformation capabilities, save matrices, and export aligned volumes
    status: completed
  - id: delete_old_files
    content: "Delete old files: utils.py, processing.py, CONSOLIDATION_PLAN.md, volume_viewer_app.py, templates/volume_viewer.html, README_volume_viewer.md"
    status: completed
  - id: write_readme
    content: Write comprehensive README.md covering setup, pipeline workflow, visualization, Napari viewer, and troubleshooting
    status: completed
---

# Add Contrast Enhancement Preprocessing

## Overview

Add a preprocessing step using ImageJ/Fiji to enhance contrast in raw OME-TIFF files before they enter the main pipeline. The enhanced files will be saved alongside the originals with an `_enhanced` suffix, and the pipeline will automatically use them if available.

## Implementation Plan

### 1. ImageJ/Fiji Batch Processing Instructions

**CRITICAL:** You MUST process each channel separately, then merge them back together before saving. If you save the active window directly, you will only get ONE channel. The enhanced file must contain BOTH channels.

#### Method 1: Manual Processing (Step-by-Step)

**For a single file:**

1. **Open the file:**

   - `Plugins` → `Bio-Formats` → `Bio-Formats Importer...`
   - Navigate to and select your `.ome.tif` file
   - Click `Open` (or `OK`)
   - In the "Bio-Formats Import Options" dialog:
     - **View stack with:** "Hyperstack"
     - **Stack order:** "XYCZT"
     - ✓ **Split channels:** CHECKED (critical - opens each channel in separate window)
     - **Color mode:** "Grayscale" or "Colorized"
     - Click `OK`
   - This opens each channel in a separate window (e.g., "beads_alpha_worm2_MMStack_Pos0.ome.tif - C=0" and "C=1")

2. **Process Channel 1 (C=0):**

   - Click on the window for Channel 1 (C=0) to make it active
   - `Process` → `Enhance Contrast...` (NOT Image → Adjust)
   - In the "Enhance Contrast" dialog:
     - **Saturated pixels:** `0.35`
     - ✓ **Normalize** (checked)
     - ✓ **Equalize histogram** (checked)
     - ✓ **Process all [N] slices** (checked - N will be your number of slices, e.g., 200)
     - ✓ **Use stack histogram** (checked)
   - Click `OK`
   - Wait for processing to complete (you'll see progress bar)

3. **Process Channel 2 (C=1):**

   - Click on the window for Channel 2 (C=1) to make it active
   - `Process` → `Enhance Contrast...`
   - Use the same settings:
     - **Saturated pixels:** `0.35`
     - ✓ **Normalize** (checked)
     - ✓ **Equalize histogram** (checked)
     - ✓ **Process all [N] slices** (checked)
     - ✓ **Use stack histogram** (checked)
   - Click `OK`
   - Wait for processing to complete

4. **Merge channels back together (CRITICAL STEP):**

   - Make sure BOTH channel windows are still open (don't close them yet!)
   - `Image` → `Color` → `Merge Channels...`
   - In the "Merge Channels" dialog:
     - **C1 (red):** Select your first channel window from dropdown (e.g., "beads_alpha_worm2_MMStack_Pos0.ome.tif - C=0")
     - **C2 (green):** Select your second channel window from dropdown (e.g., "beads_alpha_worm2_MMStack_Pos0.ome.tif - C=1")
     - Leave other channels empty
     - Click `OK`
   - This creates a NEW merged window (usually named "RGB" or similar) with BOTH channels combined

5. **Save the enhanced file:**

   - **IMPORTANT:** Click on the merged "RGB" window to make it active (not the individual channel windows)
   - `Plugins` → `Bio-Formats` →  `Bio-Formats Exporter...`
   - Navigate to the same directory as the original file
   - Enter filename: `[original_name]_enhanced.ome.tif`
     - Example: `beads_alpha_worm2_MMStack_Pos0_enhanced.ome.tif`
   - Click `Save`
   - When the "Bio-Formats Exporter - Multiple Files" dialog appears:
     - **DO NOT check any boxes** - leave ALL unchecked:
       - "Write each Z section to a separate file" - **UNCHECKED**
       - "Write each timepoint to a separate file" - **UNCHECKED**
       - "Write each channel to a separate file" - **UNCHECKED** (if checked, it will save separate files!)
       - "Use zero padding for filename indexes" - **UNCHECKED**
     - Click `OK`
   - The file will be saved as a single OME-TIFF with BOTH channels preserved

6. **Clean up:**

   - Close all windows (`File` → `Close All`)

**Repeat for all files:**

- Process both alpha and beta `.ome.tif` files
- Process each acquisition condition/run
- Each file must be processed individually using the steps above

**Note:** There is NO way to apply Enhance Contrast to both channels at once. You must process each channel separately, then merge them before saving.

#### 2. Pipeline Code Modifications

**Files to modify:**

- **[dispim_utils.py](dispim_utils.py)**: Update `discover_acquisitions()` and `load_ome_tiff()` functions
- **[dispim_pipeline.ipynb](dispim_pipeline.ipynb)**: Add optional flag to use enhanced data

**Changes:**

1. **Update `discover_acquisitions()` in [dispim_utils.py](dispim_utils.py)**:

   - Modify to check for `*_enhanced.ome.tif` files first
   - Fall back to regular `.ome.tif` files if enhanced versions don't exist
   - Return both paths in the acquisition dictionary

2. **Update `load_ome_tiff()` in [dispim_utils.py](dispim_utils.py)**:

   - No changes needed - function already accepts any path
   - Will work with enhanced files automatically

3. **Add helper function `find_enhanced_or_raw()` in [dispim_utils.py](dispim_utils.py)**:

   - Takes a base path (without `_enhanced` suffix)
   - Returns path to `*_enhanced.ome.tif` if exists, else returns original path
   - Used by `discover_acquisitions()`

4. **Update `discover_acquisitions()` logic**:

   - When finding alpha/beta pairs, check for enhanced versions first
   - Log which version is being used (enhanced vs raw)

5. **Add optional parameter to pipeline notebook**:

   - Add cell with `USE_ENHANCED_DATA = True` flag
   - When True, prefer enhanced files; when False, use raw files only

### 3. File Structure

**Current structure:**

```
datasets/
  condition/
    run/
      beads_alpha_worm2_X/
        beads_alpha_worm2_MMStack_Pos0.ome.tif
      beads_beta_worm2_X/
        beads_beta_worm2_MMStack_Pos0.ome.tif
```

**After preprocessing:**

```
datasets/
  condition/
    run/
      beads_alpha_worm2_X/
        beads_alpha_worm2_MMStack_Pos0.ome.tif          (original)
        beads_alpha_worm2_MMStack_Pos0_enhanced.ome.tif  (preprocessed)
      beads_beta_worm2_X/
        beads_beta_worm2_MMStack_Pos0.ome.tif            (original)
        beads_beta_worm2_MMStack_Pos0_enhanced.ome.tif  (preprocessed)
```

### 4. Implementation Details

**Function: `find_enhanced_or_raw(base_path)`**

- Input: Path object to `.ome.tif` file (without `_enhanced`)
- Check if `*_enhanced.ome.tif` version exists
- Return enhanced path if exists, else return original path
- Log which version is used

**Function: `discover_acquisitions()` modifications**

- After finding alpha/beta pairs, check for enhanced versions
- Update `alpha_path` and `beta_path` to use enhanced if available
- Add `'using_enhanced': bool` flag to each acquisition dict

**Pipeline notebook updates**

- Add section explaining preprocessing step
- Add `USE_ENHANCED_DATA` flag (default: `True`)
- Update discovery section to show which files are being used

### 5. Benefits

- **Non-destructive:** Original data preserved
- **Automatic:** Pipeline automatically uses enhanced data if available
- **Flexible:** Can toggle between enhanced and raw via flag
- **Reproducible:** Clear instructions for preprocessing step
- **Better quality:** Enhanced contrast should improve deskewing and alignment results

### 6. Testing

- Test with one acquisition pair first
- Verify enhanced files load correctly
- Verify pipeline uses enhanced files when available
- Verify pipeline falls back to raw files when enhanced not available
- Compare deskewing results with enhanced vs raw data