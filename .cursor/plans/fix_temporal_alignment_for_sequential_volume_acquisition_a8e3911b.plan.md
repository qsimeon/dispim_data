---
name: Fix temporal alignment for sequential volume acquisition
overview: "Update temporal alignment logic to handle two acquisition modes: (1) Sequential volumes (acquire_both_cameras_simultaneously=False): Path A completes full volume, then Path B starts after delay_before_side. (2) Simultaneous cameras (acquire_both_cameras_simultaneously=True): Path A and Path B alternate per slice. The current implementation assumes simultaneous mode but needs to handle sequential mode correctly."
todos:
  - id: update_calculate_temporal_alignment
    content: Update calculate_temporal_alignment() to check acquire_both_cameras_simultaneously flag and calculate Path B offsets correctly for sequential vs simultaneous modes
    status: completed
  - id: update_load_temporally_aligned
    content: Update load_temporally_aligned_stacks() to use correct timing calculations based on acquisition mode when computing camera slice times
    status: completed
    dependencies:
      - update_calculate_temporal_alignment
  - id: update_docstrings
    content: Update function docstrings to document both acquisition modes and their timing implications
    status: completed
  - id: add_verbose_output
    content: Add verbose output to indicate which acquisition mode is being used for each arm
    status: completed
    dependencies:
      - update_calculate_temporal_alignment
---

# Fix Temporal Alignment for Sequential vs Simultaneous Camera Acquisition

## Problem Statement

The current temporal alignment implementation assumes cameras alternate per slice, but this is only true when `acquire_both_cameras_simultaneously=True`. When this flag is `False`:

- **Path A** (Channel 0) acquires a **full volume** (all N slices) first
- After Path A completes + `delay_before_side`, **Path B** (Channel 1) starts acquiring its full volume
- The OME-TIFF stores them as separate channels, but Path B's slices correspond to later time points

## Current Implementation Issues

1. **`calculate_temporal_alignment()`**: Assumes cameras alternate per slice, calculates small offsets between cameras
2. **`load_temporally_aligned_stacks()`**: Uses slice-by-slice timing, doesn't account for sequential volume acquisition
3. **Camera offset calculation**: Only considers small delays between cameras, not full volume completion time

## Solution Approach

### 1. Update `calculate_temporal_alignment()` in [dispim_utils.py](dispim_utils.py)

**Key changes:**

- Check `acquire_both_cameras_simultaneously` flag for both alpha and beta arms
- **When False (sequential volumes)**:
  - Path A (Channel 0) starts at t=0, finishes at t = N * slice_period
  - Path B (Channel 1) starts at t = N * slice_period + delay_before_side
  - Path B finishes at t = 2*N * slice_period + delay_before_side
  - Calculate camera offsets accordingly: Path B offset = N * slice_period + delay_before_side
- **When True (simultaneous)**:
  - Use current logic: cameras alternate per slice with small timing offsets

**Return additional fields:**

- `alpha_path_b_offset_ms`: Time offset for Path B relative to Path A start (for sequential mode)
- `beta_path_b_offset_ms`: Same for beta arm
- Update `alpha_camera_offset_ms` and `beta_camera_offset_ms` to reflect correct timing based on mode

### 2. Update `load_temporally_aligned_stacks()` in [dispim_utils.py](dispim_utils.py)

**Key changes:**

- After loading data and extracting channels, check acquisition mode
- **When `acquire_both_cameras_simultaneously=False`**:
  - Path A (Channel 0) timing: `times = np.arange(N) * slice_period` (starts at t=0)
  - Path B (Channel 1) timing: `times = np.arange(N) * slice_period + (N * slice_period + delay_before_side)` (starts after Path A finishes)
  - Both channels have N slices, but Path B's slices are temporally offset
- **When `acquire_both_cameras_simultaneously=True`**:
  - Use current logic: alternate slice timing with small offsets

**Inter-arm alignment:**

- Keep current logic for aligning alpha and beta arms using StartTime differences
- Apply to both Path A and Path B channels independently

### 3. Handle Edge Cases

- **Mixed modes**: If alpha uses simultaneous but beta uses sequential (or vice versa), handle each arm independently
- **Volume duration calculation**: For sequential mode, total volume duration = 2*N*slice_period + delay_before_side (not just N*slice_period)
- **Interpolation**: Ensure interpolation handles the large time gap between Path A and Path B slices in sequential mode

### 4. Update Documentation

- Update docstrings to explain both acquisition modes
- Add comments explaining the timing differences
- Update verbose output to indicate which mode is being used

## Implementation Details

### Modified Functions

1. **`calculate_temporal_alignment()`** (lines ~582-810):
   ```python
   # Check acquisition mode
   alpha_simultaneous = alpha_meta.get('acquire_both_cameras_simultaneously', False)
   beta_simultaneous = beta_meta.get('acquire_both_cameras_simultaneously', False)
   
   # Calculate Path B offsets based on mode
   if not alpha_simultaneous:
       # Sequential: Path B starts after Path A completes + delay
       alpha_path_b_offset = (alpha_total_slices * alpha_slice_period + 
                              alpha_meta['delay_before_side'])
   else:
       # Simultaneous: small offset between cameras
       alpha_path_b_offset = alpha_camera_offset  # existing logic
   ```

2. **`load_temporally_aligned_stacks()`** (lines ~813-1000):
   ```python
   # Calculate timing for each camera based on acquisition mode
   alpha_simultaneous = alpha_meta.get('acquire_both_cameras_simultaneously', False)
   
   if not alpha_simultaneous:
       # Sequential mode: Path A first, then Path B
       times_alpha_cam0 = np.arange(num_slices) * alpha_slice_period
       times_alpha_cam1 = (np.arange(num_slices) * alpha_slice_period + 
                          temporal_info['alpha_path_b_offset_ms'] / 1000.0)
   else:
       # Simultaneous mode: alternate per slice
       # Use existing logic
   ```


## Testing Considerations

- Test with datasets where `acquire_both_cameras_simultaneously=False` (current data)
- Test with datasets where `acquire_both_cameras_simultaneously=True` (future data)
- Verify temporal alignment produces correct overlapping time windows
- Ensure interpolation handles large time gaps in sequential mode correctly

## References

- [diSPIM Plugin User Guide](http://dispim.org/docs/mm_dispim_plugin_user_guide): Documents the "Simultaneously acquire from both paths/cameras" setting
- Current metadata structure already includes `acquire_both_cameras_simultaneously` flag