# diSPIM Data Visualization Pipeline

This repository contains a complete pipeline for loading, processing, and visualizing high-resolution microscopy data from a double diSPIM (dual-view selective plane illumination microscopy) system.

## Overview

The data consists of:
- **Two imaging arms**: Alpha and Beta (each diSPIM)
- **Two channels per arm**: Two cameras/sides per diSPIM
- **Z-stacks**: 200 slices per volume
- **High resolution**: 2304×2304 pixels per slice
- **Format**: OME-TIFF files with detailed JSON metadata

## Setup

### 1. Create Conda Environment

First, create the conda environment with all necessary packages:

```bash
conda env create -f environment.yml
```

This will create an environment named `dispim_visualization` with all required dependencies.

### 2. Activate Environment

```bash
conda activate dispim_visualization
```

### 3. Launch Jupyter Notebook

```bash
jupyter notebook
```

Then open `dispim_visualization.ipynb` in your browser.

## Usage

### Basic Workflow

1. **Run the notebook cells in order** - The notebook is organized into sections:
   - Section 1: Setup and Imports
   - Section 2: Metadata Parser
   - Section 3: OME-TIFF Loader
   - Section 4: Data Discovery
   - Section 5: Temporal Alignment
   - Section 6: Spatial Information Extraction
   - Section 7: Video Creation Pipeline
   - Section 8: Interactive Visualization
   - Section 9: Example Usage

2. **Select an acquisition** - The notebook will automatically discover all alpha/beta pairs in your directory structure. Change `ACQUISITION_INDEX` to select a different acquisition.

3. **Load data** - Adjust `MAX_SLICES_FOR_DISPLAY` to control how many slices to load (for memory efficiency during initial exploration).

4. **Explore interactively** - Use the widgets to browse through slices and channels.

5. **Create videos** - Generate side-by-side video files for easy viewing.

### Key Functions

#### Metadata Parsing
- `parse_metadata(metadata_path)`: Parse JSON metadata files to extract acquisition parameters

#### Data Loading
- `load_ome_tiff(tiff_path, metadata=None, channel_idx=None, max_slices=None)`: Load OME-TIFF files with proper handling of multi-dimensional arrays

#### Data Discovery
- `discover_acquisitions(root_dir='.')`: Automatically find all alpha/beta acquisition pairs

#### Temporal Alignment
- `calculate_temporal_alignment(alpha_meta, beta_meta)`: Calculate timing offsets and frame alignment between arms

#### Spatial Information
- `extract_spatial_info(alpha_meta, beta_meta)`: Extract spatial calibration and position information

#### Video Creation
- `create_video_from_stacks(alpha_data, beta_data, output_path, ...)`: Create side-by-side video files

#### Interactive Visualization
- `InteractiveViewer`: Class for interactive exploration with widgets

## Data Structure

The data is organized as follows:

```
root_directory/
├── condition_folder/          # e.g., "1msec_worm", "20msec_worm"
│   ├── run_folder/            # e.g., "I", "II", "III"
│   │   ├── beads_alpha_*/     # Alpha arm data
│   │   │   ├── *.ome.tif      # Image stack
│   │   │   ├── *_metadata.txt # Metadata
│   │   │   └── AcqSettings.txt
│   │   └── beads_beta_*/      # Beta arm data
│   │       ├── *.ome.tif      # Image stack
│   │       ├── *_metadata.txt # Metadata
│   │       └── AcqSettings.txt
```

## Key Metadata Fields

The metadata files contain important information:

- **Dimensions**: Width, Height, Slices, Channels
- **Temporal**: StartTime, SlicePeriod_ms, VolumeDuration
- **Spatial**: PixelSize_um, z-step_um, Position_X/Y
- **Acquisition**: delayBeforeSide, numSides, MVRotations

## Memory Considerations

Full stacks can be large:
- Single slice: 2304 × 2304 × 2 bytes (16-bit) ≈ 10.6 MB
- Full stack: 200 slices × 2 channels ≈ 4.2 GB per arm

For initial exploration, consider:
- Loading subsets of slices (`max_slices` parameter)
- Loading single channels at a time
- Downsampling for visualization

## Output

The pipeline generates:
- **Side-by-side images**: Display alpha and beta channels together
- **Video files**: MP4 videos with proper frame rates based on slice periods
- **Interactive widgets**: Slice and channel selectors for exploration

## Future Extensions

This pipeline provides the foundation for:
1. **Image Registration**: Align/register alpha and beta volumes using spatial information
2. **Data Fusion**: Combine registered data for improved resolution
3. **Segmentation**: Apply ML-based segmentation tools
4. **Multi-timepoint Analysis**: Extend to handle time-lapse data
5. **Advanced Processing**: Filtering, deconvolution, etc.

## Troubleshooting

### Memory Issues
- Reduce `MAX_SLICES_FOR_DISPLAY` to load fewer slices
- Load one channel at a time
- Process data in chunks

### Video Creation Fails
- Ensure `imageio-ffmpeg` is installed (included in environment.yml)
- Check available disk space
- Try reducing number of frames or resolution

### Metadata Parsing Errors
- Check that metadata files are valid JSON
- Verify file paths are correct
- Some metadata fields may be missing - the parser handles this gracefully

## Dependencies

Key packages:
- `tifffile`: OME-TIFF reading
- `numpy`: Array operations
- `matplotlib`: Visualization
- `ipywidgets`: Interactive widgets
- `imageio`: Video creation
- `pandas`: Data organization (optional)

See `environment.yml` for complete list.

## License

This pipeline is provided as-is for research purposes.

## Contact

For questions or issues, please refer to the notebook documentation or contact the data acquisition team.

