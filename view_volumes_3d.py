"""
Napari-based 3D Volumetric Viewer for diSPIM Camera Alignment

Loads all 4 deskewed camera volumes (temporally aligned) and displays them as thin 3D volumes
for spatial alignment. The goal is to find the FIXED spatial transformations for each camera
that align all views to reconstruct the full 3D sample.

Key Concept:
- The 4 cameras have FIXED spatial relationships (they don't move relative to each other)
- Temporal alignment was already done in the processing pipeline
- This tool finds ONE spatial transform per camera that applies to ALL time slices
- Time slider is just for navigation to find frames where alignment is clearly visible
- All cameras can be transformed independently relative to each other

Features:
- 3D rotation around X, Y, Z axes (pitch, yaw, roll)
- 3D translation in X, Y, Z directions
- Transforms are GLOBAL - apply to all time slices
- Optimized slider performance with update-on-release
- Save/load camera transformation parameters
- Interactive 3D visualization with Napari

Usage:
    python view_volumes_3d.py [path_to_deskewed_folder]

If no path is provided, it will look for volumes in:
    ./processed_output/{latest_acquisition}/deskewed/
"""

# NumPy 2.0 compatibility
import numpy as np
if not hasattr(np, 'unicode_'):
    np.unicode_ = np.str_

import napari
from napari.utils.notifications import show_info
from pathlib import Path
import tifffile
import sys
import json
from qtpy.QtWidgets import (QWidget, QVBoxLayout, QPushButton, QLabel,
                            QFileDialog, QMessageBox, QSlider, QHBoxLayout,
                            QDoubleSpinBox, QComboBox, QGroupBox, QScrollArea)
from qtpy.QtCore import Qt, QTimer
import math


def find_deskewed_volumes(base_dir=None):
    """Find deskewed volumes in the output directory"""
    if base_dir is None:
        # Look for the most recent acquisition
        processed_dir = Path('./processed_output')
        if not processed_dir.exists():
            raise FileNotFoundError("No processed_output directory found. Run dispim_pipeline.ipynb first.")

        # Find all acquisition directories
        acquisition_dirs = [d for d in processed_dir.iterdir() if d.is_dir() and not d.name.startswith('.')]
        if not acquisition_dirs:
            raise FileNotFoundError("No acquisition directories found in processed_output.")

        # Use the first one (or most recent)
        base_dir = acquisition_dirs[0] / 'deskewed'
    else:
        base_dir = Path(base_dir)

    if not base_dir.exists():
        raise FileNotFoundError(f"Deskewed directory not found: {base_dir}")

    # Find all deskewed TIFF files
    tiff_files = list(base_dir.glob('*_deskewed.tif'))

    if not tiff_files:
        raise FileNotFoundError(f"No deskewed TIFF files found in {base_dir}")

    volumes = {}
    metadata = {}

    for tiff_file in tiff_files:
        # Parse filename: arm_camera_deskewed.tif
        name_parts = tiff_file.stem.replace('_deskewed', '').split('_', 1)
        if len(name_parts) == 2:
            arm, camera = name_parts
            display_name = f"{arm.capitalize()} {camera}"
        else:
            display_name = tiff_file.stem

        print(f"Loading {display_name}...")
        vol_data = tifffile.imread(tiff_file)
        volumes[display_name] = vol_data
        print(f"  Shape: {vol_data.shape}, dtype: {vol_data.dtype}")

        # Try to load metadata
        meta_file = tiff_file.with_suffix('.json')
        if meta_file.exists():
            with open(meta_file, 'r') as f:
                metadata[display_name] = json.load(f)

    return volumes, metadata, base_dir


def create_thin_volume(slice_2d, depth=10):
    """Convert 2D slice to thin 3D volume by repeating along Z axis

    Parameters:
    -----------
    slice_2d : numpy.ndarray
        2D image with shape (H, W)
    depth : int
        Number of slices in Z dimension (default: 10)

    Returns:
    --------
    numpy.ndarray : 3D volume with shape (depth, H, W)
    """
    # Repeat the 2D slice along the Z axis to create a thin volume
    return np.repeat(slice_2d[np.newaxis, :, :], depth, axis=0)


def build_3d_transform(rx_deg, ry_deg, rz_deg, tx, ty, tz, center):
    """Build 4x4 affine transform matrix for 3D rotation + translation

    Rotation order: Z -> Y -> X (roll -> yaw -> pitch)

    Parameters:
    -----------
    rx_deg : float
        Rotation around X axis in degrees (pitch: tilt forward/backward)
    ry_deg : float
        Rotation around Y axis in degrees (yaw: tilt left/right)
    rz_deg : float
        Rotation around Z axis in degrees (roll: spin in-plane)
    tx, ty, tz : float
        Translation in X, Y, Z directions
    center : tuple
        Center point (cz, cy, cx) for rotation

    Returns:
    --------
    numpy.ndarray : 4x4 affine transformation matrix

    Transform order:
    1. Translate to origin: T(-center)
    2. Rotate Z (in-plane): Rz
    3. Rotate Y (tilt left/right): Ry
    4. Rotate X (tilt forward/backward): Rx
    5. Translate back: T(center)
    6. Apply user translation: T(tx, ty, tz)
    """
    # Convert to radians
    rx = np.radians(rx_deg)
    ry = np.radians(ry_deg)
    rz = np.radians(rz_deg)

    # Build rotation matrices (4x4)
    Rx = np.array([
        [1, 0, 0, 0],
        [0, np.cos(rx), -np.sin(rx), 0],
        [0, np.sin(rx), np.cos(rx), 0],
        [0, 0, 0, 1]
    ])

    Ry = np.array([
        [np.cos(ry), 0, np.sin(ry), 0],
        [0, 1, 0, 0],
        [-np.sin(ry), 0, np.cos(ry), 0],
        [0, 0, 0, 1]
    ])

    Rz = np.array([
        [np.cos(rz), -np.sin(rz), 0, 0],
        [np.sin(rz), np.cos(rz), 0, 0],
        [0, 0, 1, 0],
        [0, 0, 0, 1]
    ])

    # Center translations
    cz, cy, cx = center
    T_neg = np.eye(4)
    T_neg[:3, 3] = [-cz, -cy, -cx]

    T_pos = np.eye(4)
    T_pos[:3, 3] = [cz, cy, cx]

    # User translation (Napari uses Z, Y, X order)
    T_user = np.eye(4)
    T_user[:3, 3] = [tz, ty, tx]

    # Compose: T_user @ T_pos @ Rx @ Ry @ Rz @ T_neg
    transform = T_user @ T_pos @ Rx @ Ry @ Rz @ T_neg

    return transform.astype(np.float64)


class VolumetricViewerControls(QWidget):
    """Widget for 3D volumetric alignment controls"""

    def __init__(self, viewer, volumes, metadata, base_dir):
        super().__init__()
        self.viewer = viewer
        self.volumes = volumes
        self.metadata = metadata
        self.base_dir = Path(base_dir)
        self.current_time_idx = 0
        self.layers = {}  # Store layer references

        # Store transform parameters per layer: {layer_name: {'rx': 0, ...}}
        # These transforms are FIXED for each camera and apply to ALL time slices
        self.transform_params = {}

        # Volume depth for thin 3D volumes (configurable)
        self.volume_depth = 10

        # Determine number of time slices (should be same for all volumes)
        if volumes:
            first_vol = list(volumes.values())[0]
            self.num_time_slices = first_vol.shape[0]
        else:
            self.num_time_slices = 0

        # Debounce timer for spinbox updates
        self.update_timer = QTimer()
        self.update_timer.setSingleShot(True)
        self.update_timer.setInterval(100)  # 100ms delay
        self.update_timer.timeout.connect(self.apply_current_transform)

        self.setup_ui()
        self.setup_connections()
        self.update_time_slice()

    def setup_ui(self):
        """Create UI elements"""
        # Create scroll area for controls
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)

        # Main widget inside scroll area
        scroll_widget = QWidget()
        layout = QVBoxLayout()
        layout.setSpacing(10)

        # Title
        title = QLabel("3D Volumetric Alignment")
        title.setStyleSheet("font-weight: bold; font-size: 14px;")
        layout.addWidget(title)

        # Status label for transform feedback
        self.status_label = QLabel("")
        self.status_label.setStyleSheet("color: #4CAF50; font-size: 10px; font-style: italic;")
        self.status_label.setWordWrap(True)
        layout.addWidget(self.status_label)

        # === Time Slice Selection ===
        slice_group = QGroupBox("Time Slice Selection")
        slice_layout = QVBoxLayout()

        self.slice_label = QLabel(f"Time Index: {self.current_time_idx} / {self.num_time_slices - 1}")
        slice_layout.addWidget(self.slice_label)

        self.slice_slider = QSlider(Qt.Horizontal)
        self.slice_slider.setMinimum(0)
        self.slice_slider.setMaximum(max(0, self.num_time_slices - 1))
        self.slice_slider.setValue(0)
        self.slice_slider.setTickPosition(QSlider.TicksBelow)
        self.slice_slider.setTickInterval(max(1, self.num_time_slices // 10))
        slice_layout.addWidget(self.slice_slider)

        nav_layout = QHBoxLayout()
        self.prev_btn = QPushButton("◄ Previous")
        self.next_btn = QPushButton("Next ►")
        nav_layout.addWidget(self.prev_btn)
        nav_layout.addWidget(self.next_btn)
        slice_layout.addLayout(nav_layout)

        slice_group.setLayout(slice_layout)
        layout.addWidget(slice_group)

        # === Layer Selection ===
        layer_group = QGroupBox("Layer Selection")
        layer_layout = QVBoxLayout()

        self.layer_combo = QComboBox()
        layer_layout.addWidget(QLabel("Select layer to transform:"))
        layer_layout.addWidget(self.layer_combo)

        layer_group.setLayout(layer_layout)
        layout.addWidget(layer_group)

        # === 3D Rotation Controls ===
        rotation_group = QGroupBox("3D Rotation")
        rotation_layout = QVBoxLayout()

        # Rotation X (Pitch)
        rotation_layout.addWidget(QLabel("X-axis (Pitch - tilt forward/back):"))
        rx_layout = QHBoxLayout()
        self.rotation_x_slider = QSlider(Qt.Horizontal)
        self.rotation_x_slider.setMinimum(-1800)
        self.rotation_x_slider.setMaximum(1800)
        self.rotation_x_slider.setValue(0)
        self.rotation_x_slider.setTickPosition(QSlider.TicksBelow)
        self.rotation_x_slider.setTickInterval(450)
        self.rotation_x_spinbox = QDoubleSpinBox()
        self.rotation_x_spinbox.setMinimum(-180.0)
        self.rotation_x_spinbox.setMaximum(180.0)
        self.rotation_x_spinbox.setValue(0.0)
        self.rotation_x_spinbox.setSingleStep(1.0)
        self.rotation_x_spinbox.setSuffix("°")
        rx_layout.addWidget(self.rotation_x_slider, stretch=3)
        rx_layout.addWidget(self.rotation_x_spinbox, stretch=1)
        rotation_layout.addLayout(rx_layout)

        # Rotation Y (Yaw)
        rotation_layout.addWidget(QLabel("Y-axis (Yaw - tilt left/right):"))
        ry_layout = QHBoxLayout()
        self.rotation_y_slider = QSlider(Qt.Horizontal)
        self.rotation_y_slider.setMinimum(-1800)
        self.rotation_y_slider.setMaximum(1800)
        self.rotation_y_slider.setValue(0)
        self.rotation_y_slider.setTickPosition(QSlider.TicksBelow)
        self.rotation_y_slider.setTickInterval(450)
        self.rotation_y_spinbox = QDoubleSpinBox()
        self.rotation_y_spinbox.setMinimum(-180.0)
        self.rotation_y_spinbox.setMaximum(180.0)
        self.rotation_y_spinbox.setValue(0.0)
        self.rotation_y_spinbox.setSingleStep(1.0)
        self.rotation_y_spinbox.setSuffix("°")
        ry_layout.addWidget(self.rotation_y_slider, stretch=3)
        ry_layout.addWidget(self.rotation_y_spinbox, stretch=1)
        rotation_layout.addLayout(ry_layout)

        # Rotation Z (Roll)
        rotation_layout.addWidget(QLabel("Z-axis (Roll - spin in-plane):"))
        rz_layout = QHBoxLayout()
        self.rotation_z_slider = QSlider(Qt.Horizontal)
        self.rotation_z_slider.setMinimum(-1800)
        self.rotation_z_slider.setMaximum(1800)
        self.rotation_z_slider.setValue(0)
        self.rotation_z_slider.setTickPosition(QSlider.TicksBelow)
        self.rotation_z_slider.setTickInterval(450)
        self.rotation_z_spinbox = QDoubleSpinBox()
        self.rotation_z_spinbox.setMinimum(-180.0)
        self.rotation_z_spinbox.setMaximum(180.0)
        self.rotation_z_spinbox.setValue(0.0)
        self.rotation_z_spinbox.setSingleStep(1.0)
        self.rotation_z_spinbox.setSuffix("°")
        rz_layout.addWidget(self.rotation_z_slider, stretch=3)
        rz_layout.addWidget(self.rotation_z_spinbox, stretch=1)
        rotation_layout.addLayout(rz_layout)

        rotation_group.setLayout(rotation_layout)
        layout.addWidget(rotation_group)

        # === 3D Translation Controls ===
        translation_group = QGroupBox("3D Translation")
        translation_layout = QVBoxLayout()

        # Translation X
        translation_layout.addWidget(QLabel("X-axis:"))
        tx_layout = QHBoxLayout()
        self.translate_x_slider = QSlider(Qt.Horizontal)
        self.translate_x_slider.setMinimum(-5000)
        self.translate_x_slider.setMaximum(5000)
        self.translate_x_slider.setValue(0)
        self.translate_x_spinbox = QDoubleSpinBox()
        self.translate_x_spinbox.setMinimum(-5000.0)
        self.translate_x_spinbox.setMaximum(5000.0)
        self.translate_x_spinbox.setValue(0.0)
        self.translate_x_spinbox.setSingleStep(10.0)
        self.translate_x_spinbox.setSuffix(" px")
        tx_layout.addWidget(self.translate_x_slider, stretch=3)
        tx_layout.addWidget(self.translate_x_spinbox, stretch=1)
        translation_layout.addLayout(tx_layout)

        # Translation Y
        translation_layout.addWidget(QLabel("Y-axis:"))
        ty_layout = QHBoxLayout()
        self.translate_y_slider = QSlider(Qt.Horizontal)
        self.translate_y_slider.setMinimum(-5000)
        self.translate_y_slider.setMaximum(5000)
        self.translate_y_slider.setValue(0)
        self.translate_y_spinbox = QDoubleSpinBox()
        self.translate_y_spinbox.setMinimum(-5000.0)
        self.translate_y_spinbox.setMaximum(5000.0)
        self.translate_y_spinbox.setValue(0.0)
        self.translate_y_spinbox.setSingleStep(10.0)
        self.translate_y_spinbox.setSuffix(" px")
        ty_layout.addWidget(self.translate_y_slider, stretch=3)
        ty_layout.addWidget(self.translate_y_spinbox, stretch=1)
        translation_layout.addLayout(ty_layout)

        # Translation Z
        translation_layout.addWidget(QLabel("Z-axis:"))
        tz_layout = QHBoxLayout()
        self.translate_z_slider = QSlider(Qt.Horizontal)
        self.translate_z_slider.setMinimum(-100)
        self.translate_z_slider.setMaximum(100)
        self.translate_z_slider.setValue(0)
        self.translate_z_spinbox = QDoubleSpinBox()
        self.translate_z_spinbox.setMinimum(-100.0)
        self.translate_z_spinbox.setMaximum(100.0)
        self.translate_z_spinbox.setValue(0.0)
        self.translate_z_spinbox.setSingleStep(1.0)
        self.translate_z_spinbox.setSuffix(" slices")
        tz_layout.addWidget(self.translate_z_slider, stretch=3)
        tz_layout.addWidget(self.translate_z_spinbox, stretch=1)
        translation_layout.addLayout(tz_layout)

        translation_group.setLayout(translation_layout)
        layout.addWidget(translation_group)

        # === Control Buttons ===
        button_layout = QVBoxLayout()

        self.reset_current_btn = QPushButton("Reset Current Layer")
        self.reset_current_btn.setToolTip("Reset currently selected layer to identity transform")
        button_layout.addWidget(self.reset_current_btn)

        self.save_btn = QPushButton("Save Transformations")
        self.save_btn.setToolTip("Save all transformation parameters to JSON")
        button_layout.addWidget(self.save_btn)

        self.load_btn = QPushButton("Load Transformations")
        self.load_btn.setToolTip("Load transformation parameters from JSON")
        button_layout.addWidget(self.load_btn)

        layout.addLayout(button_layout)

        # === Info Section ===
        info_label = QLabel(
            "<b>How it works:</b><br>"
            "• Cameras have FIXED spatial relationships<br>"
            "• Time slider = navigate to find good frame<br>"
            "• Transforms apply to ALL time slices<br>"
            "• All cameras can be transformed independently<br><br>"
            "<b>Napari 3D view:</b><br>"
            "  - Left-drag: Rotate view<br>"
            "  - Right-drag: Pan view<br>"
            "  - Scroll: Zoom"
        )
        info_label.setWordWrap(True)
        info_label.setStyleSheet("color: gray; font-size: 10px; padding: 10px;")
        layout.addWidget(info_label)

        layout.addStretch()
        scroll_widget.setLayout(layout)
        scroll.setWidget(scroll_widget)

        # Set main layout
        main_layout = QVBoxLayout()
        main_layout.addWidget(scroll)
        self.setLayout(main_layout)

    def setup_connections(self):
        """Connect UI elements to functions"""
        # Time slice navigation
        self.slice_slider.valueChanged.connect(self.on_time_slice_changed)
        self.prev_btn.clicked.connect(self.prev_time_slice)
        self.next_btn.clicked.connect(self.next_time_slice)

        # Layer selection
        self.layer_combo.currentTextChanged.connect(self.on_layer_selected)

        # Rotation controls - sliders update display on change, apply on release
        # Spinboxes: apply on Enter/focus loss (editingFinished)
        self.rotation_x_slider.valueChanged.connect(self.on_rotation_x_slider_display)
        self.rotation_x_slider.sliderReleased.connect(self.on_slider_released)
        self.rotation_x_spinbox.valueChanged.connect(self.on_rotation_x_spinbox_changed)
        self.rotation_x_spinbox.editingFinished.connect(self.apply_current_transform)

        self.rotation_y_slider.valueChanged.connect(self.on_rotation_y_slider_display)
        self.rotation_y_slider.sliderReleased.connect(self.on_slider_released)
        self.rotation_y_spinbox.valueChanged.connect(self.on_rotation_y_spinbox_changed)
        self.rotation_y_spinbox.editingFinished.connect(self.apply_current_transform)

        self.rotation_z_slider.valueChanged.connect(self.on_rotation_z_slider_display)
        self.rotation_z_slider.sliderReleased.connect(self.on_slider_released)
        self.rotation_z_spinbox.valueChanged.connect(self.on_rotation_z_spinbox_changed)
        self.rotation_z_spinbox.editingFinished.connect(self.apply_current_transform)

        # Translation controls - sliders update display on change, apply on release
        # Spinboxes: apply on Enter/focus loss (editingFinished)
        self.translate_x_slider.valueChanged.connect(self.on_translate_x_slider_display)
        self.translate_x_slider.sliderReleased.connect(self.on_slider_released)
        self.translate_x_spinbox.valueChanged.connect(self.on_translate_x_spinbox_changed)
        self.translate_x_spinbox.editingFinished.connect(self.apply_current_transform)

        self.translate_y_slider.valueChanged.connect(self.on_translate_y_slider_display)
        self.translate_y_slider.sliderReleased.connect(self.on_slider_released)
        self.translate_y_spinbox.valueChanged.connect(self.on_translate_y_spinbox_changed)
        self.translate_y_spinbox.editingFinished.connect(self.apply_current_transform)

        self.translate_z_slider.valueChanged.connect(self.on_translate_z_slider_display)
        self.translate_z_slider.sliderReleased.connect(self.on_slider_released)
        self.translate_z_spinbox.valueChanged.connect(self.on_translate_z_spinbox_changed)
        self.translate_z_spinbox.editingFinished.connect(self.apply_current_transform)

        # Buttons
        self.reset_current_btn.clicked.connect(self.reset_current_layer)
        self.save_btn.clicked.connect(self.save_transforms)
        self.load_btn.clicked.connect(self.load_transforms)

    def on_time_slice_changed(self, value):
        """Handle time slice slider change"""
        print(f"\n{'='*60}")
        print(f"[TIME_CHANGE] Changing from time {self.current_time_idx} to {value}")
        print(f"{'='*60}")

        # Save current layer's params before switching (transforms are global but need to be captured)
        self.save_current_layer_params()

        self.current_time_idx = value
        self.slice_label.setText(f"Time Index: {self.current_time_idx} / {self.num_time_slices - 1}")
        self.update_time_slice()

    def prev_time_slice(self):
        """Go to previous time slice"""
        if self.current_time_idx > 0:
            self.slice_slider.setValue(self.current_time_idx - 1)

    def next_time_slice(self):
        """Go to next time slice"""
        if self.current_time_idx < self.num_time_slices - 1:
            self.slice_slider.setValue(self.current_time_idx + 1)

    def update_time_slice(self):
        """Update displayed volumes for current time slice"""
        # Clear existing layers
        for layer_name in list(self.layers.keys()):
            if layer_name in self.viewer.layers:
                self.viewer.layers.remove(self.viewer.layers[layer_name])
        self.layers.clear()

        if self.num_time_slices == 0:
            return

        # Color scheme
        colors = {
            'Alpha HamCam2': 'red',
            'Alpha HamCam1': 'green',
            'Beta HamCam2': 'blue',
            'Beta HamCam1': 'yellow',
            'Beta HamuHam4': 'blue',
            'Beta HamuHam3': 'yellow',
        }
        color_list = ['red', 'green', 'blue', 'yellow']

        # Extract 2D slice from each volume and create thin 3D volume
        for i, (name, vol) in enumerate(self.volumes.items()):
            if self.current_time_idx >= vol.shape[0]:
                continue

            # Extract 2D slice at current time
            slice_2d = vol[self.current_time_idx, :, :]

            # Create thin 3D volume
            thin_volume = create_thin_volume(slice_2d, depth=self.volume_depth)

            # Get color
            color = colors.get(name, color_list[i % len(color_list)])

            # Add as 3D image layer
            layer = self.viewer.add_image(
                thin_volume,
                name=name,
                colormap=color,
                opacity=0.7,
                blending='additive',
                contrast_limits=(thin_volume.min(), thin_volume.max())
            )

            # Store layer reference
            self.layers[name] = layer

            # Set up identity transform initially
            if hasattr(layer, 'affine'):
                layer.affine.affine_matrix = np.eye(4)

        # Force 3D rendering
        self.viewer.dims.ndisplay = 3

        # Update layer combo
        self.update_layer_combo()

        # Restore transforms for this time slice
        self.restore_all_transforms()

        # Select first non-reference layer
        if self.layer_combo.count() > 1:
            self.layer_combo.setCurrentIndex(1)
        elif self.layer_combo.count() > 0:
            self.layer_combo.setCurrentIndex(0)

    def update_layer_combo(self):
        """Update layer selection combo box"""
        self.layer_combo.blockSignals(True)
        self.layer_combo.clear()
        for layer_name in self.layers.keys():
            self.layer_combo.addItem(layer_name)
        self.layer_combo.blockSignals(False)

    def get_current_layer_name(self):
        """Get current layer name from combo box"""
        return self.layer_combo.currentText()

    def get_current_layer(self):
        """Get currently selected layer object"""
        name = self.get_current_layer_name()
        return self.layers.get(name)

    def on_layer_selected(self, text):
        """Handle layer selection change"""
        print(f"\n[LAYER_SELECT] Layer selected: '{text}'")

        # Save previous layer's params
        self.save_current_layer_params()

        layer_name = self.get_current_layer_name()
        print(f"[LAYER_SELECT] Clean layer name: '{layer_name}'")

        # Restore params if they exist (transforms are global, not per time)
        if layer_name in self.transform_params:
            params = self.transform_params[layer_name]
            print(f"[LAYER_SELECT] Found params for '{layer_name}': rz={params['rz']}")
            self.set_controls_from_params(params)
        else:
            print(f"[LAYER_SELECT] No params for '{layer_name}', initializing to zero")
            # Initialize to zero
            self.set_controls_from_params({
                'rx': 0, 'ry': 0, 'rz': 0,
                'tx': 0, 'ty': 0, 'tz': 0
            })

        # Apply transform
        print(f"[LAYER_SELECT] Applying transform to '{layer_name}'")
        self.apply_current_transform()

    def set_controls_from_params(self, params):
        """Set UI controls from parameter dict"""
        # Block signals
        self.rotation_x_slider.blockSignals(True)
        self.rotation_x_spinbox.blockSignals(True)
        self.rotation_y_slider.blockSignals(True)
        self.rotation_y_spinbox.blockSignals(True)
        self.rotation_z_slider.blockSignals(True)
        self.rotation_z_spinbox.blockSignals(True)
        self.translate_x_slider.blockSignals(True)
        self.translate_x_spinbox.blockSignals(True)
        self.translate_y_slider.blockSignals(True)
        self.translate_y_spinbox.blockSignals(True)
        self.translate_z_slider.blockSignals(True)
        self.translate_z_spinbox.blockSignals(True)

        # Set values
        self.rotation_x_slider.setValue(int(params['rx'] * 10))
        self.rotation_x_spinbox.setValue(params['rx'])
        self.rotation_y_slider.setValue(int(params['ry'] * 10))
        self.rotation_y_spinbox.setValue(params['ry'])
        self.rotation_z_slider.setValue(int(params['rz'] * 10))
        self.rotation_z_spinbox.setValue(params['rz'])
        self.translate_x_slider.setValue(int(params['tx']))
        self.translate_x_spinbox.setValue(params['tx'])
        self.translate_y_slider.setValue(int(params['ty']))
        self.translate_y_spinbox.setValue(params['ty'])
        self.translate_z_slider.setValue(int(params['tz']))
        self.translate_z_spinbox.setValue(params['tz'])

        # Unblock signals
        self.rotation_x_slider.blockSignals(False)
        self.rotation_x_spinbox.blockSignals(False)
        self.rotation_y_slider.blockSignals(False)
        self.rotation_y_spinbox.blockSignals(False)
        self.rotation_z_slider.blockSignals(False)
        self.rotation_z_spinbox.blockSignals(False)
        self.translate_x_slider.blockSignals(False)
        self.translate_x_spinbox.blockSignals(False)
        self.translate_y_slider.blockSignals(False)
        self.translate_y_spinbox.blockSignals(False)
        self.translate_z_slider.blockSignals(False)
        self.translate_z_spinbox.blockSignals(False)

    def save_current_layer_params(self):
        """Save current layer's transform parameters (global across all time)"""
        layer_name = self.get_current_layer_name()
        if layer_name:
            self.transform_params[layer_name] = {
                'rx': self.rotation_x_spinbox.value(),
                'ry': self.rotation_y_spinbox.value(),
                'rz': self.rotation_z_spinbox.value(),
                'tx': self.translate_x_spinbox.value(),
                'ty': self.translate_y_spinbox.value(),
                'tz': self.translate_z_spinbox.value()
            }
            print(f"[SAVE] Saved params for '{layer_name}': rz={self.rotation_z_spinbox.value()}")
            print(f"[SAVE] transform_params now has {len(self.transform_params)} entries: {list(self.transform_params.keys())}")

    def restore_all_transforms(self):
        """Restore saved transforms for all layers (global across all time)"""
        print(f"[RESTORE] Restoring transforms for {len(self.layers)} layers")
        print(f"[RESTORE] Available params for: {list(self.transform_params.keys())}")
        for layer_name, layer in self.layers.items():
            if layer_name in self.transform_params:
                params = self.transform_params[layer_name]
                print(f"[RESTORE] Applying to '{layer_name}': rz={params['rz']}")
                self.apply_transform_to_layer(
                    layer, params['rx'], params['ry'], params['rz'],
                    params['tx'], params['ty'], params['tz']
                )
            else:
                print(f"[RESTORE] No params found for '{layer_name}'")

    def apply_transform_to_layer(self, layer, rx, ry, rz, tx, ty, tz):
        """Apply 3D transformation to a layer"""
        if not hasattr(layer, 'affine') or not hasattr(layer, 'data'):
            return

        # Get volume center
        depth, height, width = layer.data.shape
        center = (depth / 2.0, height / 2.0, width / 2.0)

        # Build 3D transform matrix
        transform = build_3d_transform(rx, ry, rz, tx, ty, tz, center)

        # Apply to layer and force visual refresh
        layer.affine.affine_matrix = transform

        # Force Napari to refresh the layer visually
        layer.refresh()  # Trigger re-render
        self.viewer.camera.center = self.viewer.camera.center  # Force viewer update

    def apply_current_transform(self):
        """Apply current transform values to selected layer"""
        layer = self.get_current_layer()
        if layer:
            self.status_label.setText("⏳ Applying transform...")
            rx = self.rotation_x_spinbox.value()
            ry = self.rotation_y_spinbox.value()
            rz = self.rotation_z_spinbox.value()
            tx = self.translate_x_spinbox.value()
            ty = self.translate_y_spinbox.value()
            tz = self.translate_z_spinbox.value()
            self.apply_transform_to_layer(layer, rx, ry, rz, tx, ty, tz)

            # Auto-save params after applying (ensures persistence)
            self.save_current_layer_params()

            self.status_label.setText("✓ Transform applied")
            # Clear status after 1 second
            QTimer.singleShot(1000, lambda: self.status_label.setText(""))

    # === OPTIMIZED SLIDER/SPINBOX HANDLERS ===

    def on_slider_released(self):
        """Called when any slider is released - apply transform"""
        self.apply_current_transform()

    def debounced_apply(self):
        """Restart debounce timer - apply transform after delay"""
        self.update_timer.stop()
        self.update_timer.start()

    # Rotation X handlers
    def on_rotation_x_slider_display(self, value):
        """Update spinbox display only (no transform)"""
        self.rotation_x_spinbox.blockSignals(True)
        self.rotation_x_spinbox.setValue(value / 10.0)
        self.rotation_x_spinbox.blockSignals(False)

    def on_rotation_x_spinbox_changed(self, value):
        """Update slider and debounce transform"""
        self.rotation_x_slider.blockSignals(True)
        self.rotation_x_slider.setValue(int(value * 10))
        self.rotation_x_slider.blockSignals(False)
        self.debounced_apply()

    # Rotation Y handlers
    def on_rotation_y_slider_display(self, value):
        """Update spinbox display only (no transform)"""
        self.rotation_y_spinbox.blockSignals(True)
        self.rotation_y_spinbox.setValue(value / 10.0)
        self.rotation_y_spinbox.blockSignals(False)

    def on_rotation_y_spinbox_changed(self, value):
        """Update slider and debounce transform"""
        self.rotation_y_slider.blockSignals(True)
        self.rotation_y_slider.setValue(int(value * 10))
        self.rotation_y_slider.blockSignals(False)
        self.debounced_apply()

    # Rotation Z handlers
    def on_rotation_z_slider_display(self, value):
        """Update spinbox display only (no transform)"""
        self.rotation_z_spinbox.blockSignals(True)
        self.rotation_z_spinbox.setValue(value / 10.0)
        self.rotation_z_spinbox.blockSignals(False)

    def on_rotation_z_spinbox_changed(self, value):
        """Update slider and debounce transform"""
        self.rotation_z_slider.blockSignals(True)
        self.rotation_z_slider.setValue(int(value * 10))
        self.rotation_z_slider.blockSignals(False)
        self.debounced_apply()

    # Translation X handlers
    def on_translate_x_slider_display(self, value):
        """Update spinbox display only (no transform)"""
        self.translate_x_spinbox.blockSignals(True)
        self.translate_x_spinbox.setValue(float(value))
        self.translate_x_spinbox.blockSignals(False)

    def on_translate_x_spinbox_changed(self, value):
        """Update slider and debounce transform"""
        self.translate_x_slider.blockSignals(True)
        self.translate_x_slider.setValue(int(value))
        self.translate_x_slider.blockSignals(False)
        self.debounced_apply()

    # Translation Y handlers
    def on_translate_y_slider_display(self, value):
        """Update spinbox display only (no transform)"""
        self.translate_y_spinbox.blockSignals(True)
        self.translate_y_spinbox.setValue(float(value))
        self.translate_y_spinbox.blockSignals(False)

    def on_translate_y_spinbox_changed(self, value):
        """Update slider and debounce transform"""
        self.translate_y_slider.blockSignals(True)
        self.translate_y_slider.setValue(int(value))
        self.translate_y_slider.blockSignals(False)
        self.debounced_apply()

    # Translation Z handlers
    def on_translate_z_slider_display(self, value):
        """Update spinbox display only (no transform)"""
        self.translate_z_spinbox.blockSignals(True)
        self.translate_z_spinbox.setValue(float(value))
        self.translate_z_spinbox.blockSignals(False)

    def on_translate_z_spinbox_changed(self, value):
        """Update slider and debounce transform"""
        self.translate_z_slider.blockSignals(True)
        self.translate_z_slider.setValue(int(value))
        self.translate_z_slider.blockSignals(False)
        self.debounced_apply()

    def reset_current_layer(self):
        """Reset currently selected layer to identity"""
        layer = self.get_current_layer()
        if layer:
            # Reset transform
            if hasattr(layer, 'affine'):
                layer.affine.affine_matrix = np.eye(4)

            # Reset controls
            self.set_controls_from_params({
                'rx': 0, 'ry': 0, 'rz': 0,
                'tx': 0, 'ty': 0, 'tz': 0
            })

            show_info(f"Reset {self.get_current_layer_name()} to identity transform.")

    def save_transforms(self):
        """Save all transformation parameters to JSON file"""
        # Save current layer params first
        self.save_current_layer_params()

        if not self.transform_params:
            QMessageBox.warning(self, "No Transforms", "No transformation parameters found.")
            return

        # Convert to serializable format
        # Transforms are camera-specific and apply to ALL time slices
        save_data = {
            'volume_depth': self.volume_depth,
            'transforms': self.transform_params.copy(),
            'description': 'Fixed spatial transformations for each camera (apply to all time slices)'
        }

        # Get save location
        default_path = self.base_dir.parent / f'camera_transforms.json'
        file_path, _ = QFileDialog.getSaveFileName(
            self, "Save Camera Transformation Parameters",
            str(default_path), "JSON Files (*.json)"
        )

        if file_path:
            try:
                with open(file_path, 'w') as f:
                    json.dump(save_data, f, indent=2)
                show_info(f"Camera transformations saved to:\n{file_path}")
                print(f"Saved {len(self.transform_params)} camera transformations to: {file_path}")
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Failed to save transforms:\n{str(e)}")

    def load_transforms(self):
        """Load transformation parameters from JSON file"""
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Load Camera Transformation Parameters",
            str(self.base_dir.parent), "JSON Files (*.json)"
        )

        if file_path:
            try:
                with open(file_path, 'r') as f:
                    save_data = json.load(f)

                # Load transforms (camera-specific, apply to all time)
                self.transform_params.clear()
                for layer_name, params in save_data.get('transforms', {}).items():
                    self.transform_params[layer_name] = params

                # Restore transforms to all layers
                self.restore_all_transforms()

                # Update current layer controls
                layer_name = self.get_current_layer_name()
                if layer_name in self.transform_params:
                    self.set_controls_from_params(self.transform_params[layer_name])

                show_info(f"Loaded {len(self.transform_params)} camera transformations.")
                print(f"Loaded camera transformations from: {file_path}")
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Failed to load transforms:\n{str(e)}")


def main():
    """Main function to load volumes and start Napari viewer"""
    print("=" * 70)
    print("Napari 3D Volumetric Viewer for diSPIM Alignment")
    print("=" * 70)

    # Get base directory from command line or use default
    if len(sys.argv) > 1:
        base_dir = sys.argv[1]
    else:
        base_dir = None

    try:
        volumes, metadata, vol_dir = find_deskewed_volumes(base_dir)
        print(f"\nFound {len(volumes)} volumes in: {vol_dir}")
    except FileNotFoundError as e:
        print(f"\nError: {e}")
        print("\nPlease run dispim_pipeline.ipynb first to generate deskewed volumes.")
        return

    # Create Napari viewer
    print("\nStarting Napari 3D viewer...")
    viewer = napari.Viewer(title="diSPIM 3D Volumetric Alignment", ndisplay=3)

    # Add transform controls widget
    controls = VolumetricViewerControls(viewer, volumes, metadata, vol_dir)
    viewer.window.add_dock_widget(controls, name="3D Transform Controls", area="right")

    print("\n" + "=" * 70)
    print("3D Viewer Controls:")
    print("=" * 70)
    print("CONCEPT:")
    print("  - Cameras have FIXED spatial relationships (don't move)")
    print("  - Find ONE transform per camera that applies to ALL time")
    print("  - Time slider = just for navigation to find good frames")
    print("  - Goal: Reconstruct full 3D sample from 4 camera views")
    print("\nNAPARI 3D VIEW (rotate your viewing angle):")
    print("  - Left-click + drag: Rotate 3D camera")
    print("  - Right-click + drag: Pan view")
    print("  - Mouse wheel: Zoom in/out")
    print("\nLAYER TRANSFORMATION (align camera volumes):")
    print("  - Use sliders in right panel to transform cameras")
    print("  - X/Y/Z rotation: Tilt camera volumes in 3D space")
    print("  - X/Y/Z translation: Move camera volumes")
    print("\nTIPS:")
    print("  - All cameras can be transformed independently")
    print("  - Transforms are GLOBAL - apply to all time slices")
    print("  - Use layer opacity to see overlaps (left panel)")
    print("  - Save transforms before closing!")
    print("=" * 70)

    # Start interactive viewer
    napari.run()


if __name__ == '__main__':
    main()
