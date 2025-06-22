import sys
import os
import cv2
import numpy as np
import torch
import time
import warnings
import json

warnings.filterwarnings("ignore", category=FutureWarning, module="timm.models.layers")
warnings.filterwarnings("ignore", category=FutureWarning, module="timm.models.registry")
warnings.filterwarnings("ignore", category=UserWarning, module="segment_anything.modeling.tiny_vit_sam")

from PyQt6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout,
                             QHBoxLayout, QPushButton, QLabel, QFileDialog,
                             QGroupBox, QTextEdit, QSpinBox, QCheckBox,
                             QDoubleSpinBox, QFormLayout, QScrollArea, QMessageBox, QComboBox,
                             QProgressBar, QTabWidget)
from PyQt6.QtCore import Qt, QThread, pyqtSignal, QMutex, QTimer
from PyQt6.QtGui import QPixmap, QImage

try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False
    print("Warning: 'psutil' library not found. CPU/RAM monitoring will be unavailable.")

PYNVML_AVAILABLE = False
NVML_INITIALIZED_SUCCESSFULLY = False
try:
    import pynvml
    try:
        pynvml.nvmlInit()
        PYNVML_AVAILABLE = True
        NVML_INITIALIZED_SUCCESSFULLY = True
    except pynvml.NVMLError as e:
        print(f"Warning: 'pynvml' imported but NVML could not be initialized: {e}. GPU monitoring will be unavailable.")
except ImportError:
    print("Warning: 'pynvml' (nvidia-ml-py) library not found. NVIDIA GPU monitoring will be unavailable.")


try:
    from segment_anything import sam_model_registry, SamAutomaticMaskGenerator
    SEGMENT_ANYTHING_AVAILABLE = True
except ImportError as e:
    print("Error: 'segment_anything' library not found.")
    print("Please ensure it and its dependencies (like 'timm') are correctly installed according to the environment setup guide.")
    SEGMENT_ANYTHING_AVAILABLE = False

try:
    from pycocotools import mask as mask_utils
    PYCOCOTOOLS_AVAILABLE = True
except ImportError:
    PYCOCOTOOLS_AVAILABLE = False
    print("Warning: 'pycocotools' library not found. RLE output modes ('uncompressed_rle', 'coco_rle') decoding will be unavailable.")

try:
    import tifffile
except ImportError:
    tifffile = None
    print("Warning: 'tifffile' library not found. TIFF images might not load correctly.")

try:
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
    from matplotlib.figure import Figure
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False
    print("Warning: 'matplotlib' library not found. Single image analysis will be unavailable.")


class SingleImageAnalysisWidget(QWidget):
    def __init__(self):
        super().__init__()
        self.selected_image_path = None
        self.initUI()

    def initUI(self):
        main_layout = QHBoxLayout(self)
        main_layout.setContentsMargins(15, 15, 15, 15)
        main_layout.setSpacing(15)

        left_panel = QWidget()
        left_panel.setMaximumWidth(450)
        left_layout = QVBoxLayout(left_panel)
        left_layout.setContentsMargins(0, 0, 0, 0)
        left_layout.setSpacing(15)

        file_group = QGroupBox("Image Selection")
        file_group.setStyleSheet("QGroupBox { font-weight: bold; }")
        file_layout = QVBoxLayout(file_group)
        file_layout.setContentsMargins(10, 15, 10, 10)
        file_layout.setSpacing(10)

        image_select_layout = QHBoxLayout()
        self.image_path_label = QLabel("No image selected")
        self.select_image_btn = QPushButton("Select Image")
        self.select_image_btn.setStyleSheet(self.get_button_style("green"))
        self.select_image_btn.clicked.connect(self.select_image)

        image_select_layout.addWidget(self.image_path_label, 1)
        image_select_layout.addWidget(self.select_image_btn)
        file_layout.addLayout(image_select_layout)

        analyze_btn_layout = QHBoxLayout()
        self.analyze_btn = QPushButton("Analyze Aggregation Distribution")
        self.analyze_btn.setStyleSheet(self.get_button_style("blue"))
        self.analyze_btn.setEnabled(False)
        self.analyze_btn.clicked.connect(self.analyze_image)
        analyze_btn_layout.addStretch()
        analyze_btn_layout.addWidget(self.analyze_btn)
        analyze_btn_layout.addStretch()
        file_layout.addLayout(analyze_btn_layout)

        left_layout.addWidget(file_group)

        stats_group = QGroupBox("Statistics")
        stats_group.setStyleSheet("QGroupBox { font-weight: bold; }")
        stats_layout = QFormLayout(stats_group)
        stats_layout.setContentsMargins(10, 15, 10, 10)

        self.total_pixels_label = QLabel("--")
        self.mean_aggregate_label = QLabel("--")
        self.min_aggregate_label = QLabel("--")
        self.max_aggregate_label = QLabel("--")
        self.turning_points_label = QLabel("--")
        self.turning_points_label.setWordWrap(True)
        self.turning_points_label.setStyleSheet("QLabel { color: #333333; }")

        stats_layout.addRow("Total Pixels:", self.total_pixels_label)
        stats_layout.addRow("Mean Aggregation:", self.mean_aggregate_label)
        stats_layout.addRow("Min Aggregation:", self.min_aggregate_label)
        stats_layout.addRow("Max Aggregation:", self.max_aggregate_label)
        stats_layout.addRow("Distribution Inflection Points:", self.turning_points_label)

        left_layout.addWidget(stats_group)
        left_layout.addStretch()

        chart_group = QGroupBox("Aggregation Distribution Chart")
        chart_group.setStyleSheet("QGroupBox { font-weight: bold; }")
        chart_layout = QVBoxLayout(chart_group)
        chart_layout.setContentsMargins(10, 15, 10, 10)

        if MATPLOTLIB_AVAILABLE:
            self.figure = Figure(figsize=(10, 6))
            self.canvas = FigureCanvas(self.figure)
            chart_layout.addWidget(self.canvas)

            chart_controls = QHBoxLayout()
            self.save_chart_btn = QPushButton("Save Chart")
            self.save_chart_btn.setStyleSheet(self.get_button_style("orange"))
            self.save_chart_btn.setEnabled(False)
            self.save_chart_btn.clicked.connect(self.save_chart)

            self.clear_chart_btn = QPushButton("Clear Chart")
            self.clear_chart_btn.setStyleSheet(self.get_button_style("grey"))
            self.clear_chart_btn.clicked.connect(self.clear_chart)

            chart_controls.addStretch()
            chart_controls.addWidget(self.save_chart_btn)
            chart_controls.addWidget(self.clear_chart_btn)
            chart_layout.addLayout(chart_controls)
        else:
            no_matplotlib_label = QLabel("matplotlib library not found, chart cannot be displayed")
            no_matplotlib_label.setStyleSheet("color: red; font-weight: bold; text-align: center;")
            no_matplotlib_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
            chart_layout.addWidget(no_matplotlib_label)

        main_layout.addWidget(left_panel)
        main_layout.addWidget(chart_group, 1)

    def get_button_style(self, color="blue"):
        base_style = "QPushButton { padding: 8px 12px; color: white; border: none; border-radius: 4px; font-weight: bold; } QPushButton:disabled { background-color: #cccccc; color: #666666; }"
        colors = {
            "green": ("#4CAF50", "#45a049"),
            "blue": ("#2196F3", "#0b7dda"),
            "red": ("#f44336", "#d32f2f"),
            "grey": ("#607d8b", "#455a64"),
            "orange": ("#FF9800", "#FB8C00")
        }
        bg_color, hover_color = colors.get(color, colors["blue"])
        return base_style + f"QPushButton {{ background-color: {bg_color}; }} QPushButton:hover:!disabled {{ background-color: {hover_color}; }}"

    def select_image(self):
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Select Image",
            "",
            "Image Files (*.png *.jpg *.jpeg *.tif *.tiff)"
        )
        if file_path:
            self.selected_image_path = file_path
            self.image_path_label.setText(f"Selected: {os.path.basename(file_path)}")
            self.analyze_btn.setEnabled(True)

    def find_turning_points(self, x, y, min_prominence=None):
        turning_points = []

        if len(y) < 3:
            return turning_points

        if min_prominence is None:
            min_prominence = (np.max(y) - np.min(y)) * 0.05

        for i in range(1, len(y) - 1):
            if y[i] > y[i-1] and y[i] > y[i+1] and y[i] > min_prominence:
                turning_points.append((x[i], y[i], 'max'))
            elif y[i] < y[i-1] and y[i] < y[i+1] and y[i] > 0:
                turning_points.append((x[i], y[i], 'min'))

        return turning_points

    def analyze_image(self):
        if not self.selected_image_path or not MATPLOTLIB_AVAILABLE:
            return

        try:
            grayscale_image = cv2.imread(self.selected_image_path, cv2.IMREAD_GRAYSCALE)

            if grayscale_image is None:
                if tifffile:
                    try:
                        tiff_image = tifffile.imread(self.selected_image_path)
                        if len(tiff_image.shape) == 2:
                            grayscale_image = tiff_image
                        elif len(tiff_image.shape) == 3:
                            grayscale_image = cv2.cvtColor(tiff_image[:,:,:3], cv2.COLOR_RGB2GRAY)

                        if grayscale_image.dtype != np.uint8:
                            grayscale_image = ((grayscale_image / np.max(grayscale_image)) * 255).astype(np.uint8) if np.max(grayscale_image) > 0 else grayscale_image.astype(np.uint8)
                    except Exception as e:
                        QMessageBox.critical(self, "Error", f"Could not read image file: {str(e)}")
                        return
                else:
                    QMessageBox.critical(self, "Error", "Could not read image file")
                    return

            aggregate_values = 100.0 - (grayscale_image.astype(np.float32) / 255.0) * 100.0

            hist, bin_edges = np.histogram(aggregate_values, bins=101, range=(0, 100))
            bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

            turning_points = self.find_turning_points(bin_centers, hist)

            self.figure.clear()
            ax = self.figure.add_subplot(111)
            ax.plot(bin_centers, hist, linewidth=2, color='#2196F3', label='Aggregation Distribution')
            ax.fill_between(bin_centers, hist, alpha=0.3, color='#2196F3')

            if turning_points:
                max_points = [(x, y) for x, y, t in turning_points if t == 'max']
                min_points = [(x, y) for x, y, t in turning_points if t == 'min']

                if max_points:
                    max_x, max_y = zip(*max_points)
                    ax.scatter(max_x, max_y, color='red', s=50, zorder=5, label=f'Peaks ({len(max_points)})')

                    for x, y in max_points:
                        ax.annotate(f'({x:.1f}, {y:.0f})',
                                  xy=(x, y),
                                  xytext=(5, 10),
                                  textcoords='offset points',
                                  fontsize=9,
                                  color='red',
                                  fontweight='bold',
                                  bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8, edgecolor='red'))

                if min_points:
                    min_x, min_y = zip(*min_points)
                    ax.scatter(min_x, min_y, color='orange', s=50, zorder=5, label=f'Troughs ({len(min_points)})')

                    for x, y in min_points:
                        ax.annotate(f'({x:.1f}, {y:.0f})',
                                  xy=(x, y),
                                  xytext=(5, -15),
                                  textcoords='offset points',
                                  fontsize=9,
                                  color='orange',
                                  fontweight='bold',
                                  bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8, edgecolor='orange'))

            ax.set_xlabel('Aggregation', fontsize=12)
            ax.set_ylabel('Pixel Count', fontsize=12)
            ax.set_title(f'Aggregation Distribution - {os.path.basename(self.selected_image_path)}', fontsize=14, fontweight='bold')
            ax.grid(True, alpha=0.3)
            ax.set_xlim(0, 100)
            ax.legend(loc='upper right')

            self.figure.tight_layout()
            self.canvas.draw()

            total_pixels = aggregate_values.size
            mean_agg = np.mean(aggregate_values)
            min_agg = np.min(aggregate_values)
            max_agg = np.max(aggregate_values)

            self.total_pixels_label.setText(f"{total_pixels:,}")
            self.mean_aggregate_label.setText(f"{mean_agg:.2f}")
            self.min_aggregate_label.setText(f"{min_agg:.2f}")
            self.max_aggregate_label.setText(f"{max_agg:.2f}")

            if turning_points:
                max_points = [(x, y) for x, y, t in turning_points if t == 'max']
                min_points = [(x, y) for x, y, t in turning_points if t == 'min']

                turning_info = []
                if max_points:
                    max_coords = [f"({x:.1f}, {y:.0f})" for x, y in max_points]
                    turning_info.append(f"Peaks: {', '.join(max_coords)}")
                if min_points:
                    min_coords = [f"({x:.1f}, {y:.0f})" for x, y in min_points]
                    turning_info.append(f"Troughs: {', '.join(min_coords)}")

                self.turning_points_label.setText(" | ".join(turning_info) if turning_info else "No significant inflection points")
            else:
                self.turning_points_label.setText("No significant inflection points")

            self.save_chart_btn.setEnabled(True)

        except Exception as e:
            QMessageBox.critical(self, "Analysis Error", f"An error occurred during image analysis: {str(e)}")

    def save_chart(self):
        if not MATPLOTLIB_AVAILABLE:
            return

        file_path, _ = QFileDialog.getSaveFileName(
            self,
            "Save Chart",
            f"aggregate_distribution_{os.path.splitext(os.path.basename(self.selected_image_path))[0]}.png",
            "PNG Files (*.png);;PDF Files (*.pdf);;SVG Files (*.svg)"
        )
        if file_path:
            try:
                self.figure.savefig(file_path, dpi=300, bbox_inches='tight')
                QMessageBox.information(self, "Save Successful", f"Chart saved to: {file_path}")
            except Exception as e:
                QMessageBox.critical(self, "Save Error", f"An error occurred while saving the chart: {str(e)}")

    def clear_chart(self):
        if MATPLOTLIB_AVAILABLE:
            self.figure.clear()
            self.canvas.draw()
            self.save_chart_btn.setEnabled(False)

            self.total_pixels_label.setText("--")
            self.mean_aggregate_label.setText("--")
            self.min_aggregate_label.setText("--")
            self.max_aggregate_label.setText("--")
            self.turning_points_label.setText("--")


class HQSAMProcessorThread(QThread):
    processing_finished = pyqtSignal()
    log_updated = pyqtSignal(str)
    export_data_signal = pyqtSignal(list)
    progress_updated = pyqtSignal(int, int)
    image_processed_successfully = pyqtSignal(str)

    MAX_IMAGE_SIZE = 4096

    bright_colors = [
        (0, 0, 255), (0, 165, 255), (0, 255, 255), (0, 255, 0),
        (255, 0, 0), (147, 20, 255), (255, 0, 255), (255, 255, 0),
        (128, 0, 128),
    ]

    def __init__(self, input_folder, output_folder, model_path, filter_params, sam_params):
        super().__init__()
        self.input_folder = input_folder
        self.output_folder = output_folder
        self.model_path = model_path
        self.filter_params = filter_params
        self.sam_params = sam_params
        self.running = True
        self.lock = QMutex()
        self.all_processed_data = []

    def validate_image(self, image_path):
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image file not found: {image_path}")
        if not os.access(image_path, os.R_OK):
            raise PermissionError(f"No read permission for: {image_path}")

    def run(self):
        self.all_processed_data = []
        if not SEGMENT_ANYTHING_AVAILABLE:
            self.log_updated.emit("Error: 'segment_anything' library is unavailable, cannot proceed.")
            self.export_data_signal.emit(self.all_processed_data)
            self.processing_finished.emit()
            return

        try:
            if not os.path.exists(self.input_folder):
                raise FileNotFoundError(f"Input folder not found: {self.input_folder}")

            os.makedirs(self.output_folder, exist_ok=True)
            if not os.access(self.output_folder, os.W_OK):
                raise PermissionError(f"No write permission for: {self.output_folder}")

            self.log_updated.emit(f"Output folder: {self.output_folder}")
            self.log_updated.emit("Loading SAM-HQ model...")

            device = "cuda" if torch.cuda.is_available() else "cpu"
            self.log_updated.emit(f"Using device: {device}")

            try:
                if not os.path.exists(self.model_path): raise FileNotFoundError(f"SAM model file not found: {self.model_path}")
                if not os.access(self.model_path, os.R_OK): raise PermissionError(f"No read permission for SAM model file: {self.model_path}")
                model_type = "vit_h"; model_filename_lower = os.path.basename(self.model_path).lower()
                if "vit_l" in model_filename_lower: model_type = "vit_l"
                elif "vit_b" in model_filename_lower: model_type = "vit_b"
                self.log_updated.emit(f"Attempting to load SAM model type: {model_type} (HQ version)")
                sam = sam_model_registry[model_type](checkpoint=self.model_path).to(device); sam.eval()
            except Exception as e:
                raise RuntimeError(f"Failed to load SAM model: {str(e)}")

            mask_generator = SamAutomaticMaskGenerator(model=sam, **self.sam_params)

            self.log_updated.emit("SAM-HQ Generator Parameters:")
            for key, value in self.sam_params.items(): self.log_updated.emit(f"  - {key}: {value}")
            self.log_updated.emit("Filter Parameters:")
            for key, value in self.filter_params.items(): self.log_updated.emit(f"  - {key}: {value}")

            image_files_with_paths = []
            for dirpath, dirnames, filenames in os.walk(self.input_folder):
                for filename in filenames:
                    if filename.lower().endswith((".png", ".jpg", ".jpeg", ".tif", ".tiff")):
                        image_files_with_paths.append(os.path.join(dirpath, filename))

            files = image_files_with_paths
            if not files:
                self.log_updated.emit("Warning: No image files found in the input folder or its subfolders.")
                self.progress_updated.emit(0, 0)
                self.export_data_signal.emit(self.all_processed_data)
                self.processing_finished.emit()
                return

            self.log_updated.emit(f"Found {len(files)} image file(s) in total (including subfolders).")
            self.progress_updated.emit(0, len(files))
            processed_count = 0

            for idx, full_input_path in enumerate(files):
                self.lock.lock()
                is_running = self.running
                self.lock.unlock()
                if not is_running:
                    self.log_updated.emit("Processing stopped by user.")
                    break

                self.progress_updated.emit(idx, len(files))

                try:
                    input_path = full_input_path
                    self.validate_image(input_path)

                    relative_image_path = os.path.relpath(input_path, self.input_folder)
                    output_base_name_parts = []

                    if os.path.dirname(relative_image_path) != "":
                        output_base_name_parts.append(os.path.dirname(relative_image_path).replace(os.sep, "_"))

                    output_base_name_parts.append(os.path.splitext(os.path.basename(relative_image_path))[0])
                    output_base_name = "_".join(output_base_name_parts)

                    file_ext = os.path.splitext(input_path)[1]

                    output_filename = f"{output_base_name}_sam_hq_processed{file_ext}"
                    output_path = os.path.join(self.output_folder, output_filename)

                    self.log_updated.emit(f"Processing image {idx + 1}/{len(files)}: {input_path}")
                    start_time = time.time()
                    success = self.process_image_with_filter(input_path, output_path, mask_generator)
                    elapsed_time = time.time() - start_time
                    if success:
                        processed_count += 1
                        self.log_updated.emit(f"Processed image saved to: {output_path}")
                        self.image_processed_successfully.emit(output_path)
                    self.log_updated.emit(f"Processing time: {elapsed_time:.2f} seconds")
                except Exception as e:
                    self.log_updated.emit(f"Error processing {os.path.basename(input_path)}: {str(e)}")
                    continue
                finally:
                    self.progress_updated.emit(idx + 1, len(files))

            self.lock.lock()
            is_running_final = self.running
            self.lock.unlock()
            if is_running_final: self.log_updated.emit(f"Completed: {processed_count}/{len(files)} image(s) processed.")

        except Exception as e:
            self.log_updated.emit(f"Fatal error during processing: {str(e)}")
        finally:
            if 'sam' in locals() and device == 'cuda':
                del sam
                torch.cuda.empty_cache()
                self.log_updated.emit("CUDA cache cleared.")
            self.export_data_signal.emit(self.all_processed_data)
            self.processing_finished.emit()

    def process_image_with_filter(self, input_path, output_path, mask_generator):
        try:
            original_image_bgr = cv2.imread(input_path, cv2.IMREAD_COLOR)
            grayscale_image = cv2.imread(input_path, cv2.IMREAD_GRAYSCALE)

            if original_image_bgr is None or grayscale_image is None:
                if tifffile:
                    try:
                        tiff_image = tifffile.imread(input_path)
                        if len(tiff_image.shape) == 3 and tiff_image.shape[2] >= 3:
                            original_image_bgr = cv2.cvtColor(tiff_image[:,:,:3], cv2.COLOR_RGB2BGR)
                            grayscale_image = cv2.cvtColor(original_image_bgr, cv2.COLOR_BGR2GRAY)
                        elif len(tiff_image.shape) == 2:
                            grayscale_image = tiff_image
                            original_image_bgr = cv2.cvtColor(grayscale_image, cv2.COLOR_GRAY2BGR)
                        else: raise ValueError("Unsupported TIFF structure")
                        if original_image_bgr.dtype != np.uint8: original_image_bgr = ((original_image_bgr / np.max(original_image_bgr)) * 255).astype(np.uint8) if np.max(original_image_bgr) > 0 else original_image_bgr.astype(np.uint8)
                        if grayscale_image.dtype != np.uint8: grayscale_image = ((grayscale_image / np.max(grayscale_image)) * 255).astype(np.uint8) if np.max(grayscale_image) > 0 else grayscale_image.astype(np.uint8)
                    except Exception as tiff_e: self.log_updated.emit(f"Warning: Failed to read TIFF: {tiff_e}"); return False
                else: self.log_updated.emit(f"Error: Failed to read image (cv2 failed, tifffile not installed)"); return False
            if original_image_bgr is None or grayscale_image is None: self.log_updated.emit(f"Error: Could not load image"); return False
            if original_image_bgr.shape[0] > self.MAX_IMAGE_SIZE or original_image_bgr.shape[1] > self.MAX_IMAGE_SIZE:
                self.log_updated.emit(f"Error: Image dimensions ({original_image_bgr.shape[0]}x{original_image_bgr.shape[1]}) exceed max allowed size ({self.MAX_IMAGE_SIZE}x{self.MAX_IMAGE_SIZE}). This image will be skipped.")
                return False

            self.log_updated.emit(f"  Generating masks for {os.path.basename(input_path)} (HQ)...")
            try:
                masks = mask_generator.generate(original_image_bgr)
                self.log_updated.emit(f"  Found {len(masks)} initial masks.")
            except Exception as e:
                self.log_updated.emit(f"Error generating masks: {str(e)}"); return False

            output_image = original_image_bgr.copy()
            height, width = output_image.shape[:2]
            drawn_text_boxes = []
            filtered_mask_count = 0
            use_area_filter = self.filter_params.get('use_area', True)
            use_intensity_filter = self.filter_params.get('use_intensity', True)
            a_threshold = self.filter_params.get('a_threshold', 0)
            i_threshold = self.filter_params.get('i_threshold', 0.0)
            r_threshold = self.filter_params.get('r_threshold', 0.0)
            min_aggregate = self.filter_params.get('min_aggregate', 0)
            max_aggregate = self.filter_params.get('max_aggregate', 100)

            for i, mask_info in enumerate(masks):
                segmentation_data = mask_info["segmentation"]
                segmentation_mask_bool = None; segmentation_mask_uint8 = None
                if isinstance(segmentation_data, dict):
                    if PYCOCOTOOLS_AVAILABLE:
                        try:
                            segmentation_mask_uint8 = mask_utils.decode(segmentation_data)
                            segmentation_mask_bool = segmentation_mask_uint8.astype(bool)
                        except Exception as decode_e: self.log_updated.emit(f"  Warning: Failed to decode RLE mask ({decode_e}), skipping."); continue
                    else: self.log_updated.emit(f"  Error: Detected RLE mask but 'pycocotools' is not installed."); continue
                elif isinstance(segmentation_data, np.ndarray):
                    segmentation_mask_bool = segmentation_data.astype(bool)
                    segmentation_mask_uint8 = segmentation_mask_bool.astype(np.uint8) * 255
                else: self.log_updated.emit(f"  Warning: Unknown mask format type ({type(segmentation_data)}), skipping."); continue
                if segmentation_mask_bool is None or segmentation_mask_uint8 is None: continue

                region_pixels_gray = grayscale_image[segmentation_mask_bool];
                if len(region_pixels_gray) == 0: continue
                aggregate_values_0_100 = 100.0 - (region_pixels_gray.astype(np.float32) / 255.0) * 100.0
                initial_area = len(aggregate_values_0_100)
                initial_intensity = np.sum(aggregate_values_0_100)
                initial_ratio = initial_intensity / initial_area if initial_area > 0 else 0.0
                passed_phase1 = True
                if use_area_filter and initial_area < a_threshold: passed_phase1 = False
                if passed_phase1 and use_intensity_filter and initial_intensity < i_threshold: passed_phase1 = False
                if passed_phase1 and use_area_filter and use_intensity_filter and initial_ratio < r_threshold: passed_phase1 = False
                if not passed_phase1: continue
                pixels_within_agg_range_mask = ((aggregate_values_0_100 >= min_aggregate) & (aggregate_values_0_100 <= max_aggregate))
                num_pixels_passed_phase2 = np.count_nonzero(pixels_within_agg_range_mask)

                if num_pixels_passed_phase2 > 0:
                    filtered_mask_count += 1
                    final_aggregates = aggregate_values_0_100[pixels_within_agg_range_mask]
                    final_area = num_pixels_passed_phase2
                    final_intensity = np.sum(final_aggregates)
                    final_ratio = final_intensity / final_area if final_area > 0 else 0.0
                    value_text_parts = []
                    if use_area_filter: value_text_parts.append(f"A:{final_area}")
                    if use_intensity_filter:
                        value_text_parts.append(f"I:{final_intensity:.2f}")
                        if use_area_filter and final_area > 0: value_text_parts.append(f"R:{final_ratio:.2f}")
                        elif use_area_filter: value_text_parts.append("R:N/A")
                    value_text = " ".join(value_text_parts) if value_text_parts else "Passed"

                    mask_data_entry = {
                        "image_filename": os.path.basename(input_path),
                        "mask_index": filtered_mask_count,
                        "area": final_area if use_area_filter else None,
                        "intensity": final_intensity if use_intensity_filter else None,
                        "ratio": final_ratio if use_intensity_filter and use_area_filter and final_area > 0 else None
                    }
                    self.all_processed_data.append(mask_data_entry)

                    contours, _ = cv2.findContours(segmentation_mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                    color_bgr = self.bright_colors[filtered_mask_count % len(self.bright_colors)]
                    cv2.drawContours(output_image, contours, -1, color_bgr, 1)

                    ys, xs = np.where(segmentation_mask_bool);
                    if len(xs) == 0 or len(ys) == 0: continue
                    center_x = int(np.mean(xs)); center_y = int(np.mean(ys))
                    text_x = max(5, min(center_x, width - 10)); text_y = max(15, min(center_y, height - 5))
                    font = cv2.FONT_HERSHEY_SIMPLEX; font_scale = 0.4; thickness = 1
                    (text_width, text_height), baseline = cv2.getTextSize(value_text, font, font_scale, thickness)
                    current_text_box = [text_x, text_y - text_height, text_width, text_height + baseline]; shift_amount = text_height + 2

                    attempts = 0; max_attempts = 10
                    while any(self.boxes_overlap(current_text_box, box) for box in drawn_text_boxes) and attempts < max_attempts:
                        text_y += shift_amount
                        if text_y > height - 5:
                            text_y = max(15, min(center_y, height - 5))
                            text_y -= (attempts + 1) * shift_amount;
                        if text_y < 15:
                            text_y = max(15, min(center_y, height - 5)); break
                        current_text_box = [text_x, text_y - text_height, text_width, text_height + baseline]; attempts += 1
                    cv2.putText(output_image, value_text, (text_x, text_y), font, font_scale, color_bgr, thickness, cv2.LINE_AA)
                    drawn_text_boxes.append(current_text_box)

            self.log_updated.emit(f"  Kept {filtered_mask_count} masks for visualization after filtering.")
            try:
                cv2.imwrite(output_path, output_image)
                return True
            except Exception as e:
                self.log_updated.emit(f"Error: Failed to save output image {output_path}: {str(e)}")
                return False

        except Exception as e:
            self.log_updated.emit(f"General error processing {os.path.basename(input_path)}: {str(e)}")
            if 'cuda' in str(e).lower() and torch.cuda.is_available(): torch.cuda.empty_cache()
            return False

    def boxes_overlap(self, box1, box2):
        x1, y1, w1, h1 = box1; x2, y2, w2, h2 = box2
        if (x1 + w1 < x2 or x2 + w2 < x1 or
            y1 + h1 < y2 or y2 + h2 < y1):
            return False
        return True

    def stop(self):
        self.lock.lock()
        try:
            self.running = False
            self.log_updated.emit("Attempting to stop processing...")
        finally:
            self.lock.unlock()


class HQSAMMainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.input_folder = None
        self.model_path = None
        self.process_thread = None
        self.collected_data_for_export = []
        self.current_output_folder = None
        self.last_processed_image_path = None
        self.initUI()
        self.init_system_monitor()


    def initUI(self):
        self.setWindowTitle('SAM-HQ Image Processing Tool')
        self.setGeometry(100, 100, 1400, 900)

        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        main_layout = QVBoxLayout(main_widget)
        main_layout.setContentsMargins(10, 10, 10, 10)
        main_layout.setSpacing(10)

        self.tab_widget = QTabWidget()
        self.tab_widget.setStyleSheet("""
            QTabWidget::pane {
                border: 1px solid #cccccc;
                background-color: white;
            }
            QTabBar::tab {
                background-color: #f0f0f0;
                padding: 10px 20px;
                margin-right: 2px;
                border: 1px solid #cccccc;
                border-bottom: none;
                border-top-left-radius: 4px;
                border-top-right-radius: 4px;
            }
            QTabBar::tab:selected {
                background-color: white;
                border-bottom: 1px solid white;
            }
            QTabBar::tab:hover {
                background-color: #e0e0e0;
            }
        """)

        self.single_image_widget = SingleImageAnalysisWidget()
        self.tab_widget.addTab(self.single_image_widget, "Aggregation Analysis")

        self.mask_analysis_widget = MaskAnalysisWidget()
        self.tab_widget.addTab(self.mask_analysis_widget, "Mask Analysis")

        self.batch_processing_widget = self.create_batch_processing_widget()
        self.tab_widget.addTab(self.batch_processing_widget, "Batch Processing")

        self.sam_params_widget = SAMParametersWidget()
        self.tab_widget.addTab(self.sam_params_widget, "SAM-HQ Parameters")

        self.sam_params_widget.parameters_changed.connect(self.on_sam_parameters_changed)

        main_layout.addWidget(self.tab_widget)

        self.init_system_monitor()
        self.update_button_states()

    def on_sam_parameters_changed(self, params):
        print(f"SAM-HQ parameters updated: {params}")

    def get_sam_params(self):
        return self.sam_params_widget.get_sam_params()

    def goto_sam_parameters(self):
        for i in range(self.tab_widget.count()):
            if self.tab_widget.tabText(i) == "SAM-HQ Parameters":
                self.tab_widget.setCurrentIndex(i)
                break

    def create_batch_processing_widget(self):
        batch_widget = QWidget()
        main_layout = QHBoxLayout(batch_widget)
        main_layout.setContentsMargins(15, 15, 15, 15)
        main_layout.setSpacing(15)

        left_widget = QWidget()
        left_widget.setMaximumWidth(450)
        left_layout = QVBoxLayout(left_widget)
        left_layout.setContentsMargins(0, 0, 0, 0)
        left_layout.setSpacing(10)

        control_scroll_area = QScrollArea(); control_scroll_area.setWidgetResizable(True); control_scroll_area.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        control_scroll_widget = QWidget(); control_scroll_layout = QVBoxLayout(control_scroll_widget)
        control_scroll_layout.setContentsMargins(3,3,3,3); control_scroll_layout.setSpacing(8)

        file_group = QGroupBox("File Selection"); file_group.setStyleSheet("QGroupBox { font-weight: bold; }")
        file_layout = QFormLayout(file_group); file_layout.setContentsMargins(8, 12, 8, 8); file_layout.setSpacing(8)

        self.input_folder_label = QLabel("Not Selected")
        input_folder_btn = QPushButton("Select Input Folder"); input_folder_btn.setStyleSheet(self.get_button_style("green")); input_folder_btn.clicked.connect(self.select_input_folder)
        input_folder_layout = QHBoxLayout()
        input_folder_layout.addWidget(self.input_folder_label, 1)
        input_folder_layout.addWidget(input_folder_btn)
        file_layout.addRow("Input Folder:", input_folder_layout)

        self.model_path_label = QLabel("Not Selected")
        model_path_btn = QPushButton("Select SAM Model"); model_path_btn.setStyleSheet(self.get_button_style("green")); model_path_btn.clicked.connect(self.select_model_file)
        model_path_layout = QHBoxLayout()
        model_path_layout.addWidget(self.model_path_label, 1)
        model_path_layout.addWidget(model_path_btn)
        file_layout.addRow("Model File (.pth):", model_path_layout)
        control_scroll_layout.addWidget(file_group)

        config_group = QGroupBox("Configuration"); config_group.setStyleSheet("QGroupBox { font-weight: bold; }")
        config_layout = QHBoxLayout(config_group); config_layout.setContentsMargins(8,8,8,8); config_layout.setSpacing(8)
        self.save_config_btn = QPushButton("Save Configuration"); self.save_config_btn.setStyleSheet(self.get_button_style("blue")); self.save_config_btn.clicked.connect(self.save_configuration)
        self.load_config_btn = QPushButton("Load Configuration"); self.load_config_btn.setStyleSheet(self.get_button_style("blue")); self.load_config_btn.clicked.connect(self.load_configuration)
        config_layout.addWidget(self.save_config_btn); config_layout.addWidget(self.load_config_btn)
        control_scroll_layout.addWidget(config_group)

        param_group_phase1 = QGroupBox("Filter Parameters"); param_group_phase1.setStyleSheet("QGroupBox { font-weight: bold; }")
        param_layout_phase1 = QFormLayout(param_group_phase1); param_layout_phase1.setContentsMargins(8, 12, 8, 8); param_layout_phase1.setVerticalSpacing(8); param_layout_phase1.setLabelAlignment(Qt.AlignmentFlag.AlignRight)
        self.a_threshold_spin = QSpinBox(); self.a_threshold_spin.setRange(0, 1000000); self.a_threshold_spin.setValue(0)
        param_layout_phase1.addRow("Min Area (A):", self.a_threshold_spin)
        self.i_threshold_spin = QDoubleSpinBox(); self.i_threshold_spin.setRange(0.0, 100000000.0); self.i_threshold_spin.setValue(0.00); self.i_threshold_spin.setDecimals(2)
        param_layout_phase1.addRow("Min Intensity (I):", self.i_threshold_spin)
        self.r_threshold_spin = QDoubleSpinBox(); self.r_threshold_spin.setRange(0.0, 100.0); self.r_threshold_spin.setSingleStep(0.01); self.r_threshold_spin.setValue(0.00); self.r_threshold_spin.setDecimals(2)
        param_layout_phase1.addRow("Min Ratio (R=I/A):", self.r_threshold_spin)
        self.min_aggregate_spin = QSpinBox(); self.min_aggregate_spin.setRange(0, 100); self.min_aggregate_spin.setValue(0)
        param_layout_phase1.addRow("Min Aggregation Threshold:", self.min_aggregate_spin)
        self.max_aggregate_spin = QSpinBox(); self.max_aggregate_spin.setRange(0, 100); self.max_aggregate_spin.setValue(100)
        param_layout_phase1.addRow("Max Aggregation Threshold:", self.max_aggregate_spin); control_scroll_layout.addWidget(param_group_phase1)

        sam_params_group = QGroupBox("SAM-HQ Parameters")
        sam_params_group.setStyleSheet("QGroupBox { font-weight: bold; }")
        sam_params_layout = QVBoxLayout(sam_params_group)
        sam_params_layout.setContentsMargins(8, 12, 8, 8)
        sam_params_layout.setSpacing(8)

        sam_info_label = QLabel("SAM-HQ parameters are configured in the 'SAM-HQ Parameters' tab.\nPlease click 'Apply Changes' in that tab after making adjustments.")
        sam_info_label.setWordWrap(True)
        sam_info_label.setStyleSheet("QLabel { color: #666666; background-color: #f0f8ff; padding: 6px; border: 1px solid #ddd; border-radius: 4px; font-size: 10px; }")
        sam_params_layout.addWidget(sam_info_label)

        goto_sam_btn = QPushButton("Go to SAM-HQ Parameters")
        goto_sam_btn.setStyleSheet(self.get_button_style("blue"))
        goto_sam_btn.clicked.connect(self.goto_sam_parameters)
        sam_params_layout.addWidget(goto_sam_btn)

        control_scroll_layout.addWidget(sam_params_group)

        control_scroll_layout.addStretch(1); control_scroll_area.setWidget(control_scroll_widget); left_layout.addWidget(control_scroll_area)

        process_stop_layout = QHBoxLayout()
        self.process_btn = QPushButton("Start Processing"); self.process_btn.setStyleSheet(self.get_button_style("blue")); self.process_btn.setEnabled(False); self.process_btn.clicked.connect(self.start_processing)
        process_stop_layout.addWidget(self.process_btn)
        self.stop_btn = QPushButton("Stop Processing"); self.stop_btn.setStyleSheet(self.get_button_style("red")); self.stop_btn.setEnabled(False); self.stop_btn.clicked.connect(self.stop_processing)
        process_stop_layout.addWidget(self.stop_btn)
        left_layout.addLayout(process_stop_layout)

        output_actions_layout = QHBoxLayout()
        self.open_output_btn = QPushButton("Open Output"); self.open_output_btn.setStyleSheet(self.get_button_style("grey")); self.open_output_btn.setEnabled(False); self.open_output_btn.clicked.connect(self.open_output_folder)
        output_actions_layout.addWidget(self.open_output_btn)
        self.export_data_btn = QPushButton("Export Data"); self.export_data_btn.setStyleSheet(self.get_button_style("orange")); self.export_data_btn.setEnabled(False); self.export_data_btn.clicked.connect(self.export_all_data)
        output_actions_layout.addWidget(self.export_data_btn)
        self.view_last_image_btn = QPushButton("View Output"); self.view_last_image_btn.setStyleSheet(self.get_button_style("grey")); self.view_last_image_btn.setEnabled(False); self.view_last_image_btn.clicked.connect(self.view_last_processed_image)
        output_actions_layout.addWidget(self.view_last_image_btn)
        left_layout.addLayout(output_actions_layout)

        main_layout.addWidget(left_widget)

        right_widget = QWidget()
        right_layout = QVBoxLayout(right_widget)
        right_layout.setContentsMargins(0, 0, 0, 0)
        right_layout.setSpacing(10)

        progress_group = QGroupBox("Processing Progress")
        progress_group.setStyleSheet("QGroupBox { font-weight: bold; }")
        progress_layout = QVBoxLayout(progress_group)
        progress_layout.setContentsMargins(10, 15, 10, 10)
        progress_layout.setSpacing(8)

        self.overall_progress_bar = QProgressBar()
        self.overall_progress_bar.setTextVisible(True)
        self.overall_progress_bar.setValue(0)
        self.overall_progress_bar.setFormat("Ready to process images...")
        self.overall_progress_bar.setStyleSheet(self.get_progress_bar_style())
        self.overall_progress_bar.setMinimumHeight(25)

        self.progress_details_label = QLabel("Please select an input folder and a SAM model")
        self.progress_details_label.setStyleSheet("QLabel { color: #666666; font-size: 11px; margin-top: 2px; }")

        progress_layout.addWidget(self.overall_progress_bar)
        progress_layout.addWidget(self.progress_details_label)
        right_layout.addWidget(progress_group)

        log_group = QGroupBox("Processing Log"); log_group.setStyleSheet("QGroupBox { font-weight: bold; }")
        log_layout = QVBoxLayout(log_group); log_layout.setContentsMargins(10, 15, 10, 10)
        self.log_text = QTextEdit(); self.log_text.setReadOnly(True); self.log_text.setStyleSheet(self.get_log_style())
        log_layout.addWidget(self.log_text)
        right_layout.addWidget(log_group, 1)

        monitor_group = QGroupBox("System Monitor"); monitor_group.setStyleSheet("QGroupBox { font-weight: bold; }")
        monitor_layout = QFormLayout(monitor_group); monitor_layout.setContentsMargins(10,15,10,10); monitor_layout.setLabelAlignment(Qt.AlignmentFlag.AlignLeft)
        self.cpu_usage_label = QLabel("CPU: N/A")
        self.memory_usage_label = QLabel("RAM: N/A")
        self.gpu_usage_label = QLabel("GPU: N/A")
        self.gpu_memory_label = QLabel("VRAM: N/A")
        monitor_layout.addRow("CPU Usage:", self.cpu_usage_label)
        monitor_layout.addRow("RAM Usage:", self.memory_usage_label)
        monitor_layout.addRow("GPU Usage:", self.gpu_usage_label)
        monitor_layout.addRow("VRAM Usage:", self.gpu_memory_label)
        right_layout.addWidget(monitor_group)


        main_layout.addWidget(right_widget, 1)

        return batch_widget

    def init_system_monitor(self):
        self.monitor_timer = QTimer(self)
        self.monitor_timer.timeout.connect(self.update_system_stats)

        if PSUTIL_AVAILABLE:
            if hasattr(self, 'log_text'):
                self.log_text.append("psutil library found. CPU/RAM monitoring enabled.")
        else:
            if hasattr(self, 'cpu_usage_label'):
                self.cpu_usage_label.setText("CPU: N/A (psutil not installed)")
            if hasattr(self, 'memory_usage_label'):
                self.memory_usage_label.setText("RAM: N/A (psutil not installed)")
            if hasattr(self, 'log_text'):
                self.log_text.append("Warning: psutil library not found. CPU/RAM monitoring will be disabled.")

        if PYNVML_AVAILABLE:
            if NVML_INITIALIZED_SUCCESSFULLY:
                if hasattr(self, 'log_text'):
                    self.log_text.append("pynvml library initialized. NVIDIA GPU monitoring enabled.")
            else:
                if hasattr(self, 'log_text'):
                    self.log_text.append("Warning: pynvml library found but NVML initialization failed. GPU monitoring disabled.")
                if hasattr(self, 'gpu_usage_label'):
                    self.gpu_usage_label.setText("GPU: NVML Init Error")
                if hasattr(self, 'gpu_memory_label'):
                    self.gpu_memory_label.setText("VRAM: NVML Init Error")
        else:
            if hasattr(self, 'gpu_usage_label'):
                self.gpu_usage_label.setText("GPU: N/A (pynvml not installed)")
            if hasattr(self, 'gpu_memory_label'):
                self.gpu_memory_label.setText("VRAM: N/A (pynvml not installed)")

        if PSUTIL_AVAILABLE or (PYNVML_AVAILABLE and NVML_INITIALIZED_SUCCESSFULLY):
             self.monitor_timer.start(999)

    def update_system_stats(self):
        if PSUTIL_AVAILABLE:
            cpu_percent = psutil.cpu_percent(interval=None)
            if hasattr(self, 'cpu_usage_label'):
                self.cpu_usage_label.setText(f"{cpu_percent:.1f}%")

            mem = psutil.virtual_memory()
            if hasattr(self, 'memory_usage_label'):
                self.memory_usage_label.setText(f"{mem.percent:.1f}% ({mem.used/1024**3:.1f}/{mem.total/1024**3:.1f} GB)")

        if PYNVML_AVAILABLE and NVML_INITIALIZED_SUCCESSFULLY:
            try:
                device_count = pynvml.nvmlDeviceGetCount()
                if device_count > 0:
                    handle = pynvml.nvmlDeviceGetHandleByIndex(0)

                    util = pynvml.nvmlDeviceGetUtilizationRates(handle)
                    gpu_util = util.gpu

                    mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
                    mem_used = mem_info.used // (1024**2)
                    mem_total = mem_info.total // (1024**2)
                    mem_percent = (mem_used / mem_total * 100) if mem_total > 0 else 0

                    if hasattr(self, 'gpu_usage_label'):
                        self.gpu_usage_label.setText(f"GPU 0: {gpu_util}%")
                    if hasattr(self, 'gpu_memory_label'):
                        self.gpu_memory_label.setText(f"VRAM 0: {mem_percent:.1f}% ({mem_used}/{mem_total} MiB)")
                else:
                    if hasattr(self, 'gpu_usage_label'):
                        self.gpu_usage_label.setText("GPU: No NVIDIA GPU")
                    if hasattr(self, 'gpu_memory_label'):
                        self.gpu_memory_label.setText("VRAM: No NVIDIA GPU")
            except pynvml.NVMLError as e:
                if hasattr(self, 'gpu_usage_label'):
                    self.gpu_usage_label.setText("GPU: NVML Query Error")
                if hasattr(self, 'gpu_memory_label'):
                    self.gpu_memory_label.setText("VRAM: NVML Query Error")
                print(f"NVML error during stats update: {e}")

    def get_button_style(self, color="blue"):
        base_style = "QPushButton { padding: 8px 12px; color: white; border: none; border-radius: 4px; font-weight: bold; } QPushButton:disabled { background-color: #cccccc; color: #666666; }"
        colors = {
            "green": ("#4CAF50", "#45a049"),
            "blue": ("#2196F3", "#0b7dda"),
            "red": ("#f44336", "#d32f2f"),
            "grey": ("#607d8b", "#455a64"),
            "orange": ("#FF9800", "#FB8C00")
        }
        bg_color, hover_color = colors.get(color, colors["blue"])
        return base_style + f"QPushButton {{ background-color: {bg_color}; }} QPushButton:hover:!disabled {{ background-color: {hover_color}; }}"

    def get_log_style(self):
        return "QTextEdit { border: 1px solid #cccccc; border-radius: 4px; padding: 5px; font-family: Consolas, Courier New, monospace; background-color: #f8f8f8; }"

    def get_progress_bar_style(self):
        return """
        QProgressBar {
            border: 2px solid #e0e0e0;
            border-radius: 8px;
            text-align: center;
            font-weight: bold;
            font-size: 12px;
            background-color: #f5f5f5;
            color: #333333;
        }
        QProgressBar::chunk {
            background: qlineargradient(x1: 0, y1: 0, x2: 1, y2: 0,
                                       stop: 0 #4CAF50, stop: 0.5 #45a049, stop: 1 #4CAF50);
            border-radius: 6px;
            margin: 1px;
        }
        QProgressBar[value="0"] {
            color: #888888;
        }
        """

    def get_processing_output_folder_path(self):
        if not self.input_folder: return None
        input_folder_name = os.path.basename(os.path.normpath(self.input_folder))
        parent_dir = os.path.dirname(os.path.normpath(self.input_folder))
        param_parts = []
        param_parts.append(f"A={self.a_threshold_spin.value()}")
        param_parts.append(f"I={self.i_threshold_spin.value():.2f}")
        param_parts.append(f"R={self.r_threshold_spin.value():.2f}")
        param_parts.append(f"Agg={self.min_aggregate_spin.value()}-{self.max_aggregate_spin.value()}")
        param_str = "_".join(param_parts)
        output_folder_name = f"{input_folder_name}_SAM_HQ_Output_[{param_str}]"
        return os.path.join(parent_dir, output_folder_name)



    def get_all_parameters(self):
        params = {
            "filters": self.get_filter_params(),
            "sam_hq": self.get_sam_params()
        }
        return params

    def set_all_parameters(self, params_dict):
        filter_params = params_dict.get("filters", {})
        self.a_threshold_spin.setValue(filter_params.get("a_threshold", 0))
        self.i_threshold_spin.setValue(filter_params.get("i_threshold", 0.00))
        self.r_threshold_spin.setValue(filter_params.get("r_threshold", 0.00))
        self.min_aggregate_spin.setValue(filter_params.get("min_aggregate", 0))
        self.max_aggregate_spin.setValue(filter_params.get("max_aggregate", 100))

        sam_params = params_dict.get("sam_hq", {})
        if sam_params and hasattr(self, 'sam_params_widget'):
            self.sam_params_widget.set_sam_params(sam_params)


    def save_configuration(self):
        filePath, _ = QFileDialog.getSaveFileName(self, "Save Configuration", "", "JSON Files (*.json)")
        if filePath:
            try:
                params_to_save = self.get_all_parameters()
                with open(filePath, 'w') as f:
                    json.dump(params_to_save, f, indent=4)
                if hasattr(self, 'log_text'):
                    self.log_text.append(f"Configuration saved to: {filePath}")
            except Exception as e:
                if hasattr(self, 'log_text'):
                    self.log_text.append(f"Error saving configuration: {e}")
                QMessageBox.critical(self, "Save Error", f"Could not save configuration to {filePath}.\nError: {e}")

    def load_configuration(self):
        filePath, _ = QFileDialog.getOpenFileName(self, "Load Configuration", "", "JSON Files (*.json)")
        if filePath:
            try:
                with open(filePath, 'r') as f:
                    loaded_params = json.load(f)
                self.set_all_parameters(loaded_params)
                if hasattr(self, 'log_text'):
                    self.log_text.append(f"Configuration loaded from: {filePath}")
            except Exception as e:
                if hasattr(self, 'log_text'):
                    self.log_text.append(f"Error loading configuration: {e}")
                QMessageBox.critical(self, "Load Error", f"Could not load configuration from {filePath}.\nError: {e}")




    def get_filter_params(self):
        return {
            "use_area": True,
            "use_intensity": True,
            "a_threshold": self.a_threshold_spin.value(),
            "i_threshold": self.i_threshold_spin.value(),
            "r_threshold": self.r_threshold_spin.value(),
            "min_aggregate": self.min_aggregate_spin.value(),
            "max_aggregate": self.max_aggregate_spin.value()
        }

    def select_input_folder(self):
        folder = QFileDialog.getExistingDirectory(self, "Select Input Folder")
        if folder:
            self.input_folder = folder
            self.input_folder_label.setText(f"...{folder[-30:]}")
            self.update_button_states()

    def select_model_file(self):
        file, _ = QFileDialog.getOpenFileName(self, "Select SAM Model File", "", "Model Files (*.pth)")
        if file:
            self.model_path = file
            self.model_path_label.setText(f"{os.path.basename(file)}")
            self.update_button_states()

    def update_overall_progress(self, current_value, max_value):
        self.overall_progress_bar.setMaximum(max_value)
        self.overall_progress_bar.setValue(current_value)

        if max_value > 0:
            percentage = (current_value / max_value) * 100
            self.overall_progress_bar.setFormat(f"Processing: {current_value}/{max_value} ({percentage:.0f}%)")

            if current_value == 0:
                self.progress_details_label.setText("Initializing processing...")
            elif current_value == max_value:
                self.progress_details_label.setText("✓ All images processed!")
                self.progress_details_label.setStyleSheet("QLabel { color: #4CAF50; font-size: 11px; margin-top: 2px; font-weight: bold; }")
            else:
                remaining = max_value - current_value
                self.progress_details_label.setText(f"Processing image {current_value + 1}... ({remaining} remaining)")
                self.progress_details_label.setStyleSheet("QLabel { color: #2196F3; font-size: 11px; margin-top: 2px; }")
        elif max_value == 0 and current_value == 0:
            self.overall_progress_bar.setFormat("No images found to process")
            self.progress_details_label.setText("⚠ No image files found in the selected folder")
            self.progress_details_label.setStyleSheet("QLabel { color: #FF9800; font-size: 11px; margin-top: 2px; font-weight: bold; }")
        else:
            self.overall_progress_bar.setFormat("Ready to process images...")
            self.progress_details_label.setText("Please select an input folder and a SAM model")
            self.progress_details_label.setStyleSheet("QLabel { color: #666666; font-size: 11px; margin-top: 2px; }")

    def handle_image_processed(self, image_path):
        self.last_processed_image_path = image_path
        self.view_last_image_btn.setEnabled(True)


    def view_last_processed_image(self):
        if self.last_processed_image_path and os.path.exists(self.last_processed_image_path):
            try:
                if sys.platform == "win32":
                    os.startfile(self.last_processed_image_path)
                elif sys.platform == "darwin":
                    os.system(f"open \"{self.last_processed_image_path}\"")
                else:
                    os.system(f"xdg-open \"{self.last_processed_image_path}\"")
            except Exception as e:
                if hasattr(self, 'log_text'):
                    self.log_text.append(f"Error opening last image: {e}")
                QMessageBox.warning(self, "Open Image Error", f"Could not open image: {self.last_processed_image_path}\nError: {e}")
        else:
            if hasattr(self, 'log_text'):
                self.log_text.append("No last processed image to view or path is invalid.")
            QMessageBox.information(self, "No Image", "No last processed image available to view.")


    def update_button_states(self):
        input_ready = bool(self.input_folder)
        model_ready = bool(self.model_path)
        can_process = input_ready and model_ready and SEGMENT_ANYTHING_AVAILABLE

        is_processing = self.process_thread is not None and self.process_thread.isRunning()

        self.process_btn.setEnabled(can_process and not is_processing)
        self.stop_btn.setEnabled(is_processing)
        self.open_output_btn.setEnabled(bool(self.current_output_folder) and not is_processing)
        self.export_data_btn.setEnabled(bool(self.collected_data_for_export) and not is_processing)
        self.view_last_image_btn.setEnabled(bool(self.last_processed_image_path) and not is_processing)
        self.save_config_btn.setEnabled(not is_processing)
        self.load_config_btn.setEnabled(not is_processing)


        if not SEGMENT_ANYTHING_AVAILABLE:
            if not hasattr(self, "_lib_warning_shown") or not self._lib_warning_shown:
                if hasattr(self, 'log_text'):
                    self.log_text.append("Error: 'segment_anything' library not found or failed to import. Processing is disabled.")
                self._lib_warning_shown = True

    def open_output_folder(self):
        output_folder_to_open = self.current_output_folder
        if not output_folder_to_open:
            if not self.input_folder:
                QMessageBox.warning(self, "Input Missing", "Please select an input folder first or run processing.")
                return
            output_folder_to_open = self.get_processing_output_folder_path()
            if not output_folder_to_open:
                 QMessageBox.warning(self, "Error", "Could not determine the output folder path. Process an image first.")
                 return

        if not output_folder_to_open:
            QMessageBox.warning(self, "Error", "Could not determine the output folder path.")
            return

        try:
            os.makedirs(output_folder_to_open, exist_ok=True)
            if hasattr(self, 'log_text'):
                self.log_text.append(f"Ensuring output folder exists: {output_folder_to_open}")

            if sys.platform == "win32":
                os.startfile(output_folder_to_open)
            elif sys.platform == "darwin":
                os.system(f"open \"{output_folder_to_open}\"")
            else:
                os.system(f"xdg-open \"{output_folder_to_open}\"")
        except OSError as e:
             if hasattr(self, 'log_text'):
                 self.log_text.append(f"Error: Could not create or access output folder: {e}")
             QMessageBox.warning(self, "Creation/Access Failed", f"Could not create or access the output folder:\n{output_folder_to_open}\nError: {e}")
        except Exception as e:
            if hasattr(self, 'log_text'):
                self.log_text.append(f"Error: Could not open output folder: {e}")
            QMessageBox.warning(self, "Open Failed", f"Could not open the output folder:\n{output_folder_to_open}\nError: {e}")


    def start_processing(self):
        min_agg = self.min_aggregate_spin.value(); max_agg = self.max_aggregate_spin.value()
        if min_agg > max_agg: QMessageBox.warning(self, "Invalid Thresholds", "Min Aggregate Threshold cannot be greater than Max Aggregate Threshold."); return
        if not self.input_folder or not self.model_path: QMessageBox.warning(self, "Input Missing", "Please select both an input folder and a SAM model file."); return
        if not SEGMENT_ANYTHING_AVAILABLE: QMessageBox.critical(self, "Library Missing", "'segment_anything' library is unavailable. Please install it."); return

        self.current_output_folder = self.get_processing_output_folder_path()
        if not self.current_output_folder:
            if hasattr(self, 'log_text'):
                self.log_text.append("Error: Could not determine output folder path.")
            return

        if hasattr(self, 'log_text'):
            self.log_text.append("-" * 20 + " New SAM-HQ Processing Task " + "-" * 20)
            self.log_text.append("Starting SAM-HQ image processing and filtering...")

        self.collected_data_for_export = []
        self.last_processed_image_path = None
        self.overall_progress_bar.setValue(0)
        self.progress_details_label.setText("Starting image processing...")
        self.progress_details_label.setStyleSheet("QLabel { color: #FF9800; font-size: 11px; margin-top: 2px; }")
        self.update_button_states()

        sam_params = self.get_sam_params(); filter_params = self.get_filter_params()
        if self.process_thread and self.process_thread.isRunning():
            if hasattr(self, 'log_text'):
                self.log_text.append("Stopping previous processing task (should not happen)...");
            self.process_thread.stop(); self.process_thread.wait();
            if hasattr(self, 'log_text'):
                self.log_text.append("Previous task stopped.")
        self.process_thread = HQSAMProcessorThread(self.input_folder, self.current_output_folder, self.model_path, filter_params, sam_params)
        self.process_thread.processing_finished.connect(self.on_processing_finished)
        self.process_thread.log_updated.connect(self.update_log)
        self.process_thread.export_data_signal.connect(self.on_data_ready_for_export)
        self.process_thread.progress_updated.connect(self.update_overall_progress)
        self.process_thread.image_processed_successfully.connect(self.handle_image_processed)
        self.process_thread.start()

    def stop_processing(self):
        if self.process_thread and self.process_thread.isRunning():
            self.process_thread.stop()
            self.progress_details_label.setText("⏹ Processing stopped by user")
            self.progress_details_label.setStyleSheet("QLabel { color: #f44336; font-size: 11px; margin-top: 2px; font-weight: bold; }")
        else:
            if hasattr(self, 'log_text'):
                self.log_text.append("No processing task is currently running.")
            if hasattr(self, 'progress_details_label'):
                self.progress_details_label.setText("No processing task is currently running")
                self.progress_details_label.setStyleSheet("QLabel { color: #666666; font-size: 11px; margin-top: 2px; }")

    def update_log(self, message):
        self.log_text.append(message)
        scrollbar = self.log_text.verticalScrollBar()
        scrollbar.setValue(scrollbar.maximum())

    def on_data_ready_for_export(self, data):
        self.collected_data_for_export = data
        if hasattr(self, 'log_text'):
            if data:
                self.log_text.append("Numerical data is ready for export.")
            else:
                self.log_text.append("No numerical data generated for export.")
        self.update_button_states()

    def on_processing_finished(self):
        if hasattr(self, 'log_text'):
            self.log_text.append("All image processing finished.")
        if hasattr(self, 'overall_progress_bar') and self.overall_progress_bar.maximum() > 0 :
             self.overall_progress_bar.setValue(self.overall_progress_bar.maximum())
             self.update_overall_progress(self.overall_progress_bar.maximum(), self.overall_progress_bar.maximum())
        elif hasattr(self, 'progress_details_label'):
            self.progress_details_label.setText("Processing completed")
            self.progress_details_label.setStyleSheet("QLabel { color: #4CAF50; font-size: 11px; margin-top: 2px; }")
        self.process_thread = None
        self.update_button_states()

    def export_all_data(self):
        if not self.collected_data_for_export:
            if hasattr(self, 'log_text'):
                self.log_text.append("No data available for export.")
            QMessageBox.information(self, "No Data", "No data available for export.")
            return

        if not self.current_output_folder:
            if hasattr(self, 'log_text'):
                self.log_text.append("Error: Cannot determine output folder to save data file.")
            QMessageBox.warning(self, "Error", "Cannot determine output folder path. Please run processing once.")
            return

        base_input_folder_name = os.path.basename(os.path.normpath(self.input_folder))
        export_filename = f"{base_input_folder_name}_SAM_HQ_Numerical_Data.txt"
        export_filepath = os.path.join(self.current_output_folder, export_filename)

        try:
            with open(export_filepath, 'w', encoding='utf-8') as f:
                current_image_filename = None
                for entry in self.collected_data_for_export:
                    if entry["image_filename"] != current_image_filename:
                        if current_image_filename is not None: f.write("\n")
                        current_image_filename = entry["image_filename"]
                        f.write(f"Image: {current_image_filename}\n")

                    area_str = f"Area={entry['area']}" if entry['area'] is not None else "Area=N/A"
                    intensity_str = f"Intensity={entry['intensity']:.2f}" if entry['intensity'] is not None else "Intensity=N/A"
                    ratio_str = f"Ratio={entry['ratio']:.2f}" if entry['ratio'] is not None else "Ratio=N/A"

                    f.write(f"  Mask {entry['mask_index']}: {area_str}, {intensity_str}, {ratio_str}\n")

            if hasattr(self, 'log_text'):
                self.log_text.append(f"Numerical data successfully exported to: {export_filepath}")
            QMessageBox.information(self, "Export Successful", f"Numerical data successfully exported to:\n{export_filepath}")
        except Exception as e:
            if hasattr(self, 'log_text'):
                self.log_text.append(f"Failed to export numerical data: {str(e)}")
            QMessageBox.critical(self, "Export Failed", f"Failed to export numerical data.\nError: {str(e)}")


    def closeEvent(self, event):
        reply = QMessageBox.question(self, 'Confirm Exit',
                                     "Are you sure you want to exit?",
                                     QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                                     QMessageBox.StandardButton.No)
        if reply == QMessageBox.StandardButton.Yes:
            if self.process_thread and self.process_thread.isRunning():
                self.process_thread.stop()
                self.process_thread.wait(3000)

            if PYNVML_AVAILABLE and NVML_INITIALIZED_SUCCESSFULLY:
                try:
                    pynvml.nvmlShutdown()
                    if hasattr(self, 'log_text'):
                        self.log_text.append("NVML shut down successfully.")
                except pynvml.NVMLError as e:
                    if hasattr(self, 'log_text'):
                        self.log_text.append(f"Error shutting down NVML: {e}")
            event.accept()
        else:
            event.ignore()


class SAMParametersWidget(QWidget):
    parameters_changed = pyqtSignal(dict)

    def __init__(self):
        super().__init__()
        self.initUI()

    def initUI(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(20, 20, 20, 20)
        layout.setSpacing(20)

        title_label = QLabel("SAM-HQ Model Parameters Configuration")
        title_label.setStyleSheet("QLabel { font-size: 18px; font-weight: bold; color: #2196F3; margin-bottom: 10px; }")
        layout.addWidget(title_label)

        description_label = QLabel("Configure SAM (Segment Anything Model) parameters for all analysis functions. Changes will apply to both Mask Analysis and Batch Processing.")
        description_label.setWordWrap(True)
        description_label.setStyleSheet("QLabel { color: #666666; margin-bottom: 20px; }")
        layout.addWidget(description_label)

        content_widget = QWidget()
        content_layout = QHBoxLayout(content_widget)
        content_layout.setSpacing(30)

        left_panel = self.create_parameters_panel()
        content_layout.addWidget(left_panel, 1)

        right_panel = self.create_info_panel()
        content_layout.addWidget(right_panel, 2)

        layout.addWidget(content_widget, 1)

        button_layout = QHBoxLayout()
        button_layout.addStretch()

        self.reset_btn = QPushButton("Reset to Defaults")
        self.reset_btn.setStyleSheet(self.get_button_style("grey"))
        self.reset_btn.clicked.connect(self.reset_to_defaults)
        button_layout.addWidget(self.reset_btn)

        self.apply_btn = QPushButton("Apply Changes")
        self.apply_btn.setStyleSheet(self.get_button_style("blue"))
        self.apply_btn.clicked.connect(self.apply_changes)
        button_layout.addWidget(self.apply_btn)

        layout.addLayout(button_layout)

    def create_parameters_panel(self):
        panel = QGroupBox("SAM-HQ Parameters")
        panel.setStyleSheet("QGroupBox { font-weight: bold; font-size: 14px; }")
        layout = QFormLayout(panel)
        layout.setContentsMargins(20, 25, 20, 20)
        layout.setVerticalSpacing(15)
        layout.setLabelAlignment(Qt.AlignmentFlag.AlignLeft)

        self.points_per_side_spin = QSpinBox()
        self.points_per_side_spin.setRange(1, 100)
        self.points_per_side_spin.setValue(32)
        self.points_per_side_spin.setToolTip("Number of points to be sampled along one side of the image")
        layout.addRow("Points Per Side:", self.points_per_side_spin)

        self.points_per_batch_spin = QSpinBox()
        self.points_per_batch_spin.setRange(1, 1000)
        self.points_per_batch_spin.setValue(64)
        self.points_per_batch_spin.setToolTip("Number of points to process in a batch")
        layout.addRow("Points Per Batch:", self.points_per_batch_spin)

        self.pred_iou_thresh_spin = QDoubleSpinBox()
        self.pred_iou_thresh_spin.setRange(0.0, 1.0)
        self.pred_iou_thresh_spin.setSingleStep(0.01)
        self.pred_iou_thresh_spin.setValue(0.88)
        self.pred_iou_thresh_spin.setDecimals(2)
        self.pred_iou_thresh_spin.setToolTip("Threshold for predicted IoU (Intersection over Union)")
        layout.addRow("Pred IoU Threshold:", self.pred_iou_thresh_spin)

        self.stability_score_thresh_spin = QDoubleSpinBox()
        self.stability_score_thresh_spin.setRange(0.0, 1.0)
        self.stability_score_thresh_spin.setSingleStep(0.01)
        self.stability_score_thresh_spin.setValue(0.95)
        self.stability_score_thresh_spin.setDecimals(2)
        self.stability_score_thresh_spin.setToolTip("Threshold for stability score to filter out unstable masks")
        layout.addRow("Stability Score Threshold:", self.stability_score_thresh_spin)

        self.stability_score_offset_spin = QDoubleSpinBox()
        self.stability_score_offset_spin.setRange(0.0, 10.0)
        self.stability_score_offset_spin.setSingleStep(0.01)
        self.stability_score_offset_spin.setValue(1.00)
        self.stability_score_offset_spin.setDecimals(2)
        self.stability_score_offset_spin.setToolTip("Offset for stability score calculation")
        layout.addRow("Stability Score Offset:", self.stability_score_offset_spin)

        self.box_nms_thresh_spin = QDoubleSpinBox()
        self.box_nms_thresh_spin.setRange(0.0, 1.0)
        self.box_nms_thresh_spin.setSingleStep(0.01)
        self.box_nms_thresh_spin.setValue(0.70)
        self.box_nms_thresh_spin.setDecimals(2)
        self.box_nms_thresh_spin.setToolTip("IoU threshold for Non-Maximum Suppression on bounding boxes")
        layout.addRow("Box NMS Threshold:", self.box_nms_thresh_spin)

        self.crop_n_layers_spin = QSpinBox()
        self.crop_n_layers_spin.setRange(0, 5)
        self.crop_n_layers_spin.setValue(0)
        self.crop_n_layers_spin.setToolTip("Number of layers to crop the image into for better segmentation")
        layout.addRow("Crop N Layers:", self.crop_n_layers_spin)

        self.crop_nms_thresh_spin = QDoubleSpinBox()
        self.crop_nms_thresh_spin.setRange(0.0, 1.0)
        self.crop_nms_thresh_spin.setSingleStep(0.01)
        self.crop_nms_thresh_spin.setValue(0.70)
        self.crop_nms_thresh_spin.setDecimals(2)
        self.crop_nms_thresh_spin.setToolTip("IoU threshold for NMS on crops")
        layout.addRow("Crop NMS Threshold:", self.crop_nms_thresh_spin)

        self.crop_overlap_ratio_spin = QDoubleSpinBox()
        self.crop_overlap_ratio_spin.setRange(0.0, 1.0)
        self.crop_overlap_ratio_spin.setSingleStep(0.001)
        self.crop_overlap_ratio_spin.setValue(round(512 / 1500, 3))
        self.crop_overlap_ratio_spin.setDecimals(3)
        self.crop_overlap_ratio_spin.setToolTip("Ratio of overlap between crops")
        layout.addRow("Crop Overlap Ratio:", self.crop_overlap_ratio_spin)

        self.crop_n_points_downscale_factor_spin = QSpinBox()
        self.crop_n_points_downscale_factor_spin.setRange(1, 10)
        self.crop_n_points_downscale_factor_spin.setValue(1)
        self.crop_n_points_downscale_factor_spin.setToolTip("Factor to downscale the number of points in crops")
        layout.addRow("Crop Points Downscale Factor:", self.crop_n_points_downscale_factor_spin)

        self.min_mask_region_area_spin = QSpinBox()
        self.min_mask_region_area_spin.setRange(0, 100000)
        self.min_mask_region_area_spin.setValue(0)
        self.min_mask_region_area_spin.setToolTip("Minimum area for a mask region to be kept")
        layout.addRow("Min Mask Region Area:", self.min_mask_region_area_spin)

        self.output_mode_combo = QComboBox()
        self.output_mode_combo.addItems(["binary_mask", "uncompressed_rle", "coco_rle"])
        self.output_mode_combo.setToolTip("Output format for the generated masks")
        layout.addRow("Output Mode:", self.output_mode_combo)

        return panel

    def create_info_panel(self):
        panel = QGroupBox("Parameter Descriptions")
        panel.setStyleSheet("QGroupBox { font-weight: bold; font-size: 14px; }")
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(20, 25, 20, 20)
        layout.setSpacing(15)

        info_text = QTextEdit()
        info_text.setReadOnly(True)
        info_text.setMinimumHeight(600)
        info_text.setStyleSheet("QTextEdit { background-color: #f8f8f8; border: 1px solid #ddd; font-family: Arial, sans-serif; font-size: 13px; line-height: 1.4; }")

        info_content = """
<h3 style="color: #2196F3; margin-bottom: 20px;">SAM-HQ Parameters Guide</h3>

<h4 style="color: #444; margin-top: 25px; margin-bottom: 15px;">🎯 Basic Detection Settings</h4>

<p style="margin-bottom: 15px;"><b>Points Per Side:</b><br/>
How many sample points the AI uses to scan your image. Think of it like a grid - more points mean finer detection but slower processing. Use 32-64 for detailed work, 8-16 for quick scans.<br/>
<i>Default: 32 | Range: 1-100</i></p>

<p style="margin-bottom: 15px;"><b>Points Per Batch:</b><br/>
How many points the computer processes at once. Higher numbers can be faster if you have a powerful graphics card, but use more memory.<br/>
<i>Default: 64 | Range: 1-1000</i></p>

<h4 style="color: #444; margin-top: 25px; margin-bottom: 15px;">🎛️ Quality Control Settings</h4>

<p style="margin-bottom: 15px;"><b>Pred IoU Thresh:</b><br/>
This controls how "perfect" a mask needs to be to get accepted. Higher values (like 0.9) are very picky and only keep the best masks. Lower values (like 0.7) are more forgiving but might include some imperfect results.<br/>
<i>Default: 0.88 | Range: 0.0-1.0</i></p>

<p style="margin-bottom: 15px;"><b>Stability Score Thresh:</b><br/>
Checks if the mask is consistent and reliable. Higher values mean only very stable masks are kept, but you might get fewer results overall.<br/>
<i>Default: 0.95 | Range: 0.0-1.0</i></p>

<p style="margin-bottom: 15px;"><b>Stability Score Offset:</b><br/>
A technical adjustment for stability calculation. Usually best to leave this at the default setting unless you're doing advanced fine-tuning.<br/>
<i>Default: 1.0 | Range: 0.0-10.0</i></p>

<h4 style="color: #444; margin-top: 25px; margin-bottom: 15px;">🔄 Cleanup Settings</h4>

<p style="margin-bottom: 15px;"><b>Box NMS Thresh:</b><br/>
Removes overlapping detection boxes. Lower values (0.5-0.6) are stricter about removing overlaps, higher values (0.8-0.9) allow more overlapping detections.<br/>
<i>Default: 0.7 | Range: 0.0-1.0</i></p>

<p style="margin-bottom: 15px;"><b>Min Mask Region Area:</b><br/>
Ignores tiny detections smaller than this pixel count. Set to 100 or more to filter out noise and very small objects that might not be useful.<br/>
<i>Default: 0 | Range: 0-100000</i></p>

<h4 style="color: #444; margin-top: 25px; margin-bottom: 15px;">✂️ Large Image Settings</h4>

<p style="margin-bottom: 15px;"><b>Crop N Layers:</b><br/>
For very large images, this splits the image into smaller pieces for better processing. Set to 0 for normal images. Use 1-3 for huge images (like satellite photos).<br/>
<i>Default: 0 | Range: 0-5</i></p>

<p style="margin-bottom: 15px;"><b>Crop NMS Thresh:</b><br/>
When using image cropping, this setting merges results from different pieces. Usually keep the same as Box NMS Thresh.<br/>
<i>Default: 0.7 | Range: 0.0-1.0</i></p>

<p style="margin-bottom: 15px;"><b>Crop Overlap Ratio:</b><br/>
How much the image pieces overlap when cropping. Higher overlap gives better edge detection but takes more processing time.<br/>
<i>Default: 0.341 | Range: 0.0-1.0</i></p>

<p style="margin-bottom: 15px;"><b>Crop Points Downscale Factor:</b><br/>
Reduces detection density in cropped pieces to save time. Higher numbers mean faster processing but might miss small details.<br/>
<i>Default: 1 | Range: 1-10</i></p>

<h4 style="color: #444; margin-top: 25px; margin-bottom: 15px;">💾 Output Format</h4>

<p style="margin-bottom: 15px;"><b>Output Mode:</b><br/>
How the mask data is saved:<br/>
• <b>binary_mask:</b> Standard format - easiest to work with (recommended)<br/>
• <b>uncompressed_rle:</b> Compressed format - saves space<br/>
• <b>coco_rle:</b> Special compressed format for advanced users<br/>
<i>Default: binary_mask</i></p>

<hr style="margin: 25px 0;"/>

<p style="margin-bottom: 15px;"><b>💡 Quick Tips:</b><br/>
• Start with default settings and change only what you need<br/>
• Want to find small objects? → Increase "Points Per Side"<br/>
• Want faster processing? → Decrease "Points Per Side", increase "Points Per Batch"<br/>
• Want higher quality results? → Increase "Pred IoU Thresh" and "Stability Score Thresh"<br/>
• Always click "Apply Changes" after making adjustments</p>
        """
        info_text.setHtml(info_content)
        layout.addWidget(info_text)

        layout.addStretch()

        return panel

    def get_button_style(self, color="blue"):
        base_style = "QPushButton { padding: 10px 15px; color: white; border: none; border-radius: 5px; font-weight: bold; } QPushButton:disabled { background-color: #cccccc; color: #666666; }"
        colors = {
            "green": ("#4CAF50", "#45a049"),
            "blue": ("#2196F3", "#0b7dda"),
            "red": ("#f44336", "#d32f2f"),
            "grey": ("#607d8b", "#455a64"),
            "orange": ("#FF9800", "#FB8C00")
        }
        bg_color, hover_color = colors.get(color, colors["blue"])
        return base_style + f"QPushButton {{ background-color: {bg_color}; }} QPushButton:hover:!disabled {{ background-color: {hover_color}; }}"

    def get_sam_params(self):
        return {
            "points_per_side": self.points_per_side_spin.value(),
            "points_per_batch": self.points_per_batch_spin.value(),
            "pred_iou_thresh": self.pred_iou_thresh_spin.value(),
            "stability_score_thresh": self.stability_score_thresh_spin.value(),
            "stability_score_offset": self.stability_score_offset_spin.value(),
            "box_nms_thresh": self.box_nms_thresh_spin.value(),
            "crop_n_layers": self.crop_n_layers_spin.value(),
            "crop_nms_thresh": self.crop_nms_thresh_spin.value(),
            "crop_overlap_ratio": self.crop_overlap_ratio_spin.value(),
            "crop_n_points_downscale_factor": self.crop_n_points_downscale_factor_spin.value(),
            "min_mask_region_area": self.min_mask_region_area_spin.value(),
            "output_mode": self.output_mode_combo.currentText()
        }

    def set_sam_params(self, params):
        self.points_per_side_spin.setValue(params.get("points_per_side", 32))
        self.points_per_batch_spin.setValue(params.get("points_per_batch", 64))
        self.pred_iou_thresh_spin.setValue(params.get("pred_iou_thresh", 0.88))
        self.stability_score_thresh_spin.setValue(params.get("stability_score_thresh", 0.95))
        self.stability_score_offset_spin.setValue(params.get("stability_score_offset", 1.00))
        self.box_nms_thresh_spin.setValue(params.get("box_nms_thresh", 0.70))
        self.crop_n_layers_spin.setValue(params.get("crop_n_layers", 0))
        self.crop_nms_thresh_spin.setValue(params.get("crop_nms_thresh", 0.70))
        self.crop_overlap_ratio_spin.setValue(params.get("crop_overlap_ratio", round(512/1500, 3)))
        self.crop_n_points_downscale_factor_spin.setValue(params.get("crop_n_points_downscale_factor", 1))
        self.min_mask_region_area_spin.setValue(params.get("min_mask_region_area", 0))
        self.output_mode_combo.setCurrentText(params.get("output_mode", "binary_mask"))

    def reset_to_defaults(self):
        default_params = {
            "points_per_side": 32,
            "points_per_batch": 64,
            "pred_iou_thresh": 0.88,
            "stability_score_thresh": 0.95,
            "stability_score_offset": 1.00,
            "box_nms_thresh": 0.70,
            "crop_n_layers": 0,
            "crop_nms_thresh": 0.70,
            "crop_overlap_ratio": round(512/1500, 3),
            "crop_n_points_downscale_factor": 1,
            "min_mask_region_area": 0,
            "output_mode": "binary_mask"
        }
        self.set_sam_params(default_params)
        QMessageBox.information(self, "Reset", "SAM-HQ parameters have been reset to default values.")



    def apply_changes(self):
        params = self.get_sam_params()
        self.parameters_changed.emit(params)
        QMessageBox.information(self, "Applied", "SAM-HQ parameters have been applied to all analysis functions.")


class ClickableImageLabel(QLabel):
    mask_clicked = pyqtSignal(int, bool)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.masks_data = []
        self.selected_masks = set()
        self.scale_factor = 1.0
        self.image_offset = (0, 0)
        self.original_image_size = (0, 0)
        self.displayed_image_size = (0, 0)

    def set_masks_data(self, masks_data):
        self.masks_data = masks_data
        self.selected_masks.clear()

    def set_selected_masks(self, selected_indices):
        self.selected_masks = set(selected_indices)

    def set_image_info(self, original_size, displayed_size, scale_factor, offset):
        self.original_image_size = original_size
        self.displayed_image_size = displayed_size
        self.scale_factor = scale_factor
        self.image_offset = offset

    def mousePressEvent(self, event):
        if not self.masks_data:
            return

        click_pos = event.pos()

        original_pos = self.convert_to_original_coords(click_pos)
        if original_pos is None:
            return

        clicked_mask_idx = self.find_mask_at_position(original_pos)

        if clicked_mask_idx is not None:
            is_right_click = event.button() == Qt.MouseButton.RightButton
            self.mask_clicked.emit(clicked_mask_idx, is_right_click)

    def convert_to_original_coords(self, display_pos):
        if not self.original_image_size or not self.displayed_image_size:
            return None

        label_size = self.size()
        img_width, img_height = self.displayed_image_size

        x_offset = (label_size.width() - img_width) // 2
        y_offset = (label_size.height() - img_height) // 2

        adj_x = display_pos.x() - x_offset
        adj_y = display_pos.y() - y_offset

        if adj_x < 0 or adj_y < 0 or adj_x >= img_width or adj_y >= img_height:
            return None

        orig_x = int(adj_x / self.scale_factor)
        orig_y = int(adj_y / self.scale_factor)

        return (orig_x, orig_y)

    def find_mask_at_position(self, pos):
        x, y = pos

        for i in range(len(self.masks_data) - 1, -1, -1):
            mask_info = self.masks_data[i]
            segmentation_data = mask_info["segmentation"]

            segmentation_mask_bool = None
            if isinstance(segmentation_data, dict):
                if PYCOCOTOOLS_AVAILABLE:
                    try:
                        segmentation_mask_uint8 = mask_utils.decode(segmentation_data)
                        segmentation_mask_bool = segmentation_mask_uint8.astype(bool)
                    except:
                        continue
                else:
                    continue
            elif isinstance(segmentation_data, np.ndarray):
                segmentation_mask_bool = segmentation_data.astype(bool)
            else:
                continue

            if segmentation_mask_bool is None:
                continue

            if (y < segmentation_mask_bool.shape[0] and
                x < segmentation_mask_bool.shape[1] and
                segmentation_mask_bool[y, x]):
                return i

        return None


class MaskGenerationWorker(QThread):
    progress_updated = pyqtSignal(str)
    masks_generated = pyqtSignal(list)
    error_occurred = pyqtSignal(str)

    def __init__(self, image, model_path, sam_params):
        super().__init__()
        self.image = image
        self.model_path = model_path
        self.sam_params = sam_params

    def run(self):
        try:
            self.progress_updated.emit("Loading SAM model...")

            device = "cuda" if torch.cuda.is_available() else "cpu"
            self.progress_updated.emit(f"Using device: {device}")

            model_filename_lower = os.path.basename(self.model_path).lower()
            if "vit_l" in model_filename_lower:
                model_type = "vit_l"
            elif "vit_b" in model_filename_lower:
                model_type = "vit_b"
            else:
                model_type = "vit_h"

            self.progress_updated.emit(f"Loading model type: {model_type}")
            sam = sam_model_registry[model_type](checkpoint=self.model_path).to(device)
            sam.eval()

            self.progress_updated.emit("Creating mask generator...")
            mask_generator = SamAutomaticMaskGenerator(model=sam, **self.sam_params)

            self.progress_updated.emit("Generating masks, please wait...")
            masks = mask_generator.generate(self.image)

            if device == 'cuda':
                del sam
                torch.cuda.empty_cache()
                self.progress_updated.emit("GPU memory cleared")

            self.progress_updated.emit(f"Successfully generated {len(masks)} masks")
            self.masks_generated.emit(masks)

        except Exception as e:
            self.error_occurred.emit(f"An error occurred while generating masks: {str(e)}")
            if 'cuda' in str(e).lower() and torch.cuda.is_available():
                torch.cuda.empty_cache()


class MaskAnalysisWidget(QWidget):
    def __init__(self):
        super().__init__()
        self.selected_image_path = None
        self.original_image = None
        self.grayscale_image = None
        self.generated_masks = []
        self.selected_mask_indices = set()
        self.current_sam_model = None
        self.current_mask_generator = None
        self.worker_thread = None
        self.current_results = []
        self.cached_display_params = None
        self.cached_base_pixmap = None
        self.initUI()

    def initUI(self):
        layout = QHBoxLayout(self)
        layout.setContentsMargins(15, 15, 15, 15)
        layout.setSpacing(15)

        left_widget = QWidget()
        left_widget.setMaximumWidth(450)
        left_layout = QVBoxLayout(left_widget)
        left_layout.setContentsMargins(0, 0, 0, 0)
        left_layout.setSpacing(10)

        setup_group = QGroupBox("Step 1: Setup")
        setup_group.setStyleSheet("QGroupBox { font-weight: bold; }")
        setup_layout = QFormLayout(setup_group)
        setup_layout.setSpacing(8)

        self.image_path_label = QLabel("Not Selected")
        self.select_image_btn = QPushButton("Select Image")
        self.select_image_btn.setStyleSheet(self.get_button_style("green"))
        self.select_image_btn.clicked.connect(self.select_image)
        select_image_layout = QHBoxLayout()
        select_image_layout.addWidget(self.image_path_label, 1)
        select_image_layout.addWidget(self.select_image_btn)
        setup_layout.addRow("Image File:", select_image_layout)

        self.model_path_label = QLabel("Not Selected")
        self.select_model_btn = QPushButton("Select Model")
        self.select_model_btn.setStyleSheet(self.get_button_style("green"))
        self.select_model_btn.clicked.connect(self.select_model)
        select_model_layout = QHBoxLayout()
        select_model_layout.addWidget(self.model_path_label, 1)
        select_model_layout.addWidget(self.select_model_btn)
        setup_layout.addRow("SAM Model:", select_model_layout)

        left_layout.addWidget(setup_group)

        generate_group = QGroupBox("Step 2: Generate Masks")
        generate_group.setStyleSheet("QGroupBox { font-weight: bold; }")
        generate_layout = QVBoxLayout(generate_group)
        generate_layout.setSpacing(8)

        self.generate_masks_btn = QPushButton("Generate Masks")
        self.generate_masks_btn.setStyleSheet(self.get_button_style("blue"))
        self.generate_masks_btn.setEnabled(False)
        self.generate_masks_btn.clicked.connect(self.generate_masks)
        generate_layout.addWidget(self.generate_masks_btn)

        self.progress_label = QLabel("")
        self.progress_label.setVisible(False)
        self.progress_label.setStyleSheet("QLabel { color: #2196F3; font-weight: bold; }")
        generate_layout.addWidget(self.progress_label)

        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        self.progress_bar.setRange(0, 0)
        self.progress_bar.setTextVisible(False)
        generate_layout.addWidget(self.progress_bar)

        left_layout.addWidget(generate_group)

        analysis_group = QGroupBox("Step 3: Analyze & Filter")
        analysis_group.setStyleSheet("QGroupBox { font-weight: bold; }")
        analysis_layout = QVBoxLayout(analysis_group)

        self.filter_group = QGroupBox("Filter Parameters (Always Enabled)")
        self.filter_group.setStyleSheet("QGroupBox { font-weight: normal; } QGroupBox::title { subcontrol-origin: margin; left: 7px; padding: 0 5px 0 5px; }")
        filter_layout = QFormLayout(self.filter_group)
        filter_layout.setSpacing(8)
        filter_layout.setLabelAlignment(Qt.AlignmentFlag.AlignRight)

        self.a_threshold_spin = QSpinBox(); self.a_threshold_spin.setRange(0, 1000000); self.a_threshold_spin.setValue(0)
        filter_layout.addRow("Min Area (A):", self.a_threshold_spin)
        self.i_threshold_spin = QDoubleSpinBox(); self.i_threshold_spin.setRange(0.0, 100000000.0); self.i_threshold_spin.setValue(0.00); self.i_threshold_spin.setDecimals(2)
        filter_layout.addRow("Min Intensity (I):", self.i_threshold_spin)
        self.r_threshold_spin = QDoubleSpinBox(); self.r_threshold_spin.setRange(0.0, 100.0); self.r_threshold_spin.setSingleStep(0.01); self.r_threshold_spin.setValue(0.00); self.r_threshold_spin.setDecimals(2)
        filter_layout.addRow("Min Ratio (R):", self.r_threshold_spin)
        self.min_aggregate_spin = QSpinBox(); self.min_aggregate_spin.setRange(0, 100); self.min_aggregate_spin.setValue(0)
        filter_layout.addRow("Min Aggregation:", self.min_aggregate_spin)
        self.max_aggregate_spin = QSpinBox(); self.max_aggregate_spin.setRange(0, 100); self.max_aggregate_spin.setValue(100)
        filter_layout.addRow("Max Aggregation:", self.max_aggregate_spin)

        analysis_layout.addWidget(self.filter_group)

        selection_info = QLabel("Tip: Left-click in the image to select/deselect, right-click to deselect.")
        selection_info.setWordWrap(True)
        selection_info.setStyleSheet("QLabel { color: #666666; background-color: #f0f8ff; padding: 6px; border: 1px solid #ddd; border-radius: 4px; font-size: 10px; }")
        analysis_layout.addWidget(selection_info)

        status_layout = QHBoxLayout()
        self.selection_status_label = QLabel("No masks generated yet")
        self.selection_status_label.setStyleSheet("QLabel { font-weight: bold; color: #2196F3; }")
        status_layout.addWidget(self.selection_status_label, 1)
        self.select_all_btn = QPushButton("Select All"); self.select_all_btn.setStyleSheet(self.get_button_style("grey")); self.select_all_btn.setEnabled(False); self.select_all_btn.clicked.connect(self.select_all_masks)
        status_layout.addWidget(self.select_all_btn)
        self.deselect_all_btn = QPushButton("Deselect All"); self.deselect_all_btn.setStyleSheet(self.get_button_style("grey")); self.deselect_all_btn.setEnabled(False); self.deselect_all_btn.clicked.connect(self.deselect_all_masks)
        status_layout.addWidget(self.deselect_all_btn)
        analysis_layout.addLayout(status_layout)

        self.calculate_btn = QPushButton("Calculate Selected Masks")
        self.calculate_btn.setStyleSheet(self.get_button_style("orange"))
        self.calculate_btn.setEnabled(False)
        self.calculate_btn.clicked.connect(self.calculate_selected_masks)
        analysis_layout.addWidget(self.calculate_btn)

        left_layout.addWidget(analysis_group)

        results_group = QGroupBox("Step 4: Results")
        results_group.setStyleSheet("QGroupBox { font-weight: bold; }")
        results_layout = QVBoxLayout(results_group)
        self.results_text = QTextEdit(); self.results_text.setReadOnly(True); self.results_text.setMinimumHeight(120)
        self.results_text.setStyleSheet("QTextEdit { font-family: Consolas, Courier New, monospace; background-color: #f8f8f8; font-size: 11px; }")
        results_layout.addWidget(self.results_text)
        export_layout = QHBoxLayout()
        self.export_results_btn = QPushButton("Export Results"); self.export_results_btn.setStyleSheet(self.get_button_style("blue")); self.export_results_btn.setEnabled(False); self.export_results_btn.clicked.connect(self.export_results)
        self.save_image_btn = QPushButton("Save Marked Image"); self.save_image_btn.setStyleSheet(self.get_button_style("green")); self.save_image_btn.setEnabled(False); self.save_image_btn.clicked.connect(self.save_marked_image)
        export_layout.addStretch()
        export_layout.addWidget(self.export_results_btn)
        export_layout.addWidget(self.save_image_btn)
        results_layout.addLayout(export_layout)
        left_layout.addWidget(results_group)
        left_layout.addStretch()

        right_widget = QWidget()
        right_layout = QVBoxLayout(right_widget)
        right_layout.setContentsMargins(0, 0, 0, 0)

        image_group = QGroupBox("Image Display")
        image_group.setStyleSheet("QGroupBox { font-weight: bold; }")
        image_layout = QVBoxLayout(image_group)
        image_layout.setContentsMargins(10, 15, 10, 10)

        self.image_display_label = ClickableImageLabel()
        self.image_display_label.setText("No image loaded")
        self.image_display_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.image_display_label.setStyleSheet("QLabel { border: 2px dashed #cccccc; background-color: #f9f9f9; color: #666666; }")
        self.image_display_label.setMinimumSize(600, 450)
        self.image_display_label.mask_clicked.connect(self.on_mask_clicked)
        image_layout.addWidget(self.image_display_label, 1)

        right_layout.addWidget(image_group)

        layout.addWidget(left_widget)
        layout.addWidget(right_widget, 1)

    def goto_sam_parameters(self):
        main_window = self.window()
        if hasattr(main_window, 'goto_sam_parameters'):
            main_window.goto_sam_parameters()

    def get_filter_params(self):
        return {
            "use_area": True,
            "use_intensity": True,
            "a_threshold": self.a_threshold_spin.value(),
            "i_threshold": self.i_threshold_spin.value(),
            "r_threshold": self.r_threshold_spin.value(),
            "min_aggregate": self.min_aggregate_spin.value(),
            "max_aggregate": self.max_aggregate_spin.value()
        }

    def set_filter_params(self, params):
        self.a_threshold_spin.setValue(params.get("a_threshold", 0))
        self.i_threshold_spin.setValue(params.get("i_threshold", 0.00))
        self.r_threshold_spin.setValue(params.get("r_threshold", 0.00))
        self.min_aggregate_spin.setValue(params.get("min_aggregate", 0))
        self.max_aggregate_spin.setValue(params.get("max_aggregate", 100))

    def save_configuration(self):
        file_path, _ = QFileDialog.getSaveFileName(
            self,
            "Save Configuration",
            "",
            "JSON Files (*.json)"
        )
        if file_path:
            try:
                config = {
                    "filters": self.get_filter_params(),
                    "sam_hq": {}
                }

                main_window = self.window()
                if hasattr(main_window, 'get_sam_params'):
                    config["sam_hq"] = main_window.get_sam_params()

                with open(file_path, 'w') as f:
                    json.dump(config, f, indent=4)

                QMessageBox.information(self, "Save Successful", f"Configuration saved to: {file_path}")
            except Exception as e:
                QMessageBox.critical(self, "Save Error", f"Error saving configuration: {str(e)}")

    def load_configuration(self):
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Load Configuration",
            "",
            "JSON Files (*.json)"
        )
        if file_path:
            try:
                with open(file_path, 'r') as f:
                    config = json.load(f)

                if "filters" in config:
                    self.set_filter_params(config["filters"])

                if "sam_hq" in config:
                    main_window = self.window()
                    if hasattr(main_window, 'sam_params_widget'):
                        main_window.sam_params_widget.set_sam_params(config["sam_hq"])

                QMessageBox.information(self, "Load Successful", f"Configuration loaded from: {file_path}")
            except Exception as e:
                QMessageBox.critical(self, "Load Error", f"Error loading configuration: {str(e)}")

    def get_button_style(self, color="blue"):
        base_style = "QPushButton { padding: 8px 12px; color: white; border: none; border-radius: 4px; font-weight: bold; } QPushButton:disabled { background-color: #cccccc; color: #666666; }"
        colors = {
            "green": ("#4CAF50", "#45a049"),
            "blue": ("#2196F3", "#0b7dda"),
            "red": ("#f44336", "#d32f2f"),
            "grey": ("#607d8b", "#455a64"),
            "orange": ("#FF9800", "#FB8C00")
        }
        bg_color, hover_color = colors.get(color, colors["blue"])
        return base_style + f"QPushButton {{ background-color: {bg_color}; }} QPushButton:hover:!disabled {{ background-color: {hover_color}; }}"

    def select_image(self):
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Select Image",
            "",
            "Image Files (*.png *.jpg *.jpeg *.tif *.tiff)"
        )
        if file_path:
            self.selected_image_path = file_path
            self.image_path_label.setText(f"{os.path.basename(file_path)}")
            self.load_and_display_image()
            self.update_ui_state()

    def select_model(self):
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Select SAM Model File",
            "",
            "Model Files (*.pth)"
        )
        if file_path:
            self.model_path = file_path
            self.model_path_label.setText(f"{os.path.basename(file_path)}")
            self.update_ui_state()

    def load_and_display_image(self):
        if not self.selected_image_path:
            return

        try:
            self.original_image = cv2.imread(self.selected_image_path, cv2.IMREAD_COLOR)
            self.grayscale_image = cv2.imread(self.selected_image_path, cv2.IMREAD_GRAYSCALE)

            if self.original_image is None or self.grayscale_image is None:
                if tifffile:
                    try:
                        tiff_image = tifffile.imread(self.selected_image_path)
                        if len(tiff_image.shape) == 3 and tiff_image.shape[2] >= 3:
                            self.original_image = cv2.cvtColor(tiff_image[:,:,:3], cv2.COLOR_RGB2BGR)
                            self.grayscale_image = cv2.cvtColor(self.original_image, cv2.COLOR_BGR2GRAY)
                        elif len(tiff_image.shape) == 2:
                            self.grayscale_image = tiff_image
                            self.original_image = cv2.cvtColor(self.grayscale_image, cv2.COLOR_GRAY2BGR)

                        if self.original_image.dtype != np.uint8:
                            self.original_image = ((self.original_image / np.max(self.original_image)) * 255).astype(np.uint8) if np.max(self.original_image) > 0 else self.original_image.astype(np.uint8)
                        if self.grayscale_image.dtype != np.uint8:
                            self.grayscale_image = ((self.grayscale_image / np.max(self.grayscale_image)) * 255).astype(np.uint8) if np.max(self.grayscale_image) > 0 else self.grayscale_image.astype(np.uint8)
                    except Exception as e:
                        QMessageBox.critical(self, "Error", f"Cannot load TIFF image: {str(e)}")
                        return
                else:
                    QMessageBox.critical(self, "Error", "Cannot load image file")
                    return

            self.cached_display_params = None
            self.display_image(self.original_image, force_recalculate=True)

        except Exception as e:
            QMessageBox.critical(self, "Error", f"Error loading image: {str(e)}")

    def display_image(self, cv_image, selected_masks=None, force_recalculate=False):
        if cv_image is None:
            return

        if selected_masks is None:
            selected_masks = list(self.selected_mask_indices)

        if self.cached_display_params is None or force_recalculate:
            label_size = self.image_display_label.size()
            height, width, channel = cv_image.shape

            temp_image = cv_image.copy()
            bytes_per_line = 3 * width
            q_image = QImage(temp_image.data, width, height, bytes_per_line, QImage.Format.Format_RGB888).rgbSwapped()
            base_pixmap = QPixmap.fromImage(q_image)
            scaled_pixmap = base_pixmap.scaled(label_size, Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation)

            original_size = (width, height)
            displayed_size = (scaled_pixmap.width(), scaled_pixmap.height())
            scale_factor = min(displayed_size[0] / original_size[0], displayed_size[1] / original_size[1]) if original_size[0] > 0 and original_size[1] > 0 else 1

            self.cached_display_params = {
                'original_size': original_size,
                'displayed_size': displayed_size,
                'scale_factor': scale_factor,
                'label_size': label_size
            }

        display_image = cv_image.copy()

        if self.generated_masks:
            bright_colors = [
                (0, 165, 255), (0, 255, 255), (0, 255, 0),
                (147, 20, 255), (255, 0, 255), (255, 255, 0),
                (128, 0, 128), (255, 165, 0), (0, 128, 255)
            ]

            for mask_idx in range(len(self.generated_masks)):
                if mask_idx not in selected_masks:
                    mask_info = self.generated_masks[mask_idx]
                    segmentation_data = mask_info["segmentation"]

                    if isinstance(segmentation_data, dict):
                        if PYCOCOTOOLS_AVAILABLE:
                            try:
                                segmentation_mask_uint8 = mask_utils.decode(segmentation_data)
                            except:
                                continue
                        else:
                            continue
                    elif isinstance(segmentation_data, np.ndarray):
                        segmentation_mask_uint8 = segmentation_data.astype(np.uint8) * 255
                    else:
                        continue

                    contours, _ = cv2.findContours(segmentation_mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                    color_bgr = bright_colors[mask_idx % len(bright_colors)]
                    cv2.drawContours(display_image, contours, -1, color_bgr, 1)

            for mask_idx in selected_masks:
                if mask_idx < len(self.generated_masks):
                    mask_info = self.generated_masks[mask_idx]
                    segmentation_data = mask_info["segmentation"]

                    if isinstance(segmentation_data, dict):
                        if PYCOCOTOOLS_AVAILABLE:
                            try:
                                segmentation_mask_uint8 = mask_utils.decode(segmentation_data)
                            except:
                                continue
                        else:
                            continue
                    elif isinstance(segmentation_data, np.ndarray):
                        segmentation_mask_uint8 = segmentation_data.astype(np.uint8) * 255
                    else:
                        continue

                    contours, _ = cv2.findContours(segmentation_mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                    cv2.drawContours(display_image, contours, -1, (0, 0, 255), 3)

                    if contours:
                        M = cv2.moments(contours[0])
                        if M["m00"] != 0:
                            cx = int(M["m10"] / M["m00"])
                            cy = int(M["m01"] / M["m00"])
                            cv2.putText(display_image, f"M{mask_idx+1}", (cx-15, cy),
                                      cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

        height, width, channel = display_image.shape
        bytes_per_line = 3 * width
        q_image = QImage(display_image.data, width, height, bytes_per_line, QImage.Format.Format_RGB888).rgbSwapped()

        pixmap = QPixmap.fromImage(q_image)
        cached_label_size = self.cached_display_params['label_size']
        scaled_pixmap = pixmap.scaled(cached_label_size, Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation)

        self.image_display_label.set_masks_data(self.generated_masks)
        self.image_display_label.set_image_info(
            self.cached_display_params['original_size'],
            self.cached_display_params['displayed_size'],
            self.cached_display_params['scale_factor'],
            (0, 0)
        )
        self.image_display_label.setPixmap(scaled_pixmap)

    def clear_display_cache(self):
        self.cached_display_params = None
        self.cached_base_pixmap = None

    def resizeEvent(self, event):
        super().resizeEvent(event)
        if self.original_image is not None and self.cached_display_params is not None:
            QTimer.singleShot(100, self.refresh_display)

    def refresh_display(self):
        if self.original_image is not None:
            self.clear_display_cache()
            self.display_image(self.original_image, force_recalculate=True)

    def generate_masks(self):
        if not self.selected_image_path or not hasattr(self, 'model_path') or not self.model_path or not SEGMENT_ANYTHING_AVAILABLE:
            QMessageBox.warning(self, "Missing Requirements", "Please select both image and model files.")
            return

        if self.worker_thread and self.worker_thread.isRunning():
            QMessageBox.information(self, "Processing", "Mask generation is already in progress.")
            return

        try:
            self.progress_bar.setVisible(True)
            self.progress_label.setVisible(True)
            self.progress_label.setText("Preparing to start...")

            self.generate_masks_btn.setEnabled(False)
            self.generate_masks_btn.setText("Generating...")

            main_window = self.window()
            if hasattr(main_window, 'get_sam_params'):
                sam_params = main_window.get_sam_params()
            else:
                sam_params = {
                    "points_per_side": 32,
                    "points_per_batch": 64,
                    "pred_iou_thresh": 0.88,
                    "stability_score_thresh": 0.95,
                    "stability_score_offset": 1.00,
                    "box_nms_thresh": 0.70,
                    "min_mask_region_area": 0,
                }

            self.worker_thread = MaskGenerationWorker(self.original_image, self.model_path, sam_params)
            self.worker_thread.progress_updated.connect(self.update_progress)
            self.worker_thread.masks_generated.connect(self.on_masks_generated)
            self.worker_thread.error_occurred.connect(self.on_generation_error)
            self.worker_thread.finished.connect(self.on_generation_finished)
            self.worker_thread.start()

        except Exception as e:
            self.on_generation_error(f"An error occurred when starting mask generation: {str(e)}")

    def update_progress(self, message):
        self.progress_label.setText(message)
        QApplication.processEvents()

    def on_masks_generated(self, masks):
        self.generated_masks = masks
        self.populate_mask_list()
        self.display_image(self.original_image)

    def on_generation_error(self, error_message):
        QMessageBox.critical(self, "Error", error_message)

    def on_generation_finished(self):
        self.progress_bar.setVisible(False)
        self.progress_label.setVisible(False)

        self.generate_masks_btn.setEnabled(True)
        self.generate_masks_btn.setText("Generate Masks")

        if self.generated_masks:
            QMessageBox.information(self, "Success", f"Successfully generated {len(self.generated_masks)} masks!")

        self.update_ui_state()

    def populate_mask_list(self):
        self.selected_mask_indices.clear()
        self.update_selection_status()
        self.update_ui_state()

    def update_selection_status(self):
        if not self.generated_masks:
            self.selection_status_label.setText("No masks generated yet")
        else:
            total_masks = len(self.generated_masks)
            selected_count = len(self.selected_mask_indices)
            if selected_count == 0:
                self.selection_status_label.setText(f"0/{total_masks} masks selected")
            else:
                selected_list = sorted(list(self.selected_mask_indices))
                if len(selected_list) <= 5:
                    mask_numbers = [str(i+1) for i in selected_list]
                    self.selection_status_label.setText(f"{selected_count}/{total_masks} masks selected: M{', M'.join(mask_numbers)}")
                else:
                    self.selection_status_label.setText(f"{selected_count}/{total_masks} masks selected")

    def select_all_masks(self):
        self.selected_mask_indices = set(range(len(self.generated_masks)))
        self.update_selection_status()
        self.display_image(self.original_image)
        self.update_ui_state()

    def deselect_all_masks(self):
        self.selected_mask_indices.clear()
        self.update_selection_status()
        self.display_image(self.original_image)
        self.update_ui_state()

    def on_mask_clicked(self, mask_idx, is_right_click):
        if mask_idx >= len(self.generated_masks):
            return

        if is_right_click:
            if mask_idx in self.selected_mask_indices:
                self.selected_mask_indices.remove(mask_idx)
        else:
            if mask_idx in self.selected_mask_indices:
                self.selected_mask_indices.remove(mask_idx)
            else:
                self.selected_mask_indices.add(mask_idx)

        self.update_selection_status()
        self.display_image(self.original_image)
        self.update_ui_state()

    def calculate_selected_masks(self):
        selected_masks = list(self.selected_mask_indices)

        if not selected_masks:
            QMessageBox.warning(self, "No Selection", "Please click on masks in the image to select them first.")
            return

        filter_params = self.get_filter_params()

        results = []
        filtered_count = 0
        total_area = 0
        total_intensity = 0.0

        for mask_idx in selected_masks:
            mask_info = self.generated_masks[mask_idx]
            segmentation_data = mask_info["segmentation"]

            segmentation_mask_bool = None
            if isinstance(segmentation_data, dict):
                if PYCOCOTOOLS_AVAILABLE:
                    try:
                        segmentation_mask_uint8 = mask_utils.decode(segmentation_data)
                        segmentation_mask_bool = segmentation_mask_uint8.astype(bool)
                    except:
                        continue
                else:
                    continue
            elif isinstance(segmentation_data, np.ndarray):
                segmentation_mask_bool = segmentation_data.astype(bool)
            else:
                continue

            if segmentation_mask_bool is None:
                continue

            region_pixels_gray = self.grayscale_image[segmentation_mask_bool]
            if len(region_pixels_gray) == 0:
                continue

            aggregate_values_0_100 = 100.0 - (region_pixels_gray.astype(np.float32) / 255.0) * 100.0

            initial_area = len(aggregate_values_0_100)
            initial_intensity = np.sum(aggregate_values_0_100)
            initial_ratio = initial_intensity / initial_area if initial_area > 0 else 0.0

            passed_phase1 = True
            if initial_area < filter_params["a_threshold"]:
                passed_phase1 = False
            if passed_phase1 and initial_intensity < filter_params["i_threshold"]:
                passed_phase1 = False
            if passed_phase1 and initial_ratio < filter_params["r_threshold"]:
                passed_phase1 = False

            if not passed_phase1:
                continue

            pixels_within_agg_range_mask = ((aggregate_values_0_100 >= filter_params["min_aggregate"]) &
                                            (aggregate_values_0_100 <= filter_params["max_aggregate"]))
            num_pixels_passed_phase2 = np.count_nonzero(pixels_within_agg_range_mask)

            if num_pixels_passed_phase2 == 0:
                continue

            final_aggregates = aggregate_values_0_100[pixels_within_agg_range_mask]
            final_area = num_pixels_passed_phase2
            final_intensity = np.sum(final_aggregates)
            final_ratio = final_intensity / final_area if final_area > 0 else 0.0
            filtered_count += 1

            results.append({
                'mask_idx': mask_idx + 1,
                'area': final_area,
                'intensity': final_intensity,
                'ratio': final_ratio
            })

            total_area += final_area
            total_intensity += final_intensity

        self.display_filtered_results(results, total_area, total_intensity, filtered_count, len(selected_masks))
        self.current_results = results
        self.update_ui_state()

    def display_results(self, results, total_area, total_intensity):
        result_text = "Selected Masks Analysis Results:\n"
        result_text += "=" * 50 + "\n\n"

        for result in results:
            result_text += f"Mask {result['mask_idx']}:\n"
            result_text += f"  Area (A): {result['area']}\n"
            result_text += f"  Intensity (I): {result['intensity']:.2f}\n"
            result_text += f"  Ratio (R): {result['ratio']:.2f}\n\n"

        if len(results) > 1:
            total_ratio = total_intensity / total_area if total_area > 0 else 0.0
            result_text += f"Total Summary:\n"
            result_text += f"  Total Area: {total_area}\n"
            result_text += f"  Total Intensity: {total_intensity:.2f}\n"
            result_text += f"  Overall Ratio: {total_ratio:.2f}\n"

        self.results_text.setPlainText(result_text)

    def display_filtered_results(self, results, total_area, total_intensity, filtered_count, selected_count):
        result_text = f"Analysis Results ({filtered_count}/{selected_count} masks passed filtering):\n"
        result_text += "=" * 60 + "\n\n"

        if not results:
            result_text += "No masks passed the filter.\n"
            result_text += "Please try adjusting the filter parameters.\n"
            self.results_text.setPlainText(result_text)
            return

        for result in results:
            result_text += f"Mask {result['mask_idx']}:\n"
            result_text += f"  Area (A): {result['area']}\n"
            result_text += f"  Intensity (I): {result['intensity']:.2f}\n"
            result_text += f"  Ratio (R): {result['ratio']:.2f}\n\n"

        if len(results) > 1:
            total_ratio = total_intensity / total_area if total_area > 0 else 0.0
            result_text += f"Summary:\n"
            result_text += f"  Total Area: {total_area}\n"
            result_text += f"  Total Intensity: {total_intensity:.2f}\n"
            result_text += f"  Overall Ratio: {total_ratio:.2f}\n"

        self.results_text.setPlainText(result_text)

    def export_results(self):
        if not hasattr(self, 'current_results') or not self.current_results:
            QMessageBox.warning(self, "No Results", "No calculation results to export.")
            return

        file_path, _ = QFileDialog.getSaveFileName(
            self,
            "Export Results",
            f"mask_analysis_{os.path.splitext(os.path.basename(self.selected_image_path))[0]}.txt",
            "Text Files (*.txt)"
        )

        if file_path:
            try:
                with open(file_path, 'w', encoding='utf-8') as f:
                    f.write(f"Mask Analysis Results\n")
                    f.write(f"Image: {os.path.basename(self.selected_image_path)}\n")
                    f.write(f"Analysis Time: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")

                    filter_params = self.get_filter_params()
                    f.write(f"\nApplied Filter Parameters:\n")
                    f.write(f"  Min Area Threshold: {filter_params['a_threshold']}\n")
                    f.write(f"  Min Intensity Threshold: {filter_params['i_threshold']:.2f}\n")
                    f.write(f"  Min Ratio Threshold: {filter_params['r_threshold']:.2f}\n")
                    f.write(f"  Aggregate Filter: {filter_params['min_aggregate']}-{filter_params['max_aggregate']}\n")

                    f.write("=" * 60 + "\n\n")

                    total_area = 0
                    total_intensity = 0.0

                    for result in self.current_results:
                        f.write(f"Mask {result['mask_idx']}:\n")
                        f.write(f"  Area: {result['area']}\n")
                        f.write(f"  Intensity: {result['intensity']:.2f}\n")
                        f.write(f"  Ratio: {result['ratio']:.2f}\n\n")
                        total_area += result['area']
                        total_intensity += result['intensity']

                    if len(self.current_results) > 1:
                        total_ratio = total_intensity / total_area if total_area > 0 else 0.0
                        f.write(f"Total Summary:\n")
                        f.write(f"  Total Area: {total_area}\n")
                        f.write(f"  Total Intensity: {total_intensity:.2f}\n")
                        f.write(f"  Overall Ratio: {total_ratio:.2f}\n")

                QMessageBox.information(self, "Export Successful", f"Results exported to: {file_path}")
            except Exception as e:
                QMessageBox.critical(self, "Export Error", f"Error exporting results: {str(e)}")

    def save_marked_image(self):
        if self.original_image is None:
            QMessageBox.warning(self, "No Image", "No image to save.")
            return

        selected_masks = list(self.selected_mask_indices)

        if not selected_masks:
            QMessageBox.warning(self, "No Selection", "No masks selected to mark.")
            return

        file_path, _ = QFileDialog.getSaveFileName(
            self,
            "Save Marked Image",
            f"marked_{os.path.splitext(os.path.basename(self.selected_image_path))[0]}.png",
            "PNG Files (*.png);;JPEG Files (*.jpg)"
        )

        if file_path:
            try:
                marked_image = self.original_image.copy()
                bright_colors = [
                    (0, 0, 255), (0, 165, 255), (0, 255, 255), (0, 255, 0),
                    (255, 0, 0), (147, 20, 255), (255, 0, 255), (255, 255, 0),
                    (128, 0, 128),
                ]

                for i, mask_idx in enumerate(selected_masks):
                    if mask_idx < len(self.generated_masks):
                        mask_info = self.generated_masks[mask_idx]
                        segmentation_data = mask_info["segmentation"]

                        if isinstance(segmentation_data, dict):
                            if PYCOCOTOOLS_AVAILABLE:
                                try:
                                    segmentation_mask_uint8 = mask_utils.decode(segmentation_data)
                                except:
                                    continue
                            else:
                                continue
                        elif isinstance(segmentation_data, np.ndarray):
                            segmentation_mask_uint8 = segmentation_data.astype(np.uint8) * 255
                        else:
                            continue

                        contours, _ = cv2.findContours(segmentation_mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                        color_bgr = bright_colors[i % len(bright_colors)]
                        cv2.drawContours(marked_image, contours, -1, color_bgr, 3)

                        if contours:
                            M = cv2.moments(contours[0])
                            if M["m00"] != 0:
                                cx = int(M["m10"] / M["m00"])
                                cy = int(M["m01"] / M["m00"])
                                cv2.putText(marked_image, f"M{mask_idx+1}", (cx-15, cy),
                                          cv2.FONT_HERSHEY_SIMPLEX, 1.0, color_bgr, 3)

                cv2.imwrite(file_path, marked_image)
                QMessageBox.information(self, "Save Successful", f"Marked image saved to: {file_path}")
            except Exception as e:
                QMessageBox.critical(self, "Save Error", f"Error saving marked image: {str(e)}")

    def update_ui_state(self):
        has_image = self.selected_image_path is not None
        has_model = hasattr(self, 'model_path') and self.model_path is not None
        has_masks = len(self.generated_masks) > 0
        has_selected_masks = len(self.selected_mask_indices) > 0
        has_results = hasattr(self, 'current_results') and bool(self.current_results)

        self.generate_masks_btn.setEnabled(has_image and has_model and SEGMENT_ANYTHING_AVAILABLE)
        self.select_all_btn.setEnabled(has_masks and len(self.selected_mask_indices) < len(self.generated_masks))
        self.deselect_all_btn.setEnabled(has_selected_masks)
        self.calculate_btn.setEnabled(has_selected_masks)
        self.export_results_btn.setEnabled(has_results)
        self.save_image_btn.setEnabled(has_image and has_selected_masks)


if __name__ == "__main__":
    app = QApplication(sys.argv)
    if not SEGMENT_ANYTHING_AVAILABLE:
        msg_box = QMessageBox(); msg_box.setIcon(QMessageBox.Icon.Critical); msg_box.setWindowTitle("Dependency Missing")
        msg_box.setText("The 'segment_anything' library is required to run this program.\nPlease ensure it and its dependencies (like 'timm') are installed correctly according to the environment setup guide.")
        msg_box.setStandardButtons(QMessageBox.StandardButton.Ok); msg_box.exec(); sys.exit(1)

    window = HQSAMMainWindow()
    window.show()
    sys.exit(app.exec())
