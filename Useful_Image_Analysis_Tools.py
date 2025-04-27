import sys
import os
import cv2
import numpy as np
import torch
import random
import time
from PIL import Image, ImageEnhance
from PyQt6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout,
                             QHBoxLayout, QPushButton, QLabel, QFileDialog,
                             QSlider, QComboBox,
                             QGroupBox, QTextEdit, QTabWidget, QSpinBox,
                             QCheckBox, QDoubleSpinBox, QFormLayout, QScrollArea,
                             QLineEdit, QMessageBox, QButtonGroup, QRadioButton)
from PyQt6.QtCore import Qt, QThread, pyqtSignal, QMutex
from PyQt6.QtGui import QPixmap, QImage, QFont, QColor, QTextCursor
try:
    from segment_anything import sam_model_registry, SamAutomaticMaskGenerator
    from segment_anything.utils.amg import rle_to_mask
except ImportError:
    print("Error: 'segment_anything' library not found.")
    print("Please install it using: pip install segment-anything")
    sys.exit(1)

try:
    from pycocotools import mask as mask_utils
    pycocotools_available = True
except ImportError:
    pycocotools_available = False
    print("Warning: 'pycocotools' library not found. RLE output modes ('uncompressed_rle', 'coco_rle') decoding will be unavailable.")
    print("If you need to use RLE modes, please run: pip install pycocotools")

try:
    import tifffile
except ImportError:
    tifffile = None

class ImageAggregateAnalyzer:
    @staticmethod
    def calculate_aggregate(image_path):
        try:
            if not os.path.exists(image_path):
                return f"Error: File not found - {image_path}"
            if not os.access(image_path, os.R_OK):
                return f"Error: No read permission for - {image_path}"

            img = Image.open(image_path)
            if img.mode != 'L':
                img = img.convert('L')
            img_array = np.array(img)

            avg_pixel_value = np.mean(img_array)
            normalized_aggregate = 100.0 - (avg_pixel_value / 255.0) * 100.0
            return normalized_aggregate
        except Exception as e:
            return f"Error: {str(e)}"

class HeatmapGeneratorThread(QThread):
    processing_finished = pyqtSignal()
    log_updated = pyqtSignal(str)

    def __init__(self, input_folder, output_folder, min_aggregate_threshold, max_aggregate_threshold):
        super().__init__()
        self.input_folder = input_folder
        self.output_folder = output_folder
        self.min_aggregate_threshold = min_aggregate_threshold
        self.max_aggregate_threshold = max_aggregate_threshold
        self.running = True
        self.lock = QMutex()

    def validate_image(self, image_path):
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image not found: {image_path}")
        if not os.access(image_path, os.R_OK):
            raise PermissionError(f"No read permission for: {image_path}")

    def run(self):
        try:
            if self.min_aggregate_threshold >= self.max_aggregate_threshold:
                 if not (self.min_aggregate_threshold == 100 and self.max_aggregate_threshold == 100):
                    raise ValueError("Min aggregate threshold must be strictly lower than Max aggregate threshold (unless both are 100).")

            if not os.path.exists(self.input_folder):
                raise FileNotFoundError(f"Input folder not found: {self.input_folder}")

            os.makedirs(self.output_folder, exist_ok=True)
            if not os.access(self.output_folder, os.W_OK):
                raise PermissionError(f"No write permission for: {self.output_folder}")

            self.log_updated.emit(f"Output folder: {self.output_folder}")

            files = [f for f in os.listdir(self.input_folder)
                     if f.lower().endswith((".png", ".jpg", ".jpeg", ".tif", ".tiff"))]

            if not files:
                self.log_updated.emit("Warning: No image files found in input folder")
                self.processing_finished.emit()
                return

            self.log_updated.emit(f"Found {len(files)} image files")
            processed_count = 0

            for idx, filename in enumerate(files):
                self.lock.lock()
                try:
                    if not self.running:
                        self.log_updated.emit("Processing stopped by user.")
                        break
                finally:
                    self.lock.unlock()

                try:
                    input_path = os.path.join(self.input_folder, filename)
                    self.validate_image(input_path)

                    self.log_updated.emit(f"Processing image {idx+1}/{len(files)}: {filename}")

                    start_time = time.time()
                    result = HeatmapGenerator.generate_heatmap(
                        input_path,
                        self.output_folder,
                        self.min_aggregate_threshold,
                        self.max_aggregate_threshold
                    )
                    elapsed_time = time.time() - start_time

                    if isinstance(result, str) and not result.startswith("Error"):
                        processed_count += 1
                        self.log_updated.emit(f"Heatmap saved to: {result}")
                        self.log_updated.emit(f"Processing time: {elapsed_time:.2f} seconds")
                    elif isinstance(result, str):
                         self.log_updated.emit(f"Error processing {filename}: {result}")

                except Exception as e:
                    self.log_updated.emit(f"Error processing {filename}: {str(e)}")
                    continue

            if self.running:
                self.log_updated.emit(f"Completed: {processed_count}/{len(files)} heatmaps generated")

        except Exception as e:
            self.log_updated.emit(f"Fatal error during heatmap generation: {str(e)}")
        finally:
            self.processing_finished.emit()

    def stop(self):
        self.lock.lock()
        try:
            self.running = False
        finally:
            self.lock.unlock()

class HeatmapGenerator:
    @staticmethod
    def generate_heatmap(image_path, output_folder, min_aggregate_threshold=0, max_aggregate_threshold=100):
        try:
            image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
            if image is None:
                if tifffile:
                    try:
                        tiff_image = tifffile.imread(image_path)
                        if len(tiff_image.shape) > 2:
                            image = tiff_image[:, :, 0]
                        else:
                            image = tiff_image

                        if image.dtype != np.uint8:
                            if np.max(image) > 0:
                                image = ((image / np.max(image)) * 255).astype(np.uint8)
                            else:
                                image = image.astype(np.uint8)

                    except Exception as tiff_e:
                        return f"Error: Failed to read image {os.path.basename(image_path)} with cv2 and tifffile: {tiff_e}"
                else:
                    return f"Error: Failed to read image {os.path.basename(image_path)} with cv2 (tifffile not installed)"

            if image is None:
                 return f"Error: Could not load image {os.path.basename(image_path)}"

            output_heatmap_bgr = np.zeros((image.shape[0], image.shape[1], 3), dtype=np.uint8)

            aggregate_values = 100.0 - (image.astype(np.float32) / 255.0) * 100.0

            threshold_range = max_aggregate_threshold - min_aggregate_threshold
            if threshold_range <= 0:
                print(f"Warning: generate_heatmap called with min_aggregate_threshold ({min_aggregate_threshold}) >= max_aggregate_threshold ({max_aggregate_threshold}). Coloring pixels >= {min_aggregate_threshold}.")
                below_min_mask = aggregate_values < min_aggregate_threshold
                above_max_mask = aggregate_values >= min_aggregate_threshold
                within_range_mask = np.zeros_like(below_min_mask, dtype=bool)
            else:
                below_min_mask = aggregate_values < min_aggregate_threshold
                above_max_mask = aggregate_values > max_aggregate_threshold
                within_range_mask = (~below_min_mask) & (~above_max_mask)

            output_heatmap_bgr[above_max_mask] = [0, 0, 255]

            num_pixels_in_range = np.count_nonzero(within_range_mask)
            if num_pixels_in_range > 0 and threshold_range > 0:
                aggregates_in_range = aggregate_values[within_range_mask]
                normalized_in_range = (aggregates_in_range - min_aggregate_threshold) / threshold_range
                normalized_in_range = np.clip(normalized_in_range, 0.0, 1.0)

                hue_values = (1.0 - normalized_in_range) * 135.0
                hue_values = hue_values.astype(np.uint8)

                hsv_pixels = np.zeros((num_pixels_in_range, 1, 3), dtype=np.uint8)
                hsv_pixels[:, 0, 0] = hue_values
                hsv_pixels[:, 0, 1] = 255
                hsv_pixels[:, 0, 2] = 255

                bgr_pixels = cv2.cvtColor(hsv_pixels, cv2.COLOR_HSV2BGR)

                output_heatmap_bgr[within_range_mask] = bgr_pixels[:, 0, :]

            base_name = os.path.splitext(os.path.basename(image_path))[0]
            output_filename = f"{base_name}_heatmap.png"
            output_path = os.path.join(output_folder, output_filename)
            cv2.imwrite(output_path, output_heatmap_bgr)
            return output_path
        except Exception as e:
            import traceback
            print(f"Error generating heatmap for {os.path.basename(image_path)}:")
            print(traceback.format_exc())
            return f"Error generating heatmap for {os.path.basename(image_path)}: {str(e)}"


class ContrastEnhancementThread(QThread):
    processing_finished = pyqtSignal()
    log_updated = pyqtSignal(str)

    def __init__(self, input_folder, output_folder, contrast_factor):
        super().__init__()
        self.input_folder = input_folder
        self.output_folder = output_folder
        self.contrast_factor = contrast_factor
        self.running = True
        self.lock = QMutex()

    def validate_image(self, image_path):
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image not found: {image_path}")
        if not os.access(image_path, os.R_OK):
            raise PermissionError(f"No read permission for: {image_path}")

    def run(self):
        try:
            if not os.path.exists(self.input_folder):
                raise FileNotFoundError(f"Input folder not found: {self.input_folder}")

            os.makedirs(self.output_folder, exist_ok=True)
            if not os.access(self.output_folder, os.W_OK):
                raise PermissionError(f"No write permission for: {self.output_folder}")

            self.log_updated.emit(f"Output folder: {self.output_folder}")

            files = [f for f in os.listdir(self.input_folder)
                     if f.lower().endswith((".png", ".jpg", ".jpeg", ".tif", ".tiff"))]

            if not files:
                self.log_updated.emit("Warning: No image files found in input folder")
                self.processing_finished.emit()
                return

            self.log_updated.emit(f"Found {len(files)} image files")
            processed_count = 0

            for idx, filename in enumerate(files):
                self.lock.lock()
                try:
                    if not self.running:
                        self.log_updated.emit("Processing stopped by user.")
                        break
                finally:
                    self.lock.unlock()

                try:
                    input_path = os.path.join(self.input_folder, filename)
                    self.validate_image(input_path)

                    self.log_updated.emit(f"Processing image {idx+1}/{len(files)}: {filename}")

                    start_time = time.time()
                    img = Image.open(input_path)
                    if img.mode not in ('RGB', 'L'):
                        img = img.convert('RGB')

                    enhancer = ImageEnhance.Contrast(img)
                    enhanced_img = enhancer.enhance(self.contrast_factor)

                    base_name, ext = os.path.splitext(filename)
                    output_filename = f"{base_name}_contrast{ext}"
                    output_path = os.path.join(self.output_folder, output_filename)
                    enhanced_img.save(output_path)

                    elapsed_time = time.time() - start_time
                    processed_count += 1
                    self.log_updated.emit(f"Enhanced image saved to: {output_path}")
                    self.log_updated.emit(f"Processing time: {elapsed_time:.2f} seconds")

                except Exception as e:
                    self.log_updated.emit(f"Error processing {filename}: {str(e)}")
                    continue

            if self.running:
                self.log_updated.emit(f"Completed: {processed_count}/{len(files)} images enhanced")

        except Exception as e:
            self.log_updated.emit(f"Fatal error during contrast enhancement: {str(e)}")
        finally:
            self.processing_finished.emit()

    def stop(self):
        self.lock.lock()
        try:
            self.running = False
        finally:
            self.lock.unlock()


class ImageProcessorThread(QThread):
    processing_finished = pyqtSignal()
    image_processed = pyqtSignal(str, np.ndarray)
    log_updated = pyqtSignal(str)

    MAX_IMAGE_SIZE = 4096
    LOG_MAX_LINES = 1000

    bright_colors = [
        (0, 0, 255),
        (0, 165, 255),
        (0, 255, 255),
        (0, 255, 0),
        (147, 20, 255),
    ]

    def __init__(self, input_folder, output_folder,
                 min_aggregate_threshold, max_aggregate_threshold,
                 a_threshold, i_threshold, r_threshold, model_path,
                 use_intensity=False, use_area=True, sam_params=None):
        super().__init__()
        self.input_folder = input_folder
        self.output_folder = output_folder
        self.min_aggregate_threshold = min_aggregate_threshold
        self.max_aggregate_threshold = max_aggregate_threshold
        self.a_threshold = a_threshold
        self.i_threshold = i_threshold
        self.r_threshold = r_threshold
        self.model_path = model_path
        self.running = True
        self.use_intensity = use_intensity
        self.use_area = use_area
        self.sam_params = sam_params if sam_params else {}
        self.lock = QMutex()

    def validate_thresholds(self):
        pass


    def validate_image(self, image_path):
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image not found: {image_path}")
        if not os.access(image_path, os.R_OK):
            raise PermissionError(f"No read permission for: {image_path}")

    def run(self):
        try:
            if not os.path.exists(self.input_folder):
                raise FileNotFoundError(f"Input folder not found: {self.input_folder}")

            os.makedirs(self.output_folder, exist_ok=True)
            if not os.access(self.output_folder, os.W_OK):
                raise PermissionError(f"No write permission for: {self.output_folder}")

            self.log_updated.emit(f"Output folder: {self.output_folder}")
            self.log_updated.emit("Loading SAM model...")

            device = "cuda" if torch.cuda.is_available() else "cpu"
            self.log_updated.emit(f"Using device: {device}")

            try:
                if not os.path.exists(self.model_path):
                        raise FileNotFoundError(f"SAM model file not found: {self.model_path}")
                if not os.access(self.model_path, os.R_OK):
                        raise PermissionError(f"No read permission for SAM model file: {self.model_path}")

                model_type = "vit_h"
                if "vit_l" in os.path.basename(self.model_path).lower():
                    model_type = "vit_l"
                elif "vit_b" in os.path.basename(self.model_path).lower():
                    model_type = "vit_b"
                self.log_updated.emit(f"Attempting to load SAM model type: {model_type}")

                sam = sam_model_registry[model_type](checkpoint=self.model_path).to(device)
                sam.eval()

            except KeyError:
                raise RuntimeError(f"Invalid SAM model type '{model_type}' inferred from filename. Check model file or registry.")
            except FileNotFoundError as e:
                raise e
            except PermissionError as e:
                raise e
            except Exception as e:
                if "checksum mismatch" in str(e):
                    raise RuntimeError(f"Failed to load SAM model: Checksum mismatch. Model file might be corrupt or incomplete. ({e})")
                elif "unexpected keyword argument 'checkpoint'" in str(e):
                    raise RuntimeError(f"Failed to load SAM model: Mismatch between segment-anything library version and model file. Try updating the library. ({e})")
                else:
                    raise RuntimeError(f"Failed to load SAM model: {str(e)}")

            mask_generator = SamAutomaticMaskGenerator(
                model=sam,
                points_per_side=self.sam_params.get('points_per_side', 32),
                points_per_batch=self.sam_params.get('points_per_batch', 64),
                pred_iou_thresh=self.sam_params.get('pred_iou_thresh', 0.88),
                stability_score_thresh=self.sam_params.get('stability_score_thresh', 0.95),
                stability_score_offset=self.sam_params.get('stability_score_offset', 1.0),
                box_nms_thresh=self.sam_params.get('box_nms_thresh', 0.7),
                crop_n_layers=self.sam_params.get('crop_n_layers', 0),
                crop_nms_thresh=self.sam_params.get('crop_nms_thresh', 0.7),
                crop_overlap_ratio=self.sam_params.get('crop_overlap_ratio', 512/1500),
                crop_n_points_downscale_factor=self.sam_params.get('crop_n_points_downscale_factor', 1),
                min_mask_region_area=self.sam_params.get('min_mask_region_area', 0),
                output_mode=self.sam_params.get('output_mode', 'binary_mask')
            )

            self.log_updated.emit("SAM parameters:")
            for key, value in self.sam_params.items():
                self.log_updated.emit(f"  - {key}: {value}")

            files = [f for f in os.listdir(self.input_folder)
                     if f.lower().endswith((".png", ".jpg", ".jpeg", ".tif", ".tiff"))]

            if not files:
                self.log_updated.emit("Warning: No image files found in input folder")
                self.processing_finished.emit()
                return

            self.log_updated.emit(f"Found {len(files)} image files")
            processed_count = 0

            for idx, filename in enumerate(files):
                self.lock.lock()
                try:
                    if not self.running:
                        self.log_updated.emit("Processing stopped by user.")
                        break
                finally:
                    self.lock.unlock()

                try:
                    input_path = os.path.join(self.input_folder, filename)
                    self.validate_image(input_path)

                    file_base, file_ext = os.path.splitext(filename)
                    output_filename = f"{file_base}_sam_processed{file_ext}"
                    output_path = os.path.join(self.output_folder, output_filename)

                    self.log_updated.emit(f"Processing image {idx+1}/{len(files)}: {filename}")

                    start_time = time.time()
                    processed_image = self.process_image(input_path, output_path, mask_generator)
                    elapsed_time = time.time() - start_time

                    if processed_image is not None:
                        self.image_processed.emit(output_path, processed_image)
                        processed_count += 1
                        self.log_updated.emit(f"Processed image saved to: {output_path}")

                    self.log_updated.emit(f"Processing time: {elapsed_time:.2f} seconds")

                except Exception as e:
                    self.log_updated.emit(f"Error processing {filename}: {str(e)}")
                    continue

            if self.running:
                self.log_updated.emit(f"Completed: {processed_count}/{len(files)} images processed")

        except RuntimeError as e:
            if "CUDA out of memory" in str(e):
                self.log_updated.emit("Error: GPU memory full - try smaller images or reduce SAM parameters (e.g., points_per_side, points_per_batch, disable cropping)")
            else:
                self.log_updated.emit(f"Runtime error: {str(e)}")
        except Exception as e:
            self.log_updated.emit(f"Fatal error during processing: {str(e)}")
        finally:
            if 'sam' in locals() and device == 'cuda':
                del sam
                if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                        self.log_updated.emit("CUDA cache cleared.")
            self.processing_finished.emit()

    def process_image(self, input_path, output_path, mask_generator):
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
                         else:
                             raise ValueError("Unsupported TIFF structure")

                         if original_image_bgr.dtype != np.uint8:
                             if np.max(original_image_bgr) > 0: original_image_bgr = ((original_image_bgr / np.max(original_image_bgr)) * 255).astype(np.uint8)
                             else: original_image_bgr = original_image_bgr.astype(np.uint8)
                         if grayscale_image.dtype != np.uint8:
                             if np.max(grayscale_image) > 0: grayscale_image = ((grayscale_image / np.max(grayscale_image)) * 255).astype(np.uint8)
                             else: grayscale_image = grayscale_image.astype(np.uint8)

                     except Exception as tiff_e:
                         self.log_updated.emit(f"Warning: Failed to read {os.path.basename(input_path)} with tifffile after cv2 failed: {tiff_e}")
                         return None
                 else:
                     self.log_updated.emit(f"Error: Failed to read image {os.path.basename(input_path)} (cv2 failed, tifffile not installed)")
                     return None

            if original_image_bgr is None or grayscale_image is None:
                 self.log_updated.emit(f"Error: Could not load image {os.path.basename(input_path)} in a usable format.")
                 return None

            if original_image_bgr.shape[0] > self.MAX_IMAGE_SIZE or original_image_bgr.shape[1] > self.MAX_IMAGE_SIZE:
                self.log_updated.emit(f"Error: Image {os.path.basename(input_path)} too large ({original_image_bgr.shape[0]}x{original_image_bgr.shape[1]}), max allowed is {self.MAX_IMAGE_SIZE}x{self.MAX_IMAGE_SIZE}")
                return None

            self.log_updated.emit(f"  Generating masks for {os.path.basename(input_path)}...")
            try:
                masks = mask_generator.generate(original_image_bgr)
                self.log_updated.emit(f"  Found {len(masks)} masks.")
            except torch.cuda.OutOfMemoryError:
                self.log_updated.emit(f"Error: GPU memory exhausted during mask generation for {os.path.basename(input_path)}. Try smaller image or reduce SAM parameters.")
                if torch.cuda.is_available(): torch.cuda.empty_cache()
                return None
            except Exception as e:
                self.log_updated.emit(f"Error during mask generation for {os.path.basename(input_path)}: {str(e)}")
                return None

            output_image = original_image_bgr.copy()
            height, width = output_image.shape[:2]
            drawn_text_boxes = []

            for i, mask_info in enumerate(masks):
                segmentation_data = mask_info["segmentation"]
                segmentation_mask = None

                if isinstance(segmentation_data, dict):
                    if pycocotools_available:
                        try:
                            segmentation_mask = mask_utils.decode(segmentation_data).astype(np.uint8)
                        except Exception as decode_e:
                            self.log_updated.emit(f"  Warning: Failed to decode RLE mask ({decode_e}), skipping mask.")
                            continue
                    else:
                        self.log_updated.emit(f"  Error: Detected RLE mask but 'pycocotools' not installed or failed to import. Cannot process this mask. Please install 'pycocotools' or select 'binary_mask' output mode.")
                        continue

                elif isinstance(segmentation_data, np.ndarray):
                    segmentation_mask = segmentation_data.astype(np.uint8)
                else:
                    self.log_updated.emit(f"  Warning: Unknown mask format type ({type(segmentation_data)}), skipping mask.")
                    continue

                if segmentation_mask is None:
                    continue

                region_pixels_0_255 = grayscale_image[segmentation_mask > 0]

                if len(region_pixels_0_255) == 0:
                    continue

                aggregate_values_0_100 = 100.0 - (region_pixels_0_255 / 255.0) * 100.0

                initial_area = 0
                initial_intensity = 0.0
                initial_ratio = 0.0
                passed_phase1 = True

                if self.use_area:
                    initial_area = len(aggregate_values_0_100)
                    if initial_area < self.a_threshold:
                        passed_phase1 = False

                if passed_phase1 and self.use_intensity:
                    initial_intensity = np.sum(aggregate_values_0_100)
                    if initial_intensity < self.i_threshold:
                        passed_phase1 = False

                if passed_phase1 and self.use_area and self.use_intensity and initial_area > 0:
                    initial_ratio = initial_intensity / initial_area
                    if initial_ratio < self.r_threshold:
                        passed_phase1 = False

                if passed_phase1:
                    pixels_passed_phase2_mask = (
                        (aggregate_values_0_100 >= self.min_aggregate_threshold) &
                        (aggregate_values_0_100 <= self.max_aggregate_threshold)
                    )
                    num_pixels_passed_phase2 = np.count_nonzero(pixels_passed_phase2_mask)

                    if num_pixels_passed_phase2 > 0:
                        final_area = num_pixels_passed_phase2
                        final_aggregates = aggregate_values_0_100[pixels_passed_phase2_mask]
                        final_intensity = np.sum(final_aggregates)
                        final_ratio = final_intensity / final_area if final_area > 0 else 0.0

                        value_text_parts = []
                        if self.use_area:
                            value_text_parts.append(f"A:{final_area}")
                        if self.use_intensity:
                            value_text_parts.append(f"I:{final_intensity:.0f}")
                            if self.use_area and final_area > 0:
                                value_text_parts.append(f"R:{final_ratio:.2f}")
                            elif self.use_area:
                                value_text_parts.append("R:N/A")
                        value_text = " ".join(value_text_parts) if value_text_parts else "N/A"

                        contours, _ = cv2.findContours(segmentation_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                        color_bgr = self.bright_colors[i % len(self.bright_colors)]
                        cv2.drawContours(output_image, contours, -1, color_bgr, 1)

                        ys, xs = np.where(segmentation_mask > 0)
                        if len(xs) == 0 or len(ys) == 0: continue

                        center_x = int(np.mean(xs))
                        center_y = int(np.mean(ys))

                        text_x = max(5, min(center_x, width - 10))
                        text_y = max(15, min(center_y, height - 5))

                        font = cv2.FONT_HERSHEY_SIMPLEX
                        font_scale = 0.4
                        thickness = 1
                        (text_width, text_height), baseline = cv2.getTextSize(value_text, font, font_scale, thickness)

                        current_text_box = [text_x, text_y - text_height, text_width, text_height + baseline]
                        shift_amount = text_height + 2

                        attempts = 0
                        max_attempts = 10
                        while any(self.boxes_overlap(current_text_box, box) for box in drawn_text_boxes) and attempts < max_attempts:
                            text_y += shift_amount
                            if text_y > height - 5:
                                 text_y = max(15, min(center_y, height - 5))
                                 text_y -= (attempts + 1) * shift_amount
                                 if text_y < 15:
                                     text_y = max(15, min(center_y, height - 5))
                                     break

                            current_text_box = [text_x, text_y - text_height, text_width, text_height + baseline]
                            attempts += 1

                        cv2.putText(output_image, value_text, (text_x, text_y), font, font_scale, color_bgr, thickness, cv2.LINE_AA)
                        drawn_text_boxes.append(current_text_box)

            try:
                cv2.imwrite(output_path, output_image)
                return output_image
            except Exception as e:
                self.log_updated.emit(f"Error saving output image {output_path}: {str(e)}")
                return None

        except Exception as e:
            self.log_updated.emit(f"General error processing {os.path.basename(input_path)}: {str(e)}")
            if 'cuda' in str(e).lower() and torch.cuda.is_available(): torch.cuda.empty_cache()
            return None

    def boxes_overlap(self, box1, box2):
        x1, y1, w1, h1 = box1
        x2, y2, w2, h2 = box2
        if (x1 + w1 < x2 or x2 + w2 < x1 or y1 + h1 < y2 or y2 + h2 < y1):
            return False
        return True

    def stop(self):
        self.lock.lock()
        try:
            self.running = False
        finally:
            self.lock.unlock()

class UsefulImageAnalysisTools(QMainWindow):
    def __init__(self):
        super().__init__()
        self.input_folder = None
        self.model_path = None
        self.bright_input_folder = None
        self.heatmap_input_folder = None
        self.contrast_input_folder = None

        self.process_thread = None
        self.heatmap_thread = None
        self.contrast_thread = None

        self.last_output_path = None
        self.last_output_image_array = None

        self.initUI()

    def initUI(self):
        self.setWindowTitle('Useful Image Analysis Tools')
        self.setGeometry(100, 100, 1300, 950)

        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        main_layout = QHBoxLayout(main_widget)
        main_layout.setContentsMargins(15, 15, 15, 15)
        main_layout.setSpacing(15)

        left_widget = QWidget()
        left_layout = QVBoxLayout(left_widget)
        left_layout.setContentsMargins(0, 0, 0, 0)
        left_layout.setSpacing(15)

        self.tabs = QTabWidget()
        self.tabs.setStyleSheet("""
            QTabBar::tab {
                padding: 8px 15px;
                font-weight: bold;
                border: 1px solid #cccccc;
                border-bottom: none;
                border-top-left-radius: 4px;
                border-top-right-radius: 4px;
                margin-right: 2px;
                background-color: #e0e0e0;
            }
            QTabBar::tab:selected {
                background: #ffffff;
                border-color: #aaaaaa;
                border-bottom: 1px solid #ffffff;
            }
            QTabBar::tab:!selected:hover {
                background: #f0f0f0;
            }
            QTabWidget::pane {
                border: 1px solid #aaaaaa;
                border-top: none;
                background-color: #ffffff;
                border-bottom-left-radius: 4px;
                border-bottom-right-radius: 4px;
            }
        """)

        contrast_tab = QWidget()
        aggregate_tab = QWidget()
        heatmap_tab = QWidget()
        process_tab = QWidget()

        contrast_layout = QVBoxLayout(contrast_tab)
        contrast_layout.setContentsMargins(10, 10, 10, 10)
        contrast_layout.setSpacing(15)
        contrast_file_group = QGroupBox("File Selection")
        contrast_file_group.setStyleSheet("QGroupBox { font-weight: bold; }")
        contrast_file_layout = QVBoxLayout(contrast_file_group)
        contrast_file_layout.setContentsMargins(10, 15, 10, 10)
        contrast_input_layout = QHBoxLayout()
        self.contrast_input_folder_label = QLabel("Input folder: Not selected")
        contrast_input_folder_btn = QPushButton("Select Input Folder")
        contrast_input_folder_btn.setStyleSheet(self.get_button_style("green"))
        contrast_input_folder_btn.clicked.connect(self.select_contrast_input_folder)
        contrast_input_layout.addWidget(self.contrast_input_folder_label, 1)
        contrast_input_layout.addWidget(contrast_input_folder_btn)
        contrast_file_layout.addLayout(contrast_input_layout)
        contrast_layout.addWidget(contrast_file_group)
        contrast_param_group = QGroupBox("Contrast Parameters")
        contrast_param_group.setStyleSheet("QGroupBox { font-weight: bold; }")
        contrast_param_layout = QFormLayout(contrast_param_group)
        contrast_param_layout.setContentsMargins(10, 15, 10, 10)
        contrast_param_layout.setVerticalSpacing(10)
        contrast_param_layout.setLabelAlignment(Qt.AlignmentFlag.AlignRight)
        self.contrast_spin = QDoubleSpinBox()
        self.contrast_spin.setRange(0.1, 10.0)
        self.contrast_spin.setSingleStep(0.1)
        self.contrast_spin.setValue(1.0)
        self.contrast_spin.setDecimals(1)
        contrast_param_layout.addRow("Contrast factor (0.1-10.0):", self.contrast_spin)
        contrast_layout.addWidget(contrast_param_group)
        contrast_operation_layout = QHBoxLayout()
        self.enhance_contrast_btn = QPushButton("Enhance Contrast")
        self.enhance_contrast_btn.setStyleSheet(self.get_button_style("blue"))
        self.enhance_contrast_btn.setEnabled(False)
        self.enhance_contrast_btn.clicked.connect(self.enhance_contrast)
        self.stop_contrast_btn = QPushButton("Stop Processing")
        self.stop_contrast_btn.setStyleSheet(self.get_button_style("red"))
        self.stop_contrast_btn.setEnabled(False)
        self.stop_contrast_btn.clicked.connect(self.stop_contrast_processing)
        self.open_contrast_output_btn = QPushButton("Open Output Folder")
        self.open_contrast_output_btn.setStyleSheet(self.get_button_style("grey"))
        self.open_contrast_output_btn.setEnabled(False)
        self.open_contrast_output_btn.clicked.connect(self.open_contrast_output_folder)
        contrast_operation_layout.addWidget(self.enhance_contrast_btn)
        contrast_operation_layout.addWidget(self.stop_contrast_btn)
        contrast_operation_layout.addWidget(self.open_contrast_output_btn)
        contrast_layout.addLayout(contrast_operation_layout)
        contrast_layout.addStretch()

        aggregate_layout = QVBoxLayout(aggregate_tab)
        aggregate_layout.setContentsMargins(10, 10, 10, 10)
        aggregate_layout.setSpacing(15)
        agg_file_group = QGroupBox("File Selection")
        agg_file_group.setStyleSheet("QGroupBox { font-weight: bold; }")
        agg_file_layout = QVBoxLayout(agg_file_group)
        agg_file_layout.setContentsMargins(10, 15, 10, 10)
        agg_input_layout = QHBoxLayout()
        self.agg_input_folder_label = QLabel("Input folder: Not selected")
        agg_input_folder_btn = QPushButton("Select Input Folder")
        agg_input_folder_btn.setStyleSheet(self.get_button_style("green"))
        agg_input_folder_btn.clicked.connect(self.select_aggregate_input_folder)
        agg_input_layout.addWidget(self.agg_input_folder_label, 1)
        agg_input_layout.addWidget(agg_input_folder_btn)
        agg_file_layout.addLayout(agg_input_layout)
        aggregate_layout.addWidget(agg_file_group)
        agg_operation_layout = QHBoxLayout()
        self.analyze_agg_btn = QPushButton("Analyze Aggregate")
        self.analyze_agg_btn.setStyleSheet(self.get_button_style("blue"))
        self.analyze_agg_btn.setEnabled(False)
        self.analyze_agg_btn.clicked.connect(self.analyze_folder_aggregate)
        agg_operation_layout.addWidget(self.analyze_agg_btn)
        agg_operation_layout.addStretch()
        aggregate_layout.addLayout(agg_operation_layout)
        aggregate_layout.addStretch()

        heatmap_layout = QVBoxLayout(heatmap_tab)
        heatmap_layout.setContentsMargins(10, 10, 10, 10)
        heatmap_layout.setSpacing(15)
        heatmap_file_group = QGroupBox("File Selection")
        heatmap_file_group.setStyleSheet("QGroupBox { font-weight: bold; }")
        heatmap_file_layout = QVBoxLayout(heatmap_file_group)
        heatmap_file_layout.setContentsMargins(10, 15, 10, 10)
        heatmap_input_layout = QHBoxLayout()
        self.heatmap_input_folder_label = QLabel("Input folder: Not selected")
        heatmap_input_folder_btn = QPushButton("Select Input Folder")
        heatmap_input_folder_btn.setStyleSheet(self.get_button_style("green"))
        heatmap_input_folder_btn.clicked.connect(self.select_heatmap_input_folder)
        heatmap_input_layout.addWidget(self.heatmap_input_folder_label, 1)
        heatmap_input_layout.addWidget(heatmap_input_folder_btn)
        heatmap_file_layout.addLayout(heatmap_input_layout)
        heatmap_layout.addWidget(heatmap_file_group)
        heatmap_param_group = QGroupBox("Heatmap Parameters (Aggregate 0-100)")
        heatmap_param_group.setStyleSheet("QGroupBox { font-weight: bold; }")
        heatmap_param_layout = QFormLayout(heatmap_param_group)
        heatmap_param_layout.setContentsMargins(10, 15, 10, 10)
        heatmap_param_layout.setVerticalSpacing(10)
        heatmap_param_layout.setLabelAlignment(Qt.AlignmentFlag.AlignRight)
        self.heatmap_min_aggregate_spin = QSpinBox()
        self.heatmap_min_aggregate_spin.setRange(0, 100)
        self.heatmap_min_aggregate_spin.setValue(35)
        heatmap_param_layout.addRow("Min Aggregate Threshold:", self.heatmap_min_aggregate_spin)
        self.heatmap_max_aggregate_spin = QSpinBox()
        self.heatmap_max_aggregate_spin.setRange(0, 100)
        self.heatmap_max_aggregate_spin.setValue(90)
        heatmap_param_layout.addRow("Max Aggregate Threshold:", self.heatmap_max_aggregate_spin)
        heatmap_layout.addWidget(heatmap_param_group)
        heatmap_operation_layout = QHBoxLayout()
        self.generate_heatmap_btn = QPushButton("Generate Heatmaps")
        self.generate_heatmap_btn.setStyleSheet(self.get_button_style("blue"))
        self.generate_heatmap_btn.setEnabled(False)
        self.generate_heatmap_btn.clicked.connect(self.generate_heatmaps)
        self.stop_heatmap_btn = QPushButton("Stop Heatmap Generation")
        self.stop_heatmap_btn.setStyleSheet(self.get_button_style("red"))
        self.stop_heatmap_btn.setEnabled(False)
        self.stop_heatmap_btn.clicked.connect(self.stop_heatmap_generation)
        self.open_heatmap_output_btn = QPushButton("Open Output Folder")
        self.open_heatmap_output_btn.setStyleSheet(self.get_button_style("grey"))
        self.open_heatmap_output_btn.setEnabled(False)
        self.open_heatmap_output_btn.clicked.connect(self.open_heatmap_output_folder)
        heatmap_operation_layout.addWidget(self.generate_heatmap_btn)
        heatmap_operation_layout.addWidget(self.stop_heatmap_btn)
        heatmap_operation_layout.addWidget(self.open_heatmap_output_btn)
        heatmap_layout.addLayout(heatmap_operation_layout)
        heatmap_layout.addStretch()

        process_layout = QVBoxLayout(process_tab)
        process_layout.setContentsMargins(10, 10, 10, 10)
        process_layout.setSpacing(15)
        process_scroll_area = QScrollArea()
        process_scroll_area.setWidgetResizable(True)
        process_scroll_widget = QWidget()
        process_scroll_layout = QVBoxLayout(process_scroll_widget)
        process_scroll_layout.setContentsMargins(5, 5, 5, 5)
        process_scroll_layout.setSpacing(15)
        file_group_proc = QGroupBox("File Selection")
        file_group_proc.setStyleSheet("QGroupBox { font-weight: bold; }")
        file_layout_proc = QVBoxLayout(file_group_proc)
        file_layout_proc.setContentsMargins(10, 15, 10, 10)
        file_layout_proc.setSpacing(10)
        input_layout_proc = QHBoxLayout()
        self.input_folder_label = QLabel("Input folder: Not selected")
        input_folder_btn_proc = QPushButton("Select Input Folder")
        input_folder_btn_proc.setStyleSheet(self.get_button_style("green"))
        input_folder_btn_proc.clicked.connect(self.select_input_folder)
        input_layout_proc.addWidget(self.input_folder_label, 1)
        input_layout_proc.addWidget(input_folder_btn_proc)
        file_layout_proc.addLayout(input_layout_proc)
        model_layout_proc = QHBoxLayout()
        self.model_path_label = QLabel("Model file: Not selected")
        model_path_btn_proc = QPushButton("Select SAM Model File (.pth)")
        model_path_btn_proc.setStyleSheet(self.get_button_style("green"))
        model_path_btn_proc.clicked.connect(self.select_model_file)
        model_layout_proc.addWidget(self.model_path_label, 1)
        model_layout_proc.addWidget(model_path_btn_proc)
        file_layout_proc.addLayout(model_layout_proc)
        process_scroll_layout.addWidget(file_group_proc)

        param_group_phase1 = QGroupBox("Phase 1: Region Property Filters")
        param_group_phase1.setStyleSheet("QGroupBox { font-weight: bold; }")
        param_layout_phase1 = QFormLayout(param_group_phase1)
        param_layout_phase1.setContentsMargins(10, 15, 10, 10)
        param_layout_phase1.setVerticalSpacing(10)
        param_layout_phase1.setLabelAlignment(Qt.AlignmentFlag.AlignRight)

        self.a_threshold_spin = QSpinBox()
        self.a_threshold_spin.setRange(0, 1000000)
        self.a_threshold_spin.setValue(199)
        self.a_threshold_spin.setToolTip("Phase 1: Regions with Area (pixel count) less than this value will be filtered out.")
        param_layout_phase1.addRow("Min Area (A) Threshold:", self.a_threshold_spin)

        self.i_threshold_spin = QDoubleSpinBox()
        self.i_threshold_spin.setRange(0.0, 100000000.0)
        self.i_threshold_spin.setValue(0.0)
        self.i_threshold_spin.setDecimals(0)
        self.i_threshold_spin.setToolTip("Phase 1: Regions with Intensity (sum of aggregates) less than this value will be filtered out.")
        param_layout_phase1.addRow("Min Intensity (I) Threshold:", self.i_threshold_spin)

        self.r_threshold_spin = QDoubleSpinBox()
        self.r_threshold_spin.setRange(0.0, 100.0)
        self.r_threshold_spin.setSingleStep(0.1)
        self.r_threshold_spin.setValue(37.0)
        self.r_threshold_spin.setDecimals(1)
        self.r_threshold_spin.setToolTip("Phase 1: Regions with Ratio (I/A) less than this value will be filtered out.")
        param_layout_phase1.addRow("Min Ratio (R=I/A) Threshold:", self.r_threshold_spin)

        metrics_layout = QHBoxLayout()
        metrics_layout.setSpacing(15)
        self.area_check = QCheckBox("Area")
        self.area_check.setStyleSheet("font-weight: normal;")
        self.area_check.setChecked(True)
        self.area_check.stateChanged.connect(self.on_calculation_method_changed)
        metrics_layout.addWidget(self.area_check)
        self.intensity_check = QCheckBox("Intensity")
        self.intensity_check.setStyleSheet("font-weight: normal;")
        self.intensity_check.setChecked(True)
        self.intensity_check.stateChanged.connect(self.on_calculation_method_changed)
        metrics_layout.addWidget(self.intensity_check)
        metrics_layout.addStretch()
        param_layout_phase1.addRow("Enable Phase 1 Filters:", metrics_layout)
        process_scroll_layout.addWidget(param_group_phase1)

        param_group_phase2 = QGroupBox("Phase 2: Pixel Aggregate Filter (Applied to Regions Passing Phase 1)")
        param_group_phase2.setStyleSheet("QGroupBox { font-weight: bold; }")
        param_layout_phase2 = QFormLayout(param_group_phase2)
        param_layout_phase2.setContentsMargins(10, 15, 10, 10)
        param_layout_phase2.setVerticalSpacing(10)
        param_layout_phase2.setLabelAlignment(Qt.AlignmentFlag.AlignRight)

        self.min_aggregate_spin = QSpinBox()
        self.min_aggregate_spin.setRange(0, 100)
        self.min_aggregate_spin.setValue(35)
        self.min_aggregate_spin.setToolTip("Phase 2: Pixels within passed regions with Aggregate value below this are ignored for final inclusion check.\nDisplayed A/I/R values are based on pixels passing this filter.")
        param_layout_phase2.addRow("Min Aggregate Threshold:", self.min_aggregate_spin)

        self.max_aggregate_spin = QSpinBox()
        self.max_aggregate_spin.setRange(0, 100)
        self.max_aggregate_spin.setValue(100)
        self.max_aggregate_spin.setToolTip("Phase 2: Pixels within passed regions with Aggregate value above this are ignored for final inclusion check.\nDisplayed A/I/R values are based on pixels passing this filter.")
        param_layout_phase2.addRow("Max Aggregate Threshold:", self.max_aggregate_spin)
        process_scroll_layout.addWidget(param_group_phase2)


        sam_params_group = QGroupBox("SAM Auto Mask Generator Parameters")
        sam_params_group.setStyleSheet("QGroupBox { font-weight: bold; }")
        sam_params_layout = QFormLayout(sam_params_group)
        sam_params_layout.setContentsMargins(10, 15, 10, 10)
        sam_params_layout.setVerticalSpacing(10)
        sam_params_layout.setLabelAlignment(Qt.AlignmentFlag.AlignRight)
        self.points_per_side_spin = QSpinBox()
        self.points_per_side_spin.setRange(1, 100)
        self.points_per_side_spin.setValue(32)
        self.points_per_side_spin.setToolTip("Controls the number of points sampled along each side of the image, total points is points_per_side²")
        sam_params_layout.addRow("Points Per Side:", self.points_per_side_spin)
        self.points_per_batch_spin = QSpinBox()
        self.points_per_batch_spin.setRange(1, 1000)
        self.points_per_batch_spin.setValue(64)
        self.points_per_batch_spin.setToolTip("Sets the number of points processed by the model at once, higher values may process faster but use more GPU memory")
        sam_params_layout.addRow("Points Per Batch:", self.points_per_batch_spin)
        self.pred_iou_thresh_spin = QDoubleSpinBox()
        self.pred_iou_thresh_spin.setRange(0.0, 1.0)
        self.pred_iou_thresh_spin.setSingleStep(0.01)
        self.pred_iou_thresh_spin.setValue(0.88)
        self.pred_iou_thresh_spin.setToolTip("Threshold for filtering using the model's predicted mask quality")
        sam_params_layout.addRow("Pred IoU Thresh:", self.pred_iou_thresh_spin)
        self.stability_score_thresh_spin = QDoubleSpinBox()
        self.stability_score_thresh_spin.setRange(0.0, 1.0)
        self.stability_score_thresh_spin.setSingleStep(0.01)
        self.stability_score_thresh_spin.setValue(0.95)
        self.stability_score_thresh_spin.setToolTip("Threshold for filtering using the mask's stability under changes in binarization threshold")
        sam_params_layout.addRow("Stability Score Thresh:", self.stability_score_thresh_spin)
        self.stability_score_offset_spin = QDoubleSpinBox()
        self.stability_score_offset_spin.setRange(0.0, 10.0)
        self.stability_score_offset_spin.setSingleStep(0.1)
        self.stability_score_offset_spin.setValue(1.0)
        self.stability_score_offset_spin.setToolTip("Offset used when calculating the stability score")
        sam_params_layout.addRow("Stability Score Offset:", self.stability_score_offset_spin)
        self.box_nms_thresh_spin = QDoubleSpinBox()
        self.box_nms_thresh_spin.setRange(0.0, 1.0)
        self.box_nms_thresh_spin.setSingleStep(0.01)
        self.box_nms_thresh_spin.setValue(0.7)
        self.box_nms_thresh_spin.setToolTip("IoU threshold for Non-Maximum Suppression (NMS) used to filter duplicate masks based on bounding boxes")
        sam_params_layout.addRow("Box NMS Thresh:", self.box_nms_thresh_spin)
        self.crop_n_layers_spin = QSpinBox()
        self.crop_n_layers_spin.setRange(0, 5)
        self.crop_n_layers_spin.setValue(0)
        self.crop_n_layers_spin.setToolTip("If >0, the image will be cropped into multiple layers for processing (can use more memory but handle larger images/improve detection of small objects)")
        sam_params_layout.addRow("Crop N Layers:", self.crop_n_layers_spin)
        self.crop_nms_thresh_spin = QDoubleSpinBox()
        self.crop_nms_thresh_spin.setRange(0.0, 1.0)
        self.crop_nms_thresh_spin.setSingleStep(0.01)
        self.crop_nms_thresh_spin.setValue(0.7)
        self.crop_nms_thresh_spin.setToolTip("IoU threshold for applying NMS between masks generated from different crops")
        sam_params_layout.addRow("Crop NMS Thresh:", self.crop_nms_thresh_spin)
        self.crop_overlap_ratio_spin = QDoubleSpinBox()
        self.crop_overlap_ratio_spin.setRange(0.0, 1.0)
        self.crop_overlap_ratio_spin.setSingleStep(0.01)
        self.crop_overlap_ratio_spin.setValue(round(512/1500, 3))
        self.crop_overlap_ratio_spin.setToolTip("Sets the degree of overlap between crops when Crop N Layers > 0")
        sam_params_layout.addRow("Crop Overlap Ratio:", self.crop_overlap_ratio_spin)
        self.crop_n_points_downscale_factor_spin = QSpinBox()
        self.crop_n_points_downscale_factor_spin.setRange(1, 10)
        self.crop_n_points_downscale_factor_spin.setValue(1)
        self.crop_n_points_downscale_factor_spin.setToolTip("Downscales the number of points sampled per crop layer (if Crop N Layers > 0)")
        sam_params_layout.addRow("Crop Points Downscale Factor:", self.crop_n_points_downscale_factor_spin)
        self.min_mask_region_area_spin = QSpinBox()
        self.min_mask_region_area_spin.setRange(0, 100000)
        self.min_mask_region_area_spin.setValue(0)
        self.min_mask_region_area_spin.setToolTip("If >0, post-processing will be applied to remove small disconnected regions within masks or holes smaller than this area (pixels)")
        sam_params_layout.addRow("Min Mask Region Area:", self.min_mask_region_area_spin)
        self.output_mode_combo = QComboBox()
        self.output_mode_combo.addItems(["binary_mask", "uncompressed_rle", "coco_rle"])
        self.output_mode_combo.setToolTip(
            "Controls the mask output format:\n"
            "- binary_mask: NumPy array (HW), recommended and compatible.\n"
            "- uncompressed_rle: RLE dictionary (uncompressed).\n"
            "- coco_rle: RLE dictionary (compressed).\n"
            "Note: RLE modes require 'pycocotools' to be installed for processing within this tool."
        )
        sam_params_layout.addRow("Output Mode:", self.output_mode_combo)
        reset_sam_params_btn = QPushButton("Reset SAM Parameters to Default")
        reset_sam_params_btn.setStyleSheet(self.get_button_style("grey"))
        reset_sam_params_btn.clicked.connect(self.reset_sam_params)
        sam_params_layout.addRow("", reset_sam_params_btn)
        process_scroll_layout.addWidget(sam_params_group)

        operation_layout_proc = QHBoxLayout()
        self.batch_process_btn = QPushButton("Process All Images")
        self.batch_process_btn.setStyleSheet(self.get_button_style("blue"))
        self.batch_process_btn.setEnabled(False)
        self.batch_process_btn.clicked.connect(self.batch_process_images)
        self.stop_btn = QPushButton("Stop Processing")
        self.stop_btn.setStyleSheet(self.get_button_style("red"))
        self.stop_btn.setEnabled(False)
        self.stop_btn.clicked.connect(self.stop_processing)
        self.open_proc_output_btn = QPushButton("Open Output Folder")
        self.open_proc_output_btn.setStyleSheet(self.get_button_style("grey"))
        self.open_proc_output_btn.setEnabled(False)
        self.open_proc_output_btn.clicked.connect(self.open_processing_output_folder)
        operation_layout_proc.addWidget(self.batch_process_btn)
        operation_layout_proc.addWidget(self.stop_btn)
        operation_layout_proc.addWidget(self.open_proc_output_btn)
        process_scroll_layout.addLayout(operation_layout_proc)

        process_scroll_layout.addStretch()
        process_scroll_area.setWidget(process_scroll_widget)
        process_layout.addWidget(process_scroll_area)

        self.tabs.addTab(contrast_tab, "Contrast Enhancement")
        self.tabs.addTab(aggregate_tab, "Aggregate Analysis")
        self.tabs.addTab(heatmap_tab, "Heatmap Generation")
        self.tabs.addTab(process_tab, "SAM Processing")

        left_layout.addWidget(self.tabs)
        main_layout.addWidget(left_widget, 1)

        right_widget = QWidget()
        right_layout = QVBoxLayout(right_widget)
        right_layout.setContentsMargins(0, 0, 0, 0)
        right_layout.setSpacing(15)

        contrast_log_group = QGroupBox("Contrast Enhancement Log")
        contrast_log_group.setStyleSheet("QGroupBox { font-weight: bold; }")
        contrast_log_layout = QVBoxLayout(contrast_log_group)
        contrast_log_layout.setContentsMargins(10, 15, 10, 10)
        self.contrast_log_text = QTextEdit()
        self.contrast_log_text.setStyleSheet(self.get_log_style())
        self.contrast_log_text.setReadOnly(True)
        contrast_log_layout.addWidget(self.contrast_log_text)
        right_layout.addWidget(contrast_log_group)

        aggregate_log_group = QGroupBox("Aggregate Analysis Log")
        aggregate_log_group.setStyleSheet("QGroupBox { font-weight: bold; }")
        aggregate_log_layout = QVBoxLayout(aggregate_log_group)
        aggregate_log_layout.setContentsMargins(10, 15, 10, 10)
        self.aggregate_log_text = QTextEdit()
        self.aggregate_log_text.setStyleSheet(self.get_log_style())
        self.aggregate_log_text.setReadOnly(True)
        aggregate_log_layout.addWidget(self.aggregate_log_text)
        right_layout.addWidget(aggregate_log_group)

        heatmap_log_group = QGroupBox("Heatmap Generation Log")
        heatmap_log_group.setStyleSheet("QGroupBox { font-weight: bold; }")
        heatmap_log_layout = QVBoxLayout(heatmap_log_group)
        heatmap_log_layout.setContentsMargins(10, 15, 10, 10)
        self.heatmap_log_text = QTextEdit()
        self.heatmap_log_text.setStyleSheet(self.get_log_style())
        self.heatmap_log_text.setReadOnly(True)
        heatmap_log_layout.addWidget(self.heatmap_log_text)
        right_layout.addWidget(heatmap_log_group)

        log_group_proc = QGroupBox("SAM Processing Log")
        log_group_proc.setStyleSheet("QGroupBox { font-weight: bold; }")
        log_layout_proc = QVBoxLayout(log_group_proc)
        log_layout_proc.setContentsMargins(10, 15, 10, 10)
        self.log_text = QTextEdit()
        self.log_text.setStyleSheet(self.get_log_style())
        self.log_text.setReadOnly(True)
        log_layout_proc.addWidget(self.log_text)
        right_layout.addWidget(log_group_proc)

        main_layout.addWidget(right_widget, 1)

        self.update_button_states()

    def get_button_style(self, color="blue"):
        base_style = """
            QPushButton {
                padding: 8px 12px;
                color: white;
                border: none;
                border-radius: 4px;
                font-weight: bold;
            }
            QPushButton:disabled {
                background-color: #cccccc;
                color: #666666;
            }
        """
        colors = {
            "green": ("#4CAF50", "#45a049"),
            "blue": ("#2196F3", "#0b7dda"),
            "red": ("#f44336", "#d32f2f"),
            "grey": ("#607d8b", "#455a64"),
        }
        bg_color, hover_color = colors.get(color, colors["blue"])
        return base_style + f"""
            QPushButton {{ background-color: {bg_color}; }}
            QPushButton:hover:!disabled {{ background-color: {hover_color}; }}
        """

    def get_log_style(self):
        return """
            QTextEdit {
                border: 1px solid #cccccc;
                border-radius: 4px;
                padding: 5px;
                font-family: Consolas, Courier New, monospace;
                background-color: #f8f8f8;
            }
        """

    def get_processing_output_folder_path(self):
        if not self.input_folder: return None

        input_folder_name = os.path.basename(os.path.normpath(self.input_folder))
        parent_dir = os.path.dirname(os.path.normpath(self.input_folder))

        param_parts = []
        if self.area_check.isChecked():
            param_parts.append(f"Area={self.a_threshold_spin.value()}")
        if self.intensity_check.isChecked():
            param_parts.append(f"Int={self.i_threshold_spin.value():.0f}")
            param_parts.append(f"Ratio={self.r_threshold_spin.value():.1f}")
        param_parts.append(f"Agg={self.min_aggregate_spin.value()}-{self.max_aggregate_spin.value()}")

        param_str = " ".join(param_parts)
        output_folder_name = f"{input_folder_name}_SAM_Output [{param_str}]"

        return os.path.join(parent_dir, output_folder_name)

    def get_heatmap_output_folder_path(self):
        if not self.heatmap_input_folder: return None
        input_folder_name = os.path.basename(os.path.normpath(self.heatmap_input_folder))
        parent_dir = os.path.dirname(os.path.normpath(self.heatmap_input_folder))
        min_thresh = self.heatmap_min_aggregate_spin.value()
        max_thresh = self.heatmap_max_aggregate_spin.value()
        output_folder_name = f"{input_folder_name}_Heatmap_Output [Agg={min_thresh}-{max_thresh}]"
        return os.path.join(parent_dir, output_folder_name)

    def get_contrast_output_folder_path(self):
        if not self.contrast_input_folder: return None
        input_folder_name = os.path.basename(os.path.normpath(self.contrast_input_folder))
        parent_dir = os.path.dirname(os.path.normpath(self.contrast_input_folder))
        contrast_factor = self.contrast_spin.value()
        output_folder_name = f"{input_folder_name}_Contrast_Output [Factor={contrast_factor:.1f}]"
        return os.path.join(parent_dir, output_folder_name)

    def reset_sam_params(self):
        self.points_per_side_spin.setValue(32)
        self.points_per_batch_spin.setValue(64)
        self.pred_iou_thresh_spin.setValue(0.88)
        self.stability_score_thresh_spin.setValue(0.95)
        self.stability_score_offset_spin.setValue(1.0)
        self.box_nms_thresh_spin.setValue(0.7)
        self.crop_n_layers_spin.setValue(0)
        self.crop_nms_thresh_spin.setValue(0.7)
        self.crop_overlap_ratio_spin.setValue(round(512/1500, 3))
        self.crop_n_points_downscale_factor_spin.setValue(1)
        self.min_mask_region_area_spin.setValue(0)
        self.output_mode_combo.setCurrentText("binary_mask")
        self.log_text.append("SAM parameters reset to default.")

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

    def select_input_folder(self):
        folder = QFileDialog.getExistingDirectory(self, "Select Input Folder (Processing)")
        if folder:
            self.input_folder = folder
            self.input_folder_label.setText(f"Input: ...{folder[-40:]}")
            self.update_button_states()

    def select_model_file(self):
        file, _ = QFileDialog.getOpenFileName(self, "Select SAM Model File", "", "Model files (*.pth)")
        if file:
            self.model_path = file
            self.model_path_label.setText(f"Model: {os.path.basename(file)}")
            self.update_button_states()

    def select_aggregate_input_folder(self):
        folder = QFileDialog.getExistingDirectory(self, "Select Input Folder (Aggregate Analysis)")
        if folder:
            self.bright_input_folder = folder
            self.agg_input_folder_label.setText(f"Input: ...{folder[-40:]}")
            self.analyze_agg_btn.setEnabled(True)

    def select_heatmap_input_folder(self):
        folder = QFileDialog.getExistingDirectory(self, "Select Input Folder (Heatmap)")
        if folder:
            self.heatmap_input_folder = folder
            self.heatmap_input_folder_label.setText(f"Input: ...{folder[-40:]}")
            self.generate_heatmap_btn.setEnabled(True)
            self.open_heatmap_output_btn.setEnabled(True)

    def select_contrast_input_folder(self):
        folder = QFileDialog.getExistingDirectory(self, "Select Input Folder (Contrast)")
        if folder:
            self.contrast_input_folder = folder
            self.contrast_input_folder_label.setText(f"Input: ...{folder[-40:]}")
            self.enhance_contrast_btn.setEnabled(True)
            self.open_contrast_output_btn.setEnabled(True)

    def on_calculation_method_changed(self, state):
        if not self.area_check.isChecked() and not self.intensity_check.isChecked():
             sender = self.sender()
             if sender == self.area_check and state == Qt.CheckState.Unchecked.value:
                 self.area_check.setChecked(True)
                 QMessageBox.warning(self, "Selection Required", "At least one Phase 1 filter method (Area or Intensity/Ratio) must be selected.")
             elif sender == self.intensity_check and state == Qt.CheckState.Unchecked.value:
                 self.intensity_check.setChecked(True)
                 QMessageBox.warning(self, "Selection Required", "At least one Phase 1 filter method (Area or Intensity/Ratio) must be selected.")

        self.update_button_states()

    def update_button_states(self):
        proc_input_ready = bool(self.input_folder)
        model_ready = bool(self.model_path)
        calculation_selected = self.area_check.isChecked() or self.intensity_check.isChecked()
        proc_all_ready = proc_input_ready and model_ready and calculation_selected
        self.batch_process_btn.setEnabled(proc_all_ready)
        self.open_proc_output_btn.setEnabled(proc_input_ready)

        self.generate_heatmap_btn.setEnabled(bool(self.heatmap_input_folder))
        self.open_heatmap_output_btn.setEnabled(bool(self.heatmap_input_folder))

        self.enhance_contrast_btn.setEnabled(bool(self.contrast_input_folder))
        self.open_contrast_output_btn.setEnabled(bool(self.contrast_input_folder))

        self.analyze_agg_btn.setEnabled(bool(self.bright_input_folder))


    def open_folder(self, folder_path, folder_type_name):
        if folder_path:
            if not os.path.exists(folder_path):
                 try:
                     os.makedirs(folder_path)
                 except OSError as e:
                     QMessageBox.warning(self, "Error", f"Could not create or access {folder_type_name} output folder:\n{folder_path}\nError: {e}")
                     return
            try:
                if sys.platform == "win32":
                    os.startfile(folder_path)
                elif sys.platform == "darwin":
                    os.system(f"open \"{folder_path}\"")
                else:
                    os.system(f"xdg-open \"{folder_path}\"")
            except Exception as e:
                QMessageBox.warning(self, "Error", f"Could not open {folder_type_name} output folder:\n{folder_path}\nError: {e}")
        else:
            QMessageBox.information(self, "Info", f"{folder_type_name} output folder path could not be determined (process might need to run first or input folder not selected).")

    def open_processing_output_folder(self):
        if self.input_folder:
            output_folder = self.get_processing_output_folder_path()
            self.open_folder(output_folder, "Processing")
        else:
             QMessageBox.warning(self, "Input Required", "Please select an input folder first.")

    def open_heatmap_output_folder(self):
        if self.heatmap_input_folder:
            output_folder = self.get_heatmap_output_folder_path()
            self.open_folder(output_folder, "Heatmap")
        else:
             QMessageBox.warning(self, "Input Required", "Please select an input folder for heatmaps first.")

    def open_contrast_output_folder(self):
        if self.contrast_input_folder:
            output_folder = self.get_contrast_output_folder_path()
            self.open_folder(output_folder, "Contrast")
        else:
             QMessageBox.warning(self, "Input Required", "Please select an input folder for contrast enhancement first.")


    def batch_process_images(self):
        min_agg = self.min_aggregate_spin.value()
        max_agg = self.max_aggregate_spin.value()
        if min_agg > max_agg or (min_agg == max_agg and min_agg != 100):
            QMessageBox.warning(self, "Invalid Aggregate Thresholds", "Min Aggregate Threshold must be less than or equal to Max Aggregate Threshold (equality allowed only if both are 100).")
            return

        if not self.area_check.isChecked() and not self.intensity_check.isChecked():
             QMessageBox.warning(self, "Selection Required", "Please select at least one Phase 1 calculation method (Area or Intensity/Ratio).")
             return
        if not self.input_folder or not self.model_path:
             QMessageBox.warning(self, "Input Required", "Please select both an input folder and a SAM model file.")
             return

        self.log_text.append("-" * 20 + " New Processing Job " + "-" * 20)
        self.log_text.append("Starting image processing...")
        self.batch_process_btn.setEnabled(False)
        self.stop_btn.setEnabled(True)

        use_area = self.area_check.isChecked()
        use_intensity = self.intensity_check.isChecked()
        sam_params = self.get_sam_params()
        output_folder = self.get_processing_output_folder_path()
        a_threshold = self.a_threshold_spin.value()
        i_threshold = self.i_threshold_spin.value()
        r_threshold = self.r_threshold_spin.value()

        if not output_folder:
             self.log_text.append("Error: Could not determine output folder path.")
             self.update_button_states()
             self.stop_btn.setEnabled(False)
             return

        self.log_text.append(f"Output folder: {output_folder}")
        self.log_text.append(f"Using SAM parameters: {sam_params}")
        self.log_text.append(f"Phase 1 Filters: Area={'Yes' if use_area else 'No'}, Intensity/Ratio={'Yes' if use_intensity else 'No'}")
        self.log_text.append(f"Phase 1 Thresholds: Min Area (A)={a_threshold}, Min Intensity (I)={i_threshold:.0f}, Min Ratio (R)={r_threshold:.1f}")
        self.log_text.append(f"Phase 2 Filter: Aggregate Range=[{min_agg}-{max_agg}]")


        if self.process_thread and self.process_thread.isRunning():
            self.log_text.append("Stopping previous processing job...")
            self.process_thread.stop()
            self.process_thread.wait()
            self.log_text.append("Previous job stopped.")

        self.process_thread = ImageProcessorThread(
            self.input_folder,
            output_folder,
            min_agg,
            max_agg,
            a_threshold,
            i_threshold,
            r_threshold,
            self.model_path,
            use_intensity,
            use_area,
            sam_params
        )
        self.process_thread.processing_finished.connect(self.processing_finished)
        self.process_thread.image_processed.connect(self.image_processed)
        self.process_thread.log_updated.connect(self.update_log)
        self.process_thread.start()

    def enhance_contrast(self):
        if not self.contrast_input_folder:
             QMessageBox.warning(self, "Input Required", "Please select an input folder for contrast enhancement.")
             return

        self.contrast_log_text.append("-" * 20 + " New Contrast Job " + "-" * 20)
        self.contrast_log_text.append("Starting contrast enhancement...")
        self.enhance_contrast_btn.setEnabled(False)
        self.stop_contrast_btn.setEnabled(True)

        output_folder = self.get_contrast_output_folder_path()
        if not output_folder:
            self.contrast_log_text.append("Error: Could not determine output folder.")
            self.enhance_contrast_btn.setEnabled(True)
            self.stop_contrast_btn.setEnabled(False)
            return

        self.contrast_log_text.append(f"Output folder: {output_folder}")
        contrast_factor = self.contrast_spin.value()
        self.contrast_log_text.append(f"Contrast Factor: {contrast_factor:.1f}")

        if self.contrast_thread and self.contrast_thread.isRunning():
            self.contrast_log_text.append("Stopping previous contrast job...")
            self.contrast_thread.stop()
            self.contrast_thread.wait()
            self.contrast_log_text.append("Previous job stopped.")

        self.contrast_thread = ContrastEnhancementThread(
            self.contrast_input_folder,
            output_folder,
            contrast_factor
        )
        self.contrast_thread.processing_finished.connect(self.contrast_enhancement_finished)
        self.contrast_thread.log_updated.connect(self.update_contrast_log)
        self.contrast_thread.start()

    def generate_heatmaps(self):
        min_thresh = self.heatmap_min_aggregate_spin.value()
        max_thresh = self.heatmap_max_aggregate_spin.value()
        if min_thresh > max_thresh or (min_thresh == max_thresh and min_thresh != 100):
             QMessageBox.warning(self, "Invalid Thresholds", "Min Aggregate Threshold must be less than or equal to Max Aggregate Threshold (equality allowed only if both are 100).")
             return

        if not self.heatmap_input_folder:
             QMessageBox.warning(self, "Input Required", "Please select an input folder for heatmap generation.")
             return

        self.heatmap_log_text.append("-" * 20 + " New Heatmap Job " + "-" * 20)
        self.heatmap_log_text.append("Starting heatmap generation...")
        self.generate_heatmap_btn.setEnabled(False)
        self.stop_heatmap_btn.setEnabled(True)

        output_folder = self.get_heatmap_output_folder_path()
        if not output_folder:
            self.heatmap_log_text.append("Error: Could not determine output folder.")
            self.generate_heatmap_btn.setEnabled(True)
            self.stop_heatmap_btn.setEnabled(False)
            return

        self.heatmap_log_text.append(f"Output folder: {output_folder}")
        self.heatmap_log_text.append(f"Aggregate Range: [{min_thresh}-{max_thresh}]")


        if self.heatmap_thread and self.heatmap_thread.isRunning():
            self.heatmap_log_text.append("Stopping previous heatmap job...")
            self.heatmap_thread.stop()
            self.heatmap_thread.wait()
            self.heatmap_log_text.append("Previous job stopped.")

        self.heatmap_thread = HeatmapGeneratorThread(
            self.heatmap_input_folder,
            output_folder,
            min_thresh,
            max_thresh
        )
        self.heatmap_thread.processing_finished.connect(self.heatmap_generation_finished)
        self.heatmap_thread.log_updated.connect(self.update_heatmap_log)
        self.heatmap_thread.start()

    def analyze_folder_aggregate(self):
        self.aggregate_log_text.clear()
        if not self.bright_input_folder:
            self.update_aggregate_log("Please select an input folder first.")
            return

        folder_path = self.bright_input_folder
        if not os.path.exists(folder_path):
            self.update_aggregate_log(f"Error: Directory '{folder_path}' does not exist.")
            return
        if not os.access(folder_path, os.R_OK):
            self.update_aggregate_log(f"Error: No read permission for directory '{folder_path}'.")
            return

        files = [f for f in os.listdir(folder_path)
                 if f.lower().endswith((".png", ".jpg", ".jpeg", ".tif", ".tiff"))]

        if not files:
            self.update_aggregate_log("No image files found in selected folder.")
            return

        image_count = 0
        error_count = 0
        self.update_aggregate_log(f"Analyzing aggregates in: {folder_path}")
        self.update_aggregate_log("-" * 50)
        files.sort()

        for filename in files:
            file_path = os.path.join(folder_path, filename)
            if os.path.isfile(file_path):
                try:
                    aggregate_val = ImageAggregateAnalyzer.calculate_aggregate(file_path)
                    if isinstance(aggregate_val, float):
                        self.update_aggregate_log(f"{filename}: Aggregate = {aggregate_val:.2f}")
                        image_count += 1
                    else:
                        self.update_aggregate_log(f"{filename}: {aggregate_val}")
                        error_count += 1
                except Exception as e:
                    self.update_aggregate_log(f"{filename}: Error - {str(e)}")
                    error_count += 1
                QApplication.processEvents()

        self.update_aggregate_log("-" * 50)
        self.update_aggregate_log(f"Processed {image_count} image(s) successfully.")
        if error_count > 0:
             self.update_aggregate_log(f"Encountered errors with {error_count} file(s).")


    def stop_processing(self):
        if self.process_thread and self.process_thread.isRunning():
            self.log_text.append("Attempting to stop processing...")
            self.process_thread.stop()
            self.stop_btn.setEnabled(False)
        else:
             self.log_text.append("No processing job is currently running.")

    def stop_heatmap_generation(self):
        if self.heatmap_thread and self.heatmap_thread.isRunning():
            self.heatmap_log_text.append("Attempting to stop heatmap generation...")
            self.heatmap_thread.stop()
            self.stop_heatmap_btn.setEnabled(False)
        else:
            self.heatmap_log_text.append("No heatmap generation job is running.")

    def stop_contrast_processing(self):
        if self.contrast_thread and self.contrast_thread.isRunning():
            self.contrast_log_text.append("Attempting to stop contrast enhancement...")
            self.contrast_thread.stop()
            self.stop_contrast_btn.setEnabled(False)
        else:
            self.contrast_log_text.append("No contrast enhancement job is running.")

    def update_log(self, message):
        self.log_text.append(message)
        scrollbar = self.log_text.verticalScrollBar()
        scrollbar.setValue(scrollbar.maximum())

    def update_aggregate_log(self, message):
        self.aggregate_log_text.append(message)
        scrollbar = self.aggregate_log_text.verticalScrollBar()
        scrollbar.setValue(scrollbar.maximum())

    def update_heatmap_log(self, message):
        self.heatmap_log_text.append(message)
        scrollbar = self.heatmap_log_text.verticalScrollBar()
        scrollbar.setValue(scrollbar.maximum())

    def update_contrast_log(self, message):
        self.contrast_log_text.append(message)
        scrollbar = self.contrast_log_text.verticalScrollBar()
        scrollbar.setValue(scrollbar.maximum())

    def processing_finished(self):
        self.log_text.append("Processing complete.")
        self.update_button_states()
        self.stop_btn.setEnabled(False)
        self.process_thread = None

    def heatmap_generation_finished(self):
        self.heatmap_log_text.append("Heatmap generation complete.")
        self.generate_heatmap_btn.setEnabled(bool(self.heatmap_input_folder))
        self.stop_heatmap_btn.setEnabled(False)
        self.heatmap_thread = None

    def contrast_enhancement_finished(self):
        self.contrast_log_text.append("Contrast enhancement complete.")
        self.enhance_contrast_btn.setEnabled(bool(self.contrast_input_folder))
        self.stop_contrast_btn.setEnabled(False)
        self.contrast_thread = None

    def image_processed(self, output_path, processed_image_array):
        self.last_output_path = output_path
        self.last_output_image_array = processed_image_array

    def closeEvent(self, event):
        reply = QMessageBox.question(self, 'Confirm Exit',
                                       "Are you sure you want to exit? Any running processes will be stopped.",
                                       QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                                       QMessageBox.StandardButton.No)

        if reply == QMessageBox.StandardButton.Yes:
            if self.process_thread and self.process_thread.isRunning():
                self.process_thread.stop()
                self.process_thread.wait()
            if self.heatmap_thread and self.heatmap_thread.isRunning():
                self.heatmap_thread.stop()
                self.heatmap_thread.wait()
            if self.contrast_thread and self.contrast_thread.isRunning():
                self.contrast_thread.stop()
                self.contrast_thread.wait()
            event.accept()
        else:
            event.ignore()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    app.setStyle('Fusion')
    window = UsefulImageAnalysisTools()
    window.show()
    sys.exit(app.exec())
