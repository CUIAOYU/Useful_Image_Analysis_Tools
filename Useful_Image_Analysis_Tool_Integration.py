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
    # Attempt to import Segment Anything Model (SAM) related components
    from segment_anything import sam_model_registry, SamAutomaticMaskGenerator
    from segment_anything.utils.amg import rle_to_mask # Utility for mask conversion
except ImportError:
    # Handle missing SAM library
    print("Error: 'segment_anything' library not found.")
    print("Please install it using: pip install segment-anything")
    sys.exit(1)

try:
    # Attempt to import pycocotools for RLE mask handling
    from pycocotools import mask as mask_utils
    pycocotools_available = True
except ImportError:
    # Handle missing pycocotools library
    pycocotools_available = False
    print("Warning: 'pycocotools' library not found. RLE output modes ('uncompressed_rle', 'coco_rle') decoding will be unavailable.")
    print("If you need to use RLE modes, please run: pip install pycocotools")

try:
    # Attempt to import tifffile for TIFF image support
    import tifffile
except ImportError:
    # Handle missing tifffile library
    tifffile = None

# Class to calculate image aggregate value (used in Aggregate Analysis tab)
class ImageAggregateAnalyzer:
    @staticmethod
    def calculate_aggregate(image_path):
        """
        Calculates a normalized aggregate value for a grayscale image.
        Aggregate value represents inverse brightness (0=white, 100=black).

        Args:
            image_path (str): Path to the image file.

        Returns:
            float or str: Normalized aggregate value (0-100) or an error message string.
        """
        try:
            # Validate file existence and permissions
            if not os.path.exists(image_path):
                return f"Error: File not found - {image_path}"
            if not os.access(image_path, os.R_OK):
                return f"Error: No read permission for - {image_path}"

            # Open image, convert to grayscale if needed
            img = Image.open(image_path)
            if img.mode != 'L':
                img = img.convert('L')
            img_array = np.array(img)

            # Calculate mean pixel value and normalize to 0-100 aggregate scale
            avg_pixel_value = np.mean(img_array)
            normalized_aggregate = 100.0 - (avg_pixel_value / 255.0) * 100.0
            return normalized_aggregate
        except Exception as e:
            return f"Error: {str(e)}"

# Thread for generating heatmaps based on pixel aggregate values
class HeatmapGeneratorThread(QThread):
    processing_finished = pyqtSignal() # Signal emitted when processing is done
    log_updated = pyqtSignal(str)      # Signal emitted to update the log text area

    def __init__(self, input_folder, output_folder, min_aggregate_threshold, max_aggregate_threshold):
        """
        Initializes the heatmap generation thread.

        Args:
            input_folder (str): Path to the folder containing input images.
            output_folder (str): Path to the folder where heatmaps will be saved.
            min_aggregate_threshold (int): Minimum aggregate value for heatmap coloring range.
            max_aggregate_threshold (int): Maximum aggregate value for heatmap coloring range.
        """
        super().__init__()
        self.input_folder = input_folder
        self.output_folder = output_folder
        self.min_aggregate_threshold = min_aggregate_threshold
        self.max_aggregate_threshold = max_aggregate_threshold
        self.running = True # Flag to control thread execution
        self.lock = QMutex() # Mutex for thread-safe access to the 'running' flag

    def validate_image(self, image_path):
        """ Basic validation for image file existence and read permissions. """
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image not found: {image_path}")
        if not os.access(image_path, os.R_OK):
            raise PermissionError(f"No read permission for: {image_path}")

    def run(self):
        """ Main execution logic for the heatmap generation thread. """
        try:
            # Validate threshold range
            if self.min_aggregate_threshold >= self.max_aggregate_threshold:
                 # Allow equality only if both are 100
                 if not (self.min_aggregate_threshold == 100 and self.max_aggregate_threshold == 100):
                    raise ValueError("Min aggregate threshold must be strictly lower than Max aggregate threshold (unless both are 100).")

            # Validate input folder
            if not os.path.exists(self.input_folder):
                raise FileNotFoundError(f"Input folder not found: {self.input_folder}")

            # Create and validate output folder (Folder name determined before thread start)
            os.makedirs(self.output_folder, exist_ok=True)
            if not os.access(self.output_folder, os.W_OK):
                raise PermissionError(f"No write permission for: {self.output_folder}")

            self.log_updated.emit(f"Output folder: {self.output_folder}")

            # Find image files in the input folder
            files = [f for f in os.listdir(self.input_folder)
                     if f.lower().endswith((".png", ".jpg", ".jpeg", ".tif", ".tiff"))]

            if not files:
                self.log_updated.emit("Warning: No image files found in input folder")
                self.processing_finished.emit()
                return

            self.log_updated.emit(f"Found {len(files)} image files")
            processed_count = 0

            # Process each image file
            for idx, filename in enumerate(files):
                # Check if processing should stop (thread-safe check)
                self.lock.lock()
                try:
                    if not self.running:
                        self.log_updated.emit("Processing stopped by user.")
                        break
                finally:
                    self.lock.unlock()

                try:
                    input_path = os.path.join(self.input_folder, filename)
                    self.validate_image(input_path) # Validate individual image

                    self.log_updated.emit(f"Processing image {idx+1}/{len(files)}: {filename}")

                    # Generate heatmap and measure time
                    start_time = time.time()
                    # Pass the pre-determined output folder to the static method
                    result = HeatmapGenerator.generate_heatmap(
                        input_path,
                        self.output_folder, # Pass the folder path
                        self.min_aggregate_threshold,
                        self.max_aggregate_threshold
                    )
                    elapsed_time = time.time() - start_time

                    # Log result or error
                    if isinstance(result, str) and not result.startswith("Error"):
                        processed_count += 1
                        self.log_updated.emit(f"Heatmap saved to: {result}")
                        self.log_updated.emit(f"Processing time: {elapsed_time:.2f} seconds")
                    elif isinstance(result, str):
                         self.log_updated.emit(f"Error processing {filename}: {result}")

                except Exception as e:
                    # Log errors during individual file processing
                    self.log_updated.emit(f"Error processing {filename}: {str(e)}")
                    continue # Continue with the next file

            # Log completion status if the process wasn't stopped
            if self.running:
                self.log_updated.emit(f"Completed: {processed_count}/{len(files)} heatmaps generated")

        except Exception as e:
            # Log fatal errors that stop the entire process
            self.log_updated.emit(f"Fatal error during heatmap generation: {str(e)}")
        finally:
            # Ensure the finished signal is emitted
            self.processing_finished.emit()

    def stop(self):
        """ Sets the running flag to False to signal the thread to stop. (Thread-safe) """
        self.lock.lock()
        try:
            self.running = False
        finally:
            self.lock.unlock()

# Static class containing the heatmap generation logic
class HeatmapGenerator:
    @staticmethod
    def generate_heatmap(image_path, output_folder, min_aggregate_threshold=0, max_aggregate_threshold=100):
        """
        Generates a heatmap image based on pixel aggregate values.

        Args:
            image_path (str): Path to the input grayscale image.
            output_folder (str): Folder to save the generated heatmap.
            min_aggregate_threshold (int): Min aggregate value for the color range.
            max_aggregate_threshold (int): Max aggregate value for the color range.

        Returns:
            str: Path to the saved heatmap image or an error message string.
        """
        try:
            # Read image using OpenCV, fallback to tifffile if needed
            image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
            if image is None:
                if tifffile:
                    try:
                        tiff_image = tifffile.imread(image_path)
                        # Handle multi-channel TIFFs (take first channel)
                        if len(tiff_image.shape) > 2:
                            image = tiff_image[:, :, 0]
                        else:
                            image = tiff_image

                        # Normalize non-uint8 images to uint8
                        if image.dtype != np.uint8:
                            if np.max(image) > 0:
                                image = ((image / np.max(image)) * 255).astype(np.uint8)
                            else:
                                image = image.astype(np.uint8) # Handle all-black images

                    except Exception as tiff_e:
                        return f"Error: Failed to read image {os.path.basename(image_path)} with cv2 and tifffile: {tiff_e}"
                else:
                    return f"Error: Failed to read image {os.path.basename(image_path)} with cv2 (tifffile not installed)"

            if image is None:
                 return f"Error: Could not load image {os.path.basename(image_path)}"

            # Initialize output heatmap (BGR format)
            output_heatmap_bgr = np.zeros((image.shape[0], image.shape[1], 3), dtype=np.uint8)

            # Calculate aggregate values (0-100 scale) for all pixels
            aggregate_values = 100.0 - (image.astype(np.float32) / 255.0) * 100.0

            # Define masks based on thresholds
            threshold_range = max_aggregate_threshold - min_aggregate_threshold
            if threshold_range <= 0: # Handles min == max (e.g., both 100)
                print(f"Warning: generate_heatmap called with min_aggregate_threshold ({min_aggregate_threshold}) >= max_aggregate_threshold ({max_aggregate_threshold}). Coloring pixels >= {min_aggregate_threshold}.")
                below_min_mask = aggregate_values < min_aggregate_threshold
                above_max_mask = aggregate_values >= min_aggregate_threshold # Treat max as min in this case
                within_range_mask = np.zeros_like(below_min_mask, dtype=bool) # No range if min >= max
            else:
                # Normal case: define pixels below min, above max, and within range
                below_min_mask = aggregate_values < min_aggregate_threshold
                above_max_mask = aggregate_values > max_aggregate_threshold
                within_range_mask = (~below_min_mask) & (~above_max_mask)

            # Color pixels above the max threshold red
            output_heatmap_bgr[above_max_mask] = [0, 0, 255] # BGR format for red

            # Color pixels within the threshold range using a gradient (Green to Orange/Yellow)
            num_pixels_in_range = np.count_nonzero(within_range_mask)
            if num_pixels_in_range > 0 and threshold_range > 0:
                # Get aggregate values only for pixels within the range
                aggregates_in_range = aggregate_values[within_range_mask]
                # Normalize these values from 0.0 to 1.0 within the range
                normalized_in_range = (aggregates_in_range - min_aggregate_threshold) / threshold_range
                normalized_in_range = np.clip(normalized_in_range, 0.0, 1.0) # Ensure values are within [0, 1]

                # Map normalized values to Hue (0-135: Red -> Yellow -> Green)
                # We invert (1.0 - normalized) so low aggregate (brighter) -> green, high aggregate (darker) -> red/orange
                hue_values = (1.0 - normalized_in_range) * 135.0 # Hue range (adjust 135 for different gradients)
                hue_values = hue_values.astype(np.uint8)

                # Create HSV pixels (Hue from calculation, Saturation=Max, Value=Max)
                hsv_pixels = np.zeros((num_pixels_in_range, 1, 3), dtype=np.uint8)
                hsv_pixels[:, 0, 0] = hue_values      # Hue
                hsv_pixels[:, 0, 1] = 255             # Saturation
                hsv_pixels[:, 0, 2] = 255             # Value

                # Convert HSV pixels to BGR
                bgr_pixels = cv2.cvtColor(hsv_pixels, cv2.COLOR_HSV2BGR)

                # Apply the calculated BGR colors to the output heatmap
                output_heatmap_bgr[within_range_mask] = bgr_pixels[:, 0, :]

            # --- MODIFICATION: Use new heatmap filename ---
            base_name = os.path.splitext(os.path.basename(image_path))[0]
            output_filename = f"{base_name}_heatmap.png"
            output_path = os.path.join(output_folder, output_filename)
            # --- END MODIFICATION ---
            cv2.imwrite(output_path, output_heatmap_bgr)
            return output_path
        except Exception as e:
            # Log detailed error with traceback
            import traceback
            print(f"Error generating heatmap for {os.path.basename(image_path)}:")
            print(traceback.format_exc())
            return f"Error generating heatmap for {os.path.basename(image_path)}: {str(e)}"


# Thread for applying contrast enhancement to images
class ContrastEnhancementThread(QThread):
    processing_finished = pyqtSignal() # Signal emitted when processing is done
    log_updated = pyqtSignal(str)      # Signal emitted to update the log text area

    def __init__(self, input_folder, output_folder, contrast_factor):
        """
        Initializes the contrast enhancement thread.

        Args:
            input_folder (str): Path to the folder containing input images.
            output_folder (str): Path to the folder where enhanced images will be saved.
            contrast_factor (float): Enhancement factor (1.0 = no change).
        """
        super().__init__()
        self.input_folder = input_folder
        self.output_folder = output_folder
        self.contrast_factor = contrast_factor
        self.running = True # Flag to control thread execution
        self.lock = QMutex() # Mutex for thread-safe access to the 'running' flag

    def validate_image(self, image_path):
        """ Basic validation for image file existence and read permissions. """
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image not found: {image_path}")
        if not os.access(image_path, os.R_OK):
            raise PermissionError(f"No read permission for: {image_path}")

    def run(self):
        """ Main execution logic for the contrast enhancement thread. """
        try:
            # Validate input folder
            if not os.path.exists(self.input_folder):
                raise FileNotFoundError(f"Input folder not found: {self.input_folder}")

            # Create and validate output folder (Folder name determined before thread start)
            os.makedirs(self.output_folder, exist_ok=True)
            if not os.access(self.output_folder, os.W_OK):
                raise PermissionError(f"No write permission for: {self.output_folder}")

            self.log_updated.emit(f"Output folder: {self.output_folder}")

            # Find image files in the input folder
            files = [f for f in os.listdir(self.input_folder)
                     if f.lower().endswith((".png", ".jpg", ".jpeg", ".tif", ".tiff"))]

            if not files:
                self.log_updated.emit("Warning: No image files found in input folder")
                self.processing_finished.emit()
                return

            self.log_updated.emit(f"Found {len(files)} image files")
            processed_count = 0

            # Process each image file
            for idx, filename in enumerate(files):
                # Check if processing should stop (thread-safe check)
                self.lock.lock()
                try:
                    if not self.running:
                        self.log_updated.emit("Processing stopped by user.")
                        break
                finally:
                    self.lock.unlock()

                try:
                    input_path = os.path.join(self.input_folder, filename)
                    self.validate_image(input_path) # Validate individual image

                    self.log_updated.emit(f"Processing image {idx+1}/{len(files)}: {filename}")

                    # Perform contrast enhancement using PIL
                    start_time = time.time()
                    img = Image.open(input_path)
                    # Ensure image is in a mode PIL.ImageEnhance can handle (RGB or L)
                    if img.mode not in ('RGB', 'L'):
                        img = img.convert('RGB')

                    enhancer = ImageEnhance.Contrast(img)
                    enhanced_img = enhancer.enhance(self.contrast_factor)

                    # --- MODIFICATION: Use new contrast filename ---
                    base_name, ext = os.path.splitext(filename)
                    output_filename = f"{base_name}_contrast{ext}"
                    output_path = os.path.join(self.output_folder, output_filename)
                    # --- END MODIFICATION ---
                    enhanced_img.save(output_path)

                    elapsed_time = time.time() - start_time
                    processed_count += 1
                    self.log_updated.emit(f"Enhanced image saved to: {output_path}")
                    self.log_updated.emit(f"Processing time: {elapsed_time:.2f} seconds")

                except Exception as e:
                    # Log errors during individual file processing
                    self.log_updated.emit(f"Error processing {filename}: {str(e)}")
                    continue # Continue with the next file

            # Log completion status if the process wasn't stopped
            if self.running:
                self.log_updated.emit(f"Completed: {processed_count}/{len(files)} images enhanced")

        except Exception as e:
            # Log fatal errors that stop the entire process
            self.log_updated.emit(f"Fatal error during contrast enhancement: {str(e)}")
        finally:
            # Ensure the finished signal is emitted
            self.processing_finished.emit()

    def stop(self):
        """ Sets the running flag to False to signal the thread to stop. (Thread-safe) """
        self.lock.lock()
        try:
            self.running = False
        finally:
            self.lock.unlock()


# Thread for processing images using SAM and applying filters based on Area, Intensity, Ratio, and Aggregate Range
class ImageProcessorThread(QThread):
    processing_finished = pyqtSignal() # Signal emitted when processing is done
    image_processed = pyqtSignal(str, np.ndarray) # Signal emitted after each image is processed (path, image_array)
    log_updated = pyqtSignal(str)      # Signal emitted to update the log text area

    MAX_IMAGE_SIZE = 4096 # Maximum dimension allowed for input images
    LOG_MAX_LINES = 1000 # (Not currently used, but could be for log trimming)

    # Colors for drawing bounding boxes/contours
    bright_colors = [
        (0, 0, 255),    # Red
        (0, 165, 255),  # Orange
        (0, 255, 255),  # Yellow
        (0, 255, 0),    # Lime Green
        (147, 20, 255), # Pink/Purple
    ]

    def __init__(self, input_folder, output_folder,
                 min_aggregate_threshold, max_aggregate_threshold, # Added back
                 a_threshold, i_threshold, r_threshold, model_path,
                 use_intensity=False, use_area=True, sam_params=None):
        """
        Initializes the SAM image processing thread with two-phase filtering.

        Args:
            input_folder (str): Path to the folder containing input images.
            output_folder (str): Path to the folder where processed images will be saved.
            min_aggregate_threshold (int): Min aggregate value for Phase 2 pixel filtering.
            max_aggregate_threshold (int): Max aggregate value for Phase 2 pixel filtering.
            a_threshold (int): Minimum Area threshold for Phase 1 region filtering.
            i_threshold (float): Minimum Intensity threshold for Phase 1 region filtering.
            r_threshold (float): Minimum Ratio (I/A) threshold for Phase 1 region filtering.
            model_path (str): Path to the SAM model checkpoint file (.pth).
            use_intensity (bool): Whether to filter based on Intensity and Ratio (Phase 1).
            use_area (bool): Whether to filter based on Area (Phase 1).
            sam_params (dict, optional): Dictionary of parameters for SamAutomaticMaskGenerator. Defaults to None.
        """
        super().__init__()
        self.input_folder = input_folder
        self.output_folder = output_folder
        # Phase 2 Thresholds
        self.min_aggregate_threshold = min_aggregate_threshold
        self.max_aggregate_threshold = max_aggregate_threshold
        # Phase 1 Thresholds
        self.a_threshold = a_threshold
        self.i_threshold = i_threshold
        self.r_threshold = r_threshold
        self.model_path = model_path
        self.running = True # Flag to control thread execution
        # Flags to determine which Phase 1 metrics to calculate and filter by
        self.use_intensity = use_intensity
        self.use_area = use_area
        # SAM model parameters
        self.sam_params = sam_params if sam_params else {}
        self.lock = QMutex() # Mutex for thread-safe access to the 'running' flag

    def validate_thresholds(self):
        """ Placeholder for any future threshold validation logic. """
        pass # No complex validation needed currently


    def validate_image(self, image_path):
        """ Basic validation for image file existence and read permissions. """
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image not found: {image_path}")
        if not os.access(image_path, os.R_OK):
            raise PermissionError(f"No read permission for: {image_path}")

    def run(self):
        """ Main execution logic for the SAM processing thread. """
        try:
            # Validate input folder
            if not os.path.exists(self.input_folder):
                raise FileNotFoundError(f"Input folder not found: {self.input_folder}")

            # Create and validate output folder (Folder name determined before thread start)
            os.makedirs(self.output_folder, exist_ok=True)
            if not os.access(self.output_folder, os.W_OK):
                raise PermissionError(f"No write permission for: {self.output_folder}")

            self.log_updated.emit(f"Output folder: {self.output_folder}")
            self.log_updated.emit("Loading SAM model...")

            # Determine device (GPU if available, else CPU)
            device = "cuda" if torch.cuda.is_available() else "cpu"
            self.log_updated.emit(f"Using device: {device}")

            # Load SAM model
            try:
                # Validate model file existence and permissions
                if not os.path.exists(self.model_path):
                        raise FileNotFoundError(f"SAM model file not found: {self.model_path}")
                if not os.access(self.model_path, os.R_OK):
                        raise PermissionError(f"No read permission for SAM model file: {self.model_path}")

                # Infer model type from filename (adjust if using different naming conventions)
                model_type = "vit_h" # Default
                if "vit_l" in os.path.basename(self.model_path).lower():
                    model_type = "vit_l"
                elif "vit_b" in os.path.basename(self.model_path).lower():
                    model_type = "vit_b"
                self.log_updated.emit(f"Attempting to load SAM model type: {model_type}")

                # Instantiate the SAM model registry and load the checkpoint
                sam = sam_model_registry[model_type](checkpoint=self.model_path).to(device)
                sam.eval() # Set model to evaluation mode

            except KeyError:
                # Handle case where inferred model type is invalid
                raise RuntimeError(f"Invalid SAM model type '{model_type}' inferred from filename. Check model file or registry.")
            except FileNotFoundError as e:
                raise e # Re-raise file not found errors
            except PermissionError as e:
                raise e # Re-raise permission errors
            except Exception as e:
                # Provide more specific error messages for common loading issues
                if "checksum mismatch" in str(e):
                    raise RuntimeError(f"Failed to load SAM model: Checksum mismatch. Model file might be corrupt or incomplete. ({e})")
                elif "unexpected keyword argument 'checkpoint'" in str(e):
                    # This often indicates a version mismatch between the library and the model file
                    raise RuntimeError(f"Failed to load SAM model: Mismatch between segment-anything library version and model file. Try updating the library. ({e})")
                else:
                    # General error message for other loading failures
                    raise RuntimeError(f"Failed to load SAM model: {str(e)}")

            # Initialize the automatic mask generator with specified parameters
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
                min_mask_region_area=self.sam_params.get('min_mask_region_area', 0), # Postprocessing filter
                output_mode=self.sam_params.get('output_mode', 'binary_mask') # Recommended mode
            )

            # Log the SAM parameters being used
            self.log_updated.emit("SAM parameters:")
            for key, value in self.sam_params.items():
                self.log_updated.emit(f"  - {key}: {value}")

            # Find image files in the input folder
            files = [f for f in os.listdir(self.input_folder)
                     if f.lower().endswith((".png", ".jpg", ".jpeg", ".tif", ".tiff"))]

            if not files:
                self.log_updated.emit("Warning: No image files found in input folder")
                self.processing_finished.emit()
                return

            self.log_updated.emit(f"Found {len(files)} image files")
            processed_count = 0

            # Process each image file
            for idx, filename in enumerate(files):
                # Check if processing should stop (thread-safe check)
                self.lock.lock()
                try:
                    if not self.running:
                        self.log_updated.emit("Processing stopped by user.")
                        break
                finally:
                    self.lock.unlock()

                try:
                    input_path = os.path.join(self.input_folder, filename)
                    self.validate_image(input_path) # Validate individual image

                    # --- MODIFICATION: Use new simpler SAM filename ---
                    file_base, file_ext = os.path.splitext(filename)
                    output_filename = f"{file_base}_sam_processed{file_ext}" # Changed suffix
                    # The output folder path (containing parameters) is determined before the thread starts
                    # and passed during thread initialization.
                    output_path = os.path.join(self.output_folder, output_filename)
                    # --- END MODIFICATION ---

                    self.log_updated.emit(f"Processing image {idx+1}/{len(files)}: {filename}")

                    # Process the image (SAM mask generation + filtering) and measure time
                    start_time = time.time()
                    processed_image = self.process_image(input_path, output_path, mask_generator)
                    elapsed_time = time.time() - start_time

                    # If processing was successful, emit signal and log
                    if processed_image is not None:
                        self.image_processed.emit(output_path, processed_image)
                        processed_count += 1
                        self.log_updated.emit(f"Processed image saved to: {output_path}")

                    self.log_updated.emit(f"Processing time: {elapsed_time:.2f} seconds")

                except Exception as e:
                    # Log errors during individual file processing
                    self.log_updated.emit(f"Error processing {filename}: {str(e)}")
                    continue # Continue with the next file

            # Log completion status if the process wasn't stopped
            if self.running:
                self.log_updated.emit(f"Completed: {processed_count}/{len(files)} images processed")

        except RuntimeError as e:
            # Handle specific runtime errors like CUDA OOM
            if "CUDA out of memory" in str(e):
                self.log_updated.emit("Error: GPU memory full - try smaller images or reduce SAM parameters (e.g., points_per_side, points_per_batch, disable cropping)")
            else:
                self.log_updated.emit(f"Runtime error: {str(e)}")
        except Exception as e:
            # Log fatal errors that stop the entire process
            self.log_updated.emit(f"Fatal error during processing: {str(e)}")
        finally:
            # Clean up GPU memory if CUDA was used
            if 'sam' in locals() and device == 'cuda':
                del sam # Delete the model object
                if torch.cuda.is_available():
                        torch.cuda.empty_cache() # Clear PyTorch's CUDA cache
                        self.log_updated.emit("CUDA cache cleared.")
            # Ensure the finished signal is emitted
            self.processing_finished.emit()

    def process_image(self, input_path, output_path, mask_generator):
        """
        Processes a single image: generates masks, calculates metrics, filters (Phase 1: A/I/R, Phase 2: Aggregate), and draws results.
        Calculates displayed A/I/R based on pixels passing *both* phases.

        Args:
            input_path (str): Path to the input image.
            output_path (str): Path where the processed output image will be saved.
            mask_generator (SamAutomaticMaskGenerator): Initialized SAM mask generator instance.

        Returns:
            np.ndarray or None: The processed image with drawn overlays, or None if an error occurred.
        """
        try:
            # Read image (BGR for SAM, Grayscale for aggregate calculations)
            original_image_bgr = cv2.imread(input_path, cv2.IMREAD_COLOR)
            grayscale_image = cv2.imread(input_path, cv2.IMREAD_GRAYSCALE)

            # Fallback to tifffile if cv2 fails (common for some TIFF formats)
            if original_image_bgr is None or grayscale_image is None:
                 if tifffile:
                     try:
                         tiff_image = tifffile.imread(input_path)
                         # Handle different TIFF structures (RGB, Grayscale)
                         if len(tiff_image.shape) == 3 and tiff_image.shape[2] >= 3:
                             # Assume RGB-like, convert to BGR
                             original_image_bgr = cv2.cvtColor(tiff_image[:,:,:3], cv2.COLOR_RGB2BGR)
                             grayscale_image = cv2.cvtColor(original_image_bgr, cv2.COLOR_BGR2GRAY)
                         elif len(tiff_image.shape) == 2:
                             # Assume Grayscale
                             grayscale_image = tiff_image
                             original_image_bgr = cv2.cvtColor(grayscale_image, cv2.COLOR_GRAY2BGR) # Create BGR version
                         else:
                             raise ValueError("Unsupported TIFF structure")

                         # Normalize non-uint8 images to uint8 (required by SAM/OpenCV drawing)
                         if original_image_bgr.dtype != np.uint8:
                             if np.max(original_image_bgr) > 0: original_image_bgr = ((original_image_bgr / np.max(original_image_bgr)) * 255).astype(np.uint8)
                             else: original_image_bgr = original_image_bgr.astype(np.uint8)
                         if grayscale_image.dtype != np.uint8:
                             if np.max(grayscale_image) > 0: grayscale_image = ((grayscale_image / np.max(grayscale_image)) * 255).astype(np.uint8)
                             else: grayscale_image = grayscale_image.astype(np.uint8)

                     except Exception as tiff_e:
                         self.log_updated.emit(f"Warning: Failed to read {os.path.basename(input_path)} with tifffile after cv2 failed: {tiff_e}")
                         return None # Cannot process this image
                 else:
                     # cv2 failed and tifffile is not available
                     self.log_updated.emit(f"Error: Failed to read image {os.path.basename(input_path)} (cv2 failed, tifffile not installed)")
                     return None

            # Final check if image loading succeeded
            if original_image_bgr is None or grayscale_image is None:
                 self.log_updated.emit(f"Error: Could not load image {os.path.basename(input_path)} in a usable format.")
                 return None

            # Check image dimensions against the limit
            if original_image_bgr.shape[0] > self.MAX_IMAGE_SIZE or original_image_bgr.shape[1] > self.MAX_IMAGE_SIZE:
                self.log_updated.emit(f"Error: Image {os.path.basename(input_path)} too large ({original_image_bgr.shape[0]}x{original_image_bgr.shape[1]}), max allowed is {self.MAX_IMAGE_SIZE}x{self.MAX_IMAGE_SIZE}")
                return None

            # Generate masks using SAM
            self.log_updated.emit(f"  Generating masks for {os.path.basename(input_path)}...")
            try:
                masks = mask_generator.generate(original_image_bgr) # Requires BGR image
                self.log_updated.emit(f"  Found {len(masks)} masks.")
            except torch.cuda.OutOfMemoryError:
                # Handle CUDA OOM during mask generation
                self.log_updated.emit(f"Error: GPU memory exhausted during mask generation for {os.path.basename(input_path)}. Try smaller image or reduce SAM parameters.")
                if torch.cuda.is_available(): torch.cuda.empty_cache() # Attempt to clear cache
                return None
            except Exception as e:
                # Handle other mask generation errors
                self.log_updated.emit(f"Error during mask generation for {os.path.basename(input_path)}: {str(e)}")
                return None

            # Prepare output image and list to track text positions
            output_image = original_image_bgr.copy()
            height, width = output_image.shape[:2]
            drawn_text_boxes = [] # To avoid text overlap

            # Process each generated mask
            for i, mask_info in enumerate(masks):
                segmentation_data = mask_info["segmentation"]
                segmentation_mask = None # Initialize mask for this region

                # Decode mask based on its format (binary array or RLE)
                if isinstance(segmentation_data, dict): # RLE format
                    if pycocotools_available:
                        try:
                            # Decode RLE using pycocotools
                            segmentation_mask = mask_utils.decode(segmentation_data).astype(np.uint8)
                        except Exception as decode_e:
                            self.log_updated.emit(f"  Warning: Failed to decode RLE mask ({decode_e}), skipping mask.")
                            continue # Skip this mask
                    else:
                        # RLE detected but pycocotools not available
                        self.log_updated.emit(f"  Error: Detected RLE mask but 'pycocotools' not installed or failed to import. Cannot process this mask. Please install 'pycocotools' or select 'binary_mask' output mode.")
                        continue # Skip this mask

                elif isinstance(segmentation_data, np.ndarray): # Binary mask format
                    segmentation_mask = segmentation_data.astype(np.uint8)
                else:
                    # Unknown format
                    self.log_updated.emit(f"  Warning: Unknown mask format type ({type(segmentation_data)}), skipping mask.")
                    continue # Skip this mask

                # Ensure mask decoding was successful
                if segmentation_mask is None:
                    continue

                # Get grayscale pixel values within the current mask
                region_pixels_0_255 = grayscale_image[segmentation_mask > 0]

                # Skip if mask is empty
                if len(region_pixels_0_255) == 0:
                    continue

                # Calculate aggregate values (0-100) for ALL pixels within the mask (needed for both phases)
                aggregate_values_0_100 = 100.0 - (region_pixels_0_255 / 255.0) * 100.0

                # --- Phase 1: A/I/R Filtering (Based on WHOLE region) ---
                # Calculate initial metrics based on the whole region
                initial_area = 0
                initial_intensity = 0.0
                initial_ratio = 0.0
                passed_phase1 = True # Flag to track if region passes Phase 1

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
                # --- End Phase 1 Check ---

                # Proceed only if the region passed Phase 1
                if passed_phase1:
                    # --- Phase 2: Min/Max Aggregate Filtering ---
                    # Find which pixels within the region pass the aggregate filter
                    pixels_passed_phase2_mask = (
                        (aggregate_values_0_100 >= self.min_aggregate_threshold) &
                        (aggregate_values_0_100 <= self.max_aggregate_threshold)
                    )
                    num_pixels_passed_phase2 = np.count_nonzero(pixels_passed_phase2_mask)

                    # Only draw if the region also passes Phase 2 (at least one pixel is within range)
                    if num_pixels_passed_phase2 > 0:
                        # --- MODIFICATION: Calculate final A/I/R based on pixels passing Phase 2 ---
                        final_area = num_pixels_passed_phase2
                        final_aggregates = aggregate_values_0_100[pixels_passed_phase2_mask]
                        final_intensity = np.sum(final_aggregates)
                        final_ratio = final_intensity / final_area if final_area > 0 else 0.0
                        # --- END MODIFICATION ---

                        # --- Draw Region that Passed Both Phases ---

                        # Prepare text to display (using FINAL A, I, R values)
                        value_text_parts = []
                        if self.use_area:
                            value_text_parts.append(f"A:{final_area}") # Use final_area
                        if self.use_intensity:
                            value_text_parts.append(f"I:{final_intensity:.0f}") # Use final_intensity
                            if self.use_area and final_area > 0: # Check final_area here
                                value_text_parts.append(f"R:{final_ratio:.2f}") # Use final_ratio
                            elif self.use_area:
                                value_text_parts.append("R:N/A")
                        value_text = " ".join(value_text_parts) if value_text_parts else "N/A"

                        # Find contours of the ORIGINAL mask to draw the outline
                        contours, _ = cv2.findContours(segmentation_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                        color_bgr = self.bright_colors[i % len(self.bright_colors)] # Cycle through colors
                        cv2.drawContours(output_image, contours, -1, color_bgr, 1) # Draw outline

                        # Calculate center of the ORIGINAL mask for text placement
                        ys, xs = np.where(segmentation_mask > 0)
                        if len(xs) == 0 or len(ys) == 0: continue # Should not happen if initial_area > 0

                        center_x = int(np.mean(xs))
                        center_y = int(np.mean(ys))

                        # Initial text position near the center, clamped within image bounds
                        text_x = max(5, min(center_x, width - 10))
                        text_y = max(15, min(center_y, height - 5))

                        # Get text size to check for overlaps
                        font = cv2.FONT_HERSHEY_SIMPLEX
                        font_scale = 0.4
                        thickness = 1
                        (text_width, text_height), baseline = cv2.getTextSize(value_text, font, font_scale, thickness)

                        # Define the bounding box for the current text
                        current_text_box = [text_x, text_y - text_height, text_width, text_height + baseline]
                        shift_amount = text_height + 2 # How much to shift text down if overlap occurs

                        # Attempt to reposition text vertically to avoid overlap with previous text
                        attempts = 0
                        max_attempts = 10 # Limit attempts to prevent infinite loops
                        while any(self.boxes_overlap(current_text_box, box) for box in drawn_text_boxes) and attempts < max_attempts:
                            text_y += shift_amount
                            # If text goes off bottom, try shifting upwards from original position
                            if text_y > height - 5:
                                 text_y = max(15, min(center_y, height - 5)) # Reset Y
                                 text_y -= (attempts + 1) * shift_amount # Shift up
                                 if text_y < 15: # If still overlapping after shifting up, give up repositioning
                                     text_y = max(15, min(center_y, height - 5)) # Reset to original Y
                                     break # Stop trying to reposition

                            # Update current text box position
                            current_text_box = [text_x, text_y - text_height, text_width, text_height + baseline]
                            attempts += 1

                        # Draw the text (using FINAL A/I/R) and record its bounding box
                        cv2.putText(output_image, value_text, (text_x, text_y), font, font_scale, color_bgr, thickness, cv2.LINE_AA)
                        drawn_text_boxes.append(current_text_box)
                    # --- End Phase 2 Check ---
                # --- End Phase 1 Check ---

            # Save the final processed image
            try:
                cv2.imwrite(output_path, output_image)
                return output_image # Return the processed image array
            except Exception as e:
                self.log_updated.emit(f"Error saving output image {output_path}: {str(e)}")
                return None # Indicate error during saving

        except Exception as e:
            # General error handling for the entire image processing step
            self.log_updated.emit(f"General error processing {os.path.basename(input_path)}: {str(e)}")
            # Clear CUDA cache if a CUDA-related error might have occurred
            if 'cuda' in str(e).lower() and torch.cuda.is_available(): torch.cuda.empty_cache()
            return None # Indicate error

    def boxes_overlap(self, box1, box2):
        """ Checks if two rectangular boxes overlap. """
        x1, y1, w1, h1 = box1
        x2, y2, w2, h2 = box2
        # Check for non-overlap conditions
        if (x1 + w1 < x2 or x2 + w2 < x1 or y1 + h1 < y2 or y2 + h2 < y1):
            return False
        return True # Overlap detected

    def stop(self):
        """ Sets the running flag to False to signal the thread to stop. (Thread-safe) """
        self.lock.lock()
        try:
            self.running = False
        finally:
            self.lock.unlock()

# Main application window class
class UsefulImageAnalysisToolIntegration(QMainWindow):
    def __init__(self):
        super().__init__()
        # Folder paths (initialized to None)
        self.input_folder = None # For SAM processing
        self.model_path = None   # Path to SAM model file
        self.bright_input_folder = None # For Aggregate Analysis
        self.heatmap_input_folder = None # For Heatmap Generation
        self.contrast_input_folder = None # For Contrast Enhancement

        # Thread objects (initialized to None)
        self.process_thread = None
        self.heatmap_thread = None
        self.contrast_thread = None

        # Variables to store last processed image info (optional)
        self.last_output_path = None
        self.last_output_image_array = None

        self.initUI() # Initialize the user interface

    def initUI(self):
        """ Sets up the main window UI elements and layout. """
        self.setWindowTitle('Useful Image Analysis Tool Integration')
        self.setGeometry(100, 100, 1300, 950) # Set window position and size

        # Main widget and layout (Horizontal Split: Controls | Logs)
        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        main_layout = QHBoxLayout(main_widget)
        main_layout.setContentsMargins(15, 15, 15, 15) # Add padding around main layout
        main_layout.setSpacing(15) # Space between left and right panels

        # --- Left Panel (Controls) ---
        left_widget = QWidget()
        left_layout = QVBoxLayout(left_widget)
        left_layout.setContentsMargins(0, 0, 0, 0) # No internal padding
        left_layout.setSpacing(15) # Space between elements in the left panel

        # Tab widget to organize different functionalities
        self.tabs = QTabWidget()
        self.tabs.setStyleSheet("""
            /* Style for Tab Bar */
            QTabBar::tab {
                padding: 8px 15px; /* Padding inside each tab */
                font-weight: bold; /* Bold tab titles */
                border: 1px solid #cccccc; /* Light grey border */
                border-bottom: none; /* No bottom border for tabs */
                border-top-left-radius: 4px; /* Rounded top corners */
                border-top-right-radius: 4px;
                margin-right: 2px; /* Small space between tabs */
                background-color: #e0e0e0; /* Light grey background for inactive tabs */
            }
            /* Style for Selected Tab */
            QTabBar::tab:selected {
                background: #ffffff; /* White background */
                border-color: #aaaaaa; /* Slightly darker border */
                border-bottom: 1px solid #ffffff; /* Seamless connection to pane */
            }
            /* Style for Hovered Tab (Not Selected) */
            QTabBar::tab:!selected:hover {
                background: #f0f0f0; /* Slightly lighter grey on hover */
            }
            /* Style for Tab Content Pane */
            QTabWidget::pane {
                border: 1px solid #aaaaaa; /* Border matching selected tab */
                border-top: none; /* No top border (covered by tab bar) */
                background-color: #ffffff; /* White background for content */
                border-bottom-left-radius: 4px; /* Rounded bottom corners */
                border-bottom-right-radius: 4px;
            }
        """)

        # Create individual tab widgets
        contrast_tab = QWidget()
        aggregate_tab = QWidget()
        heatmap_tab = QWidget()
        process_tab = QWidget() # SAM Processing Tab

        # --- Contrast Enhancement Tab ---
        contrast_layout = QVBoxLayout(contrast_tab)
        contrast_layout.setContentsMargins(10, 10, 10, 10)
        contrast_layout.setSpacing(15)
        # File selection group
        contrast_file_group = QGroupBox("File Selection")
        contrast_file_group.setStyleSheet("QGroupBox { font-weight: bold; }")
        contrast_file_layout = QVBoxLayout(contrast_file_group)
        contrast_file_layout.setContentsMargins(10, 15, 10, 10)
        contrast_input_layout = QHBoxLayout()
        self.contrast_input_folder_label = QLabel("Input folder: Not selected")
        contrast_input_folder_btn = QPushButton("Select Input Folder")
        contrast_input_folder_btn.setStyleSheet(self.get_button_style("green"))
        contrast_input_folder_btn.clicked.connect(self.select_contrast_input_folder)
        contrast_input_layout.addWidget(self.contrast_input_folder_label, 1) # Label takes available space
        contrast_input_layout.addWidget(contrast_input_folder_btn)
        contrast_file_layout.addLayout(contrast_input_layout)
        contrast_layout.addWidget(contrast_file_group)
        # Contrast parameters group
        contrast_param_group = QGroupBox("Contrast Parameters")
        contrast_param_group.setStyleSheet("QGroupBox { font-weight: bold; }")
        contrast_param_layout = QFormLayout(contrast_param_group) # Form layout for label-widget pairs
        contrast_param_layout.setContentsMargins(10, 15, 10, 10)
        contrast_param_layout.setVerticalSpacing(10)
        contrast_param_layout.setLabelAlignment(Qt.AlignmentFlag.AlignRight) # Align labels to the right
        self.contrast_spin = QDoubleSpinBox() # Spin box for float values
        self.contrast_spin.setRange(0.1, 10.0)
        self.contrast_spin.setSingleStep(0.1)
        self.contrast_spin.setValue(1.0) # Default contrast factor
        self.contrast_spin.setDecimals(1)
        contrast_param_layout.addRow("Contrast factor (0.1-10.0):", self.contrast_spin)
        contrast_layout.addWidget(contrast_param_group)
        # Operation buttons layout
        contrast_operation_layout = QHBoxLayout()
        self.enhance_contrast_btn = QPushButton("Enhance Contrast")
        self.enhance_contrast_btn.setStyleSheet(self.get_button_style("blue"))
        self.enhance_contrast_btn.setEnabled(False) # Disabled initially
        self.enhance_contrast_btn.clicked.connect(self.enhance_contrast)
        self.stop_contrast_btn = QPushButton("Stop Processing")
        self.stop_contrast_btn.setStyleSheet(self.get_button_style("red"))
        self.stop_contrast_btn.setEnabled(False) # Disabled initially
        self.stop_contrast_btn.clicked.connect(self.stop_contrast_processing)
        self.open_contrast_output_btn = QPushButton("Open Output Folder")
        self.open_contrast_output_btn.setStyleSheet(self.get_button_style("grey"))
        self.open_contrast_output_btn.setEnabled(False) # Disabled initially
        self.open_contrast_output_btn.clicked.connect(self.open_contrast_output_folder)
        contrast_operation_layout.addWidget(self.enhance_contrast_btn)
        contrast_operation_layout.addWidget(self.stop_contrast_btn)
        contrast_operation_layout.addWidget(self.open_contrast_output_btn)
        contrast_layout.addLayout(contrast_operation_layout)
        contrast_layout.addStretch() # Push elements upwards

        # --- Aggregate Analysis Tab ---
        aggregate_layout = QVBoxLayout(aggregate_tab)
        aggregate_layout.setContentsMargins(10, 10, 10, 10)
        aggregate_layout.setSpacing(15)
        # File selection group
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
        # Operation button layout
        agg_operation_layout = QHBoxLayout()
        self.analyze_agg_btn = QPushButton("Analyze Aggregate")
        self.analyze_agg_btn.setStyleSheet(self.get_button_style("blue"))
        self.analyze_agg_btn.setEnabled(False) # Disabled initially
        self.analyze_agg_btn.clicked.connect(self.analyze_folder_aggregate)
        agg_operation_layout.addWidget(self.analyze_agg_btn)
        agg_operation_layout.addStretch() # Push button to the left
        aggregate_layout.addLayout(agg_operation_layout)
        aggregate_layout.addStretch() # Push elements upwards

        # --- Heatmap Generation Tab ---
        heatmap_layout = QVBoxLayout(heatmap_tab)
        heatmap_layout.setContentsMargins(10, 10, 10, 10)
        heatmap_layout.setSpacing(15)
        # File selection group
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
        # Heatmap parameters group
        heatmap_param_group = QGroupBox("Heatmap Parameters (Aggregate 0-100)")
        heatmap_param_group.setStyleSheet("QGroupBox { font-weight: bold; }")
        heatmap_param_layout = QFormLayout(heatmap_param_group)
        heatmap_param_layout.setContentsMargins(10, 15, 10, 10)
        heatmap_param_layout.setVerticalSpacing(10)
        heatmap_param_layout.setLabelAlignment(Qt.AlignmentFlag.AlignRight)
        self.heatmap_min_aggregate_spin = QSpinBox() # Integer spin box
        self.heatmap_min_aggregate_spin.setRange(0, 100)
        self.heatmap_min_aggregate_spin.setValue(35) # Default min threshold
        heatmap_param_layout.addRow("Min Aggregate Threshold:", self.heatmap_min_aggregate_spin)
        self.heatmap_max_aggregate_spin = QSpinBox()
        self.heatmap_max_aggregate_spin.setRange(0, 100)
        self.heatmap_max_aggregate_spin.setValue(90) # Default max threshold
        heatmap_param_layout.addRow("Max Aggregate Threshold:", self.heatmap_max_aggregate_spin)
        heatmap_layout.addWidget(heatmap_param_group)
        # Operation buttons layout
        heatmap_operation_layout = QHBoxLayout()
        self.generate_heatmap_btn = QPushButton("Generate Heatmaps")
        self.generate_heatmap_btn.setStyleSheet(self.get_button_style("blue"))
        self.generate_heatmap_btn.setEnabled(False) # Disabled initially
        self.generate_heatmap_btn.clicked.connect(self.generate_heatmaps)
        self.stop_heatmap_btn = QPushButton("Stop Heatmap Generation")
        self.stop_heatmap_btn.setStyleSheet(self.get_button_style("red"))
        self.stop_heatmap_btn.setEnabled(False) # Disabled initially
        self.stop_heatmap_btn.clicked.connect(self.stop_heatmap_generation)
        self.open_heatmap_output_btn = QPushButton("Open Output Folder")
        self.open_heatmap_output_btn.setStyleSheet(self.get_button_style("grey"))
        self.open_heatmap_output_btn.setEnabled(False) # Disabled initially
        self.open_heatmap_output_btn.clicked.connect(self.open_heatmap_output_folder)
        heatmap_operation_layout.addWidget(self.generate_heatmap_btn)
        heatmap_operation_layout.addWidget(self.stop_heatmap_btn)
        heatmap_operation_layout.addWidget(self.open_heatmap_output_btn)
        heatmap_layout.addLayout(heatmap_operation_layout)
        heatmap_layout.addStretch() # Push elements upwards

        # --- SAM Processing Tab ---
        process_layout = QVBoxLayout(process_tab)
        process_layout.setContentsMargins(10, 10, 10, 10)
        process_layout.setSpacing(15)
        # Scroll area for potentially long list of parameters
        process_scroll_area = QScrollArea()
        process_scroll_area.setWidgetResizable(True) # Allow content widget to resize
        process_scroll_widget = QWidget() # Widget to hold the scrollable content
        process_scroll_layout = QVBoxLayout(process_scroll_widget) # Layout for the scrollable content
        process_scroll_layout.setContentsMargins(5, 5, 5, 5)
        process_scroll_layout.setSpacing(15)
        # File selection group (Input folder, Model file)
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

        # Phase 1: Analysis Parameters & Thresholds group (A, I, R)
        param_group_phase1 = QGroupBox("Phase 1: Region Property Filters")
        param_group_phase1.setStyleSheet("QGroupBox { font-weight: bold; }")
        param_layout_phase1 = QFormLayout(param_group_phase1)
        param_layout_phase1.setContentsMargins(10, 15, 10, 10)
        param_layout_phase1.setVerticalSpacing(10)
        param_layout_phase1.setLabelAlignment(Qt.AlignmentFlag.AlignRight)

        # Area Threshold
        self.a_threshold_spin = QSpinBox()
        self.a_threshold_spin.setRange(0, 1000000) # Allow large areas
        self.a_threshold_spin.setValue(199) # Default value set to 199
        self.a_threshold_spin.setToolTip("Phase 1: Regions with Area (pixel count) less than this value will be filtered out.")
        param_layout_phase1.addRow("Min Area (A) Threshold:", self.a_threshold_spin)

        # Intensity Threshold
        self.i_threshold_spin = QDoubleSpinBox()
        self.i_threshold_spin.setRange(0.0, 100000000.0) # Allow large intensity sums
        self.i_threshold_spin.setValue(0.0) # Default value (no intensity filtering)
        self.i_threshold_spin.setDecimals(0) # Display as integer
        self.i_threshold_spin.setToolTip("Phase 1: Regions with Intensity (sum of aggregates) less than this value will be filtered out.")
        param_layout_phase1.addRow("Min Intensity (I) Threshold:", self.i_threshold_spin)

        # Ratio Threshold
        self.r_threshold_spin = QDoubleSpinBox()
        self.r_threshold_spin.setRange(0.0, 100.0) # Ratio is average aggregate (0-100)
        self.r_threshold_spin.setSingleStep(0.1)
        self.r_threshold_spin.setValue(37.0) # Default value set to 37.0
        self.r_threshold_spin.setDecimals(1)
        self.r_threshold_spin.setToolTip("Phase 1: Regions with Ratio (I/A) less than this value will be filtered out.")
        param_layout_phase1.addRow("Min Ratio (R=I/A) Threshold:", self.r_threshold_spin)

        # Checkboxes to enable/disable A/I/R calculations and filtering (Phase 1)
        metrics_layout = QHBoxLayout()
        metrics_layout.setSpacing(15)
        self.area_check = QCheckBox("Area (pixel count)")
        self.area_check.setStyleSheet("font-weight: normal;") # Normal font weight for checkbox text
        self.area_check.setChecked(True) # Enabled by default
        self.area_check.stateChanged.connect(self.on_calculation_method_changed)
        metrics_layout.addWidget(self.area_check)
        self.intensity_check = QCheckBox("Intensity & Ratio") # Combined label
        self.intensity_check.setStyleSheet("font-weight: normal;")
        self.intensity_check.setChecked(True) # Enabled by default
        self.intensity_check.stateChanged.connect(self.on_calculation_method_changed)
        metrics_layout.addWidget(self.intensity_check)
        metrics_layout.addStretch() # Push checkboxes to the left
        param_layout_phase1.addRow("Enable Phase 1 Filters:", metrics_layout)
        process_scroll_layout.addWidget(param_group_phase1) # Add Phase 1 group to scroll layout

        # Phase 2: Pixel Aggregate Filter Group
        param_group_phase2 = QGroupBox("Phase 2: Pixel Aggregate Filter (Applied to Regions Passing Phase 1)")
        param_group_phase2.setStyleSheet("QGroupBox { font-weight: bold; }")
        param_layout_phase2 = QFormLayout(param_group_phase2)
        param_layout_phase2.setContentsMargins(10, 15, 10, 10)
        param_layout_phase2.setVerticalSpacing(10)
        param_layout_phase2.setLabelAlignment(Qt.AlignmentFlag.AlignRight)

        # Min Aggregate Threshold (Re-added)
        self.min_aggregate_spin = QSpinBox()
        self.min_aggregate_spin.setRange(0, 100)
        self.min_aggregate_spin.setValue(35) # Default value set to 35
        self.min_aggregate_spin.setToolTip("Phase 2: Pixels within passed regions with Aggregate value below this are ignored for final inclusion check.\nDisplayed A/I/R values are based on pixels passing this filter.")
        param_layout_phase2.addRow("Min Aggregate Threshold:", self.min_aggregate_spin)

        # Max Aggregate Threshold (Re-added)
        self.max_aggregate_spin = QSpinBox()
        self.max_aggregate_spin.setRange(0, 100)
        self.max_aggregate_spin.setValue(100) # Default to 100 (no max filtering)
        self.max_aggregate_spin.setToolTip("Phase 2: Pixels within passed regions with Aggregate value above this are ignored for final inclusion check.\nDisplayed A/I/R values are based on pixels passing this filter.")
        param_layout_phase2.addRow("Max Aggregate Threshold:", self.max_aggregate_spin)
        process_scroll_layout.addWidget(param_group_phase2) # Add Phase 2 group to scroll layout


        # SAM Auto Mask Generator Parameters group
        sam_params_group = QGroupBox("SAM Auto Mask Generator Parameters")
        sam_params_group.setStyleSheet("QGroupBox { font-weight: bold; }")
        sam_params_layout = QFormLayout(sam_params_group)
        sam_params_layout.setContentsMargins(10, 15, 10, 10)
        sam_params_layout.setVerticalSpacing(10)
        sam_params_layout.setLabelAlignment(Qt.AlignmentFlag.AlignRight)
        # Add SpinBoxes and ComboBox for each SAM parameter
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
        self.crop_n_layers_spin.setRange(0, 5) # 0 disables cropping
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
        self.crop_overlap_ratio_spin.setValue(round(512/1500, 3)) # Default overlap ratio
        self.crop_overlap_ratio_spin.setToolTip("Sets the degree of overlap between crops when Crop N Layers > 0")
        sam_params_layout.addRow("Crop Overlap Ratio:", self.crop_overlap_ratio_spin)
        self.crop_n_points_downscale_factor_spin = QSpinBox()
        self.crop_n_points_downscale_factor_spin.setRange(1, 10) # 1 means no downscaling
        self.crop_n_points_downscale_factor_spin.setValue(1)
        self.crop_n_points_downscale_factor_spin.setToolTip("Downscales the number of points sampled per crop layer (if Crop N Layers > 0)")
        sam_params_layout.addRow("Crop Points Downscale Factor:", self.crop_n_points_downscale_factor_spin)
        self.min_mask_region_area_spin = QSpinBox()
        self.min_mask_region_area_spin.setRange(0, 100000) # 0 disables this filter
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
        # Button to reset SAM parameters to defaults
        reset_sam_params_btn = QPushButton("Reset SAM Parameters to Default")
        reset_sam_params_btn.setStyleSheet(self.get_button_style("grey"))
        reset_sam_params_btn.clicked.connect(self.reset_sam_params)
        sam_params_layout.addRow("", reset_sam_params_btn) # Add button without a label
        process_scroll_layout.addWidget(sam_params_group) # Add SAM params group to scroll layout

        # Operation buttons layout (Process, Stop, Open Output)
        operation_layout_proc = QHBoxLayout()
        self.batch_process_btn = QPushButton("Process All Images")
        self.batch_process_btn.setStyleSheet(self.get_button_style("blue"))
        self.batch_process_btn.setEnabled(False) # Disabled initially
        self.batch_process_btn.clicked.connect(self.batch_process_images)
        self.stop_btn = QPushButton("Stop Processing")
        self.stop_btn.setStyleSheet(self.get_button_style("red"))
        self.stop_btn.setEnabled(False) # Disabled initially
        self.stop_btn.clicked.connect(self.stop_processing)
        self.open_proc_output_btn = QPushButton("Open Output Folder")
        self.open_proc_output_btn.setStyleSheet(self.get_button_style("grey"))
        self.open_proc_output_btn.setEnabled(False) # Disabled initially
        self.open_proc_output_btn.clicked.connect(self.open_processing_output_folder)
        operation_layout_proc.addWidget(self.batch_process_btn)
        operation_layout_proc.addWidget(self.stop_btn)
        operation_layout_proc.addWidget(self.open_proc_output_btn)
        process_scroll_layout.addLayout(operation_layout_proc)

        process_scroll_layout.addStretch() # Push elements upwards within scroll area
        process_scroll_area.setWidget(process_scroll_widget) # Set the content widget for the scroll area
        process_layout.addWidget(process_scroll_area) # Add scroll area to the process tab layout

        # Add tabs to the tab widget
        self.tabs.addTab(contrast_tab, "Contrast Enhancement")
        self.tabs.addTab(aggregate_tab, "Aggregate Analysis")
        self.tabs.addTab(heatmap_tab, "Heatmap Generation")
        self.tabs.addTab(process_tab, "SAM Processing") # Main processing tab

        left_layout.addWidget(self.tabs) # Add tab widget to the left panel
        main_layout.addWidget(left_widget, 1) # Add left panel to main layout (stretch factor 1)

        # --- Right Panel (Logs) ---
        right_widget = QWidget()
        right_layout = QVBoxLayout(right_widget)
        right_layout.setContentsMargins(0, 0, 0, 0) # No internal padding
        right_layout.setSpacing(15) # Space between log groups

        # Log area for Contrast Enhancement
        contrast_log_group = QGroupBox("Contrast Enhancement Log")
        contrast_log_group.setStyleSheet("QGroupBox { font-weight: bold; }")
        contrast_log_layout = QVBoxLayout(contrast_log_group)
        contrast_log_layout.setContentsMargins(10, 15, 10, 10)
        self.contrast_log_text = QTextEdit()
        self.contrast_log_text.setStyleSheet(self.get_log_style()) # Apply log style
        self.contrast_log_text.setReadOnly(True) # Make log read-only
        contrast_log_layout.addWidget(self.contrast_log_text)
        right_layout.addWidget(contrast_log_group)

        # Log area for Aggregate Analysis
        aggregate_log_group = QGroupBox("Aggregate Analysis Log")
        aggregate_log_group.setStyleSheet("QGroupBox { font-weight: bold; }")
        aggregate_log_layout = QVBoxLayout(aggregate_log_group)
        aggregate_log_layout.setContentsMargins(10, 15, 10, 10)
        self.aggregate_log_text = QTextEdit()
        self.aggregate_log_text.setStyleSheet(self.get_log_style())
        self.aggregate_log_text.setReadOnly(True)
        aggregate_log_layout.addWidget(self.aggregate_log_text)
        right_layout.addWidget(aggregate_log_group)

        # Log area for Heatmap Generation
        heatmap_log_group = QGroupBox("Heatmap Generation Log")
        heatmap_log_group.setStyleSheet("QGroupBox { font-weight: bold; }")
        heatmap_log_layout = QVBoxLayout(heatmap_log_group)
        heatmap_log_layout.setContentsMargins(10, 15, 10, 10)
        self.heatmap_log_text = QTextEdit()
        self.heatmap_log_text.setStyleSheet(self.get_log_style())
        self.heatmap_log_text.setReadOnly(True)
        heatmap_log_layout.addWidget(self.heatmap_log_text)
        right_layout.addWidget(heatmap_log_group)

        # Log area for SAM Processing
        log_group_proc = QGroupBox("SAM Processing Log")
        log_group_proc.setStyleSheet("QGroupBox { font-weight: bold; }")
        log_layout_proc = QVBoxLayout(log_group_proc)
        log_layout_proc.setContentsMargins(10, 15, 10, 10)
        self.log_text = QTextEdit() # Main log area
        self.log_text.setStyleSheet(self.get_log_style())
        self.log_text.setReadOnly(True)
        log_layout_proc.addWidget(self.log_text)
        right_layout.addWidget(log_group_proc)

        main_layout.addWidget(right_widget, 1) # Add right panel to main layout (stretch factor 1)

        self.update_button_states() # Set initial enabled/disabled state of buttons

    def get_button_style(self, color="blue"):
        """ Generates CSS style strings for buttons with different colors. """
        base_style = """
            QPushButton {
                padding: 8px 12px; /* Button padding */
                color: white; /* Text color */
                border: none; /* No border */
                border-radius: 4px; /* Rounded corners */
                font-weight: bold; /* Bold text */
            }
            /* Style for disabled buttons */
            QPushButton:disabled {
                background-color: #cccccc; /* Grey background */
                color: #666666; /* Dark grey text */
            }
        """
        # Define background colors and hover colors
        colors = {
            "green": ("#4CAF50", "#45a049"), # Normal, Hover
            "blue": ("#2196F3", "#0b7dda"),
            "red": ("#f44336", "#d32f2f"),
            "grey": ("#607d8b", "#455a64"),
        }
        bg_color, hover_color = colors.get(color, colors["blue"]) # Default to blue if color not found
        # Combine base style with specific colors
        return base_style + f"""
            QPushButton {{ background-color: {bg_color}; }}
            QPushButton:hover:!disabled {{ background-color: {hover_color}; }} /* Hover effect only for enabled buttons */
        """

    def get_log_style(self):
        """ Generates CSS style string for log text areas. """
        return """
            QTextEdit {
                border: 1px solid #cccccc; /* Light grey border */
                border-radius: 4px; /* Rounded corners */
                padding: 5px; /* Internal padding */
                font-family: Consolas, Courier New, monospace; /* Monospaced font */
                background-color: #f8f8f8; /* Very light grey background */
            }
        """

    # --- MODIFICATION: Updated output folder name generation ---
    def get_processing_output_folder_path(self):
        """ Determines the output folder path for SAM processing based on input folder and parameters. """
        if not self.input_folder: return None # Cannot determine without input folder

        input_folder_name = os.path.basename(os.path.normpath(self.input_folder))
        parent_dir = os.path.dirname(os.path.normpath(self.input_folder))

        # Build parameter string part
        param_parts = []
        if self.area_check.isChecked():
            param_parts.append(f"Area={self.a_threshold_spin.value()}")
        if self.intensity_check.isChecked():
            param_parts.append(f"Int={self.i_threshold_spin.value():.0f}")
            param_parts.append(f"Ratio={self.r_threshold_spin.value():.1f}")
        param_parts.append(f"Agg={self.min_aggregate_spin.value()}-{self.max_aggregate_spin.value()}")

        param_str = " ".join(param_parts)
        # Consistent folder naming structure
        output_folder_name = f"{input_folder_name}_SAM_Output [{param_str}]"

        return os.path.join(parent_dir, output_folder_name)

    # --- MODIFICATION: Updated Heatmap output folder name generation ---
    def get_heatmap_output_folder_path(self):
        """ Determines the output folder path for Heatmap generation. """
        if not self.heatmap_input_folder: return None
        input_folder_name = os.path.basename(os.path.normpath(self.heatmap_input_folder))
        parent_dir = os.path.dirname(os.path.normpath(self.heatmap_input_folder))
        # Consistent folder naming structure
        min_thresh = self.heatmap_min_aggregate_spin.value()
        max_thresh = self.heatmap_max_aggregate_spin.value()
        output_folder_name = f"{input_folder_name}_Heatmap_Output [Agg={min_thresh}-{max_thresh}]"
        return os.path.join(parent_dir, output_folder_name)

    # --- MODIFICATION: Updated Contrast output folder name generation ---
    def get_contrast_output_folder_path(self):
        """ Determines the output folder path for Contrast enhancement. """
        if not self.contrast_input_folder: return None
        input_folder_name = os.path.basename(os.path.normpath(self.contrast_input_folder))
        parent_dir = os.path.dirname(os.path.normpath(self.contrast_input_folder))
        # Consistent folder naming structure
        contrast_factor = self.contrast_spin.value()
        output_folder_name = f"{input_folder_name}_Contrast_Output [Factor={contrast_factor:.1f}]"
        return os.path.join(parent_dir, output_folder_name)
    # --- END MODIFICATION ---

    def reset_sam_params(self):
        """ Resets all SAM parameter widgets to their default values. """
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
        self.output_mode_combo.setCurrentText("binary_mask") # Default output mode
        self.log_text.append("SAM parameters reset to default.") # Log the action

    def get_sam_params(self):
        """ Collects current values from SAM parameter widgets into a dictionary. """
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

    # --- Folder/File Selection Methods ---
    def select_input_folder(self):
        """ Opens dialog to select input folder for SAM Processing. """
        folder = QFileDialog.getExistingDirectory(self, "Select Input Folder (Processing)")
        if folder:
            self.input_folder = folder
            # Display truncated path in the label for better UI
            self.input_folder_label.setText(f"Input: ...{folder[-40:]}")
            self.update_button_states() # Update button enabled states

    def select_model_file(self):
        """ Opens dialog to select SAM model file (.pth). """
        file, _ = QFileDialog.getOpenFileName(self, "Select SAM Model File", "", "Model files (*.pth)")
        if file:
            self.model_path = file
            self.model_path_label.setText(f"Model: {os.path.basename(file)}") # Display filename
            self.update_button_states() # Update button enabled states

    def select_aggregate_input_folder(self):
        """ Opens dialog to select input folder for Aggregate Analysis. """
        folder = QFileDialog.getExistingDirectory(self, "Select Input Folder (Aggregate Analysis)")
        if folder:
            self.bright_input_folder = folder
            self.agg_input_folder_label.setText(f"Input: ...{folder[-40:]}")
            self.analyze_agg_btn.setEnabled(True) # Enable analyze button

    def select_heatmap_input_folder(self):
        """ Opens dialog to select input folder for Heatmap Generation. """
        folder = QFileDialog.getExistingDirectory(self, "Select Input Folder (Heatmap)")
        if folder:
            self.heatmap_input_folder = folder
            self.heatmap_input_folder_label.setText(f"Input: ...{folder[-40:]}")
            self.generate_heatmap_btn.setEnabled(True) # Enable generate button
            self.open_heatmap_output_btn.setEnabled(True) # Enable open output button

    def select_contrast_input_folder(self):
        """ Opens dialog to select input folder for Contrast Enhancement. """
        folder = QFileDialog.getExistingDirectory(self, "Select Input Folder (Contrast)")
        if folder:
            self.contrast_input_folder = folder
            self.contrast_input_folder_label.setText(f"Input: ...{folder[-40:]}")
            self.enhance_contrast_btn.setEnabled(True) # Enable enhance button
            self.open_contrast_output_btn.setEnabled(True) # Enable open output button

    def on_calculation_method_changed(self, state):
        """
        Slot connected to state changes of Area/Intensity checkboxes.
        Ensures at least one checkbox remains checked for Phase 1.
        Updates button states.
        """
        # Prevent unchecking the last remaining checkbox for Phase 1 filters
        if not self.area_check.isChecked() and not self.intensity_check.isChecked():
             sender = self.sender() # Get the checkbox that triggered the signal
             # If user tried to uncheck Area while Intensity was already unchecked, re-check Area
             if sender == self.area_check and state == Qt.CheckState.Unchecked.value:
                 self.area_check.setChecked(True)
                 QMessageBox.warning(self, "Selection Required", "At least one Phase 1 filter method (Area or Intensity/Ratio) must be selected.")
             # If user tried to uncheck Intensity while Area was already unchecked, re-check Intensity
             elif sender == self.intensity_check and state == Qt.CheckState.Unchecked.value:
                 self.intensity_check.setChecked(True)
                 QMessageBox.warning(self, "Selection Required", "At least one Phase 1 filter method (Area or Intensity/Ratio) must be selected.")

        self.update_button_states() # Update button enabled states after change

    def update_button_states(self):
        """ Updates the enabled/disabled state of all action buttons based on inputs. """
        # SAM Processing buttons
        proc_input_ready = bool(self.input_folder)
        model_ready = bool(self.model_path)
        calculation_selected = self.area_check.isChecked() or self.intensity_check.isChecked() # Phase 1 filter selected
        proc_all_ready = proc_input_ready and model_ready and calculation_selected
        self.batch_process_btn.setEnabled(proc_all_ready)
        self.open_proc_output_btn.setEnabled(proc_input_ready) # Can open output if input is selected

        # Heatmap buttons
        self.generate_heatmap_btn.setEnabled(bool(self.heatmap_input_folder))
        self.open_heatmap_output_btn.setEnabled(bool(self.heatmap_input_folder))

        # Contrast buttons
        self.enhance_contrast_btn.setEnabled(bool(self.contrast_input_folder))
        self.open_contrast_output_btn.setEnabled(bool(self.contrast_input_folder))

        # Aggregate Analysis button
        self.analyze_agg_btn.setEnabled(bool(self.bright_input_folder))


    # --- Methods to Open Output Folders ---
    def open_folder(self, folder_path, folder_type_name):
        """ Helper function to open a folder in the system's file explorer. """
        if folder_path:
            if not os.path.exists(folder_path):
                 # Attempt to create the folder if it doesn't exist
                 try:
                     os.makedirs(folder_path)
                 except OSError as e:
                     QMessageBox.warning(self, "Error", f"Could not create or access {folder_type_name} output folder:\n{folder_path}\nError: {e}")
                     return
            # Open the folder using platform-specific commands
            try:
                if sys.platform == "win32":
                    os.startfile(folder_path) # Windows
                elif sys.platform == "darwin":
                    os.system(f"open \"{folder_path}\"") # macOS
                else:
                    os.system(f"xdg-open \"{folder_path}\"") # Linux
            except Exception as e:
                QMessageBox.warning(self, "Error", f"Could not open {folder_type_name} output folder:\n{folder_path}\nError: {e}")
        else:
            QMessageBox.information(self, "Info", f"{folder_type_name} output folder path could not be determined (process might need to run first or input folder not selected).")

    def open_processing_output_folder(self):
        """ Opens the output folder for SAM processing. """
        if self.input_folder:
            output_folder = self.get_processing_output_folder_path()
            self.open_folder(output_folder, "Processing")
        else:
             QMessageBox.warning(self, "Input Required", "Please select an input folder first.")

    def open_heatmap_output_folder(self):
        """ Opens the output folder for Heatmap generation. """
        if self.heatmap_input_folder:
            output_folder = self.get_heatmap_output_folder_path()
            self.open_folder(output_folder, "Heatmap")
        else:
             QMessageBox.warning(self, "Input Required", "Please select an input folder for heatmaps first.")

    def open_contrast_output_folder(self):
        """ Opens the output folder for Contrast enhancement. """
        if self.contrast_input_folder:
            output_folder = self.get_contrast_output_folder_path()
            self.open_folder(output_folder, "Contrast")
        else:
             QMessageBox.warning(self, "Input Required", "Please select an input folder for contrast enhancement first.")


    # --- Main Processing Methods ---
    def batch_process_images(self):
        """ Starts the SAM processing thread for all images in the input folder. """
        # --- MODIFICATION: Re-added Min/Max Aggregate Threshold validation ---
        min_agg = self.min_aggregate_spin.value()
        max_agg = self.max_aggregate_spin.value()
        if min_agg > max_agg or (min_agg == max_agg and min_agg != 100): # Allow 100==100
            QMessageBox.warning(self, "Invalid Aggregate Thresholds", "Min Aggregate Threshold must be less than or equal to Max Aggregate Threshold (equality allowed only if both are 100).")
            return
        # --- END MODIFICATION ---

        # Validate required inputs
        if not self.area_check.isChecked() and not self.intensity_check.isChecked():
             QMessageBox.warning(self, "Selection Required", "Please select at least one Phase 1 calculation method (Area or Intensity/Ratio).")
             return
        if not self.input_folder or not self.model_path:
             QMessageBox.warning(self, "Input Required", "Please select both an input folder and a SAM model file.")
             return

        # Log job start
        self.log_text.append("-" * 20 + " New Processing Job " + "-" * 20)
        self.log_text.append("Starting image processing...")
        self.batch_process_btn.setEnabled(False) # Disable process button
        self.stop_btn.setEnabled(True) # Enable stop button

        # Get parameters from UI
        use_area = self.area_check.isChecked()
        use_intensity = self.intensity_check.isChecked()
        sam_params = self.get_sam_params()
        output_folder = self.get_processing_output_folder_path() # Get the correctly formatted output folder path
        a_threshold = self.a_threshold_spin.value()
        i_threshold = self.i_threshold_spin.value()
        r_threshold = self.r_threshold_spin.value()
        # min_agg and max_agg already retrieved for validation

        # Ensure output folder path is valid
        if not output_folder:
             self.log_text.append("Error: Could not determine output folder path.")
             self.update_button_states() # Re-enable process button if possible
             self.stop_btn.setEnabled(False)
             return

        # Log parameters being used
        self.log_text.append(f"Output folder: {output_folder}") # Log the new folder path
        self.log_text.append(f"Using SAM parameters: {sam_params}")
        self.log_text.append(f"Phase 1 Filters: Area={'Yes' if use_area else 'No'}, Intensity/Ratio={'Yes' if use_intensity else 'No'}")
        self.log_text.append(f"Phase 1 Thresholds: Min Area (A)={a_threshold}, Min Intensity (I)={i_threshold:.0f}, Min Ratio (R)={r_threshold:.1f}")
        self.log_text.append(f"Phase 2 Filter: Aggregate Range=[{min_agg}-{max_agg}]")


        # Stop previous thread if it's running
        if self.process_thread and self.process_thread.isRunning():
            self.log_text.append("Stopping previous processing job...")
            self.process_thread.stop()
            self.process_thread.wait() # Wait for thread to finish stopping
            self.log_text.append("Previous job stopped.")

        # Create and start the new processing thread
        self.process_thread = ImageProcessorThread(
            self.input_folder,
            output_folder, # Pass the determined output folder path
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
        # Connect signals from thread to slots in the main window
        self.process_thread.processing_finished.connect(self.processing_finished)
        self.process_thread.image_processed.connect(self.image_processed)
        self.process_thread.log_updated.connect(self.update_log)
        self.process_thread.start() # Start the thread execution

    def enhance_contrast(self):
        """ Starts the contrast enhancement thread. """
        if not self.contrast_input_folder:
             QMessageBox.warning(self, "Input Required", "Please select an input folder for contrast enhancement.")
             return

        # Log job start
        self.contrast_log_text.append("-" * 20 + " New Contrast Job " + "-" * 20)
        self.contrast_log_text.append("Starting contrast enhancement...")
        self.enhance_contrast_btn.setEnabled(False) # Disable enhance button
        self.stop_contrast_btn.setEnabled(True) # Enable stop button

        # Get parameters
        output_folder = self.get_contrast_output_folder_path() # Get the correctly formatted output folder path
        if not output_folder:
            self.contrast_log_text.append("Error: Could not determine output folder.")
            self.enhance_contrast_btn.setEnabled(True) # Re-enable enhance button
            self.stop_contrast_btn.setEnabled(False)
            return

        self.contrast_log_text.append(f"Output folder: {output_folder}") # Log the new folder path
        contrast_factor = self.contrast_spin.value()
        self.contrast_log_text.append(f"Contrast Factor: {contrast_factor:.1f}")

        # Stop previous thread if running
        if self.contrast_thread and self.contrast_thread.isRunning():
            self.contrast_log_text.append("Stopping previous contrast job...")
            self.contrast_thread.stop()
            self.contrast_thread.wait()
            self.contrast_log_text.append("Previous job stopped.")

        # Create and start the new contrast thread
        self.contrast_thread = ContrastEnhancementThread(
            self.contrast_input_folder,
            output_folder, # Pass the determined output folder path
            contrast_factor
        )
        self.contrast_thread.processing_finished.connect(self.contrast_enhancement_finished)
        self.contrast_thread.log_updated.connect(self.update_contrast_log)
        self.contrast_thread.start()

    def generate_heatmaps(self):
        """ Starts the heatmap generation thread. """
        # Validate heatmap thresholds
        min_thresh = self.heatmap_min_aggregate_spin.value()
        max_thresh = self.heatmap_max_aggregate_spin.value()
        if min_thresh > max_thresh or (min_thresh == max_thresh and min_thresh != 100): # Allow 100==100
             QMessageBox.warning(self, "Invalid Thresholds", "Min Aggregate Threshold must be less than or equal to Max Aggregate Threshold (equality allowed only if both are 100).")
             return

        if not self.heatmap_input_folder:
             QMessageBox.warning(self, "Input Required", "Please select an input folder for heatmap generation.")
             return

        # Log job start
        self.heatmap_log_text.append("-" * 20 + " New Heatmap Job " + "-" * 20)
        self.heatmap_log_text.append("Starting heatmap generation...")
        self.generate_heatmap_btn.setEnabled(False) # Disable generate button
        self.stop_heatmap_btn.setEnabled(True) # Enable stop button

        # Get parameters
        output_folder = self.get_heatmap_output_folder_path() # Get the correctly formatted output folder path
        if not output_folder:
            self.heatmap_log_text.append("Error: Could not determine output folder.")
            self.generate_heatmap_btn.setEnabled(True) # Re-enable generate button
            self.stop_heatmap_btn.setEnabled(False)
            return

        self.heatmap_log_text.append(f"Output folder: {output_folder}") # Log the new folder path
        self.heatmap_log_text.append(f"Aggregate Range: [{min_thresh}-{max_thresh}]")


        # Stop previous thread if running
        if self.heatmap_thread and self.heatmap_thread.isRunning():
            self.heatmap_log_text.append("Stopping previous heatmap job...")
            self.heatmap_thread.stop()
            self.heatmap_thread.wait()
            self.heatmap_log_text.append("Previous job stopped.")

        # Create and start the new heatmap thread
        self.heatmap_thread = HeatmapGeneratorThread(
            self.heatmap_input_folder,
            output_folder, # Pass the determined output folder path
            min_thresh,
            max_thresh
        )
        self.heatmap_thread.processing_finished.connect(self.heatmap_generation_finished)
        self.heatmap_thread.log_updated.connect(self.update_heatmap_log)
        self.heatmap_thread.start()

    def analyze_folder_aggregate(self):
        """ Calculates and logs aggregate values for all images in the selected folder. """
        self.aggregate_log_text.clear() # Clear previous log
        if not self.bright_input_folder:
            self.update_aggregate_log("Please select an input folder first.")
            return

        folder_path = self.bright_input_folder
        # Validate folder existence and permissions
        if not os.path.exists(folder_path):
            self.update_aggregate_log(f"Error: Directory '{folder_path}' does not exist.")
            return
        if not os.access(folder_path, os.R_OK):
            self.update_aggregate_log(f"Error: No read permission for directory '{folder_path}'.")
            return

        # Find image files
        files = [f for f in os.listdir(folder_path)
                 if f.lower().endswith((".png", ".jpg", ".jpeg", ".tif", ".tiff"))]

        if not files:
            self.update_aggregate_log("No image files found in selected folder.")
            return

        # Process each file
        image_count = 0
        error_count = 0
        self.update_aggregate_log(f"Analyzing aggregates in: {folder_path}")
        self.update_aggregate_log("-" * 50)
        files.sort() # Process files alphabetically

        for filename in files:
            file_path = os.path.join(folder_path, filename)
            if os.path.isfile(file_path):
                try:
                    # Calculate aggregate value using the static method
                    aggregate_val = ImageAggregateAnalyzer.calculate_aggregate(file_path)
                    # Log result or error message
                    if isinstance(aggregate_val, float):
                        self.update_aggregate_log(f"{filename}: Aggregate = {aggregate_val:.2f}")
                        image_count += 1
                    else: # It's an error string
                        self.update_aggregate_log(f"{filename}: {aggregate_val}")
                        error_count += 1
                except Exception as e:
                    # Catch unexpected errors during calculation
                    self.update_aggregate_log(f"{filename}: Error - {str(e)}")
                    error_count += 1
                QApplication.processEvents() # Keep UI responsive during analysis

        # Log summary
        self.update_aggregate_log("-" * 50)
        self.update_aggregate_log(f"Processed {image_count} image(s) successfully.")
        if error_count > 0:
             self.update_aggregate_log(f"Encountered errors with {error_count} file(s).")


    # --- Methods to Stop Running Threads ---
    def stop_processing(self):
        """ Stops the SAM processing thread if it is running. """
        if self.process_thread and self.process_thread.isRunning():
            self.log_text.append("Attempting to stop processing...")
            self.process_thread.stop() # Signal the thread to stop
            self.stop_btn.setEnabled(False) # Disable stop button immediately
        else:
             self.log_text.append("No processing job is currently running.")

    def stop_heatmap_generation(self):
        """ Stops the heatmap generation thread if it is running. """
        if self.heatmap_thread and self.heatmap_thread.isRunning():
            self.heatmap_log_text.append("Attempting to stop heatmap generation...")
            self.heatmap_thread.stop()
            self.stop_heatmap_btn.setEnabled(False)
        else:
            self.heatmap_log_text.append("No heatmap generation job is running.")

    def stop_contrast_processing(self):
        """ Stops the contrast enhancement thread if it is running. """
        if self.contrast_thread and self.contrast_thread.isRunning():
            self.contrast_log_text.append("Attempting to stop contrast enhancement...")
            self.contrast_thread.stop()
            self.stop_contrast_btn.setEnabled(False)
        else:
            self.contrast_log_text.append("No contrast enhancement job is running.")

    # --- Log Update Slots ---
    def update_log(self, message):
        """ Appends a message to the main SAM processing log. """
        self.log_text.append(message)
        # Auto-scroll to the bottom
        scrollbar = self.log_text.verticalScrollBar()
        scrollbar.setValue(scrollbar.maximum())

    def update_aggregate_log(self, message):
        """ Appends a message to the Aggregate Analysis log. """
        self.aggregate_log_text.append(message)
        scrollbar = self.aggregate_log_text.verticalScrollBar()
        scrollbar.setValue(scrollbar.maximum())

    def update_heatmap_log(self, message):
        """ Appends a message to the Heatmap Generation log. """
        self.heatmap_log_text.append(message)
        scrollbar = self.heatmap_log_text.verticalScrollBar()
        scrollbar.setValue(scrollbar.maximum())

    def update_contrast_log(self, message):
        """ Appends a message to the Contrast Enhancement log. """
        self.contrast_log_text.append(message)
        scrollbar = self.contrast_log_text.verticalScrollBar()
        scrollbar.setValue(scrollbar.maximum())

    # --- Thread Finished Slots ---
    def processing_finished(self):
        """ Slot called when the SAM processing thread finishes. """
        self.log_text.append("Processing complete.")
        self.update_button_states() # Re-enable process button if inputs are still valid
        self.stop_btn.setEnabled(False) # Disable stop button
        self.process_thread = None # Clear the thread object reference

    def heatmap_generation_finished(self):
        """ Slot called when the heatmap generation thread finishes. """
        self.heatmap_log_text.append("Heatmap generation complete.")
        self.generate_heatmap_btn.setEnabled(bool(self.heatmap_input_folder)) # Re-enable generate button
        self.stop_heatmap_btn.setEnabled(False)
        self.heatmap_thread = None

    def contrast_enhancement_finished(self):
        """ Slot called when the contrast enhancement thread finishes. """
        self.contrast_log_text.append("Contrast enhancement complete.")
        self.enhance_contrast_btn.setEnabled(bool(self.contrast_input_folder)) # Re-enable enhance button
        self.stop_contrast_btn.setEnabled(False)
        self.contrast_thread = None

    def image_processed(self, output_path, processed_image_array):
        """ Slot called after each image is processed by the SAM thread. """
        # Store the path and image array (optional, could be used for display)
        self.last_output_path = output_path
        self.last_output_image_array = processed_image_array
        # Currently just logs the save path via log_updated signal

    # --- Window Close Event ---
    def closeEvent(self, event):
        """ Overrides the default close event to confirm exit and stop threads. """
        reply = QMessageBox.question(self, 'Confirm Exit',
                                       "Are you sure you want to exit? Any running processes will be stopped.",
                                       QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                                       QMessageBox.StandardButton.No) # Default to No

        if reply == QMessageBox.StandardButton.Yes:
            # Stop all running threads before closing
            if self.process_thread and self.process_thread.isRunning():
                self.process_thread.stop()
                self.process_thread.wait() # Wait for thread to finish
            if self.heatmap_thread and self.heatmap_thread.isRunning():
                self.heatmap_thread.stop()
                self.heatmap_thread.wait()
            if self.contrast_thread and self.contrast_thread.isRunning():
                self.contrast_thread.stop()
                self.contrast_thread.wait()
            event.accept() # Accept the close event
        else:
            event.ignore() # Ignore the close event, keep window open


# --- Main execution block ---
if __name__ == "__main__":
    app = QApplication(sys.argv)
    app.setStyle('Fusion') # Apply Fusion style for cross-platform consistency
    window = UsefulImageAnalysisToolIntegration() # Create main window instance
    window.show() # Show the window
    sys.exit(app.exec()) # Start the Qt application event loop
