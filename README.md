# Useful Image Analysis Tool Integration (based on SAM and PyQt)

This is a Graphical User Interface (GUI) tool built with PyQt6, designed to simplify and accelerate common image analysis and processing tasks in research or experimental settings. It's particularly useful for scenarios involving batch processing of large numbers of images (e.g., microscope photos, scanned tissue sections). This tool can help you automatically extract quantitative information, compare image features, or improve image visual quality.

## Key Features

* **Contrast Enhancement**:
    * **Description**: Adjusts the difference between the brightest and darkest areas of an image to enhance clarity and detail visibility.
    * **Batch Processing**: Applies contrast enhancement to all images within an input folder.
    * **Adjustable Factor**: Users can specify the enhancement factor (typically >1 for enhancement, <1 for reduction).
    * **Example Use Cases**:
        * Enhancing the contours of lesions in medical images (X-rays, MRI).
        * Making the edges and internal structures of cells or tissues in microscopy images clearer.
        * Improving the clarity of scanned electrophoresis gels, blots, or old lab records.

* **Aggregate Analysis**:
    * **Brightness to Aggregate Value Conversion**:
        * This tool defines an "Aggregate" metric (range 0-100) to quantify the "darkness" or "non-brightness" level of pixels.
        * It's calculated based on the pixel's **grayscale brightness** (range 0-255, where 0 is black, 255 is white).
        * **Calculation Steps**:
            1.  Normalize the brightness value to the 0-1 range (`grayscale_brightness / 255.0`).
            2.  Multiply the normalized value by 100.
            3.  Subtract the result from 100.
        * **Formula**:
            ```
            Aggregate Value = 100 - (grayscale_brightness / 255.0) * 100
            ```
        * **Core Relationship**: This formula means that **the lower the pixel brightness (the darker the color), the higher its Aggregate Value**. (e.g., pure black pixel has brightness 0, Aggregate Value 100; pure white pixel has brightness 255, Aggregate Value 0).
        * **Potential Correlation**: In certain applications (like biological staining), a higher Aggregate Value might correlate with higher "substance density" or "signal intensity" (specifically for dark signals).
    * **Quick Assessment**: Calculates the **average Aggregate Value** for a single image or all images in a folder, providing a quick metric for the overall "darkness" or "aggregation level" of the images.
    * **Example Use Cases**:
        * Quickly comparing the overall staining depth or signal level across images from different experimental groups.
        * Performing an initial screening of large image sets to identify images that are generally too dark or too bright.

* **Heatmap Generation**:
    * **Description**: Assigns different colors to pixels based on their "Aggregate Value" score, generating a visual color map of the aggregate value's spatial distribution.
    * **Aggregate Value Calculation**: Calculates the "Aggregate Value" (0-100 range, as described above) for each pixel in the grayscale image.
    * **Threshold-Based Heatmap Visualization**:
        * Users set a **Min Aggregate Threshold** and a **Max Aggregate Threshold** to define an aggregate score range of interest: [Min Value, Max Value].
        * **Thresholding and Color Mapping Logic (Reflecting Code Implementation)**:
            * Pixels with an Aggregate Value **below** the Min Threshold are displayed in **black** (default background).
            * Pixels with an Aggregate Value **above** the Max Threshold are displayed in **bright red**.
            * Pixels with an Aggregate Value **within** the [Min Value, Max Value] range are colored based on their relative position, transitioning smoothly from **green** (near Min Value, low aggregate/brighter) through **yellow/orange** to **red** (near Max Value, high aggregate/darker). (Code implementation based on HSV color space Hue mapping).
        * **Purpose**: Adjusting the thresholds focuses the visualization range, effectively filtering out irrelevant background or saturated areas, and highlighting the distribution and intensity variations within the target range.
    * **Example Use Cases**:
        * Visually displaying staining intensity or specific molecular expression levels (if correlated with aggregate value) across different regions in tissue section images.
        * Visualizing areas with specific density or component distribution ranges in material science scan images.

* **SAM Processing**:
    * **Description**: Combines the Segment Anything Model (AI automatic segmentation) with Aggregate Analysis. It automatically identifies distinct regions in an image, then uses a two-stage filtering process to select targets of interest, and finally performs quantitative analysis and visual marking.
    * **Model Loading**: Supports loading user-selected pre-trained Segment Anything Model (`.pth`) files.
    * **Automatic Mask Generation**: Utilizes SAM for fully automatic instance segmentation of the input image, generating multiple independent region masks.
    * **Two-Stage Threshold Filtering and Property Calculation (Reflecting Code Logic)**:
        * **Stage 1: Region Property Filtering (Controlled by A/I/R Thresholds)**:
            * The program first calculates the **initial** properties for **each** region generated by SAM (based on **all** pixels within that region): initial area (`initial_area`), initial intensity (`initial_intensity`), and initial ratio (`initial_ratio`).
            * Based on whether the user checked the "Area" and/or "Intensity" boxes, the program checks if these initial values meet the user-set minimum thresholds (`A Threshold`, `I Threshold`, `R Threshold`).
            * If a region's initial properties **fail** to meet all enabled threshold requirements, the region is **filtered out** and does not proceed to the next stage.
        * **Stage 2: Pixel Aggregate Filtering (Controlled by Min/Max Aggregate Threshold)**:
            * For regions that **passed Stage 1**, the program then examines **each pixel** within that region.
            * Only pixels whose Aggregate Values **fall within** the user-defined [Min Value, Max Value] range are considered "valid pixels" for the final analysis of that region.
            * If a region that passed Stage 1 **contains no** "valid pixels" after this check (i.e., all its pixels fall outside the aggregate range), this region **will ultimately not be marked or used for calculations**.
        * **Final Metric Calculation and Visualization**:
            * For regions that **passed both filtering stages**, the program **recalculates** the final metrics **using only the "valid pixels"** identified in Stage 2:
                * **Area (A)**: Total count of valid pixels (`final_area`).
                * **Intensity (I)**: Sum of Aggregate Values of valid pixels (`final_intensity`).
                * **Ratio (R)**: `final_intensity / final_area` (`final_ratio`).
            * The program then draws the **original outline** of the region (from the SAM segmentation) on the output image and annotates it with the **final calculated** A/I/R values using colored text.

    * **Detailed Parameter Tuning (SAM Auto Mask Generator Parameters)**: SAM offers advanced options to fine-tune its behavior. Default settings often work well, but understanding these can help address specific issues:
        * **`points_per_side`** (Default 32): Controls the density of the initial point grid used to find objects. Higher values mean a denser grid, better for finding small objects, but slower and requires more memory/VRAM. Lower values are faster but might miss small details.
        * **`points_per_batch`** (Default 64): Determines how many points the GPU processes at once. Mainly affects VRAM usage. **Lower this value (e.g., 32, 16) if you encounter "Out of Memory" errors.** Slightly increasing it might speed things up on high-end GPUs with ample VRAM.
        * **`pred_iou_thresh`** (Predicted IoU Threshold, Default 0.88): A score (0-1) indicating SAM's confidence in the quality of a found region. Higher values demand higher confidence, resulting in fewer false positives but potentially missing uncertain detections. Lower values are more lenient, potentially finding more but also including lower-quality results.
        * **`stability_score_thresh`** (Stability Threshold, Default 0.95): Measures how stable a mask's shape is under slight perturbations (0-1). Higher values require very stable, clear boundaries, potentially discarding objects with fuzzy edges. Lower values tolerate fuzzier boundaries but might yield less precise shapes.
        * **`stability_score_offset`** (Stability Offset, Default 1.0): An internal parameter for stability score calculation. **Usually does not need changing.**
        * **`box_nms_thresh`** (Box NMS Threshold, Default 0.7): Prevents detecting the same object multiple times. If two detected bounding boxes overlap more than this threshold (0-1), one might be suppressed. Lower values are stricter (might merge distinct nearby objects); higher values are more permissive (might allow duplicates). **Default is usually fine.**
        * **`crop_n_layers`** (Crop Layers, Default 0): Enables tiling for very large images. **0 means process the whole image at once.** Setting to 1 or more cuts the image into overlapping tiles, processes each, and stitches the results. Allows processing images larger than available memory but is **significantly slower** and may introduce artifacts at tile boundaries. **Consider setting to 1 only for memory errors with huge images.**
        * **`crop_nms_thresh`** (Crop NMS Threshold, Default 0.7): NMS threshold used when merging results from different tiles in crop mode. **Usually does not need changing.**
        * **`crop_overlap_ratio`** (Crop Overlap Ratio, Default ~0.341): Controls the amount of overlap between adjacent tiles in crop mode. **Usually does not need changing.**
        * **`crop_n_points_downscale_factor`** (Crop Points Downscale Factor, Default 1): Reduces the number of points sampled within each tile in crop mode. 1 means no reduction. Higher values speed up tile processing but may reduce accuracy. **Usually keep at 1.**
        * **`min_mask_region_area`** (Min Mask Region Area, Default 0): A **post-processing** step. Removes tiny, isolated segmented regions smaller than this pixel area value. **Useful for automatically cleaning up small noise/speckles.** Set to 0 to keep all regions.
        * **`output_mode`** (Output Mode, Default `binary_mask`): Controls the internal format of the output masks. **Strongly recommended to keep the default `binary_mask`**, which is standard and compatible. Other RLE formats require extra libraries and may cause issues.
    * **Example Use Cases**:
        * Automatically identifying and quantifying the size (Area) and specific protein expression (Intensity) of all cells in microscopy images, while only counting/marking cells meeting a certain expression level (corresponding to a specific aggregate value range).
        * Automatically segmenting positive signal areas in Immunohistochemistry (IHC) or Immunofluorescence (IF) images (by defining positive signal via aggregate thresholds) and measuring their area and total signal intensity.
        * Automatically identifying and measuring the number and area of specifically colored cells (corresponding to an aggregate range) in live/dead cell staining images, ignoring other colors or background.

* **Batch Processing and Usability**:
    * **Folder Processing**: All major functions support batch processing of all images within a selected folder.
    * **User Interface**: Provides an intuitive GUI (PyQt6) for easy file selection, parameter adjustment, process initiation, log viewing, and opening output folders.
    * **Logging**: Detailed operation and processing logs are displayed in the right panel of the interface.
    * **Output Management**: Automatically creates output folders next to the input folder, named descriptively based on the parameters used, for easy results management.

## Requirements

* **Python**: Developed and tested with **Python 3.11.9**. Python 3.9+ is recommended.
* **Operating System**: Windows 10 or Windows 11 recommended.
* **Hardware**: For reasonable SAM processing performance (using GPU acceleration), an NVIDIA GPU equivalent to at least a GeForce RTX 3060 is recommended. CPU execution is possible but will be significantly slower.
* **Key Dependencies**: You will need to install the following Python packages:
    * `PyQt6`
    * `opencv-python`
    * `numpy`
    * `Pillow`
    * `torch`
    * `torchvision`
    * `segment-anything`
    * `tifffile` (**Optional**: but highly recommended for TIFF image format support)
    * `pycocotools` (**Optional**: only needed for processing RLE format masks, which is not the default)

## Installation

**Prerequisites:**

* **Install Git:** You need [Git](https://git-scm.com/downloads) installed on your system to use the `git clone` command.
* **Install Python:** Ensure you have a compatible Python version installed (developed with Python 3.11.9, recommend 3.9+). You can check your version by running `python --version` or `python -V` in your terminal or command prompt.

**Installation Steps:**

1.  **Clone the Repository**: Open your terminal or command prompt and run the following commands to download the code:
    ```bash
    git clone [https://github.com/CUIAOYU/Useful-Image-Analysis-Tool-Integration.git](https://github.com/CUIAOYU/Useful-Image-Analysis-Tool-Integration.git)
    cd Useful-Image-Analysis-Tool-Integration
    ```
2.  **(Recommended) Create and Activate a Python Virtual Environment**: To avoid conflicts between project dependencies, it's highly recommended to create a virtual environment:
    ```bash
    # Create virtual environment (e.g., named venv)
    python -m venv venv

    # Activate virtual environment
    # Windows (cmd/powershell):
    .\venv\Scripts\activate
    # macOS/Linux (bash/zsh):
    source venv/bin/activate
    ```
    *All subsequent `pip install` commands should be run **after** activating the virtual environment.*
3.  **Install Dependencies**: You need to manually install the required packages using pip. Run:
    ```bash
    pip install PyQt6 opencv-python numpy Pillow torch torchvision segment-anything tifffile pycocotools
    ```
    * **Dependency Installation Notes & Common Issues**:
        * **PyTorch/Torchvision**: The installation of these libraries depends heavily on your operating system, whether you have an NVIDIA GPU, and your CUDA version. It is **strongly recommended** to first visit the [Official PyTorch Website](https://pytorch.org/), get the **official recommended installation command** for your specific environment (OS, Package, Compute Platform), and execute that command **separately first** to install PyTorch and Torchvision. Afterward, you can run the `pip install ...` command above for the remaining packages (you can omit torch and torchvision from it).
        * **Pycocotools**: Installing `pycocotools` on Windows might require pre-installing Microsoft C++ Build Tools. If you encounter compilation errors, you can try resolving them, or skip installing this library if you don't need to process RLE mask formats (the program will issue a warning if it's missing).
        * **Tifffile**: If `tifffile` is not installed, the program will not be able to read TIFF format images.
        * **Other Errors**: If you encounter issues installing other libraries, carefully read the error messages provided by pip, as they often contain clues for resolving the problem.

4.  **(Optional) Verify Installation**: After installing all dependencies, you can try running the main script (see step 3 in "Usage") to see if the GUI launches successfully. This serves as a basic verification that the installation was successful.

## SAM Model Setup - Important!

The SAM Processing feature requires a pre-trained Segment Anything Model file (`.pth`). **These model files are very large and are NOT included in this repository. You must download them separately.**

1.  **Download a Model**:
    * Download the desired SAM model file from an official source, such as the Meta AI Research SAM repository.
    * Common model checkpoints include `vit_h` (largest), `vit_l`, and `vit_b` (smallest). The choice depends on your trade-off between accuracy and speed/resource consumption.
    * **Official Download Link**: [SAM Model Checkpoints](https://github.com/facebookresearch/segment-anything#model-checkpoints)
2.  **Place the Model**:
    * Save the downloaded `.pth` file to a convenient location on your computer.
    * When running this tool, you will need to specify the path to this downloaded model file using the "Select SAM Model File (.pth)" button on the "SAM Processing" tab. The program needs to know where the model file is to load it.

## Usage

1.  **Prerequisites**: Ensure all dependencies are installed correctly, and you have downloaded a SAM model file.
2.  **Activate Environment**: Activate your Python virtual environment if you are using one.
3.  **Run the Application**: Open a terminal or command prompt in the project's root directory and run the main Python script:
    ```bash
    python [your_main_script_name].py
    # e.g., python main_gui.py or the actual script name
    ```
4.  **Familiarize Yourself with the GUI**: The application window will appear.
    * **Left Panel**: Control area with function tabs, file/folder selection buttons, parameter sliders/checkboxes, etc.
    * **Right Panel**: Log area displaying progress, information, and error messages.
5.  **Select Function**: Click the tabs at the top (e.g., "Contrast Enhancement", "Heatmap Generation", "SAM Processing") to switch between functions.
6.  **Configure and Execute**: Within the selected tab:
    * Use "Select Input Folder" to choose the folder containing images to process.
    * (For SAM Processing) Use "Select SAM Model File (.pth)" to select your downloaded model.
    * Carefully adjust the available parameters and thresholds using the GUI controls.
    * Click the blue action button (e.g., "Process All Images", "Generate Heatmaps") to start the task. Buttons are typically enabled only after all required inputs (like folders, models) are provided.
7.  **Monitor and Review Results**:
    * Monitor the output messages in the log area on the right during processing. Pay attention to progress updates and any potential errors.
    * Upon completion, the log area will indicate the task is finished.
    * Click the "Open Output Folder" button to easily access the folder containing the result files. Output folders are usually created next to the input folder and named descriptively (including parameters used).

## Contributing

Contributions are welcome!

* **Reporting Bugs or Suggesting Enhancements**: If you encounter any issues or have ideas for improvements, please submit them via the GitHub [Issues](https://github.com/CUIAOYU/Useful-Image-Analysis-Tool-Integration/issues) page for this repository. Please provide as much detail as possible, including steps to reproduce, screenshots (if applicable), your OS, and software versions.
* **Submitting Code**: If you'd like to contribute code improvements, please follow the standard GitHub Fork & Pull Request workflow. It's recommended to first open an Issue to discuss the changes you intend to make.

## Reporting Issues

If you encounter any bugs or have feature requests, please create a new issue on the [Issues](https://github.com/CUIAOYU/Useful-Image-Analysis-Tool-Integration/issues) page of this repository. This is the primary channel for support and feedback.

## License

This project is licensed under the **MIT License**. See the `LICENSE` file for details.

