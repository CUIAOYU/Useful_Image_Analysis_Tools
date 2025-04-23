# Useful Image Analysis Tool Integration (Based on SAM and PyQt)

This is a Graphical User Interface (GUI) tool built with PyQt6, designed to simplify and accelerate common image analysis and processing tasks in research or experiments. It's particularly useful for scenarios involving batch processing of large numbers of images (such as micrographs, tissue section scans, etc.). It can help you automatically extract quantitative information, compare image features, or enhance visual appearance.

## Key Features

* **Contrast Enhancement**:
    * **Purpose**: Stretches the difference between light and dark areas in an image, making details clearer.
    * **Batch Processing**: Can process all images within a selected folder at once.
    * **Adjustable**: You can set the enhancement factor (greater than 1 enhances, less than 1 reduces contrast).

* **Aggregation Analysis**:
    * **What is "Aggregation Score"?**: A custom metric (0-100) defined to measure how "dark" a pixel is. Simply put, **the darker the color, the higher the aggregation score**.
    * **Calculation**:
        1.  Normalize the pixel's grayscale value (0-255, where 0 is black, 255 is white) to a 0-1 range (`grayscale_value / 255.0`).
        2.  Multiply by 100.
        3.  Subtract the result from 100.
        * **Formula**: `Aggregation Score = 100 - (Grayscale Value / 255.0) * 100`
        * **Example**: A pure black pixel has an aggregation score of 100, while a pure white pixel has a score of 0.
    * **Potential Use Cases**: In certain contexts (e.g., biological staining images), a higher aggregation score might indicate higher "material density" or stronger "signal" (specifically, a dark signal).
    * **Quick Assessment**: Calculates the **average aggregation score** for an entire image or all images in a folder, providing a rapid overview of the sample's overall "darkness" or "aggregation level".

* **Heatmap Generation**:
    * **Purpose**: Visualizes the "Aggregation Score" of each pixel using different colors, creating a color map that intuitively shows where aggregation is high or low.
    * **How it Works**:
        1.  Calculates the aggregation score (0-100) for every pixel in the image.
        2.  You define an aggregation score **range** of interest by setting a minimum (`Min Aggregate Threshold`) and maximum (`Max Aggregate Threshold`) value.
        3.  **Color Mapping Rules**:
            * Pixels with an aggregation score **below** the minimum threshold are displayed as **black** (background).
            * Pixels with an aggregation score **above** the maximum threshold are displayed as **bright red** (saturated).
            * Pixels with an aggregation score **within** the specified range are colored on a gradient: smoothly transitioning from **green** (near the minimum, lower aggregation/brighter) through **yellow/orange** to **red** (near the maximum, higher aggregation/darker).
    * **Objective**: By adjusting the thresholds, you can focus attention on the aggregation score interval of interest, filter out background or overexposed areas, and highlight the distribution and intensity variations of target regions.

* **SAM Processing**:
    * **Purpose**: Integrates the Segment Anything Model (SAM for automatic segmentation) with aggregation analysis. It first automatically identifies distinct regions in the image, then filters these regions based on user-defined criteria (in two steps) to isolate targets of interest. Finally, it calculates metrics (Area, Intensity, Ratio) for these targets and annotates them on the image.
    * **Model**: You need to provide a pre-trained SAM model file (`.pth`).
    * **Automatic Segmentation**: SAM automatically segments the image into numerous distinct regions (generating masks).
    * **Two-Step Filtering**:
        1.  **Step 1: Filter by Initial Region Properties (A/I/R Threshold)**:
            * Calculates the **initial** area, intensity, and ratio for each region identified by SAM.
            * If you check "Area" or "Intensity", the program checks if these initial values meet the minimum thresholds you set (`A/I/R Threshold`).
            * Any region whose initial values **do not meet** the criteria is **immediately discarded** and does not proceed to the next step.
        2.  **Step 2: Filter by Pixel Aggregation Score (Min/Max Aggregate Threshold)**:
            * For regions that **passed Step 1**, the tool examines **each pixel** within that region.
            * Only pixels whose aggregation score falls within the specified [`Min Aggregate Threshold`, `Max Aggregate Threshold`] **range** are considered "valid pixels" for that region.
            * If a region passes Step 1 but has **zero "valid pixels"** after Step 2, that region **will not be annotated or included in the final calculations**.
    * **Final Calculation and Annotation**:
        * For regions that **pass both steps**, the program uses **only** the "valid pixels" found in Step 2 to **recalculate** the final:
            * **Area**: Number of valid pixels.
            * **Intensity**: Sum of aggregation scores of all valid pixels.
            * **Ratio**: Intensity / Area.
        * The program draws the **original contour** of the region on the output image and annotates it with the **final calculated** A/I/R values using colored text.
    * **Detailed Parameter Tuning (SAM Auto Mask Generator Parameters)**: SAM offers advanced options to fine-tune its behavior. Default settings are often sufficient, but understanding them can help address specific issues:
        * **`points_per_side`** (Default: 32): Imagine placing a grid over the image to find objects; this is the number of points along each side of the grid. **Higher values mean a denser grid**, better for finding small objects, but **slower and requiring more memory/VRAM**. Lower values are faster but might miss small details.
        * **`points_per_batch`** (Default: 64): Determines how many grid points the GPU processes at once. **Mainly affects VRAM usage**. If you encounter **"Out of Memory" errors, decrease this value** (e.g., to 32 or 16). If you have ample VRAM, slightly increasing it might speed things up.
        * **`pred_iou_thresh`** (Predicted IoU Threshold, Default: 0.88): SAM's confidence score (0-1) for the regions it finds. **Higher values (closer to 1) mean stricter requirements**, resulting in fewer noisy detections but potentially missing ambiguous or uncertain targets. **Lower values mean looser requirements**, possibly recovering some targets but also potentially introducing more incorrect segmentations.
        * **`stability_score_thresh`** (Stability Threshold, Default: 0.95): Measures how stable the boundary of a found region is (0-1). Does the shape change drastically with slight variations in criteria? **Higher values demand more stable, clear boundaries**, leading to more reliable results but potentially discarding targets with fuzzy edges. **Lower values tolerate fuzzier boundaries**, possibly recovering such targets but also risking oddly shaped results.
        * **`stability_score_offset`** (Default: 1.0): An internal parameter used in stability score calculation. **Usually does not need adjustment**.
        * **`box_nms_thresh`** (Bounding Box NMS Threshold, Default: 0.7): Prevents drawing multiple bounding boxes for the same object. If two boxes overlap more than this threshold, one might be suppressed. **Lower values mean more aggressive duplicate removal**, potentially merging nearby distinct objects; **higher values tolerate more overlap**, possibly resulting in multiple boxes for one object. **The default is usually effective; typically no need to change**.
        * **`crop_n_layers`** (Default: 0): A switch for processing very large images in "tiles". **0 means no tiling**; the entire image is processed at once. **Setting it to 1 or higher** divides the large image into many overlapping smaller tiles, processes them individually, and then stitches the results back together. Advantage: **can handle images too large for memory/VRAM**. Disadvantage: **significantly slower**, and seams between tiles might be visible. **Consider setting to 1 only if encountering memory issues with very large images**.
        * **`crop_nms_thresh`** (Default: 0.7): The non-maximum suppression threshold used when merging results from tiles in crop mode. **Usually does not need adjustment**.
        * **`crop_overlap_ratio`** (Default: ~0.341): The proportion of overlap between adjacent tiles in crop mode. **Usually does not need adjustment**.
        * **`crop_n_points_downscale_factor`** (Default: 1): In crop mode, determines if the grid points within each tile should be sparser. 1 means no downscaling. **Higher values make tile processing faster but may reduce accuracy. Generally keep at 1**.
        * **`min_mask_region_area`** (Default: 0): A **post-processing** step. Setting a value **greater than 0** (e.g., 50) automatically removes any small, isolated segmentation results with an area smaller than this pixel count. **Very useful for automatically cleaning up small noise artifacts** in the background. Setting to 0 keeps all results.
        * **`output_mode`** (Default: `binary_mask`): Controls the internal data format of the masks output by SAM. **Please keep the default value `binary_mask`**. This is the most standard and widely compatible format; other formats might cause errors in subsequent processing steps.

* **Batch Processing & Usability**:
    * **Folder Processing**: Most key features support processing an entire folder of images at once.
    * **Graphical Interface**: An intuitive GUI built with PyQt6 makes selecting files, setting parameters, viewing logs, and opening results convenient.
    * **Logging**: The right panel of the interface displays detailed operational logs and processing information.
    * **Output Management**: After processing, a new folder containing the results is automatically created next to the input folder. The output folder name includes parameter information for easy organization.

## System Requirements

* **Python**: **Python 3.9 or higher** is recommended (developed and tested with 3.11.9).
* **Operating System**: Windows 10 or 11 recommended.
* **Hardware**: For SAM processing with GPU acceleration (faster), an NVIDIA graphics card (e.g., GeForce RTX 3060 or comparable) is recommended. CPU-only execution is possible but significantly slower.
* **Required Libraries**:
    * `PyQt6`
    * `opencv-python`
    * `numpy`
    * `Pillow`
    * `torch`
    * `torchvision`
    * `segment-anything`
    * `tifffile` (**Recommended**: Enables support for TIFF image formats)
    * `pycocotools` (**Optional**: Not required if you don't need to handle RLE format masks)

## Installation Guide

**Prerequisites:**

* **Install Git:** You need [Git](https://git-scm.com/downloads) installed to clone the repository using `git clone`.
* **Install Python:** Ensure your Python version meets the requirements (3.9+ recommended). Check by typing `python --version` in your command line.

**Steps:**

1.  **Download the Code**: Open your command line/terminal and run:
    ```bash
    git clone [https://github.com/CUIAOYU/Useful-Image-Analysis-Tool-Integration.git](https://github.com/CUIAOYU/Useful-Image-Analysis-Tool-Integration.git)
    cd Useful-Image-Analysis-Tool-Integration
    ```

2.  **(Recommended) Create a Virtual Environment**: To avoid library version conflicts with other projects, it's best practice to create a dedicated virtual environment for this tool:
    ```bash
    # Create the environment (e.g., named 'venv')
    python -m venv venv

    # Activate the environment
    # Windows (cmd/powershell):
    .\venv\Scripts\activate
    # macOS/Linux (bash/zsh):
    source venv/bin/activate
    ```
    *All subsequent `pip install` commands should be run **after activating** the environment.*

3.  **Install Dependencies**: Use pip to install the required libraries. Run:
    ```bash
    pip install PyQt6 opencv-python numpy Pillow torch torchvision segment-anything tifffile pycocotools
    ```
    * **Installation Tips & Common Issues**:
        * **PyTorch/Torchvision**: Installing these can be tricky depending on your OS, NVIDIA driver/CUDA version. **It is highly recommended** to first visit the [official PyTorch website](https://pytorch.org/), select your environment specifics (OS, package manager, compute platform - CUDA or CPU), and use the **exact command provided there** to install PyTorch and Torchvision **first**. Then, run the `pip install ...` command above for the remaining libraries (you can omit `torch` and `torchvision` from that command if already installed).
        * **Pycocotools**: Installation on Windows might require Microsoft C++ Build Tools. If you encounter errors, either resolve the build environment issues or, if RLE format is not needed, you can try omitting `pycocotools` (the program might issue a warning but should run).
        * **Tifffile**: If not installed, the tool won't be able to read `.tif` or `.tiff` image files.
        * **Other Library Errors**: Carefully read the error messages from pip; they usually provide clues about the problem.

4.  **(Optional) Verify Installation**: After installation, try running the main script (see Step 3 in "How to Use" below). If the GUI opens successfully, the installation is likely correct.

## SAM Model Setup - Important!

The SAM Processing feature requires pre-trained model files (`.pth`). **These files are large and not included in the code repository; you must download them yourself.**

1.  **Download a Model**:
    * Go to the official SAM repository by Meta AI Research or another trusted source to download the model files.
    * Common models include `vit_h` (largest, most accurate, slowest), `vit_l` (medium), and `vit_b` (smallest, fastest, potentially less accurate). Choose one based on your needs and hardware capabilities.
    * **Official Link**: [SAM Model Checkpoints](https://github.com/facebookresearch/segment-anything#model-checkpoints)

2.  **Place the Model File**:
    * Save the downloaded `.pth` file to a convenient location on your computer.
    * When running the tool, navigate to the "SAM Processing" tab and click the "Select SAM Model File (.pth)" button to specify the path to your downloaded model file.

## How to Use

1.  **Prerequisites**: Ensure all dependencies are installed and the SAM model file is downloaded.
2.  **Activate Environment**: If you created a virtual environment, activate it first.
3.  **Run the Application**: Open your command line/terminal in the project's root directory and execute the main script:
    ```bash
    python your_main_script_name.py
    # Example: python main_gui.py (check your actual script filename)
    ```
4.  **GUI Layout**: Once the program opens, the left side contains controls (function selection, file selection, parameter adjustment), and the right side displays logs (runtime information and errors).
5.  **Select Functionality**: Click the tabs at the top to switch between features (e.g., "Contrast Enhancement", "SAM Processing").
6.  **Configure and Run**:
    * Click "Select Input Folder" to choose the folder containing images to process.
    * (If using SAM) Click "Select SAM Model File (.pth)" to specify the model file path.
    * Adjust the various parameters in the GUI as needed.
    * Click the blue processing button (e.g., "Process All Images") to start the task. These buttons are usually enabled only after selecting the input folder and (if applicable) the SAM model.
7.  **Check Results**:
    * Monitor the output in the log panel on the right during processing.
    * A completion message will appear in the log panel when finished.
    * Click the "Open Output Folder" button to directly open the folder containing the results. The results folder is automatically created next to the input folder, with a name indicating the parameters used.

## How to Contribute

Contributions of all forms are highly welcome!

* **Reporting Bugs or Suggesting Features**: If you find a problem or have an idea for improvement, please open an issue on the project's GitHub [Issues](https://github.com/CUIAOYU/Useful-Image-Analysis-Tool-Integration/issues) page. Provide as much detail as possible, including steps to reproduce, screenshots, your OS/software versions, etc.
* **Contributing Code**: Want to modify the code directly? Great! Please follow the standard GitHub Fork & Pull Request workflow. It's generally a good idea to open an issue first to discuss the changes you intend to make.

## Reporting Issues

If you encounter bugs or have feature requests, please open a new issue on the repository's [Issues](https://github.com/CUIAOYU/Useful-Image-Analysis-Tool-Integration/issues) page.

## License

This project is licensed under the **MIT License**. See the `LICENSE` file for details.
