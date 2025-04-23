# Useful Image Analysis Tool Integration (Based on SAM and PyQt)

This is a Graphical User Interface (GUI) tool built with PyQt6, designed to simplify and accelerate common image analysis and processing tasks in research or experiments, especially for scenarios involving batch processing of large numbers of images (such as micrographs, scanned tissue sections, etc.). It can help you automatically extract quantitative information, compare image features, or improve image visual effects.

## Key Features

* **Contrast Enhancement**:
    * **Purpose**: Increases the difference between light and dark areas in an image, making details clearer.
    * **Batch Processing**: Can process all images in a folder at once.
    * **Adjustment**: You can set the enhancement level yourself (greater than 1 enhances, less than 1 reduces).

* **Aggregation Analysis**:
    * **What is "Aggregation Degree"?**: I've defined a metric called "Aggregation Degree" (0-100) to measure how "dark" a pixel is. Simply put, **the darker the color, the higher the aggregation degree**.
        * **Formula**: `Aggregation Degree = 100 - (Grayscale Value / 255.0) * 100`
        * **Example**: A pure black pixel has an aggregation degree of 100, pure white is 0.
    * **Quick Assessment**: Calculates the **average aggregation degree** for the entire image or all images in a folder, providing a quick understanding of the overall "darkness" or "aggregation level" of the samples.

* **Heatmap Generation**:
    * **Purpose**: Represents the "Aggregation Degree" of each pixel with different colors, generating a color map that visually shows where the aggregation degree is high or low.
    * **How it works**:
        1.  Calculate the aggregation degree (0-100) for each pixel in the image.
        2.  You set an aggregation degree **range** of interest (minimum value `Min Aggregate Threshold` and maximum value `Max Aggregate Threshold`).
        3.  **Color Rules**:
            * Pixels with aggregation degree **below** the minimum threshold: Displayed as **black** (background).
            * Pixels with aggregation degree **above** the maximum threshold: Displayed as **bright red** (saturated).
            * Pixels with aggregation degree **within** the range: Color transitions smoothly from **green** (near the minimum, low aggregation/brighter) through **yellow/orange** to **red** (near the maximum, high aggregation/darker).
    * **Goal**: By adjusting the thresholds, focus attention on the aggregation degree interval of interest, filter out background or overexposed areas, and highlight the distribution and intensity changes of target regions.

* **SAM Processing**:
    * **Purpose**: Combines the Segment Anything Model (AI automatic segmentation) with aggregation analysis. It first automatically identifies individual regions in the image, then filters the regions you care about based on criteria you set (in two steps), and finally calculates metrics (area, intensity, ratio) for these targets and marks them on the image.
    * **Model**: You need to provide your own pre-trained SAM model file (`.pth`).
    * **Automatic Segmentation**: SAM automatically segments the image into many independent regions (generating masks).
    * **Two-Step Filtering**:
        1.  **Step 1: Filter by Overall Region Properties (A/I/R Threshold)**:
            * First, calculate the **initial** area, intensity, and ratio for each region found by SAM.
            * If you check "Area" or "Intensity", the program checks if these initial values meet the minimum thresholds you set (`A/I/R Threshold`).
            * Any region whose initial values **do not meet the criteria** is **immediately discarded** and does not proceed to the next step.
        2.  **Step 2: Filter by Pixel Aggregation Degree (Min/Max Aggregate Threshold)**:
            * For regions that **passed Step 1**, examine **each pixel** within them.
            * Only pixels whose aggregation degree falls within the \[minimum, maximum] **range** you set are considered "valid pixels" for that region.
            * If a region passed Step 1 but has **no "valid pixels"** after Step 2, that region **will not be marked or calculated** in the final output.
    * **Final Calculation and Annotation**:
        * For regions that **passed both steps**, the program uses **only** the "valid pixels" found in Step 2 to **recalculate** the final:
            * **Area (A)**: Number of valid pixels.
            * **Intensity (I)**: Sum of aggregation degrees of all valid pixels.
            * **Ratio (R)**: Intensity / Area.
        * The program draws the **original contour** of this region on the output image and labels it with the **finally calculated** A/I/R values in colored text. The text color matches the contour color, allowing you to visually select regions of interest and read their values.
    * **Detailed Parameter Tuning (SAM Auto Mask Generator Parameters)**: SAM offers advanced options to fine-tune its behavior. Understanding them helps solve specific problems:
        * **`points_per_side`** (Default 32): Imagine placing a grid over the image to find objects; this is the number of points on each side of the grid. **Higher value = denser grid**, better for finding small objects, but **slower and uses more memory/VRAM**. Lower is faster but might miss small things.
        * **`points_per_batch`** (Default 64): Determines how many grid points the GPU processes at once. **Mainly affects VRAM usage**. If you encounter **"Out of Memory" errors, lower this value** (e.g., 32 or 16). If you have ample VRAM, slightly increasing it might speed things up.
        * **`pred_iou_thresh`** (Predicted IoU Threshold, Default 0.88): SAM's confidence score (0-1) for a found region. **Higher value (closer to 1) = stricter requirement**, fewer noisy results, but might miss ambiguous or uncertain targets. **Lower value = looser requirement**, might recover some targets but could introduce more incorrect segmentations.
        * **`stability_score_thresh`** (Stability Threshold, Default 0.95): Measures how stable the boundary of a found region is (0-1). Does the shape change drastically with slight changes in criteria? **Higher value = requires more stable, clear boundaries**, more reliable results, but might discard targets with fuzzy edges. **Lower value = tolerates fuzzier boundaries**, might recover such targets but could yield oddly shaped results.
        * **`stability_score_offset`** (Default 1.0): An internal parameter used in stability score calculation.
        * **`box_nms_thresh`** (Bounding Box NMS Threshold, Default 0.7): Prevents drawing multiple boxes for the same object. If two boxes overlap more than this threshold, only one might be kept. **Lower value = more aggressive deduplication**, might merge nearby distinct objects; **Higher value = more tolerant of overlap**, might result in multiple boxes for one object.
        * **`crop_n_layers`** (Default 0): "Tiling" switch for processing very large images. **0 means no tiling**, process the whole image. **Set to 1 or more** to cut the large image into many overlapping smaller tiles, process them individually, and stitch the results. Advantage: **Can handle images too large for memory/VRAM**. Disadvantage: **Much, much slower**, and stitching seams might be visible. **Consider setting to 1 only when facing memory issues with huge images**.
        * **`crop_nms_thresh`** (Default 0.7): Deduplication threshold used when merging results from tiles in crop mode.
        * **`crop_overlap_ratio`** (Default \~0.341): The proportion of overlap between adjacent tiles in crop mode.
        * **`crop_n_points_downscale_factor`** (Default 1): In crop mode, whether to use fewer points within each tile. 1 means no downscaling. **Higher value = faster tile processing, but potentially lower accuracy. Usually keep at 1**.
        * **`min_mask_region_area`** (Default 0): A **post-processing** step. Set a value **greater than 0** (e.g., 50) to automatically remove all tiny, isolated segmentation results smaller than this pixel area. **Excellent for automatically cleaning up small noise specks in the background**. Set to 0 to keep all results.
        * **`output_mode`** (Default `binary_mask`): Controls the internal data format of the masks output by SAM.

* **Batch Processing & Usability**:
    * **Folder Processing**: Most features support processing an entire folder of images at once.
    * **Graphical Interface**: An intuitive interface built with PyQt6 makes selecting files, setting parameters, viewing logs, and opening result folders convenient.
    * **Logging**: The right panel displays detailed operation logs and processing information.
    * **Result Output**: After processing, a new folder containing the results is automatically created next to your input folder. The folder name includes parameter information for easy management.

## System Requirements

* **Python**: **Python 3.9 or higher** is recommended (developed and tested with 3.11.9).
* **Operating System**: Windows 10 or 11 recommended.
* **Hardware**: If using SAM processing and want faster speeds (GPU acceleration), an NVIDIA graphics card is recommended, at least GeForce RTX 3060 or equivalent. CPU-only execution is possible but will be significantly slower.
* **Required Libraries**:
    * `PyQt6`
    * `opencv-python`
    * `numpy`
    * `Pillow`
    * `torch`
    * `torchvision`
    * `segment-anything`
    * `tifffile` (**Recommended**: Adds support for TIFF image format)
    * `pycocotools` (**Optional**: Not needed if you don't require processing RLE format masks)

## Installation Guide

**Prerequisites:**

* **Install Git:** You need [Git](https://git-scm.com/downloads) installed to clone the repository using `git clone`.
* **Install Python:** Ensure your Python version meets the requirements (3.9+ recommended). Check by typing `python --version` in your command line.

**Steps:**

1.  **Download the Code**: Open your command line and run:
    ```bash
    git clone [https://github.com/CUIAOYU/Useful-Image-Analysis-Tool-Integration.git](https://github.com/CUIAOYU/Useful-Image-Analysis-Tool-Integration.git)
    cd Useful-Image-Analysis-Tool-Integration
    ```

2.  **(Recommended) Create a Virtual Environment**: To avoid library version conflicts with other projects, it's best to create a separate virtual environment for this project:
    ```bash
    # Create (e.g., named venv)
    python -m venv venv

    # Activate
    # Windows (cmd/powershell):
    .\venv\Scripts\activate
    # macOS/Linux (bash/zsh):
    source venv/bin/activate
    ```
    *All subsequent `pip install` commands should be run **after activating the environment**.*

3.  **Install Dependencies**: Install the required libraries using pip. Run:
    ```bash
    pip install PyQt6 opencv-python numpy Pillow torch torchvision segment-anything tifffile pycocotools
    ```
    * **Installation Tips and Common Issues**:
        * **PyTorch/Torchvision**: Installing these can be tricky and depends on your OS, NVIDIA card presence, and CUDA version. **Highly recommended**: Go to the [official PyTorch website](https://pytorch.org/), find the **official installation command** based on your environment (OS, package manager, compute platform - CUDA or CPU), and **run that command first** to install PyTorch and Torchvision correctly. Then, run the `pip install ...` command above for the remaining libraries (you can omit `torch` and `torchvision` from it).
        * **Pycocotools**: Installation on Windows might require Microsoft C++ Build Tools first. If you encounter errors, either resolve the build environment issues or, if you don't need RLE format support, simply don't install it (the program will issue a warning but should run).
        * **Tifffile**: Without this, you cannot read `.tif` or `.tiff` image files.
        * **Other Library Errors**: Carefully read the error messages from pip; they usually contain clues.

4.  **(Optional) Check Installation**: After installation, try running the main script (see "How to Use" step 3 below). If the GUI opens successfully, the installation is likely correct.

## SAM Model Setup - Important!

The SAM processing feature requires pre-trained model files (`.pth`). **These files are large, not included in the code repository, and must be downloaded by you.**

1.  **Download Models**:
    * Go to the official Meta AI Research SAM repository or another trusted source to download the model files.
    * Common models include `vit_h` (largest, most accurate, slowest), `vit_l` (medium), and `vit_b` (smallest, fastest, potentially less accurate). Choose one based on your needs and hardware.
    * **Official Link**: [SAM Model Checkpoints](https://github.com/facebookresearch/segment-anything#model-checkpoints)

2.  **Place the Model**:
    * After downloading, save the `.pth` file to an easily accessible location on your computer.
    * When running the tool, go to the "SAM Processing" tab and click the "Select SAM Model File (.pth)" button to tell the program where you saved the model file.

## How to Use

1.  **Prepare**: Ensure all dependencies are installed and the SAM model is downloaded.
2.  **Activate Environment**: Activate your virtual environment if you created one.
3.  **Run**: Open a command line in the project's root directory and run the main script:
    ```bash
    python your_main_script_name.py
    # Example: python main_gui.py (check your actual filename)
    ```
4.  **Interface**: Once the program opens, the left side is the control area (select features, files, adjust parameters), and the right side is the log area (view runtime information and errors).
5.  **Select Feature**: Click the tabs at the top to switch between features (e.g., "Contrast Enhancement", "SAM Processing").
6.  **Set Up and Run**:
    * Click "Select Input Folder" to choose the folder containing images to process.
    * (If using SAM) Click "Select SAM Model File (.pth)" to select the model file.
    * Carefully adjust the various parameters in the interface.
    * Click the blue processing button (e.g., "Process All Images") to start. Usually, you need to select the folder and model first for the button to become active.
7.  **View Results**:
    * Monitor the output in the log area on the right during processing.
    * A message will appear in the log area upon completion.
    * Click the "Open Output Folder" button to directly open the results folder. The results folder is automatically created next to the input folder, with parameters included in its name.

## How to Contribute

Contributions of all forms are very welcome!

* **Report Bugs or Suggest Enhancements**: Found an issue or have an idea for improvement? Please open an issue on the GitHub [Issues](https://github.com/CUIAOYU/Useful-Image-Analysis-Tool-Integration/issues) page. Provide as much detail as possible, such as steps to reproduce, screenshots, your system and software versions, etc.
* **Contribute Code**: Want to modify the code directly? Great! Please follow the standard GitHub Fork & Pull Request workflow. It's best to open an Issue first to discuss the changes you intend to make.

## Reporting Issues

If you encounter bugs or have feature requests, please open a new issue directly on the repository's [Issues](https://github.com/CUIAOYU/Useful-Image-Analysis-Tool-Integration/issues) page.

## License

This project is licensed under the **MIT License**. See the `LICENSE` file for details.
