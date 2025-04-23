# Useful Image Analysis Tool Integration (Based on SAM and PyQt)

This is a Graphical User Interface (GUI) tool built with PyQt6, designed to simplify and accelerate common image analysis and processing tasks in research or experiments. It's particularly useful for scenarios requiring batch processing of numerous images (like micrographs, tissue slide scans, etc.). It can help you automatically extract quantitative information, compare image features, or enhance image visual quality.

## Key Features

* **Contrast Enhancement**:
    * **Purpose**: Stretches the difference between light and dark areas in an image, making details clearer.
    * **Batch Processing**: Process all images within a selected folder at once.
    * **Adjustment**: You can set the enhancement level (greater than 1 enhances, less than 1 reduces).

* **Aggregation Analysis**:
    * **What is "Aggregation Degree"?**: A custom metric (0-100) defined to measure how "dark" a pixel is. Simply put, **the darker the color, the higher the aggregation degree**.
        * **Formula**: `Aggregation Degree = 100 - (Grayscale Value / 255.0) * 100`
        * **Example**: A pure black pixel has an aggregation degree of 100, while pure white is 0.
    * **Quick Assessment**: Calculates the **average aggregation degree** for an entire image or a folder of images, providing a quick overview of the sample's overall "darkness" or "aggregation level".

* **Heatmap Generation**:
    * **Purpose**: Represents the "aggregation degree" of each pixel using different colors, generating a color map that visually shows where aggregation is high or low.
    * **How it works**:
        1.  Calculates the aggregation degree (0-100) for every pixel in the image.
        2.  You define an aggregation degree **range** of interest by setting a minimum (`Min Aggregate Threshold`) and maximum (`Max Aggregate Threshold`) value.
        3.  **Color Rules**:
            * Pixels with aggregation **below** the minimum threshold: Displayed as **black** (background).
            * Pixels with aggregation **above** the maximum threshold: Displayed as **bright red** (saturated).
            * Pixels with aggregation **within** the range: Colors transition smoothly from **green** (near the minimum, low aggregation/brighter) through **yellow/orange** to **red** (near the maximum, high aggregation/darker).
    * **Goal**: By adjusting the thresholds, you can focus attention on the aggregation degree interval of interest, filtering out background or overexposed areas, and highlighting the distribution and intensity variations of target regions.

* **SAM Processing**:
    * **Purpose**: Combines the Segment Anything Model (AI automatic segmentation) with aggregation analysis. It first automatically identifies distinct regions in the image, then filters these regions based on criteria you set (in two steps) to isolate targets of interest. Finally, it calculates metrics (Area, Intensity, Ratio) for these targets and annotates them on the image.
    * **Model**: You need to provide a pre-trained SAM model file (`.pth`).
    * **Automatic Segmentation**: SAM automatically segments the image into numerous independent regions (generating masks).
    * **Two-Step Filtering**:
        1.  **Step 1: Filter by Initial Region Properties (A/I/R Threshold)**:
            * Calculates the **initial** area, intensity, and ratio for each region found by SAM.
            * If you check "Area" or "Intensity", the program checks if these initial values meet the minimum thresholds you set (`A/I/R Threshold`).
            * Any region failing to meet **any** of the selected initial thresholds is **immediately discarded** and does not proceed to the next step.
        2.  **Step 2: Filter by Pixel Aggregation Degree (Min/Max Aggregate Threshold)**:
            * For regions that **passed Step 1**, it examines **every pixel** within that region.
            * Only pixels whose aggregation degree falls **within** the specified [`Min Aggregate Threshold`, `Max Aggregate Threshold`] range are considered "valid pixels" for that region.
            * If a region passed Step 1 but ends up with **zero "valid pixels"** after Step 2, that region **will not be ultimately marked or calculated**.
    * **Final Calculation and Annotation**:
        * For regions that **pass both steps**, the program uses **only** the "valid pixels" found in Step 2 to **recalculate** the final:
            * **Area**: Number of valid pixels.
            * **Intensity**: Sum of the aggregation degrees of all valid pixels.
            * **Ratio**: Intensity / Area.
        * The program draws the **original contour** of this region on the output image and annotates it with the **finally calculated** A/I/R values in colored text.
    * **Detailed Parameter Tuning (SAM Auto Mask Generator Parameters)**: SAM offers advanced options to fine-tune its behavior. Default settings usually work well, but understanding them helps troubleshoot specific issues:
        * **`points_per_side`** (Default 32): Imagine casting a grid over the image to find objects; this is the number of points along each side of the grid. **Higher values mean a denser grid**, better for finding small objects, but also **slower and uses more memory/VRAM**. Lower values are faster but might miss small details.
        * **`points_per_batch`** (Default 64): Determines how many grid points the GPU processes at once. **Mainly affects VRAM usage**. If you encounter **"Out of Memory" errors, lower this value** (e.g., to 32 or 16). If you have ample VRAM, slightly increasing it might speed things up.
        * **`pred_iou_thresh`** (Predicted IoU Threshold, Default 0.88): A score (0-1) SAM gives itself for how confident it is about a found region. **Higher values (closer to 1) are stricter**, resulting in less noise but potentially missing ambiguous or uncertain targets. **Lower values are more lenient**, possibly recovering some targets but also potentially introducing more incorrect segmentations.
        * **`stability_score_thresh`** (Stability Threshold, Default 0.95): Measures how stable the boundary of a found region is (0-1). Does the shape change drastically if the criteria are slightly altered? **Higher values demand more stable, clear boundaries**, leading to more reliable results but potentially discarding targets with fuzzy edges. **Lower values tolerate fuzzier boundaries**, possibly recovering such targets but also potentially yielding oddly shaped results.
        * **`stability_score_offset`** (Default 1.0): An internal parameter used in calculating the stability score.
        * **`box_nms_thresh`** (Bounding Box NMS Threshold, Default 0.7): Prevents drawing multiple boxes around the same object. If two boxes overlap more than this threshold, only one might be kept. **Lower values mean more aggressive duplicate removal**, potentially merging nearby distinct objects; **higher values tolerate more overlap**, possibly resulting in multiple boxes for a single object.
        * **`crop_n_layers`** (Default 0): A switch for processing very large images in "tiles". **0 means no tiling**, process the whole image at once. **Setting to 1 or more** will divide the large image into many overlapping smaller crops, process them individually, and then stitch the results. Advantage: **Can handle images too large to fit in memory/VRAM**. Disadvantage: **Significantly slower**, and stitching artifacts might appear at seams. **Consider setting to 1 only when encountering memory issues with very large images**.
        * **`crop_nms_thresh`** (Default 0.7): In tiling mode, the overlap threshold used when merging results from the small crops.
        * **`crop_overlap_ratio`** (Default ~0.341): In tiling mode, the proportion of overlap between adjacent crops.
        * **`crop_n_points_downscale_factor`** (Default 1): In tiling mode, whether to use fewer points within each small crop. 1 means no downscaling. **Higher values make crop processing faster but may reduce accuracy. Generally keep at 1**.
        * **`min_mask_region_area`** (Default 0): A **post-processing** step. Setting a value **greater than 0** (e.g., 50) automatically removes any small, isolated segmentation results with an area smaller than this pixel count. **Excellent for automatically cleaning up small noise specks in the background**. Set to 0 to keep all results.
        * **`output_mode`** (Default `binary_mask`): Controls the internal data format of the masks output by SAM.

* **Batch Processing & Usability**:
    * **Folder Processing**: Most core functions support processing an entire folder of images at once.
    * **Graphical Interface**: An intuitive interface built with PyQt6 makes selecting files, setting parameters, viewing logs, and opening result folders easy.
    * **Logging**: The right panel displays detailed operation logs and processing information.
    * **Output**: After processing, a new folder containing the results is automatically created next to your input folder. The output folder name includes parameter information for easy management.

## System Requirements

* **Python**: **Python 3.9 or higher** is recommended (developed and tested with 3.11.9).
* **Operating System**: Windows 10 or 11 recommended.
* **Hardware**: For SAM processing with GPU acceleration (faster), an NVIDIA graphics card (e.g., GeForce RTX 3060 or similar) is recommended. CPU-only execution is possible but will be much slower.
* **Required Libraries**:
    * `PyQt6`
    * `opencv-python`
    * `numpy`
    * `Pillow`
    * `torch`
    * `torchvision`
    * `segment-anything`
    * `tifffile` (**Recommended**: For TIFF image format support)
    * `pycocotools` (**Optional**: Not needed if you don't require RLE format mask handling)

## Installation Guide

**Prerequisites:**

* **Install Git:** You need [Git](https://git-scm.com/downloads) installed to clone the repository using `git clone`.
* **Install Python:** Ensure your Python version meets the requirements (3.9+ recommended). Check by typing `python --version` in your command line/terminal.

**Steps:**

1.  **Download the Code**: Open your command line or terminal and run:
    ```bash
    git clone [https://github.com/CUIAOYU/Useful-Image-Analysis-Tool-Integration.git](https://github.com/CUIAOYU/Useful-Image-Analysis-Tool-Integration.git)
    cd Useful-Image-Analysis-Tool-Integration
    ```

2.  **(Recommended) Create a Virtual Environment**: To avoid library version conflicts with other projects, it's best to create a dedicated virtual environment for this tool:
    ```bash
    # Create (e.g., named 'venv')
    python -m venv venv

    # Activate
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
        * **PyTorch/Torchvision**: Installing these can be tricky and depends on your OS, NVIDIA GPU presence, and CUDA version. **Strongly recommended**: Go to the [official PyTorch website](https://pytorch.org/), select your environment specifics (OS, package manager, compute platform - CUDA or CPU), get the **official installation command**, and **run that command first** to install PyTorch and Torchvision correctly. Then, run the `pip install ...` command above for the remaining libraries (you can omit `torch` and `torchvision` from it if already installed).
        * **Pycocotools**: On Windows, installing this might require Microsoft C++ Build Tools first. If you encounter errors, either resolve the build environment issues or, if you don't need RLE format support, simply don't install it (the program will show a warning but should still run).
        * **Tifffile**: If not installed, you won't be able to read `.tif` or `.tiff` image files.
        * **Other Library Errors**: Carefully read the error messages from pip; they usually contain clues.

4.  **(Optional) Verify Installation**: After installation, try running the main script (see step 3 in "How to Use" below). If the GUI opens successfully, the installation is likely complete.

## SAM Model Setup - Important!

The SAM processing feature requires pre-trained model files (`.pth`). **These files are large, not included in the repository, and must be downloaded separately.**

1.  **Download a Model**:
    * Go to the official Meta AI Research SAM repository or another trusted source to download a model file.
    * Common models include `vit_h` (largest, most accurate, slowest), `vit_l` (medium), and `vit_b` (smallest, fastest, potentially less accurate). Choose one based on your needs and hardware.
    * **Official Link**: [SAM Model Checkpoints](https://github.com/facebookresearch/segment-anything#model-checkpoints)

2.  **Place the Model**:
    * Save the downloaded `.pth` file to a convenient location on your computer.
    * When running the tool, navigate to the "SAM Processing" tab and click the "Select SAM Model File (.pth)" button to specify the path to your downloaded model file.

## How to Use

1.  **Prepare**: Ensure all dependencies are installed and the SAM model is downloaded.
2.  **Activate Environment**: If you created a virtual environment, activate it first.
3.  **Run**: Navigate to the project's root directory in your command line/terminal and run the main script:
    ```bash
    python your_main_script_name.py
    # Example: python main_gui.py (check your actual script file name)
    ```
4.  **Interface**: Once the program opens, the left side contains controls (select features, files, parameters), and the right side is the log area (shows runtime information and errors).
5.  **Select Function**: Click the tabs at the top to switch between features (e.g., "Contrast Enhancement", "SAM Processing").
6.  **Configure and Run**:
    * Click "Select Input Folder" to choose the folder containing images to process.
    * (If using SAM) Click "Select SAM Model File (.pth)" to select your model file.
    * Carefully adjust the various parameters available in the interface for the selected function.
    * Click the blue processing button (e.g., "Process All Images") to start. These buttons are usually enabled only after selecting the input folder and model file (if applicable).
7.  **Check Results**:
    * Monitor the output in the log area on the right during processing.
    * A message will appear in the log area upon completion.
    * Click the "Open Output Folder" button to directly open the folder containing the results. The output folder is automatically created next to the input folder, with a name indicating the parameters used.

## How to Contribute

Contributions of all kinds are welcome!

* **Report Bugs or Suggest Features**: Found a problem or have an idea for improvement? Please open an issue on the GitHub [Issues](https://github.com/CUIAOYU/Useful-Image-Analysis-Tool-Integration/issues) page. Provide as much detail as possible (how to reproduce, screenshots, your OS/software versions, etc.).
* **Contribute Code**: Want to modify the code directly? Great! Please follow the standard GitHub Fork & Pull Request workflow. It's best to open an Issue first to discuss the changes you propose.

## Reporting Issues

If you encounter bugs or have feature requests, please open a new issue directly on the repository's [Issues](https://github.com/CUIAOYU/Useful-Image-Analysis-Tool-Integration/issues) page.

## License

This project is licensed under the **MIT License**. See the `LICENSE` file for details.
