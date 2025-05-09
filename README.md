# Useful Image Analysis Tools

This is a Graphical User Interface (GUI) tool built with PyQt6, designed to simplify and accelerate common image analysis and processing tasks in research or experiments, especially for scenarios requiring batch processing of large numbers of images (such as micrographs, scanned tissue sections, etc.). It can help you automatically extract quantitative information, compare image features, or improve image visual effects.

## Main Features

* **Contrast Enhancement**:
    * **Purpose**: Increases the difference between light and dark areas of an image, making details clearer.
    * **Batch**: Can process all images in a folder at once.
    * **Adjustment**: You can set the enhancement level yourself (greater than 1 enhances, less than 1 reduces).

* **Aggregation Analysis**:
    * **What is "Aggregation Degree"?**: An index (0-100) defined to measure how "dark" a pixel is. Simply put, **the darker the color, the higher the aggregation degree**.
        * **Formula**: `Aggregation Degree = 100 - (Grayscale Value / 255.0) * 100`
        * **Example**: A pure black pixel has an aggregation degree of 100, pure white is 0.
    * **Quick Assessment**: Calculates the **average aggregation degree** for an entire image or all images in a folder, quickly providing an understanding of the overall "darkness" or "aggregation level" of the sample.

* **Heatmap Generation**:
    * **Purpose**: Represents the "aggregation degree" of each pixel with different colors, generating a colored map that visually shows where the aggregation degree is high or low.
    * **How it works**:
        1.  Calculate the aggregation degree (0-100) for each pixel in the image.
        2.  You set an aggregation degree **range** of interest (minimum value `Min Aggregate Threshold` and maximum value `Max Aggregate Threshold`).
        3.  **Color Rules**:
            * Pixels with aggregation degree **below** the minimum threshold: Displayed as **black** (background).
            * Pixels with aggregation degree **above** the maximum threshold: Displayed as **bright red** (saturated).
            * Pixels with aggregation degree **within** the range: Color smoothly transitions from **green** (near the minimum, low aggregation/brighter) through **yellow/orange**, to **red** (near the maximum, high aggregation/darker).
    * **Goal**: By adjusting the thresholds, you can focus on the aggregation degree interval of interest, filter out background or overexposed areas, and highlight the distribution and intensity changes of target regions.

* **SAM Processing**:
    * **Purpose**: Combines the Segment Anything Model (AI automatic segmentation) with aggregation analysis. It first automatically identifies various independent regions in the image, then uses conditions you set (in two steps) to filter for the targets you care about, and finally calculates metrics (area, intensity, ratio) for these targets and marks them on the image.
    * **Model Required**: You need to provide a pre-trained SAM model file (`.pth`).
    * **Automatic Segmentation**: SAM automatically segments the image into many independent regions (generating masks).
    * **Two-Step Filtering**:
        1.  **Step 1: Filter by Overall Region Properties (A/I/R Threshold)**:
            * Calculate the **initial** area, intensity, and ratio for each region found by SAM.
            * If you check "Area" or "Intensity", the program checks if these initial values meet the minimum threshold you set (`A/I/R Threshold`).
            * Any region whose initial values **do not meet** the criteria is **immediately discarded** and does not proceed to the next step.
        2.  **Step 2: Filter by Pixel Aggregation Degree (Min/Max Aggregate Threshold)**:
            * For regions that **passed Step 1**, examine **each pixel** within them.
            * Only pixels whose aggregation degree falls within the \[minimum, maximum] **interval** you set are considered "valid pixels" for that region.
            * If a region passed Step 1 but has **no "valid pixels"** after Step 2, this region **will not be marked or calculated** in the final output.
    * **Final Calculation and Annotation**:
        * For regions that **passed both steps**, the program uses **only** the "valid pixels" found in Step 2 to **recalculate** the final:
            * **Area**: Number of valid pixels.
            * **Intensity**: Sum of aggregation degrees of all valid pixels.
            * **Ratio**: Intensity / Area.
        * The program will draw the **original contour** of this region on the output image and label it with the **finally calculated** A/I/R values in colored text. The text color matches the contour color, allowing you to visually select regions of interest and read their values.
    * **Detailed Parameter Tuning (SAM Auto Mask Generator Parameters)**: SAM offers advanced options to fine-tune its behavior. Understanding them helps solve specific issues:
        * **`points_per_side`** (Default 32): Imagine laying a grid over the image to find objects; this is the number of points on each side of the grid. **Higher value = denser grid**, better at finding small things, but **slower and uses more memory/VRAM**. Lower value is faster but might miss small objects.
        * **`points_per_batch`** (Default 64): Determines how many grid points the GPU processes at once. **Mainly affects VRAM usage**. If you encounter **"Out of Memory" errors, lower this value** (e.g., 32 or 16). If you have ample VRAM, slightly increasing it might speed things up.
        * **`pred_iou_thresh`** (Default 0.88): SAM's confidence score (0-1) for a found region. **Higher value (closer to 1) = stricter requirement**, fewer noisy results, but might miss ambiguous or uncertain targets. **Lower value = looser requirement**, might recover some targets but could introduce more incorrect segmentations.
        * **`stability_score_thresh`** (Default 0.95): Measures how stable the boundary of a found region is (0-1). Does the shape change drastically if the criteria shift slightly? **Higher value = requires more stable, clear boundaries**, more reliable results, but might discard targets with fuzzy boundaries. **Lower value = tolerates fuzzier boundaries**, might recover such targets but could yield oddly shaped results.
        * **`stability_score_offset`** (Default 1.0): An internal parameter used in stability score calculation.
        * **`box_nms_thresh`** (Default 0.7): Prevents drawing multiple boxes for the same object. If two boxes overlap more than this threshold, only one might be kept. **Lower value = more aggressive deduplication**, might merge close but distinct objects; **Higher value = tolerates more overlap**, might result in multiple boxes for one object.
        * **`crop_n_layers`** (Default 0): "Tile processing" switch for very large images. **0 means no tiling**, process the whole image. **Setting to 1 or more** cuts the large image into many overlapping smaller tiles, processes them individually, then stitches the results. Advantage: **can process images too large for memory/VRAM**. Disadvantage: **much, much slower**, and stitching artifacts might appear at tile boundaries. **Only consider setting to 1 if you encounter memory issues with very large images**.
        * **`crop_nms_thresh`** (Default 0.7): Deduplication threshold used when merging results from tiles in crop mode.
        * **`crop_overlap_ratio`** (Default ~0.341): The proportion of overlap between adjacent tiles in crop mode.
        * **`crop_n_points_downscale_factor`** (Default 1): In crop mode, whether to reduce the density of points sampled within each tile. 1 means no reduction. **Higher value = faster tile processing, but potentially lower accuracy. Usually keep at 1**.
        * **`min_mask_region_area`** (Default 0): A **post-processing** step. Setting a value **greater than 0** (e.g., 50) automatically removes all isolated, small segmentation results with an area less than this pixel count. **Very useful for automatically cleaning up small noise spots in the background**. Set to 0 to keep all results.
        * **`output_mode`** (Default `binary_mask`): Controls the internal data format of the masks output by SAM.

* **Batch Processing & Usability**:
    * **Folder Processing**: Main features support processing an entire folder of images at once.
    * **Graphical Interface**: An intuitive interface built with PyQt6 makes selecting files, setting parameters, viewing logs, and opening result folders convenient.
    * **Logging**: The right panel displays detailed operation logs and processing information.
    * **Result Output**: After processing, a new folder containing the results is automatically created next to your input folder. The folder name includes parameter information for easy management.

## System Requirements

* **Operating System**: Windows 10 or 11 recommended.
* **Anaconda**: Anaconda Navigator.
* **Python Version**: **Python 3.11** is recommended.
* **Hardware**:
    * If you want to use the SAM processing feature and achieve faster speeds (GPU acceleration), an **NVIDIA GPU** is recommended, at least GeForce RTX 3060 or equivalent performance level.
    * If you don't have an NVIDIA GPU, the SAM feature can still run on the CPU, but it will be **much slower**. Other non-SAM features have low hardware requirements.
* **Required Python Libraries**:
    * `PyQt6` (for the GUI)
    * `opencv-python` (for image processing)
    * `numpy` (for numerical computation)
    * `Pillow` (for image file I/O)
    * `torch` (PyTorch core library)
    * `torchvision` (PyTorch vision library)
    * `segment-anything` (Meta AI's SAM library)
    * `tifffile` (**Recommended**: for supporting `.tif` or `.tiff` image formats)
    * `pycocotools` (**Optional**: Not needed if you don't process COCO dataset format or RLE format masks. Installation on Windows might require additional compiler setup.)

## Installation Guide (Using Anaconda Navigator and Pip)

**Step 1: Install Anaconda**

* If you don't have Anaconda installed, go to the [Anaconda Distribution website](https://www.anaconda.com/products/distribution) to download the installer for your Windows system.
* Run the installer and follow the prompts. Default settings are usually fine for most options.

**Step 2: Create a New Working Environment (Using Anaconda Navigator)**

An environment is like an isolated "room" where we install all the software libraries needed for this tool, preventing conflicts with other Python projects on your computer.

1.  **Open Anaconda Navigator**: Find and launch "Anaconda Navigator" from your Windows Start Menu. It might take a moment to start up.
2.  **Go to Environments**: Click the **"Environments"** tab on the left sidebar of the Navigator window.
3.  **Create New Environment**:
    * Click the **"Create"** button (with a plus icon) at the bottom of the environment list (which might initially only show "base (root)").
    * A small window titled "Create new environment" will appear.
    * In the **"Name"** field, enter a name for your new environment, for example, `image_tool`.
    * In the "Packages" section below, ensure **Python** is selected. Click the version number dropdown menu next to it and select **3.11**.
    * Confirm the name and Python version, then click the **"Create"** button in the bottom right corner of the window.
    * Anaconda will start creating the environment and installing the specified Python version along with some basic packages. Please wait patiently; this might take a few minutes. Once completed, you will see your newly created `image_tool` environment in the list on the left.

**Step 3: Install Required Libraries (Using Pip within the Environment)**

Now we need to enter the `image_tool` environment we just created and install all the necessary Python libraries using the `pip` package manager.

1.  **Open the Environment's Terminal**:
    * In Anaconda Navigator's "Environments" view, **click to select** the `image_tool` environment you just created from the list on the left (make sure it's highlighted).
    * Look for a **green play button (▶️)** or a **small down arrow (🔽)** next to the environment name `image_tool` at the top of the middle section. Click this button/arrow.
    * Select **"Open Terminal"** from the dropdown menu.
    * A **black command prompt window** (CMD or PowerShell) will open. You'll notice the prompt starts with `(image_tool)`, indicating that you are now operating within this isolated environment. **All subsequent installation commands must be executed in this window**.

2.  **Install PyTorch (Core Library, Following Official Recommendations)**:
    * **Very Important**: PyTorch installation is closely tied to your hardware (especially GPU) and the required CUDA version. It is **strongly recommended** to follow the latest instructions from the official PyTorch website.
    * **Open your web browser** and go to the official PyTorch website: <https://pytorch.org/>
    * Find the "Get Started" or similar installation guide section on the homepage.
    * Make selections based on your system:
        * **PyTorch Build**: Usually select **Stable**.
        * **Your OS**: Select **Windows**.
        * **Package**: **Select Pip** (Based on my previous experience and official guidance, Pip is often recommended).
        * **Language**: Select **Python**.
        * **Compute Platform**:
            * If your computer has an **NVIDIA GPU** (e.g., GeForce RTX series) and you want to use GPU acceleration (recommended for speed), select a **CUDA** version (e.g., CUDA 11.8 or 12.1).
            * If your computer **does not have an NVIDIA GPU**, or you are unsure, or you only want to use the CPU, select **CPU**.
    * After making your selections, the page will **automatically generate** an installation command under the **"Run this Command:"** section. This command usually starts with `pip install torch torchvision torchaudio ...`.
    * **Carefully and completely copy this generated `pip install ...` command**.
    * **Return to the black command prompt window** that has the `(image_tool)` prefix.
    * **Paste the copied command** into the window (you can usually right-click in the window to paste).
    * Press the **Enter** key to execute the command.
    * `pip` will start downloading and installing PyTorch and its related libraries. This may take a significant amount of time depending on your internet speed and the size of the files. Wait patiently until the command finishes executing without errors.

3.  **Install Other Required Libraries**:
    * In the **same black command prompt window** (after the PyTorch installation is successfully completed), **copy and paste** the following command:
        ```bash
        pip install PyQt6 opencv-python numpy Pillow segment-anything tifffile pycocotools
        ```
    * Press the **Enter** key.
    * `pip` will download and install the remaining libraries: PyQt6 (GUI), opencv-python (image processing), numpy (scientific computing), Pillow (basic image library), segment-anything (SAM library), tifffile (TIFF support), pycocotools (COCO tools).
    * Wait patiently for all libraries to download and install.
    * *(Installation Tips & Common Issues)*:
        * `pycocotools`: Installation on Windows sometimes fails due to a missing C++ build environment. If you encounter errors related to `cl.exe` or `Microsoft C++ Build Tools`, you might need to install Microsoft C++ Build Tools first (available via the Visual Studio Installer). If you are sure you don't need COCO dataset or RLE mask format support, you can omit `pycocotools` from the command above: `pip install PyQt6 opencv-python numpy Pillow segment-anything tifffile`. The program will issue a warning if it's missing but core functionality should still work.
        * `tifffile`: If not installed, you won't be able to read `.tif` or `.tiff` format images.
        * Network Issues: If downloads are slow or fail, it might be due to network problems.

4.  **Close the Command Prompt Window**: Once all libraries are successfully installed (the command prompt stops scrolling and returns to the `(image_tool) C:\Users\YourUsername>` prompt), you can type `exit` and press Enter, or simply click the "X" button in the top-right corner to close the window.

## SAM Model Setup - Important!

The SAM processing feature requires a pre-trained model file (`.pth`). **These files are large, not included in the code repository, and must be downloaded manually.**

1.  **Download the Model**:
    * Go to the official SAM repository by Meta AI Research or another trusted source to download the model file.
    * Common models include `vit_h` (largest, most accurate, but slowest), `vit_l` (medium), and `vit_b` (smallest, fastest, but potentially less accurate). Choose one based on your needs and computer specifications.
    * **Official Link**: [SAM Model Checkpoints](https://github.com/facebookresearch/segment-anything#model-checkpoints)

2.  **Place the Model**:
    * After downloading, save the `.pth` file to an easily accessible location on your computer.
    * When running the tool, on the "SAM Processing" tab, click the "Select SAM Model File (.pth)" button and navigate to where you saved the model file.

## How to Use

1.  **Preparation**: Ensure all dependencies are installed and the SAM model is downloaded.
2.  **Activate Environment**: If using a virtual environment, activate it first. (The Terminal opened via Anaconda Navigator already activates the `image_tool` environment automatically).
3.  **Run**: Open a command prompt/terminal in the project's root directory (refer to Installation Step 3.1 to open the Terminal for the `image_tool` environment) and run the main script:
    ```bash
    python your_main_script_filename.py
    # Example: python main_gui.py (check your actual filename)
    ```
4.  **Interface**: Once the program opens, the left side is the control area (select functions, files, adjust parameters), and the right side is the log area (view runtime information and errors).
5.  **Select Function**: Click the tabs at the top to switch between functions (e.g., "Contrast Enhancement", "SAM Processing").
6.  **Setup and Run**:
    * Click "Select Input Folder" to choose the folder containing the images to process.
    * (If using SAM) Click "Select SAM Model File (.pth)" to choose the model file.
    * Carefully adjust the various parameters on the interface.
    * Click the blue processing button (e.g., "Process All Images") to start. Usually, the input folder (and model, if applicable) must be selected first for the button to be enabled.
7.  **View Results**:
    * Monitor the output in the log area on the right during processing.
    * A completion message will appear in the log area when finished.
    * Click the "Open Output Folder" button to directly open the results folder. The results folder is automatically created next to the input folder, and its name includes parameter information.

## How to Contribute

Contributions of all forms are very welcome!

* **Report Bugs or Suggest Enhancements**: Found an issue or have an idea for improvement? Please submit it on the GitHub [Issues](https://github.com/CUIAOYU/Useful-Image-Analysis-Tool-Integration/issues) page. The more detail you provide (how to reproduce, screenshots, your system/software versions, etc.), the better.
* **Contribute Code**: Want to modify the code directly? Welcome! Please follow the standard GitHub Fork & Pull Request workflow. It's best to open an Issue first to discuss the changes you intend to make.

## Reporting Issues

For bugs or feature requests, please open a new Issue directly on the repository's [Issues](https://github.com/CUIAOYU/Useful-Image-Analysis-Tool-Integration/issues) page.
