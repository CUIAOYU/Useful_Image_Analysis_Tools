# SAM-HQ Image Batch Processing & Analysis Tool

This is a comprehensive GUI application built with PyQt6, designed to streamline and accelerate common image analysis and processing tasks in research and experimental work. It's particularly useful for scenarios requiring batch processing of large image datasets, such as microscopy photos, tissue slice scans, and similar scientific imagery. The tool combines the powerful segmentation capabilities of SAM-HQ (Segment Anything in High Quality) with custom image analysis algorithms to help you automatically extract quantitative information, compare image features, and enhance visual presentation.

## Key Features

The tool consists of **four core functional modules** organized through a tabbed interface:

### 1. Aggregation Analysis
* **Aggregation Metric Definition**: We've defined an "aggregation" score (0-100) that measures how "dark" pixels are. **Darker colors have higher aggregation values**.
    - **Formula**: `Aggregation = 100 - (grayscale_value / 255.0) × 100.0`
    - **Example**: Pure black pixels have aggregation of 100, pure white pixels have aggregation of 0
* **Single Image Distribution Analysis**: Select any image to analyze the distribution of aggregation values across all pixels and generate histograms.
* **Automatic Feature Detection**: Automatically identifies and marks peaks and troughs in the distribution, helping you understand the grayscale characteristics of your image content.
* **Detailed Statistics**: Provides key statistics including total pixel count, mean/min/max aggregation values, and more.
* **Chart Export**: Save generated analysis charts as PNG, PDF, or SVG formats.

### 2. Mask Analysis
* **Interactive Mask Generation**: Use SAM-HQ models to generate all possible masks for single images.
* **Visual Mask Selection**: Click directly on images to select/deselect masks, with left-click for selection and right-click for deselection.
* **Real-time Filtering & Calculation**: Apply custom filtering parameters to selected masks and calculate area, intensity, and ratios in real-time.
* **Results Export**: Export analysis results as text files or save marked images.
* **Bulk Selection Operations**: "Select All" and "Deselect All" functions for efficient batch operations.

### 3. Batch Processing
* **Smart File Traversal**: Select a folder and automatically discover all images (including subfolders), supporting `.png`, `.jpg`, `.jpeg`, `.tif`, `.tiff` formats.
* **SAM-HQ Model Integration**: Load SAM-HQ pretrained models (`.pth` files) for fully automated high-quality image mask generation.
* **Two-Stage Filtering System**:
    * **Stage 1: Region Property Filtering**: Initial screening based on area (A), total aggregation intensity (I), and ratio (R = I/A).
    * **Stage 2: Pixel-Level Aggregation Filtering**: Fine-grained filtering based on pixel-level aggregation value ranges.
* **Smart Output Management**: Processed images are automatically saved to parameter-named output folders to avoid file conflicts.
* **Data Export**: All filtered mask data can be exported as structured `.txt` files.
* **Real-time Progress Monitoring**: Detailed progress bars and logging information.

### 4. SAM-HQ Parameter Configuration
* **Complete Parameter Control**: Access to all adjustable parameters of the SAM-HQ automatic mask generator.
* **Parameter Documentation**: Detailed explanations and usage recommendations for each parameter.
* **Preset Management**: Support for resetting to defaults and applying changes across all functional modules.

### Universal Features
* **Configuration Management**: **Save/load** all filtering parameters and SAM-HQ model parameters to JSON files.
* **System Monitoring**: Real-time display of CPU, memory, GPU, and VRAM usage (requires appropriate libraries).
* **Multi-format Support**: Support for common image formats, including special handling for TIFF files.
* **Error Handling**: Comprehensive error handling and user notification mechanisms.
* **Batch Processing & Usability**: Core functions support processing entire image folders at once.
* **Graphical Interface**: Intuitive PyQt6-based interface makes file selection, parameter adjustment, log viewing, and result folder access convenient.

## System Requirements

* **Operating System**: Windows 10 or Windows 11 recommended
* **Anaconda**: Anaconda Navigator must be pre-installed
* **Python Version**: **Python 3.11** recommended
* **Hardware Requirements**:
    * For SAM processing with optimal speed (GPU acceleration), we recommend an **NVIDIA graphics card**, at least GeForce RTX 3060 or equivalent performance level
    * Without NVIDIA GPU, SAM functions will still work on CPU but will be **significantly slower**
    * Other non-SAM features have minimal hardware requirements

### Core Dependencies
* **PyQt6** - GUI framework
* **PyTorch** - Deep learning framework for SAM-HQ model support
* **segment_anything** - SAM-HQ core library
* **OpenCV (opencv-python)** - Image processing
* **NumPy** - Numerical computing
* **Pillow** - Image file I/O
* **matplotlib** - Chart generation (required for aggregation analysis)
* **pycocotools** - Mask data processing

### Optional Dependencies (Recommended)
* **psutil** - CPU and memory monitoring
* **nvidia-ml-py (pynvml)** - NVIDIA GPU monitoring
* **tifffile** - TIFF file support

## Installation Guide

### Step 1: Install Anaconda

If you don't already have Anaconda installed, visit the [Anaconda Distribution website](https://www.anaconda.com/products/distribution) to download the installer for your Windows system. Run the installer and follow the prompts. For most options, keeping the default settings is fine.

### Step 2: Create a New Environment

Think of an environment as an isolated "room" where we install all the software libraries this tool needs, preventing conflicts with other Python projects on your computer.

1. **Open Anaconda Navigator**: Find and launch "Anaconda Navigator" from the Windows Start menu. The first startup may take some time.

2. **Navigate to Environment Management**: In the Navigator window, click the **"Environments"** tab in the left sidebar.

3. **Create New Environment**:
   * Below the environment list (initially may only show "base (root)"), find and click the **"Create"** button (with plus icon)
   * A "Create new environment" dialog will appear
   * In the **"Name"** field, give your new environment a name, e.g., `sam_hq_env`
   * In the "Packages" section below, ensure **Python** is selected. Click the version dropdown and select **3.11**
   * After confirming the name and Python version, click the **"Create"** button in the bottom right
   * Anaconda will begin creating the environment and installing the specified Python version plus basic packages. Please be patient, this may take several minutes

### Step 3: Install PyTorch (Important)

1. **Open Environment-Specific Terminal**:
   * In Anaconda Navigator's "Environments" interface, **click to select** the `sam_hq_env` environment you just created (ensure it's highlighted)
   * In the top area next to the environment name `sam_hq_env`, you'll see a **green play button (▶️)** or **dropdown arrow (🔽)**. Click this button/arrow
   * In the popup menu, select **"Open Terminal"**
   * The system will open a **black command-line window**. Notice the prompt starts with `(sam_hq_env)`, indicating you're operating within this isolated environment

2. **Install PyTorch**:
   * **Important Note**: PyTorch installation is closely tied to your hardware (especially graphics cards) and required CUDA versions
   * Open your web browser and visit the PyTorch official website: https://pytorch.org/
   * Find the "Get Started" or similar installation guide section on the homepage
   * Make selections based on your situation:
     - **PyTorch Build**: Usually select **Stable**
     - **Your OS**: Select **Windows**
     - **Package**: Select **Pip**
     - **Language**: Select **Python**
     - **Compute Platform**:
       * If your computer has an **NVIDIA graphics card** and you want GPU acceleration (recommended), select a **CUDA** version (like CUDA 11.8 or 12.1)
       * If your computer **doesn't have NVIDIA GPU**, or you're unsure, or you only want CPU computation, select **CPU**
   * After completing the selections above, an installation command will be **automatically generated** at the bottom of the page
   * **Copy the entire generated command**, return to the command-line window, paste, and execute

### Step 4: Install SAM-HQ Core Library

In the same command-line window, execute the following commands:

```bash
git clone https://github.com/SysCV/sam-hq.git
cd sam-hq
pip install -e .
```

**Note**: This will install the SAM-HQ version of the segment_anything library, which will override the original segment_anything library.

### Step 5: Install Other Required Libraries

In the same command-line window, execute:

```bash
pip install PyQt6 opencv-python numpy Pillow matplotlib psutil tifffile
```

### Step 6: Install Optional Libraries

```bash
# GPU monitoring library (NVIDIA graphics card users only)
pip install nvidia-ml-py

# COCO tools library (for certain mask formats)
pip install pycocotools
```

**Installation Tips & Common Issues**:
* **pycocotools**: On Windows, installation may fail due to missing C++ build environment. If you encounter errors about `cl.exe` or `Microsoft C++ Build Tools`, try:
  ```bash
  pip install pycocotools-windows
  ```
* **nvidia-ml-py**: If you don't have an NVIDIA graphics card, you can skip this package
* **tifffile**: Without this, you won't be able to read `.tif` or `.tiff` format images

### Step 7: Verify Installation

Run the following Python code in the command line to verify key libraries are correctly installed:

```bash
python -c "import torch; print('PyTorch version:', torch.__version__); print('CUDA available:', torch.cuda.is_available())"
python -c "import segment_anything; print('SAM-HQ library imported successfully')"
python -c "import PyQt6; print('PyQt6 imported successfully')"
```

If there are no error messages, installation was successful.

## SAM-HQ Model Setup

SAM processing features require pretrained model files (`.pth`). **These files are large and not included in the code repository - you need to download them yourself.**

### Download Model Files
* Visit the **SAM-HQ official repository**: [https://github.com/SysCV/sam-hq](https://github.com/SysCV/sam-hq)
* **Important**: These are **SAM-HQ** models, not original SAM models!

### Available SAM-HQ Model Versions
Based on supported model types in the code:

* **`vit_h`**: ViT-H SAM-HQ model (largest and most accurate but slowest, recommended for high-quality requirements)
* **`vit_l`**: ViT-L SAM-HQ model (balanced performance and speed)
* **`vit_b`**: ViT-B SAM-HQ model (smaller and faster)

### Model Download Instructions
1. **Create Model Directory**: Create a folder in the SAM-HQ project directory to store models
   ```bash
   mkdir pretrained_checkpoint
   ```

2. **Download Model Files**:
   * Visit the Model Checkpoints section of the [SAM-HQ GitHub repository](https://github.com/SysCV/sam-hq)
   * The repository provides multiple download links, including official links and Hugging Face mirrors
   * Choose the appropriate model version based on your needs and computer configuration

### Model File Management
* After downloading, save the `.pth` files to an easily accessible location on your computer (recommended: store in the `pretrained_checkpoint` folder)
* When running the tool, click the "Select SAM Model File (.pth)" button in the appropriate tab to tell the program where you've placed the model files
* **File Naming Examples**:
  - `sam_hq_vit_h.pth` (ViT-H version)
  - `sam_hq_vit_l.pth` (ViT-L version)
  - `sam_hq_vit_b.pth` (ViT-B version)

## Interface Guide

The main program interface contains four tabs, each with specific functionality:

### Tab 1: Aggregation Analysis

Used for analyzing pixel aggregation value distribution characteristics of single images.

#### Operation Steps:
1. **Select Image**: Click "Select Image" to choose the image you want to analyze
2. **Start Analysis**: Click "Analyze Aggregation Distribution" to begin analysis
3. **View Results**: A histogram of aggregation value distribution will appear on the right, with statistics displayed on the left

#### Feature Details:
* **Aggregation Calculation**: Aggregation = 100 - (grayscale_value / 255.0) × 100.0, range 0-100
* **Distribution Chart**:
    - **X-axis**: Aggregation values (0-100), higher values represent darker pixels
    - **Y-axis**: Pixel count
    - **Red dots**: Automatically identified peak positions
    - **Orange dots**: Automatically identified trough positions
* **Statistical Information**:
    - Total pixel count
    - Mean/minimum/maximum aggregation values
    - Specific coordinates of distribution inflection points
* **Export Functions**:
    - "Save Chart": Save analysis charts as PNG/PDF/SVG formats
    - "Clear Chart": Clear current chart

### Tab 2: Mask Analysis

Provides interactive single-image mask generation and analysis functionality.

#### Operation Steps:
1. **Step 1 - Setup**:
   - Select image file
   - Select SAM-HQ model file (.pth)
2. **Step 2 - Generate Masks**:
   - Click "Generate Masks" to start generation (first time may be slow)
3. **Step 3 - Analyze & Filter**:
   - Click on masks in the image to select them (left-click to select/deselect, right-click to deselect)
   - Adjust filtering parameters
   - Click "Calculate Selected Masks" to analyze selected masks
4. **Step 4 - View Results**:
   - Review calculation results
   - Export results or save marked images

#### Interactive Features:
* **Mask Selection**:
    - Left-click masks in the image to select/deselect
    - Right-click to directly deselect
    - Selected masks display with red borders and numbered annotations
    - Unselected masks show colored thin borders
* **Batch Operations**:
    - "Select All": Select all masks
    - "Deselect All": Clear all selections
* **Filtering Parameters**: Uses the same filtering logic as batch processing

### Tab 3: Batch Processing

Core batch image processing functionality.

#### File Selection:
* **Input Folder**: Select folder containing images to process (supports recursive subfolder search)
* **Model File**: Select SAM-HQ model weight file (.pth)

#### SAM Processing Two-Stage Filtering Explained:
Combines the Segment Anything Model (AI auto-segmentation) with aggregation analysis. It first automatically identifies individual regions in images, then filters for your targets using your specified conditions (in two stages).

**Stage 1: Regional Property Filtering (A/I/R Thresholds)**:
* Calculate **initial** area, intensity, and ratio for each region found by SAM
* If you've enabled corresponding options, the program checks whether these initial values meet your minimum thresholds
* Any region with **subpar** initial values is **immediately eliminated** and doesn't proceed to the next stage

**Stage 2: Pixel-Level Aggregation Filtering (Min/Max Aggregate Thresholds)**:
* For regions that **passed Stage 1**, examine **every pixel** within them
* Only pixels with aggregation values falling within your specified [minimum, maximum] **range** count as "valid pixels" for that region
* If a region passed Stage 1 but has **no "valid pixels"** after Stage 2 checking, that region **won't be marked or calculated in the final results**

#### Filter Parameter Settings:
* **Stage 1 Filtering**:
    - **Min Area (A)**: Minimum area threshold (pixel count)
    - **Min Intensity (I)**: Minimum total aggregation value threshold
    - **Min Ratio (R=I/A)**: Minimum ratio threshold
    - **Aggregation Range**: Aggregation value range filter (Min-Max)

* **Stage 2 Filtering**:
    - Only pixels within the aggregation range participate in final calculations
    - Complete mask contours are retained for visualization

#### Operation Buttons:
* **Start Processing**: Begin batch processing
* **Stop Processing**: Stop current processing
* **Open Output**: Open output folder
* **Export Data**: Export all numerical data
* **View Output**: View the last processed image

### Tab 4: SAM-HQ Parameter Configuration

Detailed configuration of all SAM-HQ model parameters.

#### Main Parameter Categories:

**🎯 Basic Detection Settings**
* **Points Per Side**: Imagine spreading a grid across the image to find objects - this value is the number of points along each side of the grid. **Higher values mean denser grids**, making it easier to find small objects, but also **slower and more memory/VRAM intensive**
* **Points Per Batch**: Determines how many grid points the GPU processes at once. **Mainly affects VRAM**. If you encounter **"Out of Memory" errors, reduce this value**

**🎛️ Quality Control**
* **Pred IoU Thresh**: SAM's confidence score for regions it finds (0-1). **Higher values are more demanding**, resulting in less noise but potentially missing some targets
* **Stability Score Thresh**: Measures how stable the found region boundaries are. **Higher values require more stable, clearer boundaries**
* **Stability Score Offset**: Internal parameter used in stability score calculation

**🔄 Post-processing Settings**
* **Box NMS Thresh**: Prevents the same object from getting multiple boxes. **Lower values are more aggressive** at removing duplicates
* **Min Mask Region Area**: Setting a value **greater than 0** will automatically remove all tiny segmentation results smaller than this pixel count

**✂️ Large Image Processing**
* **Crop N Layers**: "Tiling" switch for processing very large images. **0 means no tiling**, **setting to 1 or higher** will split large images into many overlapping small tiles for separate processing
* **Crop-related Parameters**: Control tiling overlap and merging strategies

**💾 Output Format**
* **Output Mode**: Choose mask output format
  - `binary_mask`: Standard format, easiest to use (recommended)
  - `uncompressed_rle`: Compressed format, saves space
  - `coco_rle`: Special compressed format for advanced users

#### Usage Tips:
* Parameter changes must be followed by clicking "Apply Changes" to take effect
* "Reset to Defaults" restores default settings
* Right panel provides detailed parameter descriptions and usage recommendations

## System Monitoring

The system monitoring panel at the bottom of the tool displays real-time:
* **CPU Usage**: Processor utilization
* **RAM Usage**: Memory utilization and specific values
* **GPU Usage**: NVIDIA GPU utilization (requires nvidia-ml-py)
* **VRAM Usage**: Video memory utilization and specific values

## Configuration File Management

### Save Configuration
Click "Save Configuration" to save all current parameters as a JSON file, including:
* Filter parameters (area, intensity, ratio, aggregation ranges)
* SAM-HQ model parameters
* All tab settings

### Load Configuration
Click "Load Configuration" to restore all parameter settings from a previously saved JSON file.

## Output File Organization

### Batch Processing Output
Output folder naming convention:
```
[original_folder_name]_SAM_HQ_Output_[A=value_I=value_R=value_Agg=range]
```

### File Types
* **Processed Images**: Original images with filtered mask contours and annotations
* **Numerical Data Files**: `[folder_name]_SAM_HQ_Numerical_Data.txt`
* **Analysis Charts**: Charts generated from aggregation analysis

## How to Use

1. **Preparation**: Ensure all dependencies are installed and SAM-HQ models are downloaded
2. **Activate Environment**: Open the `sam_hq_env` environment Terminal through Anaconda Navigator
3. **Run**: In the project root directory, run the main script:
   ```bash
   python sam_hq_process.py
   ```
4. **Interface Operation**: When the program opens, the left side is the control area (select functions, files, adjust parameters), the right side is the log area (view processing information and errors)
5. **Select Function**: Click the tabs at the top to switch functions
6. **Setup and Run**:
   * Click "Select Input Folder" to choose the image folder to process
   * Click "Select SAM Model File (.pth)" to choose the model file
   * Carefully adjust various parameters in the interface
   * Click the appropriate processing button to start
7. **View Results**:
   * Pay attention to the output in the right log area during processing
   * After completion, click "Open Output Folder" to directly open the results folder

## Technical Limitations & Notes

* **Maximum Image Size**: Single edge not exceeding 4096 pixels
* **Memory Requirements**: Processing large images may require substantial memory
* **GPU Acceleration**: Supports CUDA acceleration, NVIDIA graphics cards recommended
* **Processing Time**: Initial model loading is slow, subsequent processing speed depends on image size and parameter settings
* **First-time Model Loading**: Or when processing particularly large images, may take some time - please be patient
* **Disk Space**: Ensure your computer has sufficient disk space to save processed images and exported data files

## Troubleshooting

### Common Issues
1. **"segment_anything library not found"**: Ensure SAM-HQ library is correctly installed
2. **GPU monitoring shows "N/A"**: Check NVIDIA driver and nvidia-ml-py installation
3. **Charts not displaying**: Ensure matplotlib is correctly installed
4. **TIFF files won't load**: Ensure tifffile library is installed
5. **pycocotools installation fails**: On Windows, try `pip install pycocotools-windows` first

### Performance Optimization Tips
* For large batch processing, GPU acceleration is recommended
* Adjust "Points Per Side" parameter to balance accuracy and speed
* Large images can enable "Crop N Layers" functionality
* Regularly clean output folders to avoid disk space issues

## Contributing

We welcome contributions of all kinds!

### Report Issues
* Report bugs on the GitHub [Issues](https://github.com/CUIAOYU/Useful-Image-Analysis-Tool-Integration/issues) page
* Provide detailed reproduction steps, screenshots, and system information

### Contribute Code
* Fork the repository and create feature branches
* Follow existing code style
* Test before submitting Pull Requests

### Improve Documentation
* Enhance usage instructions
* Add more examples
* Translate to other languages

## License

This project uses an open source license. Please see the LICENSE file for details. 
