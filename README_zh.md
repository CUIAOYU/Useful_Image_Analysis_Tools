# SAM-HQ 图像批量处理与分析工具

这是一个使用 PyQt6 构建的图形用户界面工具，目的是为了简化和加速科研或实验中常见的图像分析与处理任务，特别是针对需要批量处理大量图像（如显微照片、组织切片扫描图等）的场景。它结合了 SAM-HQ (Segment Anything in High Quality) 模型的强大分割能力和自定义的图像分析算法，可以帮你自动提取定量信息、比较图像特征或改善图像视觉效果。

## 主要功能

本工具包含**四个核心功能模块**，通过标签页进行组织：

### 1. 聚合分析 (Aggregation Analysis)
* **聚合度定义**：我们定义了一个叫"聚合度"的指标 (0-100)，用来衡量像素有多"暗"。**颜色越深，聚合度越高**。
    - **计算公式**：`聚合度 = 100 - (灰度值 / 255.0) × 100.0`
    - **举例说明**：纯黑像素聚合度是 100，纯白像素聚合度是 0
* **单图像聚合值分布分析**：选择单张图片，工具可以分析其所有像素的"聚合值"分布情况，并生成直方图。
* **关键节点自动识别**：自动在分布图上识别并标注出波峰(Peaks)和波谷(Troughs)，帮助你理解图像内容的灰度分布特征。
* **详细统计数据**：提供总像素数、平均/最小/最大聚合值等关键统计信息。
* **图表导出功能**：可以将生成的分析图表保存为 PNG, PDF 或 SVG 格式。

### 2. 掩码分析 (Mask Analysis) 
* **交互式掩码生成**：使用SAM-HQ模型为单张图像生成所有可能的掩码。
* **可视化掩码选择**：在图像上直接点击选择/取消选择掩码，支持左键选择、右键取消。
* **实时过滤与计算**：对选定的掩码应用自定义过滤参数，实时计算面积、强度和比率。
* **结果导出**：可导出分析结果为文本文件，或保存带有标记的图像。
* **批量选择操作**：提供"全选"和"全不选"功能，便于批量操作。

### 3. 批量处理 (Batch Processing)
* **智能文件遍历**：选择一个文件夹，自动找出所有图片（包括子文件夹），支持 `.png`, `.jpg`, `.jpeg`, `.tif`, `.tiff` 格式。
* **SAM-HQ 模型支持**：加载SAM-HQ预训练模型 (`.pth` 文件)，全自动生成超高质量的图像掩码。
* **两阶段过滤系统**：
    * **阶段一：区域属性过滤**：根据面积(Area)、总聚合值(Intensity)和比率(Ratio = Intensity/Area)进行初步筛选。
    * **阶段二：像素聚合值过滤**：基于像素级别的聚合值范围进行更精细的筛选。
* **智能输出管理**：处理后的图像自动保存到按参数命名的输出文件夹，避免文件覆盖。
* **数据导出**：所有筛选通过的掩码数据可导出为结构化的 `.txt` 文件。
* **实时进度监控**：提供详细的处理进度条和日志信息。

### 4. SAM-HQ 参数配置
* **完整参数控制**：提供SAM-HQ自动掩码生成器的所有可调参数。
* **参数说明文档**：每个参数都有详细的中文说明和使用建议。
* **预设管理**：支持重置为默认值和应用更改到所有功能模块。

### 通用功能
* **配置管理**：可以**保存/加载**所有过滤参数和SAM-HQ模型参数到JSON文件。
* **系统监控**：实时显示CPU、内存、GPU及显存的使用率（需安装相应库）。
* **多格式支持**：支持常见图像格式，包括TIFF文件的特殊处理。
* **错误处理**：完善的错误处理和用户提示机制。
* **批量处理与易用性**：主要功能都支持一次处理整个文件夹的图片。
* **图形界面**：基于 PyQt6 开发的直观界面，选文件、设参数、看日志、打开结果文件夹都方便。

## 系统要求

* **操作系统**：推荐 Windows 10 或 Windows 11
* **Anaconda**：需要预先安装 Anaconda Navigator
* **Python 版本**：推荐使用 **Python 3.11**
* **硬件要求**：
    * 如果希望使用 SAM 处理功能并获得较快速度 (GPU 加速)，建议配备 **NVIDIA 显卡**，至少是 GeForce RTX 3060 或同等性能水平
    * 如果没有 NVIDIA 显卡，SAM 功能仍然可以在 CPU 上运行，但速度会**慢很多**
    * 其他非 SAM 功能对硬件要求不高

### 核心依赖库
* **PyQt6** - 图形界面框架
* **PyTorch** - 深度学习框架，支持SAM-HQ模型
* **segment_anything** - SAM-HQ核心库
* **OpenCV (opencv-python)** - 图像处理
* **NumPy** - 数值计算
* **Pillow** - 图像文件读写
* **matplotlib** - 图表绘制（聚合分析功能必需）
* **pycocotools** - 掩码数据处理

### 可选依赖库（推荐安装）
* **psutil** - CPU和内存监控
* **nvidia-ml-py (pynvml)** - NVIDIA GPU监控
* **tifffile** - TIFF文件支持

## 安装指南

### 第 1 步：安装 Anaconda

如果你电脑上还没有安装 Anaconda，请前往 [Anaconda Distribution 官网](https://www.anaconda.com/products/distribution) 下载适合你 Windows 系统的安装包。下载后运行安装程序，按照提示完成安装。对于大多数选项，保持默认设置即可。

### 第 2 步：创建新的工作环境

环境就像一个独立的"房间"，我们在这个"房间"里安装这个工具需要的所有软件库，这样可以避免和你电脑上其他 Python 项目的库发生冲突。

1. **打开 Anaconda Navigator**：在 Windows 开始菜单中找到并启动 "Anaconda Navigator"。它第一次启动可能需要一些时间。

2. **进入环境管理界面**：在 Navigator 窗口的左侧菜单栏中，点击 **"Environments"** 选项卡。

3. **创建新环境**：
   * 在环境列表（初始可能只有 "base (root)"）的下方，找到并点击 **"Create"**（带有加号图标）按钮
   * 会弹出一个名为 "Create new environment" 的小窗口
   * 在 **"Name"** 输入框中，为你的新环境起一个名字，例如 `sam_hq_env`
   * 在下方的 "Packages" 部分，确保 **Python** 被选中。点击右侧的版本号下拉菜单，选择 **3.11**
   * 确认名称和 Python 版本无误后，点击窗口右下角的 **"Create"** 按钮
   * Anaconda 会开始创建环境并安装指定版本的 Python 及一些基础包。请耐心等待，这可能需要几分钟

### 第 3 步：安装 PyTorch（重要）

1. **打开环境专属的命令行终端**：
   * 在 Anaconda Navigator 的 "Environments" 界面，从左侧列表中**点击选中**你刚刚创建的 `sam_hq_env` 环境（确保它处于高亮状态）
   * 在中间区域顶部，环境名称 `sam_hq_env` 的右侧，你会看到一个**绿色的播放按钮 (▶️)** 或者一个**向下的小箭头 (🔽)**。点击这个按钮/箭头
   * 在弹出的菜单中，选择 **"Open Terminal"**
   * 系统会打开一个**黑色背景的命令行窗口**。你会注意到窗口的提示符最前面带有 `(sam_hq_env)` 字样，这表示你当前的操作都是在这个独立的环境中进行的

2. **安装 PyTorch**：
   * **重要提示**：PyTorch 的安装与你的硬件（特别是显卡）和所需的 CUDA 版本紧密相关
   * 打开你的网页浏览器，访问 PyTorch 官方网站：https://pytorch.org/
   * 在官网首页找到 "Get Started" 或类似的安装指引区域
   * 根据你的情况进行选择：
     - **PyTorch Build**：通常选择 **Stable**（稳定版）
     - **Your OS**：选择 **Windows**
     - **Package**：选择 **Pip**
     - **Language**：选择 **Python**
     - **Compute Platform**：
       * 如果你的电脑装有 **NVIDIA 显卡** 并且希望使用 GPU 加速（推荐），请选择一个 **CUDA** 版本（如 CUDA 11.8 或 12.1）
       * 如果你的电脑**没有 NVIDIA 显卡**，或者你不确定，或者你只想使用 CPU 进行计算，请选择 **CPU**
   * 当你完成以上选择后，页面下方会**自动生成**一条安装命令
   * **完整地复制这条生成的命令**，回到命令行窗口粘贴并执行

### 第 4 步：安装 SAM-HQ 核心库

在同一个命令行窗口中，执行以下命令：

```bash
git clone https://github.com/SysCV/sam-hq.git
cd sam-hq
pip install -e .
```

**注意**：这将安装 SAM-HQ 版本的 segment_anything 库，它会覆盖原始的 segment_anything 库。

### 第 5 步：安装其他必需库

在同一个命令行窗口中，执行以下命令：

```bash
pip install PyQt6 opencv-python numpy Pillow matplotlib psutil tifffile
```

### 第 6 步：安装可选库

```bash
# GPU 监控库（仅适用于 NVIDIA 显卡用户）
pip install nvidia-ml-py

# COCO 工具库（用于某些掩码格式）
pip install pycocotools
```

**安装提示与常见问题**：
* **pycocotools**：在 Windows 上安装可能会因为缺少 C++ 编译环境而失败。如果遇到关于 `cl.exe` 或 `Microsoft C++ Build Tools` 的错误，可以尝试：
  ```bash
  pip install pycocotools-windows
  ```
* **nvidia-ml-py**：如果你没有 NVIDIA 显卡，可以跳过这个包
* **tifffile**：如果不安装，将无法读取 `.tif` 或 `.tiff` 格式的图像

### 第 7 步：验证安装

在命令行中运行以下 Python 代码来验证关键库是否正确安装：

```bash
python -c "import torch; print('PyTorch version:', torch.__version__); print('CUDA available:', torch.cuda.is_available())"
python -c "import segment_anything; print('SAM-HQ library imported successfully')"
python -c "import PyQt6; print('PyQt6 imported successfully')"
```

如果没有错误信息，说明安装成功。

## SAM-HQ 模型设置

SAM 处理功能需要预训练的模型文件 (`.pth`)。**这些文件很大，代码仓库里不带，需要你自己下载。**

### 下载模型文件
* 访问 **SAM-HQ 官方仓库**：[https://github.com/SysCV/sam-hq](https://github.com/SysCV/sam-hq)
* **重要**：这里是 **SAM-HQ** 的模型，不是原始 SAM 的模型！

### 可用的 SAM-HQ 模型版本
根据代码支持的模型类型：

* **`vit_h`**：ViT-H SAM-HQ 模型（最大最准但也最慢，推荐用于高质量需求）
* **`vit_l`**：ViT-L SAM-HQ 模型（中等性能和速度平衡）
* **`vit_b`**：ViT-B SAM-HQ 模型（较小较快）

### 模型下载说明
1. **创建模型文件夹**：在 SAM-HQ 项目目录下创建文件夹存放模型
   ```bash
   mkdir pretrained_checkpoint
   ```

2. **下载模型文件**：
   * 访问 [SAM-HQ GitHub 仓库](https://github.com/SysCV/sam-hq) 的 Model Checkpoints 部分
   * 仓库提供了多种下载链接，包括官方链接和 Hugging Face 镜像
   * 根据你的需求和电脑配置选择合适的模型版本

### 模型文件管理
* 下载后，把 `.pth` 文件存到你电脑上一个好找的地方（建议存放在 `pretrained_checkpoint` 文件夹中）
* 运行工具时，在相应的标签页点击 "Select SAM Model File (.pth)" 按钮，告诉程序你把模型文件放在哪里了
* **文件命名示例**：
  - `sam_hq_vit_h.pth`（ViT-H 版本）
  - `sam_hq_vit_l.pth`（ViT-L 版本）
  - `sam_hq_vit_b.pth`（ViT-B 版本）

## 界面操作详解

程序主界面包含四个标签页，每个都有特定的功能：

### 标签页 1: 聚合分析 (Aggregation Analysis)

用于分析单张图片的像素聚合值分布特征。

#### 操作步骤：
1. **选择图像**：点击"Select Image"选择要分析的图片
2. **开始分析**：点击"Analyze Aggregation Distribution"开始分析
3. **查看结果**：右侧会生成聚合值分布的直方图，左侧显示统计数据

#### 功能说明：
* **聚合值计算**：聚合值 = 100 - (灰度值 / 255.0) × 100.0，范围为0-100
* **分布图表**：
    - **X轴**：聚合值 (0-100)，数值越高代表像素越暗
    - **Y轴**：像素数量
    - **红点**：自动识别的波峰位置
    - **橙点**：自动识别的波谷位置
* **统计信息**：
    - 总像素数
    - 平均/最小/最大聚合值
    - 分布拐点的具体坐标
* **导出功能**：
    - "Save Chart"：保存分析图表为PNG/PDF/SVG格式
    - "Clear Chart"：清除当前图表

### 标签页 2: 掩码分析 (Mask Analysis)

提供交互式的单图像掩码生成和分析功能。

#### 操作步骤：
1. **Step 1 - 设置**：
   - 选择图像文件
   - 选择SAM-HQ模型文件(.pth)
2. **Step 2 - 生成掩码**：
   - 点击"Generate Masks"开始生成（首次可能较慢）
3. **Step 3 - 分析与过滤**：
   - 在图像上点击掩码进行选择（左键选择/取消，右键取消选择）
   - 调整过滤参数
   - 点击"Calculate Selected Masks"分析选定的掩码
4. **Step 4 - 查看结果**：
   - 查看计算结果
   - 导出结果或保存标记图像

#### 交互功能：
* **掩码选择**：
    - 在图像上左键点击掩码可选择/取消选择
    - 右键点击可直接取消选择
    - 选中的掩码显示为红色边框并标注编号
    - 未选中的掩码显示为彩色细边框
* **批量操作**：
    - "Select All"：选择所有掩码
    - "Deselect All"：取消所有选择
* **过滤参数**：与批量处理使用相同的过滤逻辑

### 标签页 3: 批量处理 (Batch Processing)

核心的批量图像处理功能。

#### 文件选择：
* **Input Folder**：选择包含待处理图片的文件夹（支持子文件夹递归）
* **Model File**：选择SAM-HQ模型权重文件(.pth)

#### SAM 处理的两步筛选详解：
把 Segment Anything Model (AI 自动分割) 和聚合度分析结合起来。它先自动找出图里的各个独立区域，然后用你设定的条件（分两步）筛选出你关心的目标。

**第一步：按区域整体属性筛选 (A/I/R Threshold)**：
* 先算每个 SAM 找到的区域的**初始**面积、强度、比率
* 如果你勾选了相应选项，程序会检查这些初始值是否达到你设置的最小阈值
* 任何一个初始值**不达标**的区域，**直接淘汰**，不进入下一步

**第二步：按像素聚合度筛选 (Min/Max Aggregate Threshold)**：
* 对于**通过第一步**的区域，再检查它里面的**每一个像素**
* 只有聚合度落在你设定的 [最低值, 最高值] **区间内**的像素，才算这个区域的"有效像素"
* 如果一个区域通过了第一步，但在第二步检查后**一个"有效像素"都没有**，那这个区域**最终也不会被标记和计算**

#### 过滤参数设置：
* **阶段一过滤**：
    - **Min Area (A)**：最小面积阈值（像素数）
    - **Min Intensity (I)**：最小总聚合值阈值
    - **Min Ratio (R=I/A)**：最小比率阈值
    - **Aggregation Range**：聚合值范围过滤（Min-Max）

* **阶段二过滤**：
    - 只有在聚合值范围内的像素才参与最终计算
    - 但保留完整的掩码轮廓用于可视化

#### 操作按钮：
* **Start Processing**：开始批量处理
* **Stop Processing**：停止当前处理
* **Open Output**：打开输出文件夹
* **Export Data**：导出所有数值数据
* **View Output**：查看最后处理的图像

### 标签页 4: SAM-HQ 参数配置

详细配置SAM-HQ模型的所有参数。

#### 主要参数分类：

**🎯 基础检测设置**
* **Points Per Side**：想象在图上撒网格找物体，这个值是网格每边的点数。**值越高，网格越密**，更容易找到小东西，但也**更慢、更吃内存/显存**
* **Points Per Batch**：决定了显卡 (GPU) 一次处理多少个网格点。**主要影响显存**。如果遇到 **"Out of Memory" 错误，就降低这个值**

**🎛️ 质量控制**
* **Pred IoU Thresh**：SAM 对自己找到的区域打的分数 (0-1)。**值越高，要求越严**，结果噪点少，但可能漏掉一些目标
* **Stability Score Thresh**：衡量找到的区域边界稳不稳固。**值越高，要求边界越稳定清晰**
* **Stability Score Offset**：计算稳定性分数时用的内部参数

**🔄 后处理设置**
* **Box NMS Thresh**：防止同一个物体被画上好几个框。**值越低，去重越狠**
* **Min Mask Region Area**：设一个**大于 0** 的值可以自动删掉所有面积小于这个像素数的、细小的分割结果

**✂️ 大图像处理**
* **Crop N Layers**：处理超大图片的"切块"开关。**0 表示不切**，**设为 1 或更大**会把大图切成很多重叠的小块分别处理
* **Crop相关参数**：控制裁剪重叠和合并策略

**💾 输出格式**
* **Output Mode**：选择掩码输出格式
  - `binary_mask`：标准格式，最容易使用（推荐）
  - `uncompressed_rle`：压缩格式，节省空间
  - `coco_rle`：高级用户专用的特殊压缩格式

#### 使用提示：
* 参数修改后必须点击"Apply Changes"才能生效
* "Reset to Defaults"可恢复默认设置
* 右侧面板提供详细的参数说明和使用建议

## 系统监控功能

工具底部的系统监控面板实时显示：
* **CPU Usage**：处理器使用率
* **RAM Usage**：内存使用率和具体数值
* **GPU Usage**：NVIDIA GPU使用率（需要nvidia-ml-py）
* **VRAM Usage**：显存使用率和具体数值

## 配置文件管理

### 保存配置
点击"Save Configuration"可将当前所有参数保存为JSON文件，包括：
* 过滤参数（面积、强度、比率、聚合值范围）
* SAM-HQ模型参数
* 所有标签页的设置

### 加载配置
点击"Load Configuration"可从JSON文件恢复之前保存的所有参数设置。

## 输出文件组织

### 批量处理输出
输出文件夹命名规则：
```
[原文件夹名]_SAM_HQ_Output_[A=值_I=值_R=值_Agg=范围]
```

### 文件类型
* **处理后图像**：带有筛选掩码轮廓和标注的原图
* **数值数据文件**：`[文件夹名]_SAM_HQ_Numerical_Data.txt`
* **分析图表**：聚合分析生成的图表文件

## 如何使用

1. **准备**：确保依赖库都装好了，SAM-HQ 模型也下载好了
2. **激活环境**：通过 Anaconda Navigator 打开 `sam_hq_env` 环境的 Terminal
3. **运行**：在项目文件夹根目录，运行主脚本：
   ```bash
   python sam_hq_process.py
   ```
4. **界面操作**：程序打开后，左边是控制区（选功能、选文件、调参数），右边是日志区（看运行信息和报错）
5. **选功能**：点顶部的标签切换功能
6. **设置和运行**：
   * 点 "Select Input Folder" 选要处理的图片文件夹
   * 点 "Select SAM Model File (.pth)" 选模型文件
   * 仔细调整界面上的各种参数
   * 点相应的处理按钮开始运行
7. **查看结果**：
   * 处理时留意右边日志区的输出
   * 处理完成后点 "Open Output Folder" 按钮可以直接打开结果文件夹

## 技术限制与注意事项

* **最大图像尺寸**：单边不超过4096像素
* **内存要求**：处理大图像时可能需要较大内存
* **GPU加速**：支持CUDA加速，建议使用NVIDIA显卡
* **处理时间**：首次加载模型较慢，后续处理速度取决于图像大小和参数设置
* **第一次加载模型**：或者处理特别大的图片时，可能会花点时间，请耐心等待
* **硬盘空间**：确保你的电脑硬盘有足够的空间来保存处理完的图片和导出的数据文件

## 故障排除

### 常见问题
1. **"segment_anything library not found"**：确保正确安装了SAM-HQ库
2. **GPU监控显示"N/A"**：检查NVIDIA驱动和nvidia-ml-py安装
3. **图表不显示**：确保matplotlib已正确安装
4. **TIFF文件无法加载**：确保tifffile库已安装
5. **pycocotools 安装失败**：在 Windows 上优先试试 `pip install pycocotools-windows`

### 性能优化建议
* 对于大批量处理，建议使用GPU加速
* 调整"Points Per Side"参数平衡精度和速度
* 大图像可启用"Crop N Layers"功能
* 定期清理输出文件夹避免磁盘空间不足

## 贡献指南

欢迎各种形式的贡献！

### 报告问题
* 到GitHub的[Issues](https://github.com/CUIAOYU/Useful-Image-Analysis-Tool-Integration/issues)页面报告Bug
* 提供详细的复现步骤、截图和系统信息

### 贡献代码
* Fork仓库并创建feature分支
* 遵循现有代码风格
* 提交Pull Request前请先测试

### 改进文档
* 完善使用说明
* 添加更多示例
* 翻译为其他语言

## 许可证

本项目采用开源许可证，具体请查看LICENSE文件。
