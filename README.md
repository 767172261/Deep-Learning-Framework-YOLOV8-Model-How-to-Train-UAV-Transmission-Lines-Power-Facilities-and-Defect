无人机输电线电力设施及缺陷检测数据集

---![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/da7c5af1e07d4cc7ac1edf58a5d26aaa.png)


### 📊 无人机电力设施及缺陷检测数据集概览表

| **项目**                | **描述**                                                                 |
|------------------------|--------------------------------------------------------------------------|
| **数据集名称**          | 无人机电力设施及缺陷检测数据集                                            |
| **图像总数**            | 8,932 张                                                                 |
| **采集平台**            | 无人机（未指定型号，典型用于电力巡检）                                    |
| **应用场景**            | 输电线路巡检、电力设备缺陷识别、智能电网运维                              |
| **标注格式**            | ✅ **YOLO 格式**（每张图对应 `.txt` 文件）<br>✅ **PASCAL VOC 格式**（每张图对应 `.xml` 文件） |
| **类别总数**            | 19 类（含正常设备与缺陷类型）                                             |
| **总标注实例数**        | **62,705 个**（所有类别之和）                                             |
| **分辨率/尺寸**         | 未明确提供（典型为 1920×1080 或更高，适用于细节缺陷识别）                 |
| **数据划分建议**        | 可按 7:2:1 划分训练集/验证集/测试集（约 6252 / 1786 / 894 张）           |

---
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/2e9cf9c950c14b80ad092af62e8ccae8.png)

### 🏷️ 类别分布详情表

| **序号** | **英文类别名**                     | **中文名称**               | **标注数量** |
|---------|----------------------------------|--------------------------|------------|
| 1       | Insulator                        | 绝缘子                   | 3,408      |
| 2       | Broken Insulator Cap             | 破损的绝缘子帽           | 2,011      |
| 3       | Insulator Cap                    | 绝缘子帽                 | 22,722     |
| 4       | Cable                            | 电缆                     | 24,605     |
| 5       | Tower (Wooden)                   | 木塔                     | 875        |
| 6       | Transformer                      | 变压器                   | 317        |
| 7       | Front Insulator Cap              | 前绝缘子帽               | 4,386      |
| 8       | Tower (Lattice)                  | 格构塔                   | 1,011      |
| 9       | Tower (Tucohy)                   | Tucohy塔                 | 483        |
| 10      | Transmission Line                | 输电线路                 | 517        |
| 11      | Damaged Line                     | 损坏的线路               | 247        |
| 12      | Bus                              | 总线                     | 125        |
| 13      | Faulty Transformer               | 故障变压器               | 6          |
| 14      | Spacer between Dampers           | 阻尼器之间的间隔         | 51         |
| 15      | Broken Insulator                 | 破损的绝缘子             | 68         |
| 16      | Bent Insulator                   | 弯曲的绝缘子             | 20         |
| 17      | Object on TL                     | 输电线路上的物体         | 120        |
| 18      | Capacitor Bank                   | 电容器组                 | 106        |
| 19      | Damaged Splice                   | 损坏的接头               | 83         |
|         | **总计**                         |                          | **62,705** |

> 🔍 注：  
> - “Insulator Cap” 和 “Front Insulator Cap” 占比最高，合计超 27,000 个，反映其在电力杆塔中的密集出现。  
> - 缺陷类样本（如破损、弯曲、故障）数量较少，属于**长尾分布**，训练时建议采用**数据增强**或**重采样策略**。

---
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/11ff6c7587584cbfa2640b4c9872b980.png)

### 🗂️ 文件组织结构示例

```
power_inspection_dataset/
├── images/                     # 原始图像（.jpg / .png）
│   ├── img_0001.jpg
│   └── ...
├── labels_yolo/                # YOLO 格式标签（.txt）
│   ├── img_0001.txt
│   └── ...
├── annotations_voc/            # PASCAL VOC 格式标签（.xml）
│   ├── img_0001.xml
│   └── ...
└── classes.txt                 # 19 类名称列表（按索引顺序）
```

---

### ✅ 数据集优势

- **双标注格式支持**：兼容 YOLO 系列（v5/v8）与传统检测框架（如 Faster R-CNN + VOC）；
- **覆盖全面**：包含正常设备 + 多种典型缺陷，适用于**异常检测**与**状态评估**；
- **高实用价值**：可直接用于电力巡检 AI 系统开发、缺陷预警、自动化报告生成；
- **类别语义清晰**：命名规范，便于模型解释与后期维护。

---
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/c05fb285d7f24f56b670a106ffbd105c.png)
基于YOLOv8的电力设备检测系统，我们可以分步骤来进行：从准备环境、调用预训练模型、训练数据集到推理和用户界面。以下是一个简化版的指南和代码示例，旨在提供一个大致框架。

以下文字及代码仅供参考
### 1. 准备工作

首先，确保你的环境中安装了必要的库，比如`ultralytics`（YOLOv8所在库），你可以通过pip来安装：

```bash
pip install ultralytics
```

### 2. 加载预训练模型

接下来，我们加载YOLOv8的预训练模型。这里以YOLOv8n为例，你也可以选择其他大小的模型如YOLOv8s, YOLOv8m等。

```python
from ultralytics import YOLO

# 加载预训练模型
model = YOLO('yolov8n.pt')
```

### 3. 数据集准备

根据你的描述，你的数据集是已经标注好的，并且支持YOLO和VOC两种格式。对于YOLOv8，你需要将标签转换为YOLO格式（如果尚未完成）。假设你的数据集目录结构如下：

```
power_inspection_dataset/
├── images/
│   ├── img_0001.jpg
│   └── ...
├── labels_yolo/
│   ├── img_0001.txt
│   └── ...
```

创建一个YAML文件来定义数据集路径和类别信息：

```yaml
train: ./power_inspection_dataset/images/train
val: ./power_inspection_dataset/images/val

nc: 19 # 类别数量
names: ['Insulator', 'Broken Insulator Cap', 'Insulator Cap', ...] # 类别名称列表
```

### 4. 训练模型

使用指定的数据集和配置进行训练：

```python
model.train(data='path/to/your/dataset.yaml', epochs=100, imgsz=640)
```

### 5. 推理与验证

训练完成后，可以对新的图像进行推理：

```python
results = model.predict(source='path/to/new/image.jpg', save=True)
```

### 6. 构建用户界面（简易版本）

为了快速搭建一个界面，可以考虑使用Streamlit，它非常适合快速原型开发。

首先，安装Streamlit：

```bash
pip install streamlit
```

然后，创建一个简单的应用脚本：

```python
import streamlit as st
from ultralytics import YOLO

# 加载模型
model = YOLO('path/to/best_model.pt')

st.title("电力设备缺陷检测系统")

uploaded_file = st.file_uploader("选择一张图片", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    results = model.predict(source=uploaded_file.getvalue(), save=True)
    st.image(results[0].plot(), caption="检测结果")
```

运行这个应用：

```bash
streamlit run your_script.py
```


