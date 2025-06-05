# ❗COOLANT 改进复现项目

本项目为 [Wang et al., 2023] 提出的多模态虚假信息检测模型 **COOLANT**（Cross-modal Contrastive Learning for Multimodal Fake News Detection）的复现与优化版本。  
原始代码仓库地址：[https://github.com/ictalab/COOLANT](https://github.com/wishever/COOLANT)

在复现实验中我们发现原始代码在 Weibo 数据集上未能复现论文中的最佳性能，因此我们对数据处理流程进行了改进。同时，扩展至 **CFND 数据集**。

---

## 📁 文件结构
```
COOLANT/
├── weibo/              # Weibo 数据集主程序
│   ├── weibo.py        # 主训练脚本
│   └── save_features.py# BERT特征提取脚本
├── cfnd/               # CFND 数据集主程序
│   ├── cfnd.py
│   └── save_features.py
├── Data/               # 存放 Weibo 补充数据（需用户手动补全）
│   └── weibo/
│       ├── dataformat.txt
│       ├── w2v.pickle
│       ├── word_embedding.pickle
│       └── stop_words.txt
├── CFND/               # CFND 原始数据文件夹 + 必要补充文件（同 Weibo 所需）
│   ├── [原始 CFND 数据]
│   ├── dataformat.txt
│   ├── w2v.pickle
│   ├── word_embedding.pickle
│   └── stop_words.txt
```
**说明：**  
`Data/` 和 `CFND/` 均为下载获得的原始数据集目录，仅需将必要补充文件放置在其下即可完成运行准备。

---

## 📦 数据集准备

### 🧾 Weibo 数据集

- **部分数据集来源：**  
  https://github.com/yaqingwang/EANN-KDD18/tree/master/data

- **完整数据集下载（推荐）：**  
  [Google Drive 链接 (约1.3GB)](https://drive.google.com/file/d/14VQ7EWPiFeGzxp3XC2DeEHi-BEisDINn/view?usp=sharing)

- **补充文件（来源于部分数据集处, 放入 `Data/weibo/`）：**
  - `dataformat.txt`
  - `w2v.pickle`
  - `word_embedding.pickle`
  - `stop_words.txt`

---

### 🧾 CFND 数据集

- **下载链接：**  
  [Google Drive](https://drive.google.com/file/d/1J4rcWcVavTY5GGw29ZBr17bdjyBmTHpE/view?usp=drive_link)

- **补充文件（同 Weibo 数据集，放入 `CFND/` 文件夹）：**
  - `dataformat.txt`
  - `w2v.pickle`
  - `word_embedding.pickle`
  - `stop_words.txt`

---

## 🚀 快速开始

### ✅ 训练与测试 Weibo 数据集

```bash
# Step 1: 提取 BERT 特征（可选，仅首次需要）
python weibo/save_features.py

# Step 2: 模型训练与评估
python weibo/weibo.py
```
### ✅ 训练与测试 CFND 数据集

```bash
# Step 1: 提取 BERT 特征（可选，仅首次需要）
python cfnd/save_features.py

# Step 2: 模型训练与评估
python cfnd/cfnd.py
```
## 引用文献
### COOLANT 原始论文：
```
@inproceedings{10.1145/3581783.3613850,
  author = {Wang, Longzheng and Zhang, Chuang and Xu, Hongbo and Xu, Yongxiu and Xu, Xiaohan and Wang, Siqi},
  title = {Cross-Modal Contrastive Learning for Multimodal Fake News Detection},
  booktitle = {Proceedings of the 31st ACM International Conference on Multimedia},
  year = {2023},
  pages = {5696–5704}
}
```
### CFND 数据集（NLIN）论文：
```
@article{NLIN,
  title     = {Natural Language-centered Inference Network for Multi-modal Fake News Detection},
  author    = {Zhang, Qiang and Liu, Jiawei and Zhang, Fanrui and Xie, Jingyi and Zha, Zheng-Jun},
  booktitle = {Proceedings of the Thirty-Third International Joint Conference on Artificial Intelligence (IJCAI-24)},
  pages     = {2542--2550},
  year      = {2024}
}
```
