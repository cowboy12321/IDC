# IDC  
This is our project management for Image Difference Captioning (IDC).  

## 简介  
Image Difference Captioning（IDC）是一个旨在生成描述两幅图像之间差异的自然语言任务。该任务在多个领域具有重要应用价值，例如：  
- **智能监控：** 实现场景变化检测与解释，提高视频监控的效率和智能化程度。  
- **医疗影像分析：** 提供直观的变化描述，辅助医生诊断。  
- **多模态内容生成：** 支持生成图文结合的解释内容，增强用户体验。  

该项目通过参考最新的研究文献，对现有方法进行复现和改进，同时结合实际需求（如独居老人跌倒检测报警），对模型进行微调和优化。最终目标是开发一个高效且可扩展的 IDC 框架，并将其应用于下游任务。  

---

## 参考文献分配  
**任务：** 根据各自文献总结模型思路  

- **分配给 user1：**  
   [http://arxiv.org/abs/2402.19119](http://arxiv.org/abs/2402.19119)  
   [https://doi.org/10.48550/arXiv.2312.02974](https://doi.org/10.48550/arXiv.2312.02974)  
   [https://aclanthology.org/2022.aacl-short.5](https://aclanthology.org/2022.aacl-short.5)  
   [https://doi.org/10.1109/CVPR46437.2021.00275](https://doi.org/10.1109/CVPR46437.2021.00275)  

- **分配给 user2：**  
   [http://arxiv.org/abs/2407.05645](http://arxiv.org/abs/2407.05645)  
   [https://doi.org/10.48550/arXiv.1808.10584](https://doi.org/10.48550/arXiv.1808.10584)  
   [http://arxiv.org/abs/2408.04594](http://arxiv.org/abs/2408.04594)  
   [https://doi.org/10.48550/arXiv.1901.02527](https://doi.org/10.48550/arXiv.1901.02527)  

- **分配给 user3：**  
   [https://doi.org/10.48550/arXiv.2309.16283](https://doi.org/10.48550/arXiv.2309.16283)  
   [https://doi.org/10.18653/v1/2021.findings-acl.6](https://doi.org/10.18653/v1/2021.findings-acl.6)  
   [https://doi.org/10.1609/aaai.v36i3.20218](https://doi.org/10.1609/aaai.v36i3.20218)  
   [https://openreview.net/forum?id=eiGs5VCsYM](https://openreview.net/forum?id=eiGs5VCsYM)  

### Architectural analysis
	每篇文章需要有对应的架构分析

---

## 任务复现  

**目标：**  
在各自文献中挑选出最合适一种的或者融合多种方法，复现分配文献中的关键模型，掌握实现细节并进行性能验证，为后续改进奠定基础。  

### 复现步骤：  
1. **环境配置：**  
   - 安装所需依赖项（ PyTorch）。  
   - 确保 GPU 环境可用，以支持大规模训练与推理。  

2. **代码获取与审查：**  
   - Git 仓库中有不同分支，用于各自任务的代码提交和调试。  

3. **模型复现：**  
   - 根据文献提供的模型架构和实验设置，逐步实现相关方法。  
   - 对照文献中的实验结果，验证复现是否准确。  

4. **日志与问题记录：**  
   - 使用日志记录工具（如 TensorBoard）监控训练进展。  
   - 汇总复现过程中遇到的问题及解决方案，便于后续总结与优化。  

---

## **代码提交提示设计**  

### **提交方式：**  
- **代码提交：**  
 建议使用 `GitHub Desktop` ：  

- **跳转方式：**  
- **user1：** 提交路径:[src/Reproduction/user1.py](src/Reproduction/user1.py)
- **user2：** 提交路径:[src/Reproduction/user2.py](src/Reproduction/user2.py)
- **user3：** 提交路径:[src/Reproduction/user3.py](src/Reproduction/user3.py)

### **提交说明：**   
1. **注释要求：**  
 代码需附注释，在开头处简要说明模型的框架和出处，便于组员理解和后续改进。  
2. **README 文档：**  
 在 `README.md`中说明模型架构、运行方法及依赖项。  

---  

## 数据自动标注生成  
**任务：** 基于任务复现的模型进行数据自动标注生成工作。  

---

## 围绕独居老人跌倒检测报警应用进行微调  
**任务：** 围绕独居老人跌倒检测报警这个应用对已做任务进行微调。  

---

## 共同商讨模型改进方向  
**任务：** 三人共同商讨出模型改进方向。  

---

## 搭建框架  
**任务：** 根据商讨出的模型改进方向一起搭建框架。  

---

## 分工实现框架  
**任务：**  
- **人员 a：** 负责框架中的一部分任务实现。  
- **人员 b：** 承担框架中的另一部分任务实现。  
- **人员 c：** 完成框架中剩余部分的任务实现。  

---

## 汇总应用到下游任务  
**任务：** 将汇总后的成果应用到下游任务中。  

---  

## 论文撰写  
**任务：** 整理成果并撰写论文。  
