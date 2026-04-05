# 数据集红色边界区域 (Interface Margin) 的细胞类型定义

在 `panel_g_*.py` 脚本绘制的空间相互作用图中，**红色的三角网状区域**代表了“肿瘤-微环境交界区”（Tumor-Microenvironment Interface），也就是肿瘤扩张/侵袭的前沿（Invasive Margin）。

其算法逻辑是：如果一个空间位点（Spot）处于“肿瘤细胞”主导，且它的相邻或周围位点有处于“基质细胞”或“免疫细胞”主导的；或是反过来，自身为间质或免疫但周围有肿瘤，该点就会被标记为界面点（Interface Spot）。然后程序对这些相互临近的点进行 Delaunay 三角剖分描点连线，最终形成了一道拦截在肿瘤与微环境之间的边界区。

为了防遗忘，下面是各个数据集中，被分配为 **Tumor (肿瘤)**、**Stromal (基质)** 以及 **Immune (免疫)** 三大阵营的具体细胞类型名称列表：

---

### 1. **GSE243275**
*脚本: `panel_g_gse243275.py`*
* 此切片展示了具有侵袭和导管原位特征的乳腺区域及其微环境。

- **Tumor (肿瘤)**: 
  - `DCIS 1`
  - `DCIS 2`
  - `Invasive Tumor`
  - `Prolif Invasive Tumor`
- **Stromal (基质)**: 
  - `Stromal` 
  - `CAFs` (若有)
- **Immune (免疫)**: 
  - `B Cells`
  - `CD4+ T Cells`
  - `CD8+ T Cells`
  - `Macrophages 1`
  - `Macrophages 2`

---

### 2. **CID44971**
*脚本: `panel_g_cid44971.py`*
* 这是乳腺癌图谱 (Wu et al.) 中的一片高分辨率空间切片。

- **Tumor (肿瘤/上皮)**: 
  - `Cancer Epithelial`
  - `Normal Epithelial`
- **Stromal (基质/成纤维)**: 
  - `CAFs`
- **Immune (免疫)**: 
  - `B-cells`
  - `Myeloid`
  - `Plasmablasts`
  - `T-cells`

---

### 3. **GSE211956 (P3 切片)**
*脚本: `panel_g_gse211956.py`*
* 该数据集细胞类型更为细分，尤其是成纤维细胞群体被划分为多个带标记基因的亚群。

- **Tumor (肿瘤)**: 
  - `Tumour cells`
- **Stromal (基质)**:包含任何名字里带有 `Fibro` 的细胞以及肌成纤维：
  - `Fibro1 (EIF4A3, STAR)`
  - `Fibro2 (RBP1, DCN)`
  - `Fibro3 (RAMP1, CFD)`
  - `Fibro5 (FN1, COL3A1)`
  - `Myofibroblasts`
- **Immune (免疫)**:包含巨噬细胞，T细胞和髓系：
  - `Macrophages`
  - `T cells`
  - `Myeloid cells` (以及其他带此类字符名字的细胞等)

---

### 📝 论文引用的备忘说明
**这段红色的网格线在论文里的核心主旨：**
红线是为了用来**佐证模型算法预测出的细胞通信具有高度的“拓扑空间聚集性”**。借由高分值的“相互作用灰线（Interaction）”频繁且密集地**横跨/贴合**在红线边界上（如 DLL4-NOTCH3 / PDGFD-PDGFRB等），能直观说服审稿人：该工具发现的通讯事件并非没有规律地洒满全切片，而是准确捕捉到了发生在边缘微环境中由肿瘤和间质/免疫相互拉扯而产生的具有显著病理特征的“真实细胞交互行为”。
