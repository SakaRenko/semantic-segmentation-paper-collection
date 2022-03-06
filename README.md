# semantic-segmentation-paper-collection

## 2021
| Title | Venue | 关键词 | Backbone | PDF | CODE | 
| :-----|:-----:|:-----:|:---:|:---:|:----:|
|SegFormer: Simple and Efficient Design for SemanticSegmentation with Transformers|NeurIPS|stride设计,聚合token并用mlp降维提高效率，Mix-FNN，MLP decoder|Transformer|https://arxiv.org/pdf/2105.15203.pdf|https://github.com/NVlabs/SegFormer|
|Rethinking Semantic Segmentation from a Sequence-to-Sequence Perspective with Transformers|CVPR|首先使用transformer作为backbone，三种decoder设计，多阶段特征融合|Transformer|https://arxiv.org/abs/2012.15840|https://fudan-zvg.github.io/SETR/
|Per-Pixel Classification is Not All You Need for Semantic Segmentation|NeurIPS|分类logits和mask两条支路，使用Transformer Head得到N个proposal|Transformer head|https://papers.nips.cc/paper/2021/file/950a4152c2b4aa3ad78bdd6b366cc179-Paper.pdf|https://github.com/facebookresearch/MaskFormer
|FaPN: Feature-aligned Pyramid Network for Dense Image Prediction|ICCV|使用DCN对齐对应层次的特征|ResNet|https://arxiv.org/pdf/2108.07058v2.pdf|https://github.com/EMI-Group/FaPN
|Interlaced Sparse Self-Attention for Semantic Segmentation|IJCV|把注意力分解为远距离和局部两个部分，把两个部分结合从而得到稀疏的整体注意力|--|https://arxiv.org/pdf/1907.12273.pdf|--
|ISNet: Integrate Image-Level and Semantic-Level Context for Semantic Segmentation|ICCV|聚合图像整体和类别内部的相似度|https://arxiv.org/abs/2108.12382|https://github.com/SegmentationBLWX/sssegmentation



## 2020
| Title | Venue | 关键词 | Backbone | PDF | CODE | 
| :-----|:-----:|:-----:|:---:|:---:|:----:|
|SegFix: Model-Agnostic Boundary Refinement for Segmentation|ECCV|模型无关的refine方法，两个额外分支预测边缘和方向，和densecrf能一起使用|--|https://link.springer.com/chapter/10.1007%2F978-3-030-58610-2_29|https://github.com/openseg-group/openseg.pytorch
|Context Prior for Scene Segmentation|CVPR|使用gt生成affinity map直接监督attention map(由feature直接生成)，双分支分别融合类间和类内注意力，使用大卷积核和空间分离卷积|https://arxiv.org/pdf/2004.01547.pdf|https://github.com/ycszen/ContextPrior
Disentangled Non-Local Neural Networks|ECCV|注意到注意力可以被分为pair-wise和unary两种，pair-wise经过均值标准化，一定程度上解释了gcnet中为何attention相似，原因为unary过强，pair-wise学内部，unary学边缘，提出分出支路学unary，最后相加|resnet|https://arxiv.org/pdf/2006.06668.pdf|https://github.com/yinmh17/DNL-Semantic-Segmentation|
Deep High-Resolution Representation Learning for Visual Recognition|TPAMI|不同分辨率不同支路，类似group convolution，但是不同group分辨率不同|resnet|https://arxiv.org/pdf/1908.07919.pdf|https://github.com/HRNet

## 2019
| Title | Venue | 关键词 | Backbone | PDF | CODE | 
| :-----|:-----:|:-----:|:---:|:---:|:----:|
Dual Attention Network for Scene Segmentation|CVPR|position和channel双重注意力机制|ResNet|https://arxiv.org/pdf/1809.02983.pdf|github.com/junfu1115/DANet/
Asymmetric Non-local Neural Networks for Semantic Segmentation|ICCV|基于non-local的注意力机制，利用global pooling减少K和V的维度，可理解为为不同像素寻找同一个特征，此head用于融合特征以及处理多尺度|resnet|https://arxiv.org/pdf/1908.07678.pdf|https://github.com/MendelXu/ANN
GCNet: Non-local Networks Meet Squeeze-Excitation Networks and Beyond|ECCV|位置注意力转化为全局注意力，发现不同位置的attention几乎一样，与DANet和OCNet结果矛盾，为什么？作者认为：与任务和模块位置相关，对于分类任务而言，nl模块接近cls，因此不同位置att相近。与SENet类似。在分割上表现不算好，低于DANet。|resnet|https://arxiv.org/pdf/1904.11492.pdf|https://github.com/xvjiarui/GCNet
|Dynamic Multi-scale Filters for Semantic Segmentation|ICCV|多尺度分支，使用不同尺度的特征作为层分离动态卷积核参数达到适应不同尺度的效果|resnet|https://openaccess.thecvf.com/content_ICCV_2019/papers/He_Dynamic_Multi-Scale_Filters_for_Semantic_Segmentation_ICCV_2019_paper.pdf|https://github.com/Junjun2016/DMNet


## 2017
| Title | Venue | 关键词 | Backbone | PDF | CODE | 
| :-----|:-----:|:-----:|:---:|:---:|:----:|
|Rethinking Atrous Convolution for Semantic Image Segmentation|Arxiv|使用带孔卷积保持深层特征分辨率，使用ASPP来获取多尺度特征|resnet|https://arxiv.org/pdf/1706.05587.pdf|https://github.com/tensorflow/models/tree/master/research/deeplab
