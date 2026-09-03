# 第二轮审稿补充实验命令

所有命令均在仓库的 `benchmark/` 目录下执行。脚本内部使用
`conda run`，无需预先切换 Conda 环境。为避免 PowerShell 续行符问题，下面全部使用
单行命令。

## 1. 无身份标签的任务内基线

训练 CSV 只有 `image` 列，因此主表不运行 Triplet、ArcFace、监督对比或 TransReID
微调。这些方法需要身份标签，强行使用会改变论文的问题设定。补充的任务内基线均以
同一图像的不同增强作为正样本：

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File scripts\run_self_supervised_baselines.ps1 -Models simsiam_r50,vicreg_r50,dino_vits16 -Phase all
```

建议先用 DryRun 检查：

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File scripts\run_self_supervised_baselines.ps1 -Models simsiam_r50,vicreg_r50,dino_vits16 -Phase all -DryRun
```

中断后可按阶段恢复，已经生成的检查点不会因只运行后续阶段而重新训练：

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File scripts\run_self_supervised_baselines.ps1 -Models simsiam_r50 -Phase train
powershell -NoProfile -ExecutionPolicy Bypass -File scripts\run_self_supervised_baselines.ps1 -Models simsiam_r50 -Phase extract
powershell -NoProfile -ExecutionPolicy Bypass -File scripts\run_self_supervised_baselines.ps1 -Models simsiam_r50 -Phase stamp
powershell -NoProfile -ExecutionPolicy Bypass -File scripts\run_self_supervised_baselines.ps1 -Models simsiam_r50 -Phase evaluate
```

模型权重保存到 `artifacts\checkpoints\<模型名>`，预训练权重缓存保存到
`artifacts\models`，最终指标保存到 `artifacts\results`。

## 2. 拓扑结构与 Backbone 消融

结构消融比较最大腐蚀深度 2、3、4，并保留“无腐蚀恒等映射”和“关闭 CBAM”
两个直接对照。配置中的 `num_erosion_levels` 包含未腐蚀的 $E_0$，所以
`levels_2` 和 `levels_4` 分别对应 3 张和 5 张层级特征图：

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File scripts\run_topology_ablations.ps1 -Variants levels_2,levels_4,operator_identity,cbam_off -Phase all
```

骨干实验比较 ResNet50、ConvNeXt-Tiny、SwinV2-Tiny 与 ViT-B/16：

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File scripts\run_topology_ablations.ps1 -Variants reference,backbone_convnext_tiny,backbone_swin_v2_t,backbone_vit_b_16 -Phase all
```

上述矩阵采用单因素设计，每个变体只改变一个因素；所有变体继续读取仅含 `image`
列的 CSV。不同骨干统一输出空间特征图，再使用相同的视觉头、拓扑分支、训练数据与
完整图库评估器。

## 3. 随机种子重复

主结果至少重复三个种子。种子 42 沿用无后缀产物，其他种子自动带
`_seed123` 或 `_seed3407` 后缀，不会覆盖已有结果：

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File scripts\run_topology_ablations.ps1 -Variants reference -Seed 42 -Phase all
powershell -NoProfile -ExecutionPolicy Bypass -File scripts\run_topology_ablations.ps1 -Variants reference -Seed 123 -Phase all
powershell -NoProfile -ExecutionPolicy Bypass -File scripts\run_topology_ablations.ps1 -Variants reference -Seed 3407 -Phase all
```

对最终选入论文的最强新基线执行同样的三个种子。例如 SimSiam：

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File scripts\run_self_supervised_baselines.ps1 -Models simsiam_r50 -Seed 42 -Phase all
powershell -NoProfile -ExecutionPolicy Bypass -File scripts\run_self_supervised_baselines.ps1 -Models simsiam_r50 -Seed 123 -Phase all
powershell -NoProfile -ExecutionPolicy Bypass -File scripts\run_self_supervised_baselines.ps1 -Models simsiam_r50 -Seed 3407 -Phase all
```

## 4. 分割缺损与成像鲁棒性

主鲁棒性实验只扰动 1075 张查询图像，图库保持干净。这样测到的是查询前景缺损导致
的性能变化，不会把查询和图库同时破坏而掩盖退化。

先比较本文无 ITA 模型和 MoCo V3+CBAM：

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File scripts\run_robustness.ps1 -Models single_topo_plain,moco_v3_cbam -Phase all
```

默认执行三档掩膜腐蚀、三档掩膜膨胀和三档局部遮挡。若只运行最关键的中等强度条件：

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File scripts\run_robustness.ps1 -Models single_topo_plain,moco_v3_cbam -Conditions mask_erode_s2,mask_dilate_s2,branch_occlusion_s2 -Phase all
```

补充旋转、模糊与 JPEG 压缩：

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File scripts\run_robustness.ps1 -Models single_topo_plain,moco_v3_cbam -Conditions rotation_s1,rotation_s2,rotation_s3,gaussian_blur_s1,gaussian_blur_s2,gaussian_blur_s3,jpeg_s1,jpeg_s2,jpeg_s3 -Phase all
```

任务内新基线完成后，可直接加入同一鲁棒性评估，例如：

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File scripts\run_robustness.ps1 -Models single_topo_plain,moco_v3_cbam,simsiam_r50 -Conditions mask_erode_s1,mask_erode_s2,mask_erode_s3,branch_occlusion_s1,branch_occlusion_s2,branch_occlusion_s3 -Phase all
```

扰动图像、查询特征和结果分别位于 `artifacts\robustness\images`、
`artifacts\robustness\features` 与 `artifacts\robustness\results`。

## 5. SSI 复算

当前 271 组协议的 SSI 已重新计算。需要复现时运行：

```powershell
conda run --no-capture-output -n gsam python -u scripts\analyze_ssi.py --cache artifacts\features\validated\single_topo_tta_271_1075.npz --query-groups artifacts\manifests\query_groups.json --output-json artifacts\analysis\ssi\single_topo_tta_271_1075.json --output-csv artifacts\analysis\ssi\single_topo_tta_271_1075.csv --figure artifacts\analysis\ssi\single_topo_tta_271_1075.png
```

SSI 是同一检索嵌入上的事后组内一致性指标，不是独立形态标注，论文中不能把 SSI
与 mAP 的关系写成因果关系。

## 尚未自动化的实验

“原始图像 vs. 前景提取图像”需要每张训练图、测试图和统一图库图像之间的一一对应
原图路径。当前提供的训练 CSV 只指向前景图像，仓库中尚无经过审计的原图映射表。
在映射明确前不应按文件名猜测配对，否则会引入新的数据泄漏或错配。现有鲁棒性实验
可以回答对掩膜缺损的敏感性，但不能替代原图/前景图的严格受控消融。
