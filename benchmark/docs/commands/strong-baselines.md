# 强基线运行命令

本页只给出可复现命令，不包含任何虚构指标。DINOv2、SigLIP 2、CLIP、SwinV2 和 ConvNeXt V2 均作为冻结预训练编码器，在相同的 271 组 / 1075-query / 12787-gallery 协议上评估。全部使用官方预处理、关闭 TTA，并对最终特征做 L2 归一化。

DINOv3 的模型访问申请已被拒绝，因此已经从默认可运行配置中移除，不再运行，也不会让批量命令因 403 错误中断。

Hugging Face 模型统一保存在
`benchmark/artifacts/models/huggingface/hub`，
DINOv2 的独立本地副本保存在 `artifacts\models\dinov2_base`。运行时会依次输出
扫描数据、检查/下载模型、加载模型、批量提取、验证保存五个阶段；批量提取阶段约每
1% 输出一次 ASCII 进度条、已处理图像数、耗时和 ETA。

## 1. 基础准备

以下命令均为单行 PowerShell 命令，不需要先 `conda activate`：

```powershell
Set-Location /absolute/path/to/ginseng/benchmark
```

```powershell
if (-not (Test-Path .env)) { Copy-Item .env.example .env }
```

```powershell
conda run -n gsam python -m pip install -e .
```

## 2. DINOv2（现有 `gsam` 环境）

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File scripts\setup_gsam_dinov2.ps1
```

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File scripts\run_strong_baselines.ps1 -DryRun -Models dinov2_base
```

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File scripts\run_strong_baselines.ps1 -Models dinov2_base -Phase all
```

## 3. 四个公开现代基线

环境脚本使用独立的 `ginseng-baselines` 环境，不会修改 `gsam`：

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File scripts\setup_modern_env.ps1
```

```powershell
conda run --no-capture-output -n ginseng-baselines python scripts\check_modern_env.py
```

一次检查四个模型的完整命令：

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File scripts\run_strong_baselines.ps1 -DryRun -Models "siglip2_base,clip_vit_b16,swinv2_base,convnextv2_base"
```

建议逐个运行。每条命令都会依次下载或复用缓存、提取特征、盖章并评估：

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File scripts\run_strong_baselines.ps1 -Models siglip2_base -Phase all
```

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File scripts\run_strong_baselines.ps1 -Models clip_vit_b16 -Phase all
```

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File scripts\run_strong_baselines.ps1 -Models swinv2_base -Phase all
```

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File scripts\run_strong_baselines.ps1 -Models convnextv2_base -Phase all
```

如果希望一次连续运行四个模型：

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File scripts\run_strong_baselines.ps1 -Models "siglip2_base,clip_vit_b16,swinv2_base,convnextv2_base" -Phase all
```

若下载中断，重新执行同一条命令即可复用项目内的 Hugging Face 缓存。日志位于
`artifacts\logs`，原始特征、已验证特征和指标分别位于 `artifacts\features\raw`、
`artifacts\features\validated` 和 `artifacts\results`。

## 4. 汇总结果

允许尚未完成的模型显示为 `PENDING`：

```powershell
conda run -n gsam python scripts\summarize_results.py --config configs\strong_models.json --input artifacts\results --output-markdown artifacts\tables\strong_models.md --output-csv artifacts\tables\strong_models.csv --allow-missing
```

## 5. TransReID 的边界

TransReID 是有身份监督的重识别架构，不能把普通 ViT 或人物重识别 checkpoint 冒充成人参 TransReID 结果。当前训练库无法仅凭平铺文件名恢复完整身份标签，因此该项仍保持 pending；只有获得来源明确、配置匹配且标签协议公平的 checkpoint 后再运行。

已有适配器的 DryRun 命令：

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File scripts\setup_transreid.ps1 -DryRun
```
