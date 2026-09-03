# 现有五个模型：271 组 / 1075 query 重评估

所有命令均在 PowerShell 中执行。先进入独立仓库并准备本地配置：

```powershell
Set-Location /absolute/path/to/ginseng/benchmark
if (-not (Test-Path .env)) { Copy-Item .env.example .env }
```

确认 `.env` 中的数据目录和 `MAIN_CODE_ROOT` 指向本机真实路径。不要把密钥提交到 Git。

## 1. 重建最新 query 协议

```powershell
conda run -n gsam python scripts\build_query_groups.py `
  --env .env `
  --output artifacts\manifests\query_groups.json
```

预期协议为 271 个身份组、1075 个 query、12787 张 gallery 图片。该命令会先执行数据审计，任何数量、路径或哈希不一致都会终止。

## 2. 先检查命令，不运行模型

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File scripts\run_existing_models.ps1 -DryRun
```

## 3. 分阶段执行

建议按阶段运行，某一步失败时不要继续下一步：

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File scripts\run_existing_models.ps1 -Phase extract
powershell -NoProfile -ExecutionPolicy Bypass -File scripts\run_existing_models.ps1 -Phase stamp
powershell -NoProfile -ExecutionPolicy Bypass -File scripts\run_existing_models.ps1 -Phase evaluate
```

也可以只运行一个模型：

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File scripts\run_existing_models.ps1 `
  -Models single_topo_plain `
  -Phase all
```

可选模型 ID：`simclr`、`moco_v3`、`moco_v3_cbam`、`single_topo_plain`、`single_topo_tta`。

其中 `single_topo_plain` 明确关闭 TTA 并只使用 `[1.0]` 权重；`single_topo_tta` 单独保留论文中的三路 TTA。两者不能混写成同一个结果。

## 4. 汇总论文表格

五个结果全部生成后执行：

```powershell
conda run -n gsam python scripts\summarize_results.py
```

若只想查看当前已完成的模型：

```powershell
conda run -n gsam python scripts\summarize_results.py --allow-missing
```

主要输出：

- `artifacts\features\raw`：新提取的本地原始特征；
- `artifacts\features\validated`：绑定数据 manifest 和 query 协议的标准缓存；
- `artifacts\results`：完整 gallery 排序的 JSON 与逐 query CSV；
- `artifacts\tables`：带身份组 bootstrap 95% 置信区间的 Markdown/CSV 表格；
- `artifacts\logs`：每个模型的执行日志。

历史缓存没有数据 manifest，且旧 mAP/MRR 使用了截断排序，不能复制到这里继续评估。

## 5. 同一 checkpoint 的受控分支消融

该实验使用同一个 MGCL checkpoint、同一个 `stretch224` 输入和关闭 TTA 的推理设置，
分别导出 256 维视觉特征与 128 维形态结构特征。它不重新训练模型，是回应
“增益是否真正来自形态结构分支”的关键因果控制。

以下均为单行 PowerShell 命令：

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File scripts\run_existing_models.ps1 -Config configs\controlled_ablations.json -Models "single_topo_visual_plain,single_topo_topology_plain" -Phase extract
```

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File scripts\run_existing_models.ps1 -Config configs\controlled_ablations.json -Models "single_topo_visual_plain,single_topo_topology_plain" -Phase stamp
```

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File scripts\run_existing_models.ps1 -Config configs\controlled_ablations.json -Models "single_topo_visual_plain,single_topo_topology_plain" -Phase evaluate
```

```powershell
conda run -n gsam python scripts\summarize_results.py --config configs\controlled_ablations.json --input artifacts\results --output-markdown artifacts\tables\controlled_ablations.md --output-csv artifacts\tables\controlled_ablations.csv
```
