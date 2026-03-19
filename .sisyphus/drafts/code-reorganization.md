# Draft: Steering Geometry Code Reorganization

## Requirements (confirmed from user)

### Core Requirement
- **Goal**: 将代码按实验功能重新组织，每个实验一个独立文件
- **保留**: `utils.py` 作为公共工具模块
- **结构**: 
  - `src/steering_geometry/` - 每个实验一个 `.py` 文件（除了 utils.py）
  - `scripts/` - 每个实验有对应的子目录和 `.sh` 启动文件
- **约束**: sh 文件必须能实际运行（不放到后台）

## Current Code Analysis

### 公共基础设施 (必须保留)
| 文件 | 功能 | 行数 |
|------|------|------|
| `utils.py` | 公共工具函数 | 123 |
| `types.py` | 核心类型定义 | 360 |
| `config.py` | 配置数据类 | 207 |
| `models.py` | 模型加载与激活提取 | 260 |

### 当前实验/功能模块
| 文件 | 功能描述 | 行数 | 独立实验? |
|------|---------|------|----------|
| `extract.py` | 转向向量提取 (5概念 + 4方法) | 701 | ✅ 核心实验 |
| `apply_steering.py` | 应用转向向量生成文本 | 351 | ✅ 核心实验 |
| `evaluation.py` | 评估转向效果 (Judge + MMLU) | 473 | ⚠️ 被 apply_steering 依赖 |
| `tdnv.py` | TDNV分离性指标分析 | 411 | ✅ 独立实验 |
| `unembed_analysis.py` | 反嵌入矩阵投影分析 | 540 | ✅ 独立实验 |
| `vector_analysis.py` | 向量稳定性分析 (cosine similarity) | 444 | ✅ 独立实验 |
| `token_analysis.py` | Token级别分析 (visualize + probe) | 660 | ✅ 独立实验 |
| `stability_comparison.py` | 稳定性对比实验 | 254 | ✅ 独立实验 |

### 当前 scripts/ 结构
```
scripts/
├── run_pipeline.sh           # 完整流程
├── run_discriminative.sh     # discriminative提取
├── run_tdnv.sh               # TDNV分析
├── run_weighted_mean.sh      # weighted_mean提取
├── run_unembed_analysis.sh   # 反嵌入分析
├── complete_plan.sh          # 计划完成工具
├── quick/
│   ├── quick_eval.sh
│   ├── quick_extract.sh
│   └── quick_steering.sh
└── vector_analysis/          # (待确认)
```

## Proposed New Structure

### src/steering_geometry/ 新结构
```
src/steering_geometry/
├── __init__.py               # 包入口，导出公共API
├── utils.py                  # 公共工具函数 (保留)
├── types.py                  # 核心类型 (保留)
├── config.py                 # 配置类 (保留)
├── models.py                 # 模型加载 (保留)
│
├── extract.py                # 实验1: 转向向量提取
├── apply_steering.py         # 实验2: 应用转向
├── tdnv.py                   # 实验3: TDNV分析
├── unembed_analysis.py       # 实验4: 反嵌入分析
├── vector_analysis.py        # 实验5: 向量稳定性分析
├── token_analysis.py         # 实验6: Token级别分析
└── stability_comparison.py   # 实验7: 稳定性对比
```

### scripts/ 新结构
```
scripts/
├── complete_plan.sh          # 保留
├── validate_analysis_json.py # 保留
│
├── extract/                  # 实验1
│   └── run_extract.sh
│
├── steering/                 # 实验2
│   └── run_steering.sh
│
├── tdnv/                     # 实验3
│   └── run_tdnv.sh
│
├── unembed/                  # 实验4
│   └── run_unembed.sh
│
├── vector_analysis/          # 实验5
│   ├── run_diff_means.sh
│   └── run_discriminative.sh
│
├── token_analysis/           # 实验6
│   ├── run_visualize.sh
│   └── run_probe.sh
│
├── stability/                # 实验7
│   └── run_stability.sh
│
├── pipeline/                 # 完整流程
│   └── run_pipeline.sh
│
└── quick/                    # 快速脚本 (保留)
    ├── quick_extract.sh
    ├── quick_steering.sh
    └── quick_eval.sh
```

## Dependencies Analysis

### Module Dependency Graph
```
extract.py
    ├── config.py (ExtractionConfig, ModelConfig, ConceptConfig)
    ├── models.py (HookedModel)
    ├── types.py (ContrastPair, SteeringVector)
    └── utils.py (ensure_dir, safe_model_name, sample_with_seed, select_token_activations)

apply_steering.py
    ├── config.py (SteeringConfig, JudgeConfig, MMLUConfig)
    ├── evaluation.py (JudgeEvaluator, MMLUEvaluator)
    ├── extract.py (load_contrast_pairs)
    ├── models.py (HookedModel)
    ├── types.py (EvaluationResult, SteeringVector)
    └── utils.py (ensure_dir, safe_model_name)

evaluation.py
    ├── config.py (JudgeConfig, MMLUConfig)
    ├── types.py (JudgeScore, MMLUResult, EvaluationResult)
    └── utils.py (clamp_score, ensure_dir)

tdnv.py
    ├── config.py (TDNVConfig, ModelConfig)
    ├── extract.py (load_contrast_pairs, VALID_CONCEPTS)
    ├── models.py (HookedModel)
    ├── types.py (TDNVLayerMetrics, TDNVResult)
    └── utils.py (ensure_dir, safe_model_name, select_token_activations)

unembed_analysis.py
    ├── config.py (ModelConfig)
    ├── models.py (HookedModel)
    ├── extract.py (extract_vector)
    ├── types.py (ConceptAnalysisResult, UnembedAnalysisResult)
    └── utils.py (ensure_dir)

vector_analysis.py
    ├── config.py (ExtractionConfig, ModelConfig)
    ├── extract.py (extract_steering_vector, extract_vector, load_contrast_pairs)
    ├── models.py (HookedModel)
    └── utils.py (ensure_dir)

token_analysis.py
    ├── config.py (ModelConfig, TokenAnalysisConfig)
    ├── extract.py (load_contrast_pairs)
    ├── models.py (HookedModel)
    ├── types.py (多个token相关类型)
    └── utils.py (ensure_dir, safe_model_name)

stability_comparison.py
    ├── config.py (ExtractionConfig, ModelConfig, StabilityComparisonConfig)
    ├── extract.py (extract_steering_vector, load_contrast_pairs)
    ├── models.py (HookedModel)
    ├── types.py (ContrastPair)
    ├── utils.py (ensure_dir)
    └── vector_analysis.py (compute_cosine_similarity_matrix, plot_heatmap)
```

## Open Questions

1. **evaluation.py 处理方式**:
   - 当前被 apply_steering.py 依赖
   - 选项A: 合并到 apply_steering.py
   - 选项B: 作为独立模块保留，被 apply_steering 导入

2. **旧脚本处理**:
   - 当前 scripts/ 下有很多 .sh 文件
   - 是否删除还是移动到对应子目录?

3. **__init__.py 导出策略**:
   - 导出哪些公共 API?
   - 每个实验模块的 CLI 是否需要通过 __init__ 暴露?

## Scope Boundaries

### INCLUDE (要做的):
- 创建 scripts/ 下的子目录结构
- 为每个实验创建/移动对应的 .sh 启动脚本
- 确保 sh 脚本能实际运行（前台运行，同步输出）
- 更新 __init__.py 导出

### EXCLUDE (不做的):
- 修改核心 Python 模块代码逻辑
- 修改模块间的依赖关系
- 添加新功能
