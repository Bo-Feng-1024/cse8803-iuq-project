# Project Plan: LLM-Informed Bayesian Priors

> 基于 `course-project.md` 的 deliverable 时间线制定。今天: 2026-04-12。

### 数据源调整说明 (2026-04-12)

Proposal 原计划使用 Horton (2023) 的 dictator/ultimatum game 数据。经调查发现：
- Horton repo (`johnjosephhorton/homo_silicus`) 的 `data/` 仅有 placeholder
- 论文中的 dictator game 复制的是 Charness & Rabin (2002) 的 **二元离散选择**，非连续 sharing ratio
- 与 proposal 中的 Beta(α, β) likelihood 模型不兼容

**调整方案**：
- **Human data**: 改用 Chowdhury et al. (2023) dictator game 数据（R package `dictator`，137 个体观测，earnings 0-10 → sharing ratio = amount_given/10）
  - 源: https://journaldata.zbw.eu/dataset/giving-and-taking-in-dictator-games-replication
- **聚焦 dictator game**：先完成 dictator game 全流程，ultimatum game 作为 extension 放到 final report
- **LLM prior**: GPT-4o 模拟标准 dictator game（"You have $10 to split..."），与 proposal 一致

---

## Phase 0: Check-in 提交 (Due: ~Apr 13) — URGENT

Check-in 占 5%，需要提交 ~3 页 PDF + GitHub repo 链接。

### Grading rubric 对照 (每项 1%)
1. Problem definition and scope clarity
2. Baseline implementation correctness and initial results
3. Frontier method implementation progress and preliminary comparison to baseline
4. Evaluation plan quality and evidence (metrics, calibration, compute/accuracy tradeoff)
5. Reproducibility and project hygiene (clear write-up + working repo with instructor/TA access)

### Tasks

- [x] **P0-1: 数据准备 (data preparation)**
  - 从 Chowdhury et al. (2023) 获取 dictator game 人类实验数据
    - 源: R package `dictator` 或 https://journaldata.zbw.eu/dataset/giving-and-taking-in-dictator-games-replication
  - 转换为 sharing ratio: y = amount_given / 10 ∈ [0, 1]
  - 整理为 `data/` 目录下的 CSV 格式
  - 写数据加载 utility (`src/data_loader.py`)

- [x] **P0-2: LLM prior 构建 (LLM prior construction)**
  - 编写 GPT-4o elicitation 脚本 (`scripts/elicit_llm_prior.py`)
  - Prompt: 标准 dictator game（"You have $10 to split with a stranger..."）
  - 收集 M=500 个 simulated sharing ratios
  - 从 pseudo-observations 推导 Beta prior 参数 (α, β)
  - 用 weight w 缩放 pseudo-observation count (calibrated to ESS)
  - 保存 prior 参数到 `results/llm_prior_params.json`

- [x] **P0-3: Baseline 实现 — Flat prior + MCMC**
  - 实现 Beta likelihood + flat prior 的 Metropolis-Hastings (`src/mcmc.py`)
  - 在不同 sample sizes n ∈ {5, 10, 20, 50, 100} 上运行 (max 100 < 137)
  - 通过插值得到 n*_flat (posterior variance 达到 target 的最小 n)
  - 生成 posterior trace plots + convergence diagnostics
  - 保存结果到 `results/baseline/`

- [x] **P0-4: Frontier 实现 — LLM prior + SVGD**
  - 实现 SVGD (`src/svgd.py`): RBF kernel + gradient updates
  - 用 LLM-informed Beta prior 替换 flat prior
  - 同样在 n ∈ {5, 10, 20, 50, 100} 上运行
  - 计算 n*_LLM 和 reduction ratio ρ = n*_LLM / n*_flat
  - 保存结果到 `results/frontier/`

- [x] **P0-5: 初步对比 & failure mode**
  - 绘制 baseline vs frontier 的 posterior 对比图 (`figures/`)
  - 计算 sample size reduction ratio ρ
  - 实现 prior misspecification sensitivity: shift LLM prior mean ±1, ±2 std
  - 测量 coverage degradation → 识别 failure mode
  - 生成关键图表 (`figures/comparison/`)

- [x] **P0-6: Check-in 报告 (3 pages PDF)**
  - 创建 `project/checkin/checkin.tex`
  - Section 1: Problem definition + 数据源调整说明
  - Section 2: Baseline results (posterior plots, n*_flat)
  - Section 3: Frontier results (SVGD posteriors, n*_LLM, ρ)
  - Section 4: Preliminary failure mode analysis
  - Section 5: Evaluation plan & remaining work (含 ultimatum game extension)
  - 编译: `latexmk -pdf checkin.tex`

- [ ] **P0-7: GitHub repo 设置**
  - 在 github.gatech.edu 创建 private repo
  - Push 当前代码 (src/, scripts/, data/, results/, figures/)
  - 写 `README.md` (运行说明 + reproduce 命令)
  - 写 `environment.yml` 或 `requirements.txt`
  - 添加 collaborators: `pchen402`, `psi6`
  - 在 check-in PDF 中包含 repo 链接

---

## Phase 1: 完善实验 & 评估 (Apr 14 – Apr 20)

Final presentation 和 report 都在 Apr 27，这一周需要完成所有实验。

### Tasks

- [ ] **P1-1: 完善 calibration 评估**
  - 实现 empirical coverage of 95% credible intervals
  - 用 held-out split of Chowdhury et al. data 做 validation
  - 绘制 calibration plot (expected vs observed coverage)

- [ ] **P1-2: Compute-accuracy tradeoff 分析**
  - 记录 SVGD vs MCMC 的 wall-clock time
  - 计算 effective sample size (ESS) at matched accuracy
  - 绘制 accuracy vs compute budget 曲线

- [ ] **P1-3: 完整 failure mode 分析**
  - Prior shift experiment: ±1, ±2, ±3 std deviations
  - 绘制 coverage vs prior shift 的 degradation 曲线
  - 识别 "LLM prior 有害" 的条件边界
  - 讨论 training data leakage 的影响

- [ ] **P1-4: Ultimatum game extension**
  - 寻找公开的 ultimatum game 个体数据 (OSF meta-analysis 或其他)
  - 如有数据：复制 dictator game 全流程，报告第二个 ρ
  - 如无合适数据：用 giving game vs taking game (Chowdhury 数据的两个 variant) 做对比
  - 比较不同 game variant 下的 LLM prior quality

- [ ] **P1-5: 整理所有图表**
  - 统一图表风格 (matplotlib style)
  - 确保所有图表有清晰的 labels, legends, captions
  - 保存高分辨率版本到 `figures/final/`

---

## Phase 2: Final Presentation 准备 (Apr 21 – Apr 26)

15 min talk + 5 min Q&A, 占 20%。

### Grading rubric 对照
- 5% Clarity and story (problem → method → results)
- 6% Technical correctness (methods, assumptions, comparisons)
- 6% Experimental evidence (plots/tables, calibration, ablations)
- 3% Discussion/Q&A quality (limitations, what you learned)

### Tasks

- [ ] **P2-1: 制作 slides**
  - 1-2 slides: Problem & motivation (为什么 LLM prior 能降低实验成本)
  - 2-3 slides: Method (Bayesian framework, prior construction, SVGD)
  - 3-4 slides: Results (ρ, calibration, compute tradeoff)
  - 1-2 slides: Failure modes & limitations
  - 1 slide: Takeaways & future work
  - 目标: 12-13 slides for 15 min

- [ ] **P2-2: 准备 Q&A**
  - 预想可能的问题:
    - "为什么换了数据源" → Horton 的 dictator game 是 binary choice 不适配 Beta model
    - "LLM 可能已经在 dictator game 文献上训练过了" → training leakage 分析
    - "为什么用 SVGD 而不是 HMC" → scalability to higher dimensions
    - "ρ < 1 的统计显著性" → 多次实验的 confidence interval
    - "Prior/likelihood separation 是否足够" → Ludwig et al. framework
  - 准备 backup slides

- [ ] **P2-3: 排练 & 计时**
  - 独立排练 2-3 次
  - 确保 15 min 以内

---

## Phase 3: Final Report (Apr 23 – Apr 27, Due: Apr 27)

~10 pages (excluding references/appendix), 占 15%。

### Grading rubric 对照
- 4% Method description and correctness
- 4% Evaluation rigor (calibration, baselines, compute/accuracy)
- 4% Reproducibility (clean repo, instructions, deterministic seeds, figure scripts)
- 3% Insight (failure modes, when it works/doesn't, design choices)

### Tasks

- [ ] **P3-1: 撰写 final report**
  - Abstract
  - Introduction & problem statement
  - Related work (扩展 proposal 中的 literature review)
  - Methodology: prior construction, baseline (MCMC), frontier (SVGD)
  - Experiments & results: ρ, calibration, compute tradeoff, failure modes
  - Discussion: limitations, when LLM prior helps vs hurts
  - Conclusion & future work
  - 创建 `project/report/report.tex`

- [ ] **P3-2: 最终代码清理 & reproducibility**
  - 确保所有 scripts 可以从头运行并生成结果
  - 更新 `README.md` with exact commands
  - 固定 random seeds
  - 确认 `environment.yml` 完整
  - 测试: 从 clean clone 能否 reproduce main figure/table

- [ ] **P3-3: 最终提交**
  - 编译 report PDF
  - Push 所有代码到 GT GitHub
  - 提交 report PDF (Canvas 或指定平台)
  - 最终检查 repo access (pchen402, psi6)

---

## 优先级总结

| 优先级 | 时间 | 内容 | 权重 |
|--------|------|------|------|
| **P0 (紧急)** | Apr 12-13 | Check-in: 代码 + 3页报告 + repo | 5% |
| **P1** | Apr 14-20 | 完善实验 & 评估 | — |
| **P2** | Apr 21-26 | Final presentation | 20% |
| **P3** | Apr 23-27 | Final report + code | 15% |

**Total project weight**: 50% of course grade (Proposal 10% + Check-in 5% + Presentation 20% + Report 15%)

---

## 当前状态 (2026-04-12)

- [x] Proposal 已提交 (Mar 2)
- [x] `src/` — data_loader.py, mcmc.py, svgd.py, evaluation.py
- [x] `scripts/` — run_experiment.py, elicit_llm_prior.py
- [x] `data/` — dictator_game.csv, llm_prior_samples.npy
- [x] `results/` — experiment_results.json, llm_prior_raw.json
- [x] `figures/` — posterior_n*.png/pdf, nstar_curve, failure_mode
- [x] Check-in report — checkin/checkin.pdf (4 pages)
- [ ] GT GitHub repo — 需要手动创建并 push
