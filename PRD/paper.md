题目：Pretraining and Trajectory Balance Fine-Tuning for Target-Specific and Synthesizable Molecular Generation

摘要：在目标导向的小分子生成中，模型既需要在高度复杂的化学空间中进行有效探索，又必须同时满足分子可合成性与靶标相关生物活性这两类关键约束。本文提出一种结合合成感知预训练与轨迹平衡微调的分子生成框架，用于统一建模分子结构、合成可行性与靶标特异活性之间的关系。该框架首先通过基于逆合成轨迹的预训练，使生成策略学习分子构建过程中的合成约束与结构先验；在此基础上，模型在蛋白条件信息（序列或结合口袋结构）的约束下，采用轨迹平衡目标进行微调，从而将生成分布逐步引导至特定靶标的活性化学空间。实验结果表明，在酪氨酸酶抑制剂设计任务中，该方法在保持生成多样性的同时，能够稳定地产生具有良好可合成性且表现出亚微摩尔级活性的候选分子，验证了预训练与轨迹平衡微调范式在目标导向分子生成中的有效性与实用价值。

### 数据集创建

  【逆合成数据集构造需要回答的问题】
  基于 AiZynthFinder 的反应模板逆合成分解，是否“只用在 zinc 库里的反应物砌块分子做分解”？
  - 是的：数据集抽取时将 AiZynthFinder 路线树里 `in_stock=True` 的叶子分子视为“可采购砌块”（在代码里默认 stock 选择为 `zinc`，由 `scripts/build_all_routes_dataset.py --stock` 控制；stock 的具体来源/规则由 `config.yml` 中 AiZynthFinder 的 stock 定义决定）。
  - 但并不要求所有反应物都在 stock：对于某一步反应，路线树中 `in_stock=False` 的反应物会被记录为“中间体反应物”，用于表达“该步需要先合成一个中间体，再与 stock 砌块反应”。

  具体抽取规则（与代码一致）：
  - **节点类型**：AiZynthFinder 的 route 以“分子节点（molecule node）”和“反应节点（reaction node）”交替组成。分子节点含 `in_stock` 字段；反应节点不含该字段。
  - **一步记录的语义**（在 `scripts/build_all_routes_dataset.py:extract_route_steps`）：
    - `当前状态分子`：某个分子节点（该步产物/父节点）的 SMILES。
    - `在zinc库里的反应物砌块分子`：该分子节点下某个反应节点的子节点中，所有 `in_stock=True` 的反应物分子 SMILES（用 `" + "` 连接；并可用 `--max-stock-mw` 过滤过大砌块，默认 200 Da）。
    - `和反应物砌块分子反应的中间体分子`：同一反应节点的子节点中，所有 `in_stock=False` 的反应物分子 SMILES（用 `" + "` 连接；若无则记为 `N/A`）。
    - `反应模版`：来自反应节点 `metadata.template` 或 `metadata.reaction_name`（若缺失则 `N/A`）。
  - **前向顺序**：抽取时先递归进入 `in_stock=False` 的反应物（先把中间体的上游步骤写入），再记录当前产物对应的反应；最后整体 `reverse()`，得到“从起始砌块到目标分子”的前向步骤序列。

  中间体/产物之间的关系如何表示？
  - 对于每一步，`当前状态分子` 是产物；其反应物由两类组成：stock 砌块（`在zinc库里的反应物砌块分子`）和非 stock 的中间体反应物（`和反应物砌块分子反应的中间体分子`）。
  - 当存在中间体反应物时，意味着该步的“状态”应理解为中间体（需要先由更早步骤合成）；当不存在中间体反应物时，意味着该步直接由 stock 砌块合成（对应 Plan B 的起始语义）。

  存放在哪些文件？
  - **逆合成步骤级数据（AiZynthFinder 路线展开后的 CSV）**：`data/reaction_paths_all_routes.csv`，由 `scripts/build_all_routes_dataset.py` 生成。
  - **前向一步轨迹（用于 BC / TB 的状态-动作-结果）**：`data/forward_trajectories*.csv`，由 `scripts/forward_trajectories.py` 从 `reaction_paths_all_routes.csv` 转换生成。

我们基于公开反应与配体–蛋白互作数据，构建用于条件分子生成与在线优化的多源数据集，覆盖“可行合成空间”“靶标条件”与“构象/结合能”三类信息。

## 小分子可合成性数据（用于预训练和微调两种方式的小分子合成轨迹）

  - 从项目内的逆合成路线展开脚本与前向轨迹脚本构建两份核心 CSV：`data/reaction_paths_all_routes.csv` 与 `data/forward_trajectories*.csv`。
    - `reaction_paths_all_routes.csv`：由 `scripts/build_all_routes_dataset.py` 调用 AiZynthFinder 产生路线树并抽取为“前向步骤序列”（见上节问答）。
    - `forward_trajectories*.csv`：由 `scripts/forward_trajectories.py` 将上述步骤序列转换为可监督/可采样的“一步转移集合”，其统一语义为：
      - **状态**：`state_smiles`
      - **动作**：`(action_building_block, action_reaction_template)`
      - **结果**：`result_smiles`
      并用 `assign_forward_order()` 在每条路线内选取一条可链接的主链（`is_in_forward_chain=True`），供训练优先使用。
  - 为提高“出圈”能力（覆盖数据外组合），集成反应模板库 `data/top100/template_top100.csv`（可通过参数替换为任意模板文件）。
    - **推理阶段**（`leadgflownet_infer.py`）：开启 `--template-walk` 时，会在数据集边之外额外生成模板提案；外部砌块默认从 `--extra-blocks-csv data/building_blocks_frag_mw250.csv` 加载（并非强制 `building_blocks_inland.csv`，可配置为任意砌块库）。
    - **在线训练阶段**（`LeadGFlowNet/online_tb_train.py`）：开启 `--template-walk/--free-walk` 且设置 `--template-prob>0` 时，按概率用模板扩展产生候选下一状态；外部砌块由 `--extra-blocks-csv` 提供。

## 靶标数据（用于靶标蛋白在预训练和微调两种方式的输入）

- 预训练方式（靶蛋白嵌入输入模型编码）
  - 训练与推理统一使用蛋白序列字符串（命令行 `--protein`）。默认采用本地 ESM2 轻量模型（`lib/models--facebook--esm2_t30_150M_UR50D`）在 `LeadGFlowNet/protein_encoder.py` 中编码；亦支持简单编码器（`--protein-encoder simple`）以降低依赖与耗时。

- 蛋白质和小分子的构象/结合能数据（微调方式用于精排与奖励）
  - 引入 PLANTAIN 预训练模型作为高效对接评分函数（`lib/plantain/`，权重 `lib/plantain/data/plantain_final.pt`）。推理阶段需要口袋文件（用户在 `test/<PDBID>/<PDBID>_pocket.pdb` 提供），候选配体来自在线生成或 `runs/*.csv` 导出的 `.smi`。在线训练阶段，PLANTAIN 对“终端产物”给出最小能量分数，映射为奖励用于策略更新。

- 药化性质辅助项
  - 使用 RDKit 在线计算的 QED 与合成可及性（SA）代理项，以及可选的 Lipinski 违规计数，作为奖励的稳定化与药化约束项（参数如 `--add-qed`、`--sub-sa`、`--lipinski-penalty`）。

### 方法论

方法以条件 GFlowNet（Trajectory Balance, TB）为核心，并由一个离线行为克隆预训练的合成策略网络提供初始化。结合代码实现后，更准确的表述应为：“以前向合成轨迹定义可合成空间，以蛋白条件策略建模靶点相关性，以终端奖励主导的 TB 微调学习 reward-aligned 分布，并辅以模板扩展、外部砌块和可选逐步 shaping 增强探索。”

- 合成策略网络与预训练
  - 基础策略网络位于 `SynthPolicyNet/models.py`，状态编码器与砌块编码器均采用基于 `GCNConv + global_mean_pool` 的图编码器。严格来说，当前实现是一个轻量 GCN-style MPNN，而非更复杂的通用消息传递框架。
  - 训练样本来自 `data/forward_trajectories*.csv`：每条记录包含 `state_smiles`、`action_building_block`、`action_reaction_template` 与 `result_smiles`。`ForwardTrajectoryDataset` 默认只取 `is_in_forward_chain=True` 且非 `is_start_state` 的一步转移。
  - 行为克隆训练在 `SynthPolicyNet/train_policy.py` 中实现。默认因子分解为先选砌块、再选反应模板：
    \[
    P_F(a_t|s_t)=P_\text{block}(b_t|B_t)\cdot P_\text{rxn}(r_t|B_t,b_t)
    \]
    其中 `compute_block_logits()` 负责砌块分布，`reaction_head` 负责条件反应模板分布。
  - 代码同时支持一个可选的“reaction-first”变体（`--rxn-first`）：先由 `uncond_rxn_head` 预测 \(P_\text{rxn}(r_t|B_t)\)，再由 `compute_block_logits_given_rxn_h()` 预测 \(P_\text{block}(b_t|B_t,r_t)\)。因此论文表述宜将“block-first factorization”说明为默认实现，而不是唯一实现。
  - 预训练目标本质上是行为克隆的最大似然估计，对应 block 与 reaction 两个交叉熵项的加权和；它为后续 TB 微调提供合成合理性的先验初始化。

- 蛋白条件策略与 TB 目标
  - 条件策略位于 `LeadGFlowNet/conditional_policy.py`。其并非简单拼接蛋白向量，而是对状态表示施加 FiLM-style 条件化：
    \[
    h'(B_t,A)=\gamma(h(A))\odot h(B_t)+\beta(h(A))
    \]
    再基于该条件状态表示预测 block / reaction 分布。
  - 蛋白编码在实现上默认使用本地 ESM2（`LeadGFlowNet/protein_encoder.py`；模型目录 `lib/models--facebook--esm2_t30_150M_UR50D`），也支持 `simple` 编码器作为低依赖回退。
  - TB 损失在 `LeadGFlowNet/trainer.py` 中定义为
    \[
    \mathcal{L}_{TB}=(\log Z+\sum_t \log P_F-\log R-\sum_t \log P_B)^2
    \]
    其中 \(\log Z\) 为可学习常数。默认的 \(P_B\) 可由简单近似给出；同时 `online_tb_train.py` 还支持更强的 child-conditioned learned backward policy（如 `--pb-learned`、`--pb-source-aware`、`--pb-logsumexp`）。

- 终端奖励与可选 shaping
  - 代码实现中，终端奖励控制器是 `LeadGFlowNet/trainer.py` 的 `MixedRewardController`。因此更准确的说法是：**TB 的主奖励信号来自终态分子，但具体采用 PLANTAIN、QSAR 还是 Vina-refined reward，取决于训练配置。**
  - 当启用 PLANTAIN 且不启用 Vina 主奖励时，`LeadGFlowNet/oracle.py` 使用最优构象分数 \(\text{score}_{\min}\) 构造
    \[
    R_\text{plant}=\exp(-\text{score}_{\min}/\text{scale})
    \]
    并在实现中裁剪到 \([0,1]\) 区间；`scale` 由 `--plantain-scale` 控制。
  - 当启用 Vina 精炼作为主奖励时，当前实现并不是把 Vina 直接加到 \(R_\text{plant}\) 上，而是优先使用 Vina 优化后的能量 \(E\) 形成
    \[
    R_\text{vina}=\max(0,\,-w_\text{vina}E+w_\text{QED}\cdot \text{QED}-w_\text{SA}\cdot \text{SA}-w_\text{Lip}\cdot \text{Lipinski})
    \]
    也就是说，Vina 分支在在线训练代码里是一个“替代型主奖励路径”，而不是纯粹的后处理附加项。
  - 若走 PLANTAIN/QSAR 路径，药化修正项仍统一写作
    \[
    R=\alpha R_\text{model}+w_\text{QED}\cdot \text{QED}-w_\text{SA}\cdot \text{SA}-w_\text{Lip}\cdot \text{Lipinski}
    \]
    其中 \(R_\text{model}\) 可以是 \(R_\text{plant}\) 或 QSAR 映射后的 reward。
  - `online_tb_train.py` 默认还开启了可选的局部 shaping：`--use-local-reward` 与 `--perstep-plantain` 默认打开，前者基于结构合理性提供逐步局部奖励，后者可每隔若干步对中间状态加入轻量的 PLANTAIN shaping。因此“奖励仅在轨迹终点使用”适合作为**主训练目标的默认叙述**，但若要与代码完全一致，需要补充“实现中支持可选逐步 shaping”。

- 开放空间探索与动作来源
  - 在线训练与推理都支持模板扩展，但实现细节不同。训练脚本 `LeadGFlowNet/online_tb_train.py` 通过 `--template-walk/--free-walk`、`--template-prob` 和 `--extra-blocks-csv` 将模板产物或外部砌块引入 open-space 转移。
  - 推理脚本 `leadgflownet_infer.py` 在 `--template-walk` 打开时，会基于模板库与外部砌块生成额外候选，并通过 `--open-max-proposals`、`--template-try-templates` 等参数限制扩展规模。
  - 训练脚本还支持 `--free-connect`：在模板不可用或希望增强出圈能力时，将当前状态与外部砌块随机加单键连接，作为额外动作来源。
  - 条件策略网络内部专门保留了 action-source embedding（内部数据集边 / template / free-connect），这也与可选 learned backward policy 的 source-aware 版本相互配套。

- 推理与精排
  - 推理阶段在给定蛋白条件下采样多条前向合成轨迹，并保存树结构 JSON。若启用 `--export-ranked`，会先做多样性筛选，再对候选终态计算 PLANTAIN 分数；可选地继续调用 Vina 精炼并导出 `plantain_min`、`plantain_reward`、`vina_affinity`、`vina_affinity_raw` 等字段。
  - 因此，从整体流程看，本项目当前最准确的统一描述是：**离线 BC 预训练学习合成先验，在线 TB 微调学习蛋白条件下的 reward-aligned 合成分布，而 PLANTAIN/Vina 既可以用于终端奖励，也可以用于推理阶段的统一精排；此外，代码还支持可选的逐步 shaping 与开放空间探索。**

