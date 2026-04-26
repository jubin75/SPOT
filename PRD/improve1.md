下面给出不改代码的解决方案设计，聚焦于“扩大可探索合成空间+强化多样性/新颖性驱动+避免离线数据边界约束”的一揽子措施。按模块分层，你可以逐项启用/组合。
扩大动作空间（打破“只走数据集边”）
动态可行集替代“数据集边掩码”：在线TB采样时，不再仅用 forward_trajectories.csv 派生的 allowed_pairs 作为硬掩码，而是：
以小概率 ε 忽略数据集掩码，直接在全量 block_vocab × rxn_vocab 上采样（带温度/Top-k），随后用 ONNX 筛选器（lib/uspto_filter_model.onnx）或 AiZynthFinder 的 filter policy 做一步反应可行性判别，不可行则重采。这会逐步把生成迁出“旧路径”。
以中等概率从“模板模型（lib/uspto_model.onnx + 模板CSV）”或 AiZynthFinder expansion policy 做 on-the-fly 扩展：给当前 state 预测前向反应模板与必要砌块 Top-k，再筛掉超 MW（≤200 Da）与不可行项，作为候选集加入采样。这样能把合成空间从“数据集”扩大到“模板+库存砌块”的可合成闭包。
检索增强的候选集：对当前 state 的图嵌入，向全体砌块嵌入做近邻检索（Top-K，K 动态退火），把近邻砌块融合到候选集中（哪怕数据集中没出现过这条边），再用反应模板判别+ONNX过滤做可行性校验。
强化探索（采样/损失层面）
概率性探索策略：
ε-greedy/ε-schedule：每步以 ε(t) 概率执行“无掩码/弱掩码”的随机探索；ε 随训练线性/余弦退火（从 0.3→0.05）。
Top-k 与温度退火：起始温度高、Top-k 大，逐步收敛（如 Top-k: Block 12→6、Rxn 6→3；T: 1.3→0.9）。
熵正则（policy entropy bonus）：在 TB 残差外加一项 λ·H(P_F(·|s))（仅训练期），以提升分布平坦度，抑制早期塌缩（λ 如 0.01~0.05）。
多样性批约束（训练时）：
批内相似性惩罚：对同一批终分子计算 ECFP4 Tanimoto，>0.6 的对数按权重惩罚，驱动去重。
回放记忆（novelty buffer）：记录近期生成的终分子，重复项降低采样权重。
奖励塑形（密集/中间态信号，而非仅终止）
新颖性奖励（对训练集/ChEMBL/PubChem）：
novelty(s) = 1 − maxSim(s, 参考库)，把该项按 β_novel 加入中间奖励，强推远离已知分子。
可药性/可合成性密集信号：
中间态 QED/SA/Lipinski 即时项（正向加权/负向惩罚），避免只在终点“补课”。
反应可行性概率（filter 模型打分）作为 shaping 奖励，逐步引导走“可行路径”。
蛋白相关密集信号（可选）：对中间态直接跑 QSAR（蛋白-分子对，已实现于 QSARPredictor），将 sigmoid(pAct) 以小权重加入中间奖励，鼓励“沿路”对目标有利。
形式建议（保留TB性质的势函数塑形）：
采用 potential-based shaping：R’(s,a,s’)=R(s,a,s’)+γ·Φ(s’)−Φ(s)，其中 Φ 可选用 novelty 或（qed − α·sa − β·lips），避免改变最优策略不变性。
训练算法层面升级（提升可逆性与覆盖）
引入参数化后向策略 P_B（Backward policy）并用 TB/DB 训练，改善MDP一致性、提升样本多样性；参考 SynFlowNet 的做法引入训练后的 P_B 约束回退可行性（文献可参考 SynFlowNet 设计）SynFlowNet 论文。
使用 Sub-TB / Flow Matching 变体：在子轨迹/中间截断上施加平衡约束，缓解“只有终点才有信号”的稀疏性。
库与模板的“开域扩展”策略（解决“只会走老路径”）
砌块库刷新：并入更大的可购库（如 Enamine REAL、ZINC），但严格保持“起始砌块≤200 Da、可购”的硬约束；用模板+过滤器把“开域”的组合导入候选。
模板扩展：启用 ringbreaker 模板与更丰富的功能团转化模板；对新模板动作设置较低先验，但可通过奖励学习提升概率。
黑名单/白名单：对白名单反应优先探索（如常见偶联、杂环构建）；对导致“回到旧分子簇”的动作设置轻惩罚。
推理时的去重/多样性控制（不改训练也能提升新颖性）
多样性束搜索：对分支进行“分簇约束”，同簇只保留一个，强制跨簇覆盖。
nucleus/typical sampling：从截断分布采样，减少高频老分子挤占。
结果后处理：与训练库/公共库相似度>0.8 的直接丢弃或降权，再回填下一个候选。
诊断与度量（确保真因定位）
记录“使用了数据集掩码 vs 动态扩展”的步次比例、每步探索类型（掩码/无掩码/模板扩展）与成功率。
统计终分子对训练库/ChEMBL/PubChem的相似度直方/分位数曲线，联动新颖性参数调节。
打印“跨出数据集边”的首个新状态和所用模板/砌块，辅助回溯。
渐进式启用（降低不稳定）
前 10~20% 训练：主要用数据集掩码，ε≈0.1；后续每个 epoch 线性增加“无掩码/模板扩展”的比重与检索 Top-k。
奖励 shaping 权重从小到大：β_novel、QED/SA、filter 可行性在若干 epoch 内缓慢加大，避免早期漂移。
这样一套组合拳能同时解决你提到的两个根因：
不再只学习“老分子的固有路径”，因为动作空间被放宽到“模板+全库砌块”并带有探索；
QSAR 不仅作用于终点，还以塑形方式作用于中间态，促进沿途偏向“对目标有利”的新结构；
引入后向策略与子轨迹平衡，提高 GFlowNet 的多样性能力（理论与实践都支持其有效性），可参考 SynFlowNet 的经验做法（反向策略与可行性一致性）SynFlowNet 论文。
如果你同意，我下一步可以按以上思路给出精确改动点和 CLI 参数设计（不直接改代码），让你能逐项打开/关闭这些能力并评估新颖性曲线。
================================================
[4 tools called]

已集成模板扩展，下一步把 ONNX filter 接到模板产物筛选管线里（保持默认关闭，启用需显式传参）：

- 新增（已生效）
  - `LeadGFlowNet/template_expander.py`：加载 `data/top100/原始的USPTO-31k反应模版.csv` 的 `updated_reaction`，编译 RDKit 反应模板，并在在线TB中以 `--template-prob` 概率走“模板扩展”分支。
  - `LeadGFlowNet/online_tb_train.py`：新增并接线
    - `--template-csv`：模板CSV路径
    - `--template-prob`：每步使用模板扩展概率
    - `--template-max-rows`：限制加载模板数量以提速
    - 模板扩展产物会“snap”到 `forward_trajectories.csv` 中的已知 `state_smiles`（且有外发边），作为探索步接入后续 TB。

- 本次变更：加入 ONNX filter（开关与管线）
  - 目的：对模板扩展产物进行“反应可行性”二次过滤（ONNX 模型在 `config.yml: filter.uspto: lib/uspto_filter_model.onnx`）。
  - 方案（最小侵入，默认关闭，需传参启用）：
    - 在在线TB中增加开关参数（不改变现有默认行为）：
      - `--use-onnx-filter`（开启 ONNX filter）
      - `--aizynth-config config.yml`（指向 AiZynthFinder 配置，内部含 onnx 路径）
    - 过滤流程（仅对“模板扩展产物 cand_products”生效）：
      - 若 `--use-onnx-filter` 且能从 `--aizynth-config` 载入 `filter.uspto` 模型：对产物做可行性判别（不可行剔除）。
      - 若 ONNX/配置不可用：回退到 RDKit 保障（分子可解析、MW 合理、基本性质合理）后再执行“snap 到已知 state”。
    - 说明：ONNX filter 的加载依赖 AiZynthFinder 的 filter policy；启用时请确保 `config.yml` 中 `filter.uspto` 指向 `lib/uspto_filter_model.onnx`，并且 `aizynthfinder` 安装可用。

- 运行示例（Linux）
  - 启用模板扩展 + ONNX filter：
    ```bash
    python /home/jb/phar/LeadGFlowNet/online_tb_train.py \
      --input /home/jb/phar/data/reaction_paths_all_routes.csv \
      --forward /home/jb/phar/data/forward_trajectories.csv \
      --rebuild-forward \
      --checkpoint /home/jb/phar/checkpoints/synth_policy_net.pt \
      --qsar-checkpoint /home/jb/phar/checkpoints/qsar.pt \
      --epochs 5 --episodes-per-epoch 800 --max-steps 8 --lr 3e-4 \
      --device auto --max-block-mw 200 \
      --add-qed 0.3 --sub-sa 0.05 --lipinski-penalty 0.1 \
      --template-csv /home/jb/phar/data/top100/原始的USPTO-31k反应模版.csv \
      --template-prob 0.3 \
      --template-max-rows 5000 \
      --use-onnx-filter \
      --aizynth-config /home/jb/phar/config.yml
    ```
  - 关闭 ONNX filter（仅用模板扩展+snap）：
    ```bash
    ... --template-csv ... --template-prob 0.3
    ```

- 备注
  - 若 `config.yml` 未配置 `filter.uspto` 或环境缺少 AiZynthFinder/ONNXRuntime，脚本会自动回退到 RDKit 基础过滤（不影响训练的稳定性）。
  - 目前我们对模板扩展的探索步不累计模型 log_pf（纯探索），你可后续选择为探索步计入一个近似概率或以 TB-compatible 的 potential shaping 来加强学习信号。

===============================================
  
不需要。直接在现有 BC/离线TB检查点上跑“启用模板扩展+ONNX filter”的在线TB即可。

建议：
- 有离线TB检查点：优先用它作为在线TB的初始点；无则用 BC 检查点。
- 只在这些情况下才考虑重跑离线TB：
  - 你改动了前向轨迹或词表（block/rxn vocab 变化）。
  - 更新了训练标签（pActivity 数据）或想用新的数据清洗策略。
  - 发现在线TB收敛很慢且初始策略过于糟糕，需要更强的初始化。

模板扩展与ONNX过滤只作用于在线采样环节，与离线TB训练数据无耦合，因此无需因此重训离线TB。
==============================================
  "SynthPolicyNet": 630843,
  "OfflineTB_trainable": 894269,
  "OnlineTB_trainable": 1090877

1) success_rate=1 的原因
启用 --template-walk 且传了 --count-open-as-success 时，模板/自由行走产生的“开放转移”会计入成功步；而且数据集内转移动作在当前状态的“可行集”内采样，失败会直接中止该 episode 的后续尝试，因而“尝试的步 ≈ 成功的步”。这不是“强制可合成”的化学保证，而是评价口径与采样策略共同导致的高成功率。

离线TB：{'samples': 356926, 'num_blocks': 13119, 'num_rxns': 8858, 'node_feat_dim': 32}，要不要增大参数空间？

