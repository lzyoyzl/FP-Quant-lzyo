# Group-wise 旋转搜索算法总体流程图

```mermaid
flowchart TD
    A["预训练 Transformer 模型"] --> B["校准数据前向采样"]
    B --> C["按 Transformer block 划分搜索单元"]

    C --> D["划分输入槽位<br/>qkv / o / gate_up / down"]
    D --> E["按 group_size 切分输入维度"]

    E --> F["构造候选旋转集合<br/>identity / hadamard / DCT / DST / GSR / Householder"]
    F --> G["逐 group 评估候选旋转矩阵"]

    G --> H{"搜索目标函数"}
    H --> H1["权重量化误差<br/>MSE"]
    H --> H2["激活协方差加权输出误差<br/>COV"]
    H --> H3["激活量化重构误差<br/>ACT-MSE"]
    H --> H4["尾部分位点加权目标<br/>J-tail"]

    H1 --> I["选择每个 group 的最优旋转"]
    H2 --> I
    H3 --> I
    H4 --> I

    I --> J["组成 MixedGroupTransform<br/>每个 group 可使用不同旋转"]
    J --> K["保持线性等价<br/>激活侧: x -> xT<br/>权重侧: W -> W T^{-T}"]

    K --> L["执行 RTN / GPTQ 量化"]
    L --> M["导出 pseudoquant / realquant 模型"]
    M --> N["下游任务评测与结果分析"]
```

## 图示说明

该流程图概括了当前项目中 group-wise 旋转搜索的核心实现。算法以 Transformer block 为基本单位，将线性层输入划分为 `qkv`、`o`、`gate_up` 和 `down` 四类槽位，并在每个槽位内按量化 group 对输入维度分块。对于每个 group，算法在结构化正交变换候选集合中搜索最优旋转矩阵。

搜索目标支持四类形式：`MSE` 关注权重量化误差，`COV` 使用校准激活协方差近似局部输出扰动，`ACT-MSE` 直接衡量旋转后激活的量化重构误差，`J-tail` 在基础目标上加入分位点尾部加权项。最终，各 group 的最优旋转被组合为 `MixedGroupTransform`，并通过激活侧旋转与权重侧逆变换折叠保持线性等价，再进入 RTN 或 GPTQ 量化流程。
