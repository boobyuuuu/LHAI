# Further Work

可以从以下几个方面进行优化：

1. Model

    （1）尝试transformer

    （2）尝试resnet（残差网络）

2. LOSS

    （1）应对高/低流强差的情况：加入log loss

    （2）每个格点都不要差过10%：加框方差

    （3）加入物理机制/约束：LOSS加物理机制限制（非常FURTHER）

3. Data

    （1）整理目前的训练-验证Dataset，目前数据有点太乱，建立数据库

    （2）数据集清洗工作。建立数据预处理流程框架

4. Evaluation

    （1）加个框再进行评估，验证LOSS-（2）的效果

    （2）根据人工打分标签，训练出 evaluation model（非常FURTHER）

5. 训练机制：半监督方法

    （1）自我循环迭代半监督

    （2）根据人工打分标签，训练出 evaluation model；根据eval model进行大量数据集生成；进行循环训练（input data足够多）