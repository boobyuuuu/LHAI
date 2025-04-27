# Further Work

可以从以下几个方面进行优化：

1. Model

    （1）diffusion换transformer

    （2）尝试resnet

2. LOSS

    （1）应对高/低流强差的情况：加入log loss

    （2）每个格点都不要差过10%：加框方差

    （3）加入物理机制/约束：LOSS加物理机制限制

3. Data

4. Evaluation

    （1）加个框再进行评估

    （2）根据人工打分标签，训练出 evaluation model

5. 训练机制：半监督方法

    （1）自我循环迭代半监督

    （2）根据人工打分标签，训练出 evaluation model；根据eval model进行大量数据集生成；进行循环训练（input data足够多）