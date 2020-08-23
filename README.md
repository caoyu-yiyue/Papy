# 某归档的金融数据代码项目

该项目为某金融数据计算的个人代码项目。流程和数据计算基本已经全部完成，后因结果不符合预期归档，代码目前不再维护。这是个人比较早的 Python 项目，很多地方可能不完善，项目依赖和 VSCode 配置也比较旧，敬请谅解。若能帮助有类似问题的朋友，不胜荣幸。

Papy is just Paper Python. 因为前面还有一个 R 版本233。

关于项目结构请见[structure.md](structure.md)。

## 项目在做什么

简单来说，计算了反转组合收益和波动、流动性的关系。这里计算了按规模分五组、每组内按历史滚动标准化收益率分 10 组，然后高低收益组对减后获得反转组合收益。获得一个（5 * 5）的反转组合收益后，和不同的波动、流动性指标进行线性回归计算它们之间的关系。

一般来看，这是一个分组回归问题。因变量是需要分组的（这里是一个 5 * 5 分组），自变量的形式可能是每组不同的，也可能是各组相同的。此外，自变量分为几种不同的种类：主要研究的自变量、作为 Dummy 的自变量、单纯放入回归的控制变量。当 Dummy 变量存在时，需要与主要变量相乘构建一个 cross feature。

虽然计算所需数据并不算难，但希望努力完成解决类似问题的通用方法，能够通过简单设置自动完成如上功能的自动化。

代码功能主要包括了：

1. 使用 numba 加速 rolling 计算反转组合收益的过程；
2. 统一接口计算不同的回归 features(X)；
3. 对象化的分组回归接口，同时兼容分组 features 和各组相同的 features；
4. 自由指定不同的主要自变量、Dummy 自变量、控制变量，按照参数命名规则输入不同的 features，自动组合并可以完成新的回归模型。(配置在 `src/models/grouped_ols.py` 中的 Enum `OLSFeatures` 中实现。)

## 使用方法

1. 一键计算所有数据：

    ```shell
    make all
    ```

    但如上方法不计算稳健型检验所需数据。稳健型检验时，使用不同的反转组合的backword 和forward 窗口长度，计算将会比较耗时。使用如下命令计算稳健型检验所需数据：

    ```shell
    make all_verbose
    ```

2. 一键生成报告：

    在（VSCode 的）jupyter 环境下运行脚本 `src/model/view_result.py`，需要一个参数，即保存一组回归 target&features 数据的路径。可以是主流程中的数据路径或一组稳健型检验的数据所在路径。输出一个 jupyter 文件，包括了 `OLSFeatures` 类下所有的 features 组合的回归结果。(不过并不是生成最终报告或是论文哦，是用于检查所有主要结果显著性的表格。）

3. 扩展：可以在目前的基础上计算更多的 features、自动组合、直接计算新 OLS。(对于非本任务下的通用情况，有一些hard code 的部分，如行名列名等，可能需要修改)

## 其他注意事项

新的对象化 `GroupedOLS` 接口还未加入 makefile 指定的 pipline 中，不过组件还算独立，可以单独使用（实际上自动生成报告时已经使用了新的接口，只是还没有整合到 pipline）。

在 VSCode 中运行可以最大程度保证配置完备。
