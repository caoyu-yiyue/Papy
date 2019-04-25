# 论文数据的测试项目

该项目为针对论文数据的算法进行的测试项目。这里简要说明项目的组织结构

## 项目结构图

```shell
.
├── Makefile
├── README.md
├── data
│   ├── external
│   ├── interim
│   │   ├── prepared_data.pickle
│   │   ├── reverse_port_ret.pickle
│   │   └── reverse_ret_use_exc.pickle
│   ├── processed
│   │   ├── rm_features.pickle
│   │   ├── std_features.pickle
│   │   └── targets.pickle
│   └── raw
│       └── raw_data.h5
├── models
│   ├── ols_with_delta_std.pickle
│   ├── ols_with_delta_std_and_rm.pickle
│   ├── ols_with_market_ret.pickle
│   └── ols_with_rolling_std_log.pickle
├── notebooks
│   ├── explore
│   │   ├── build_featrues.py
│   │   ├── ols.py
│   │   └── t_test_for_sum_coef.py
│   └── report
│       └── initial_report.py
├── papy.env
├── requirements.txt
└── src
    ├── __init__.py
    ├── data
    │   ├── __init__.py
    │   ├── preparing_data.py
    │   └── reading_csv_to_hdfs.py
    ├── features
    │   ├── __init__.py
    │   ├── process_data_api.py
    │   ├── process_features.py
    │   ├── process_targets.py
    │   ├── reverse_exc_ret.py
    │   └── reverse_port_ret.py
    ├── models
    │   ├── __init__.py
    │   └── ols_model.py
    └── visualization
        └── __init__.py
```

## data/

data 文件夹用来存储各种数据，目前包括：

* `raw/`： 从csv 文件读取后直接保存的原始数据
  * `raw_data.h5`：从csv 直接保存成的h5 对象。
* `interim/`： 构建用于建模的数据之前产生的中间数据。
  * `prepared_data.h5`：对原始数据进行整理形成的清理并增加必要所需列的数据
  * `reverse_port_ret.pickle`：反转组合收益的时间序列数据
  * `reverse_ret_use_exc.pickle`：使用**超额收益率**计算的反转组合收益率时间序列数据
* `processed/`： 经过处理后，可以用于建模的数据。
  * `rm_features.pickle`：OLS 所需使用的未来五天的超额市场收益率数据。
  * `std_features.pickle`：OLS 所需的与波动率相关的数据，包括一个过去n 天计算的滚动历史波动率，未来五天每天波动率的变动值。
  * `targets.pickle`：用于OLS 回归时所使用的targets
  虽然features 和targets 分开存贮，但其长度与index 保证为一致。分开存储是为了保持数据独立，以及避免同日期不同股票保存大量相同的features。
* `external/`：一些外部数据

考虑到数据文件的体积，data/ 文件夹中的文件目前暂未上传到GitHub  上，只在本地存有。

## models/

保存了OLS 模型的拟合结果。由于使用了分组回归，所以这些结果保存在一个pandas.DataFrame 中。目前保存的结果包括：

* `ols_with_market_ret.pickle`：使用市场超额收益率作为OLS features 的OLS 模型结果
* `ols_with_rolling_std_log.pickle`：使用滚动波动率作为OLS features 的OLS 模型结果
* `ols_with_delta_std.pickle`：使用波动率变动作为OLS features 的OLS features 的OLS 模型结果
* `ols_with_delta_std_and_rm.pickle`：使用波动率变动和市场超额收益率作为features 的OLS 模型结果

## src/

src/ 文件夹保存了处理数据所用的Python 源代码，用于作为module 在notebooks 或其他脚本中使用，或直接作为脚本进行自动化执行（生成所需数据等）。

* `data/`： 用于生成和准备数据的脚本
  * `reading_cav_to_hdfs.py`：读取原始数据，保存为hdfs 对象。生成的数据保存在`data/raw/raw_data.h5`
  * `preparing_data.py`：进行数据准备的脚本，生成的文件保存在`data/interim/prepared_data.h5`
* `features/`： 准备模型features 的脚本合集
  * `reverse_port_ret.py`：生成反转组合收益率所用的一些函数，以及直接作为脚本生成**反转组合**收益的时间序列。脚本生成的数据在`data/interim/reverse_port_ret.pickle`
  * `reverse_ext_ret.py`：生成使用**超额收益率**计算所得的反转组合收益率时间序列数据。生成的数据保存在`reverse_ret_use_exc.pickle`
  * `process_data_api.py`：用于生成后续建模时需要的`features` 和`targets` 的一些函数和接口，在其他`process_`开头的脚本中调用。
  * `process_features.py`：仅直接用作命令行脚本，调用`process_data_api.py` 中的接口，生成建模所用的`features`。当向脚本输入不同参数时，可以生成不同的`features`。目前接收两个参数：
    * `--rmfeatures`，生成市场超额收益率`features`，生成的数据保存在`data/processed/rm_features.pickle`
    * `--stdfeatures`，生成波动率相关的`features`，包括一个滚动波动率，和五个未来五天的波动率变动值。生成的数据保存在`data/processed/std_features.pickle`
  * `process_targets.py`：仅直接作用在命令行中调用，使用`process_data_api.py` 中的接口，生成建模所用的`targets`，数据保存在`data/processed/features.pickle`
* `models`：进行模型建立的脚本
  * `ols_model.py`：进行OLS 模型计算的一些函数，计算获得的OlS 模型结果组成的数据框保存在根目录下的`models/`文件夹下。
* `visualization`：用于进行一些可视化操作的脚本

## notebooks/

用于探索数据和生成报告的Jupyter notebooks

* `explore/`：用于数据探索的jupyter notebook（转换并保存为python 脚本）
  * `build_features.py`：用于探索生成OLS 模型features 和targets 等数据的notebook。
  * `ols.py`：用于探索OLS 建模过程的notebook。
  * `t_test_for_sum_coef.py`：用于探索`statsmodels` 中对于参数和的t 检验，与手动更改回归式将参数和放入时结果相同。
* `report`：用于报告的jupyter notebook
  * `initial_report.py`：第一步结论的初始报告。这里包括了超额收益计算的反转收益率对市场超额收益、对数化的历史滚动波动率、滚动波动率的日间变动值、波动的日间变动值控制市场超额收益这四种features，进行回归后的结果。

## 其他文件

* `papy.env`： 配置vscode 的一些环境变量，把`src/` 文件夹增加到 `PYTHONPATH` 中。
* `requirements.txt`：pip 自动生成的依赖关系文本
* `Makefile`：使用`Make` 进行pipline 管理的命令文件
* `README.md`：项目的结构说明
