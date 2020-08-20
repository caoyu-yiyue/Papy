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
│   │   └── reverse_port_ret.pickle
│   ├── processed
│   │   ├── 3_fac_features.pickle
│   │   ├── amihud_features.pickle
│   │   ├── ret_sign_features.pickle
│   │   ├── rm_features.pickle
│   │   ├── std_features.pickle
│   │   ├── targets.pickle
│   │   └── turnover_features.pickle
│   ├── raw
│   │   └── raw_data.h5
│   └── robost
│       ├── b20_f10
│       │   ├── 3_fac_features.pickle
│       │   ├── amihud_features.pickle
│       │   ├── ret_sign_features.pickle
│       │   ├── rm_features.pickle
│       │   ├── std_features.pickle
│       │   ├── targets.pickle
│       │   └── turnover_features.pickle
│       ├── b20_f20
│       │   ├── ...
│           └── ...
├── models
│   ├── ols_on_delta_std.pickle
│   ├── ols_on_delta_std_full.pickle
│   ├── ols_on_delta_std_full_sign.pickle
│   ├── ols_on_delta_std_full_sign_rm.pickle
│   ├── ols_on_delta_std_rm.pickle
│   ├── ols_on_mkt.pickle
│   ├── ols_on_std.pickle
│   └── ols_on_std_with_sign.pickle
├── notebooks
│   ├── explore
│   │   ├── build_featrues.py
│   │   ├── ols.py
│   │   └── t_test_for_sum_coef.py
│   └── report
│       ├── initial_report.py
│       └── second_rep.py
├── papy.env
├── requirements.txt
├── src
│   ├── __init__.py
│   ├── data
│   │   ├── __init__.py
│   │   ├── preparing_data.py
│   │   └── reading_csv_to_hdfs.py
│   ├── features
│   │   ├── __init__.py
│   │   ├── process_data_api.py
│   │   ├── process_features.py
│   │   ├── reverse_exc_ret.py
│   │   └── reverse_port_ret.py
│   ├── models
│   │   ├── __init__.py
│   │   ├── grouped_ols.py
│   │   ├── ols_model.py
│   │   └── view_result.py
│   └── visualization
│       └── __init__.py
├── structure.md
└── test
    ├── features_test.py
    └── ols_test.py
```

## data/

data 文件夹用来存储各种数据，目前包括：

* `raw/`： 从csv 文件读取后直接保存的原始数据
  * `raw_data.h5`：从csv 直接保存成的h5 对象。
* `interim/`： 构建用于建模的数据之前产生的中间数据。
  * `prepared_data.pickle`：对原始数据进行整理形成的清理并增加必要所需列的数据
  * `reverse_port_ret.pickle`：反转组合收益的时间序列数据
  * `reverse_ret_use_exc.pickle`：使用**超额收益率**计算的反转组合收益率时间序列数据
* `processed/`： 经过处理后，可以用于建模的数据。
  * `targets.pickle`：用于OLS 回归时所使用的targets。本质上是使用**超额收益率**计算的反转组合收益率时间序列数据
  * `rm_features.pickle`：OLS 所需使用的未来五天的超额市场收益率数据。
  * `std_features.pickle`：OLS 所需的与波动率相关的数据，包括一个过去n 天计算的滚动历史波动率，未来五天每天波动率的变动值。
  * `ret_sign_features.pickle`：反转收益正负的Dummy Variable.
  * `3_fac_features.pickle`, `amihud_features.pickle`, `turnover_features.pickle`：分别为三因子、滚动amihud 指标、滚动历史换手率。
  
  虽然features 和targets 分开存贮，但其长度与index 保证为一致。分开存储是为了保持数据独立，以及避免同日期不同股票保存大量相同的features。
* `external/`：一些外部数据，但实际上为空。
* `robost/`：用于稳健型检验的数据，文件夹名称为`b\d_f\d`，两个数字部分分别指定计算反转收益时，向前窗口的长度、向后窗口的长度。该文件夹中的结构与`processed` 中相同，只是更换了计算反转时的窗口。

考虑到数据文件的体积和版权问题，data/ 文件夹中的文件未上传到GitHub  上，只在本地存有。

## models/

保存了OLS 模型的拟合结果。由于使用了分组回归，所以这些结果保存在一个pandas.DataFrame 中。命名规则为`ols_on_*.pickle`，*分别是进行OLS 时的features。几种features 名称含义：

* mkt 市场收益
* std 标准差
* delta_std 反转组合持有期期间，每日的标准差差值
* delta_std_full 反转组合持有期期间，整体的标准差差值
* sign 表示收益率正负Dummy。

## src/

src/ 文件夹保存了处理数据所用的Python 源代码，用于作为module 在notebooks 或其他脚本中使用，或直接作为脚本进行自动化执行（生成所需数据等）。

* `data/`： 用于生成和准备数据的脚本
  * `reading_csv_to_hdfs.py`：读取原始数据，保存为hdfs 对象。生成的数据保存在`data/raw/raw_data.h5`
  * `preparing_data.py`：进行数据准备的脚本，生成的文件保存在`data/interim/prepared_data.pickle`
* `features/`： 准备模型features 的脚本合集
  * `process_data_api.py`：用于生成后续建模时需要的`features` 和`targets` 的一些函数和接口。
  * `process_features.py`：接受不同的参数，生成不同类型的features 的脚本。
  * `reverse_port_ret.py`：生成反转组合收益率所用的一些函数，以及直接作为脚本生成**反转组合**收益的时间序列。
  * `reverse_ext_ret.py`：生成使用**超额收益率**计算所得的反转组合收益率时间序列数据，实际上作为了 OLS 回归的 target。
* `models`：进行模型建立的脚本
  * `ols_model.py`：进行分组 OLS 的旧接口。目前只适配了新接口的一部分（OLSFeatures Enum 类）。Makefile 目前还在调用这个脚本，后续适配未结束。计算获得的OlS 模型结果组成的数据框保存在根目录下的`models/`文件夹下。
  * `grouped_ols.py`：对象化的分组 OLS 新接口：更方便地指定 OLS 回归，更灵活地设定不同的 features 组合（select_features 方法从 OLSFeautres Enum 直接解析需要的features 组合，避免了hard code）。仅需要添加 OLSFeatures 的值便可直接对新 features 组合计算新的 OLS，不必更改其他代码。但该部分还未加入到 build 流程中。
  * `view_result.py`：一键查看所有 OLS 结果。调用对象化的新接口，传入 target 和 features 的数据存储路径，一键输出一个 notebook 查看所有 features 组合的 OLS 结果。(本质上并非 build 流程的一部分，但方便直接生成报告)
* `visualization`：用于进行一些可视化操作的脚本目前为空。

## test/

对代码进行的测试，使用pytest。

* `features_tests.py`：构建 features 时的测试，包括feature 计算是否正确、一些功能函数是否符合预期等。
* `ols_test.py`：测试进行分组 OLS 的相关代码。包括对象构建、获取features、拟合、读取、查看结果等部分。

## notebooks/

用于探索数据和生成报告的Jupyter notebooks

* `explore/`：用于数据探索的jupyter notebook（转换并保存为python 脚本），只做备份目的，无其他用处。
  * `build_features.py`：用于探索生成OLS 模型features 和targets 等数据的notebook。
  * `ols.py`：用于探索OLS 建模过程的notebook。
  * `t_test_for_sum_coef.py`：用于探索`statsmodels` 中对于参数和的t 检验，与手动更改回归式将参数和放入时结果相同。
* `report`：用于报告的jupyter notebook
  * `initial_report.py`：第一步结论的初始报告。
  * `second_rep.py`：第二次报告。

## 其他文件

* `papy.env`： 配置vscode 的一些环境变量，把`src/` 文件夹增加到 `PYTHONPATH` 中。
* `requirements.txt`：pip 自动生成的依赖关系文本
* `Makefile`：使用`Make` 进行pipline 管理的命令文件
* `README.md`：项目说明
* `structure.md`：项目结构记录和说明，方便后续整理和重读。
