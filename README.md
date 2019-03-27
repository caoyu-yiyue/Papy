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
│   │   ├── prepared_data.h5
│   │   └── reverse_port_ret.pickle
│   ├── processed
│   └── raw
│       └── raw_data.h5
├── notebooks
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
    │   └── reverse_port_ret.py
    ├── models
    │   └── __init__.py
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
* `processed/`： 经过处理后，可以用于建模的数据。
* `external/`：一些外部数据

考虑到数据文件的体积，data/ 文件夹中的文件目前暂未上传到GitHub  上，只在本地存有。

## src/

src/ 文件夹保存了处理数据所用的Python 源代码，用于作为module 在notebooks 或其他脚本中使用，或直接作为脚本进行自动化执行（生成所需数据等）。

* `data/`： 用于生成和准备数据的脚本
  * `reading_cav_to_hdfs.py`：读取原始数据，保存为hdfs 对象。生成的数据保存在`data/raw/raw_data.h5`
  * `preparing_data.py`：进行数据准备的脚本，生成的文件保存在`data/interim/prepared_data.h5`
* `features/`： 准备模型features 的脚本合集
  * `reverse_port_ret.py`：生成反转组合收益率所用的一些函数，以及直接作为脚本生成**反转组合**收益的时间序列。脚本生成的数据在`data/interim/reverse_port_ret.pickle`
* `models`：进行模型建立的脚本
* `visualization`：用于进行一些可视化操作的脚本

## notebooks/

用于探索数据和生成报告的Jupyter notebooks

## 其他文件

* `papy.env`： 配置vscode 的一些环境变量，把`src/` 文件夹增加到 `PYTHONPATH` 中。
* `requirements.txt`：pip 自动生成的依赖关系文本
* `Makefile`：使用`Make` 进行pipline 管理的命令文件
* `README.md`：项目的结构说明