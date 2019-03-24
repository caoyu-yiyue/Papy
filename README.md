# 论文数据的测试项目

该项目为针对论文数据的算法进行的测试项目。这里简要说明项目的组织结构

## data/

data 文件夹用来存储各种数据，目前包括：

* `raw_data.h5`： 从csv 文件读取后直接保存的原始数据
* `prepared_data.h5`： 进行一些数据整理、添加所需变量形成的准备后的数据
* `reverse_portfolie.h5`： 目前用来保存**反转组合收益**结果的表格

考虑到数据文件的体积，data/ 文件夹中的文件目前暂未上传到GitHub  上，只在本地存有。

## modules/

modules/ 文件夹封装了一些计算中需要用到的函数，便于直接在其他脚本中调用。目前包括：

* `__init__.py`： 用以将文件夹识别为modules
* `reverse_func.py`： 一些用于计算反转组合收益率的函数

## notebooks/

该文件夹保存了一些由python 脚本生成的Juputer Notebook 文件，用于保存结果。目前包括：

* `01_calculateReversePortfolie.ipynb`：目前为最早期的计算反转组合收益的代码生成的`ipynb` 文件

## Scripts/

包括了很多运行时需要的脚本，为主要的脚本存储文件夹。目前包括：

* `00_dataReading.py`：从原始数据的.csv 文件读取数据，然后保存到`data/rawData.h5` 文件中。
* `00_dataPreparing.py`：从原始数据读取，然后去掉一些无用数据、增加一些后续需要的数据的脚本。处理过后的数据保存到`data/prepared_data.h5` 文件中。
* `01_calculateReversePortfolie.py`：计算**反转组合收益率** 的脚本，目前结果保存在`data/reverse_portfolie.h5`

## 其他文件

* `papy.env` 配置vscode 的一些环境变量，把`modules` 文件夹增加到 `PYTHONPATH` 中。
