# Regression analysis and generalized linear models Presentation
# 回归分析与广义线性模型大作业汇报


## 说明
1. 任务说明
我们的大作业主要是利用所学知识对波士顿房价数据进行分析、检验和估计

2. 编程语言说明
该大作业主要用到python和matlab两种编程语言，其中python主要用于数据预处理、变量筛选、模型构建和模型验证， matlab主要用于绘制图、检验（残差检验等等）和分析（异方差分析等等），
同时matlab也有做估计，对比与python所求的结果是否一致，python也有对数据进行一定的检验和图的绘制，同时进行、相互验证

3. 文件说明
(1)`HousingData.csv`为原始数据
(2)文件名包括`matlab`的txt文件是matlab的脚本
(3)`whole_process.py`是输出全流程结果的脚本，通过运行variable_selection.py可以获取全流程结果，下面会详细说明
(4)`logging`目录是存放全流程运行结果的日志文件


### 全流程
1. 下载代码和相应的文件
`git clone https://github.com/pangyd/multipy_regression_analysis.git`

2. 安装模型所需要的库
`pip install -r requirements.txt`
**注：我们所用的python版本是3.9.7，因为用到的依赖库并不多，且都是常用库，因此一般情况下用任意的python3版本都可以运行**

3. 运行脚本`whole_process.py`输出全流程(`数据预处理`、`变量筛选`、`模型构建`和`模型验证`)结果的日志文件，
```shell
python whole_process.py
       --data_path './HousingData.csv'
       --y_transformation 'log'
       --handle_nan 'nearest_interpolate'
       --outlier_type 'DEFITS'
       --select_criteria 'p_value'
```
- 结果可以在logging目录下查看日志，具体文件名可以参考whole_process.py文件中`第450行`
**注：以上默认配置是最终效果最好的配置，其中`--select_criteria`是前向逐步回归的变量筛选的标准，可将`p_value`更改为`AIC`生成以AIC值为标准的变量筛选结果**