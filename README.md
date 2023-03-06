# Reversal portfolio return vs liquidity and volatility

This is a finance data modeling of my personal research. The workflow and data modeling part is almost completed and now is archived. It's an early Python project on mine and the profile for VSCode may be also outdated now. So don't mind if there's some not sufficient part. I'll be happy if it can help other people who have similar problems.

The name of Papy is just Paper in Python. Because there was an R version before that I didn't open source.

For project structure in Chinese, see [structure.md](structure.md).

## What's the project for?

Frankly speaking, it models the relationship of the return of reversal portfolios with liquidity and/or volatility. We calculated the return of the reversal portfolios, by first splitting the stocks into 5 groups by their size, and then splitting every group into 10 sub-groups by their historical rolling standardized return, finally subtracting the high from the low. We can then get the 5*5 reversal portfolios, and model them using OLS with different indices of liquidity and volatility.

Generally, it's a grouping regression problem. The dependent variable needs to be split into groups (5*5 here), and the independent variable may be split or not. And we also need to calculate different types of features, like dummy variables, and control variables. We dummy variables are in the regression, we also need to construct cross features with the main feature.

It's easy to calculate the data we need, but I want to automate the workflow and write a general framework for the same kind of problem. As the result, we can run the whole project with one line in CLI.

Here're the main functionalities:

1. Using `numba` to increase the speed of the rolling window calculation (For the time before Pandas 1.0, we need to setting it manually).
2. Same API for different features.
3. Objective grouping regression API, can use in features that need to be grouped or not.
4. Freely set different main X variables, Dummy variables, and control variable combinations. By the rule of features setting in `OLSFeatures` in `src/models/grouped_ols.py`, the code can automatically combine different features and fit a new regression model.

## Usage

1. One line for all calculation

    ```shell
    make all
    ```
    
    But it doesn't generate the result of the robust tests. We set different backward and forward window lengths of reversal portfolios in robust tests, and it's time costing. Using the following command for robust test results.
    
    ```shell
    make all_verbose
    ```
2. Automated quick report generating 
    
    Run the script `src/model/view_result.py` in the Jupyter environment (of VSCode), and pass a parameter of the path of data of a group of target&features. It will generate a jupyter notebook file which includes all regression results configured in `OLSFeatures`.
 
 3. It can also be used to calculate more features, automated combinations, and new OLS models. (For more general tasks, there may be some hard codes like row or column names to refactor)

## Other Information

The new objective API `GroupedOLS` didn't add into the pipeline of the makefile when the project was archived but is a standalone part and now can be used by itself.

Run the project in VSCode for most complete configurations.

It's a so early project for me and it's over-engineering for a data science project when I look back to it now. But it really helps me to learn DS and even programming and CS concepts. It's really a happy time coding these, if you reading the project, I hope you are, too : )
