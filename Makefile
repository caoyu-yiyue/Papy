# set .PHONY
.PHONY: all all_from_h5 all_verbose clean clean_targets clean_features\
clean_models clean_all build_from_h5 features ols_models

# clean models' targets
clean_targets:
	rm -f data/processed/*targets.pickle

# clean models' feature
clean_features:
	rm -f data/processed/*features.pickle

# clean the models results
clean_models:
	rm -f models/*.pickle

# clean all files but no raw data file
clean:
	rm -f data/interim/*.pickle
	rm -f data/processed/*.pickle
	rm -f data/external/*.pickle
	rm -f models/*.pickle


# clean all generated file
clean_all:
	rm -f data/raw/*.h5
	rm -f data/interim/*.pickle
	rm -f data/processed/*.pickle
	rm -f data/external/*.pickle
	rm -f models/*.pickle

# read raw csv to hdfs
data/raw/raw_data.h5:
	python3 src/data/reading_csv_to_hdfs.py

# prepare data
data/interim/prepared_data.pickle: data/raw/raw_data.h5
	python3 src/data/preparing_data.py $< $@

# caculate a reverse portfolie return for 60-5
data/interim/reverse_port_ret.pickle: data/interim/prepared_data.pickle
	python3 src/features/reverse_port_ret.py $< $@

# ===================================== process data ============================================== #

# caculate a reverse porfolie return as target using the excess return for cumulate.
data/processed/targets.pickle: data/interim/prepared_data.pickle
	python3 src/features/reverse_exc_ret.py --windows 60 5 $@

# process OLS rm_features data frame
data/processed/rm_features.pickle: data/processed/targets.pickle
	python3 src/features/process_features.py --which rm_features --windows 60 5 $< $@

# process OLS std_features data frame
data/processed/std_features.pickle: data/processed/targets.pickle
	python3 src/features/process_features.py --which std_features --windows 60 5 $< $@

# process OlS turnover features
data/processed/turnover_features.pickle: data/processed/targets.pickle \
data/interim/prepared_data.pickle data/raw/raw_data.h5
	python3 src/features/process_features.py --which turnover --windows 60 5 $< $@

# process amihud features series
data/processed/amihud_features.pickle: data/processed/targets.pickle data/interim/prepared_data.pickle 
	python3 src/features/process_features.py --which amihud --windows 60 5 $< $@

data/processed/ret_sign_features.pickle: data/processed/targets.pickle
	python3 src/features/process_features.py --which ret_sign $< $@

data/processed/three_factors_features.pickle: data/processed/targets.pickle
	python3 src/features/process_features.py --which 3_fac $< $@

# process OLS features and targets data frame
features: data/processed/rm_features.pickle data/processed/std_features.pickle \
data/processed/turnover_features.pickle data/processed/amihud_features.pickle \
data/processed/ret_sign_features.pickle data/processed/three_factors_features.pickle

# ======================================================================================================= #
# contruct ols models data frame
# 1. ols with market excess return
models/ols_on_mkt.pickle: data/processed/rm_features.pickle data/processed/targets.pickle
	python3 src/models/ols_model.py --featurestype mkt $@

# 2. ols with rolling std log
models/ols_on_std.pickle: data/processed/std_features.pickle data/processed/targets.pickle
	python3 src/models/ols_model.py --featurestype std $@

# 3. ols with delta_std
models/ols_on_delta_std.pickle: data/processed/std_features.pickle data/processed/targets.pickle
	python3 src/models/ols_model.py --featurestype delta_std $@

# 4. ols with delta_std and market return
models/ols_on_delta_std_rm.pickle: data/processed/std_features.pickle data/processed/targets.pickle
	python3 src/models/ols_model.py --featurestype delta_std_rm $@

# 5. ols on delta std in full forward interval
models/ols_on_delta_std_full.pickle: data/processed/std_features.pickle data/processed/targets.pickle
	python3 src/models/ols_model.py --featurestype delta_std_full $@

# 6. ols on std log and ret sign dummy
models/ols_on_std_with_sign.pickle: data/processed/std_features.pickle data/processed/ret_sign_features.pickle \
data/processed/targets.pickle
	python3 src/models/ols_model.py --featurestype std_with_sign $@

# 7. ols on delta std full and ret sign dummy
models/ols_on_delta_std_full_sign.pickle: data/processed/std_features.pickle data/processed/ret_sign_features.pickle \
data/processed/targets.pickle
	python3 src/models/ols_model.py --featurestype delta_std_full_sign $@

# 8. ols on delta std full, return sign dummy and mkt
models/ols_on_delta_std_full_sign_rm.pickle: data/processed/std_features.pickle data/processed/ret_sign_features.pickle \
data/processed/targets.pickle
	python3 src/models/ols_model.py --featurestype delta_std_full_sign_rm $@


# all ols models
ols_models: models/ols_on_mkt.pickle models/ols_on_std.pickle models/ols_on_delta_std.pickle \
 models/ols_on_delta_std_rm.pickle models/ols_on_delta_std_full.pickle models/ols_on_std_with_sign.pickle \
 models/ols_on_delta_std_full_sign.pickle models/ols_on_delta_std_full_sign_rm.pickle

##################################################################################################################
# 从raw_data.h5 开始build，但不包括ols 拟合的结果
all_from_h5: data/interim/prepared_data.pickle data/interim/reverse_port_ret.pickle\
data/processed/targets.pickle features

# 所有的文件，但不包括ols 拟合的结果
all: data/raw/raw_data.h5 build_from_h5

# 包括raw_data.h5 和ols 拟合结果在内的全部文件
all_verbose: data/raw/raw_data.h5 build_from_h5 ols_model