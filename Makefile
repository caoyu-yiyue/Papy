# set .PHONY
.PHONY: all clean clean_targets clean_features clean_models\
 build_from_h5 features ols_models

# clean models' targets
clean_targets:
	rm -f data/processed/*targets.pickle

# clean models' feature
clean_features:
	rm -f data/processed/*features.pickle

# clean the models results
clean_models:
	rm -f models/*.pickle

# clean all generated file
clean:
	rm -f data/raw/*.h5
	rm -f data/interim/*.h5
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

# caculate a reverse porfolie return time series using the excess return for cumulate.
data/interim/reverse_ret_use_exc.pickle: data/interim/prepared_data.pickle
	python3 src/features/reverse_exc_ret.py $@

# process OLS rm_features data frame
data/processed/rm_features.pickle: data/interim/reverse_ret_use_exc.pickle
	python3 src/features/process_features.py --which rm_features $< $@

# process OLS std_features data frame
data/processed/std_features.pickle: data/interim/reverse_ret_use_exc.pickle
	python3 src/features/process_features.py --which std_features $< $@

# process OLS features and targets data frame
features: data/processed/rm_features.pickle data/processed/std_features.pickle

data/processed/targets.pickle: data/interim/reverse_ret_use_exc.pickle
	python3 src/features/process_targets.py $< $@

# contruct ols models data frame
# ols with market excess return
models/ols_with_market_ret.pickle: data/processed/rm_features.pickle data/processed/targets.pickle
	python3 src/models/ols_model.py --featurestype market_ret $@

# ols with rolling std
models/ols_with_rolling_std_log.pickle: data/processed/std_features.pickle data/processed/targets.pickle
	python3 src/models/ols_model.py --featurestype rolling_std_log $@

# ols with delta_std
models/ols_with_delta_std.pickle: data/processed/std_features.pickle data/processed/targets.pickle
	python3 src/models/ols_model.py --featurestype delta_std $@

# ols with delta_std and market return
models/ols_with_delta_std_and_rm.pickle: data/processed/std_features.pickle data/processed/targets.pickle
	python3 src/models/ols_model.py --featurestype delta_std_and_rm $@

# all ols models
ols_models: models/ols_with_market_ret.pickle models/ols_with_rolling_std_log.pickle \
models/ols_with_delta_std.pickle models/ols_with_delta_std_and_rm.pickle

build_from_h5: data/interim/prepared_data.pickle data/interim/reverse_port_ret.pickle\
data/interim/reverse_ret_use_exc.pickle features data/processed/targets.pickle ols_models

all: data/raw/raw_data.h5 build_from_h5
