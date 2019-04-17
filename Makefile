# set .PHONY
.PHONY: all clean build_from_h5 features

# clean all generated file
clean:
	rm -f data/raw/*.h5
	rm -f data/interim/*.h5
	rm -f data/interim/*.pickle

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

# process OLS features and targets data frame
features: data/interim/reverse_ret_use_exc.pickle
	python3 src/features/process_features.py $< data/processed/rm_features.pickle data/processed/std_features.pickle

data/processed/targets.pickle: data/interim/reverse_ret_use_exc.pickle
	python3 src/features/process_targets.py $< $@

build_from_h5: data/interim/prepared_data.pickle data/interim/reverse_port_ret.pickle\
data/interim/reverse_ret_use_exc.pickle data/processed/features.pickle\
data/processed/targets.pickle

all: data/raw/raw_data.h5 build_from_h5
