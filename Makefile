
PYTHONC = ../miniconda3/envs/carnd-term1/bin/python
PYTHONG = LD_LIBRARY_PATH=/usr/local/cuda-8.0/lib64 ../miniconda3/envs/carnd-term1-gpu/bin/python

UNAME = ${shell uname}
ifeq ($(UNAME),Darwin)
PYTHONC = ../../../src/miniconda3/envs/carnd-term1/bin/python
PYTHONG = ../../../src/miniconda3/envs/carnd-term1/bin/python
endif


all: 
	$(PYTHONG) model.py -i data/driving_log.csv -d data

train:
	$(PYTHONG) model_NVIDIA3.py -i data/driving_log.csv -d data
	$(PYTHONG) model_NVIDIA2.py -i data/driving_log.csv -d data
	$(PYTHONG) model_NVIDIA.py  -i data/driving_log.csv -d data
	$(PYTHONG) model_CNN.py     -i data/driving_log.csv -d data
	$(PYTHONG) model_flat.py    -i data/driving_log.csv -d data

XXX:
	#$(PYTHONG) model_flat.py -i record/driving_log.csv -d record
	#$(PYTHONG) model_NVIDIA.py -i record/driving_log.csv -d record
	#$(PYTHONG) model.py -i record/driving_log.csv -d record
	#$(PYTHONG) model.py -i data/driving_log.csv -d data

#dataset:
#	$(PYTHONG) record2pickle.py -i record/driving_log.csv -d record -o pickle_sample.p
#	#$(PYTHONG) record2pickle.py -i data/driving_log.csv -d data -o pickle_sample.p

sim:
	$(PYTHONC) drive.py model_CNN.h5
	#$(PYTHONC) drive.py model_NVIDIA3.h5
	#$(PYTHONC) drive.py model_flat.h5
	#$(PYTHONC) drive.py model_NVIDIA_record_100.h5
	#$(PYTHONC) drive.py model_NVIDIA_data.h5
	#$(PYTHONC) drive.py model.h5
	#./linux_sim/linux_sim.x86_64

video:
	$(PYTHONG) drive.py model_NVIDIA_data_1.h5 model_NVIDIA; $(PYTHONG) video.py model_NVIDIA
	#$(PYTHONG) drive.py model_CNN2_data.h5 model_CNN2
	#$(PYTHONG) drive.py model_CNN_data.h5 model_CNN
