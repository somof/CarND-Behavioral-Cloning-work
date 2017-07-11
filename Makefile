
PYTHONC = ../miniconda3/envs/carnd-term1/bin/python
PYTHONG = LD_LIBRARY_PATH=/usr/local/cuda-8.0/lib64 ../miniconda3/envs/carnd-term1-gpu/bin/python

UNAME = ${shell uname}
ifeq ($(UNAME),Darwin)
PYTHONC = ../../../src/miniconda3/envs/carnd-term1/bin/python
PYTHONG = ../../../src/miniconda3/envs/carnd-term1/bin/python
endif


all: 
	rm -f driving_log_all.csv;
	cp -f record/driving_log.csv driving_log_all.csv;
	cat challenge/driving_log.csv >> driving_log_all.csv;
	$(PYTHONG) model.py -i driving_log_all.csv -d .
	#$(PYTHONG) model.py -i challenge/driving_log.csv -d challenge
	#$(PYTHONG) model.py -i record/driving_log.csv -d record
	#$(PYTHONG) model.py -i data/driving_log.csv -d data

train:
	#$(PYTHONG) model_NVIDIA3.py -i data/driving_log.csv -d data
	#$(PYTHONG) model_NVIDIA2.py -i data/driving_log.csv -d data
	#$(PYTHONG) model_NVIDIA.py  -i data/driving_log.csv -d data
	#$(PYTHONG) model_CNN.py     -i data/driving_log.csv -d data
	#$(PYTHONG) model_flat.py    -i data/driving_log.csv -d data

XXX:
	#$(PYTHONG) model_flat.py -i record/driving_log.csv -d record
	#$(PYTHONG) model_NVIDIA.py -i record/driving_log.csv -d record
	#$(PYTHONG) model.py -i record/driving_log.csv -d record
	#$(PYTHONG) model.py -i data/driving_log.csv -d data

#dataset:
#	$(PYTHONG) record2pickle.py -i record/driving_log.csv -d record -o pickle_sample.p
#	#$(PYTHONG) record2pickle.py -i data/driving_log.csv -d data -o pickle_sample.p

scene:
	$(PYTHONC) camera_images.py -i record/driving_log.csv -d record

sim:
	$(PYTHONC) drive.py model.h5 # track1 & track 2
	#$(PYTHONC) drive.py model_challenge_2.h5 # track 2
	#$(PYTHONC) drive.py model_record_aug.h5  # model for report
	#$(PYTHONC) drive.py model_record.h5
	#$(PYTHONC) drive.py model_data.h5
	#$(PYTHONC) drive.py model_CNN.h5
	#$(PYTHONC) drive.py model_NVIDIA3.h5
	#$(PYTHONC) drive.py model_flat.h5
	#$(PYTHONC) drive.py model_NVIDIA_record_100.h5
	#$(PYTHONC) drive.py model_NVIDIA_data.h5
	#$(PYTHONC) drive.py model.h5
	#./linux_sim/linux_sim.x86_64

video:
	$(PYTHONG) drive.py model.h5 model_all;
	#$(PYTHONG) video.py model_all
	#$(PYTHONG) drive.py model_record_aug.h5 model_record_aug; $(PYTHONG) video.py model_record_aug
	#$(PYTHONG) drive.py model_NVIDIA_data_1.h5 model_NVIDIA; $(PYTHONG) video.py model_NVIDIA
	#$(PYTHONG) drive.py model_CNN2_data.h5 model_CNN2
	#$(PYTHONG) drive.py model_CNN_data.h5 model_CNN
