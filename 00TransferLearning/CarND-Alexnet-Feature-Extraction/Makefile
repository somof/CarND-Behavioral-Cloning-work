
PYTHONC = ../../../miniconda3/envs/carnd-term1/bin/python
PYTHONG = LD_LIBRARY_PATH=/usr/local/cuda-8.0/lib64 ../../../miniconda3/envs/carnd-term1-gpu/bin/python

UNAME = ${shell uname}
ifeq ($(UNAME),Darwin)
PYTHONC = ../../../../src/miniconda3/envs/carnd-term1/bin/python
PYTHONG = ../../../../src/miniconda3/envs/carnd-term1/bin/python
endif


all:
	$(PYTHONG) train_feature_extraction.py

feature_extraction:
	$(PYTHONG) feature_extraction.py

traffic_sign_inference:
	$(PYTHONG) traffic_sign_inference.py

imagenet:
	$(PYTHONG) imagenet_inference.py
