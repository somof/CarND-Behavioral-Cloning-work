
PYTHONC = ../../miniconda3/envs/carnd-term1/bin/python
PYTHONG = LD_LIBRARY_PATH=/usr/local/cuda-8.0/lib64 ../../miniconda3/envs/carnd-term1-gpu/bin/python

UNAME = ${shell uname}
ifeq ($(UNAME),Darwin)
PYTHONC = ../../../../src/miniconda3/envs/carnd-term1/bin/python
PYTHONG = ../../../../src/miniconda3/envs/carnd-term1/bin/python
endif


all:
	$(PYTHONG) test_saved.py -sf modelfile1
	$(PYTHONG) test_saved.py -mf modelfile1 -sf modelfile2
	#$(PYTHONC) test.py
