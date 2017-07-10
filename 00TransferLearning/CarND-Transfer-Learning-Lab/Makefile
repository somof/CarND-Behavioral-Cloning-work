
PYTHONC = ../../../miniconda3/envs/carnd-term1/bin/python
PYTHONG = LD_LIBRARY_PATH=/usr/local/cuda-8.0/lib64 ../../../miniconda3/envs/carnd-term1-gpu/bin/python

UNAME = ${shell uname}
ifeq ($(UNAME),Darwin)
PYTHONC = ../../../../src/miniconda3/envs/carnd-term1/bin/python
PYTHONG = ../../../../src/miniconda3/envs/carnd-term1/bin/python
endif


all: traffic # cifar

cifar:
	$(PYTHONG) feature_extraction.py --training_file ../data/vgg_cifar10_100_bottleneck_features_train.p --validation_file ../data/vgg_cifar10_bottleneck_features_validation.p
	$(PYTHONG) feature_extraction.py --training_file ../data/inception_cifar10_100_bottleneck_features_train.p --validation_file ../data/inception_cifar10_bottleneck_features_validation.p
	$(PYTHONG) feature_extraction.py --training_file ../data/resnet_cifar10_100_bottleneck_features_train.p --validation_file ../data/resnet_cifar10_bottleneck_features_validation.p

traffic:
	$(PYTHONG) feature_extraction.py --training_file ../data/vgg_traffic_100_bottleneck_features_train.p --validation_file ../data/vgg_traffic_bottleneck_features_validation.p
	$(PYTHONG) feature_extraction.py --training_file ../data/inception_traffic_100_bottleneck_features_train.p --validation_file ../data/inception_traffic_bottleneck_features_validation.p
	$(PYTHONG) feature_extraction.py --training_file ../data/resnet_traffic_100_bottleneck_features_train.p --validation_file ../data/resnet_traffic_bottleneck_features_validation.p
