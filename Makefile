OUTNAME_RELEASE = yolov5
OUTNAME_DEBUG   = yolov5_debug
EXTRA_DIRECTORIES = ../common
SAMPLE_DIR_NAME = $(shell basename $(dir $(abspath $(firstword $(MAKEFILE_LIST)))))
MAKEFILE ?= ../Makefile.config
include $(MAKEFILE)
