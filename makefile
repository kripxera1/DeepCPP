# Variables
CC = g++
CFLAGS = -fopenmp -O3 -Iinclude -c
LDFLAGS = -fopenmp -O3 -Iinclude
OBJ_DIR = obj
BIN_DIR = bin
SRC_DIR = src
EXAMPLE_DIR = examples
IMG_DIR = images
CPP_VERSION = -std=c++17

OBJS = $(OBJ_DIR)/bitmap.o $(OBJ_DIR)/algebra.o $(OBJ_DIR)/NNUtils.o \
       $(OBJ_DIR)/losses.o $(OBJ_DIR)/layers.o $(OBJ_DIR)/optimizers.o \
       $(OBJ_DIR)/LRScheduler.o

all: $(BIN_DIR)/classifier $(BIN_DIR)/vae $(BIN_DIR)/denoising-vae

# Targets
$(BIN_DIR)/classifier: $(OBJS) $(OBJ_DIR)/classifier.o
	$(CC) $(LDFLAGS) -o $@ $^

$(BIN_DIR)/vae: $(OBJS) $(OBJ_DIR)/vae.o
	$(CC) $(LDFLAGS) -o $@ $^

$(BIN_DIR)/denoising-vae: $(OBJS) $(OBJ_DIR)/denoising-vae.o
	$(CC) $(LDFLAGS) -o $@ $^

$(OBJ_DIR)/%.o: $(SRC_DIR)/%.cpp
	$(CC) $(CFLAGS) $(CPP_VERSION) -o $@ $<

$(OBJ_DIR)/%.o: $(EXAMPLE_DIR)/%.cpp
	$(CC) $(CFLAGS) $(CPP_VERSION) -o $@ $<

.PHONY: clean cleanimages mrproper

clean:
	-rm $(BIN_DIR)/*

cleanimages:
	-rm $(IMG_DIR)/vae/* $(IMG_DIR)/denoising-vae/* $(IMG_DIR)/classifier/*

mrproper: clean
	-rm $(OBJ_DIR)/*