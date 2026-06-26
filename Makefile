# BAFE Makefile - builds libbafe.so and runs tests
#
# Usage:
#   make            build libbafe.so
#   make test       run pytest
#   make clean      remove build artifacts
#   make debug      build with -O0 -g -fsanitize=address,undefined
#   make install    install python package in editable mode

CC      ?= cc
AR      ?= ar
CFLAGS  ?= -O2 -std=c11 -Wall -Wextra -Wno-unused-function \
           -fPIC -fvisibility=hidden -Iinclude
LDFLAGS ?= -shared
LIBS    ?= -ldl

BUILD_DIR  := build
SRC_DIR    := src
INC_DIR    := include
PY_DIR     := python

LIB        := $(BUILD_DIR)/libbafe.so

SRCS       := $(wildcard $(SRC_DIR)/*.c)
OBJS       := $(patsubst $(SRC_DIR)/%.c,$(BUILD_DIR)/%.o,$(SRCS))

HEADERS    := $(wildcard $(INC_DIR)/bafe/*.h)

.PHONY: all clean test debug install python

all: $(LIB)

$(BUILD_DIR):
	mkdir -p $(BUILD_DIR)

$(BUILD_DIR)/%.o: $(SRC_DIR)/%.c $(HEADERS) | $(BUILD_DIR)
	$(CC) $(CFLAGS) -c $< -o $@

$(LIB): $(OBJS)
	$(CC) $(LDFLAGS) -o $@ $(OBJS) $(LIBS)

# expose the .so to python via env var (binding auto-discovers)
test: $(LIB)
	BAFE_LIB=$$(pwd)/$(LIB) python3 -m pytest tests/ -v

debug: CFLAGS = -O0 -g -std=c11 -Wall -Wextra -Wno-unused-function \
                 -fPIC -fvisibility=hidden -Iinclude \
                 -fsanitize=address,undefined -fno-omit-frame-pointer
debug: LDFLAGS = -shared -fsanitize=address,undefined
debug: $(LIB)
	@echo "Built debug build with ASAN/UBSAN at $(LIB)"

install: $(LIB)
	pip install -e .

python: $(LIB)
	@echo "BAFE_LIB=$$(pwd)/$(LIB) python3 -c 'import bafe; print(bafe.__version__)'"

clean:
	rm -rf $(BUILD_DIR) .bafecache __pycache__ */__pycache__ */*/__pycache__ \
	       *.pyc tests/*.pyc python/bafe/*.pyc
