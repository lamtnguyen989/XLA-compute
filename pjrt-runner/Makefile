CFLAGS  ?= -O3 -Wall -Wextra -flto
SRC_DIR = src
BUILD_DIR = build

ifeq ($(XLA_SOURCE_DIR),)
	$(error Set XLA_SOURCE_DIR=/path/to/xla, e.g. make XLA_SOURCE_DIR=... <target>)
endif

pjrt_runner: $(SRC_DIR)/pjrt_runner.c
	mkdir -p $(BUILD_DIR)
	$(CC) $(CFLAGS) -I $(XLA_SOURCE_DIR) $< -ldl -o $(BUILD_DIR)/$@

clean:
	rm -rf $(BUILD_DIR)