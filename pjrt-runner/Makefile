CFLAGS  ?= -O3 -Wall -Wextra
SRC_DIR = src/

ifeq ($(XLA_SOURCE_DIR),)
	$(error Set XLA_SOURCE_DIR=/path/to/xla, e.g. make XLA_SOURCE_DIR=../xla)
endif

pjrt_runner: $(SRC_DIR)/pjrt_runner.c
	$(CC) $(CFLAGS) -I$(XLA_SOURCE_DIR) $@ $< -ldl

clean:
	rm -f pjrt_runner