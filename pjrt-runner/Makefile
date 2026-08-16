CFLAGS  ?= -O3 -Wall -Wextra -flto
SRC_DIR = src
BUILD_DIR = build

pjrt_runner: $(SRC_DIR)/pjrt_runner.c
	@if [ -z "$(XLA_SOURCE_DIR)" ]; then \
		echo "Error: Set XLA_SOURCE_DIR=/path/to/xla, e.g. make XLA_SOURCE_DIR=... pjrt_runner"; \
		exit 1; \
	fi
	
	mkdir -p $(BUILD_DIR)
	$(CC) $(CFLAGS) -I $(XLA_SOURCE_DIR) $< -ldl -o $(BUILD_DIR)/$@

clean:
	rm -rf $(BUILD_DIR)