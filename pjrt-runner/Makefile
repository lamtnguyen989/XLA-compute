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

make run_pjrt_runner: pjrt_runner
	@if [ -z "$(PJRT_SO)" ] || [ -z "$(MORSE_MLIR)" ]; then \
		echo "Error: Set PJRT_GPU_SO=/path/to/pjrt_c_api_{cpu|gpu}_plugin.so and MORSE_MLIR=/path/to/morse/mlir/source"; \
		exit 1; \
	fi

	$(BUILD_DIR)/pjrt_runner $(MORSE_MLIR) $(PJRT_SO)

clean:
	rm -rf $(BUILD_DIR)