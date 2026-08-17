CFLAGS  ?= -O3 -flto -Wall -Wextra -Werror
SRC_DIR = src
BUILD_DIR = build

# Compile protobuf for internal MLIR compile
$(BUILD_DIR)/compile_options.bin: $(SRC_DIR)/compile_options.textproto
	@if [ -z "$(XLA_SOURCE_DIR)" ]; then \
		echo "Error: Set XLA_SOURCE_DIR=<path-to-xla>, e.g. make XLA_SOURCE_DIR=... pjrt_runner"; \
		exit 1; \
	fi

	mkdir -p $(BUILD_DIR)

	protoc --proto_path=$(XLA_SOURCE_DIR) \
	   --encode=xla.CompileOptionsProto \
	   $(XLA_SOURCE_DIR)/xla/pjrt/proto/compile_options.proto \
	   < $< > $@

# Make PJRT runner
pjrt_runner: $(SRC_DIR)/pjrt_runner.c $(BUILD_DIR)/compile_options.bin
	mkdir -p $(BUILD_DIR)
	$(CC) $(CFLAGS) -I $(XLA_SOURCE_DIR) $< -ldl -o $(BUILD_DIR)/$@

# Run the compiled PJRT runner
run_pjrt_runner: pjrt_runner
	@if [ -z "$(PJRT_SO)" ] || [ -z "$(MORSE_MLIR)" ]; then \
		echo "Error: Set PJRT_SO=<path-to-pjrt_c_api_{cpu|gpu}_plugin.so> and MORSE_MLIR=<path-to-morse-mlir-source>"; \
		exit 1; \
	fi

	$(BUILD_DIR)/pjrt_runner $(MORSE_MLIR) $(PJRT_SO)

# Clean
clean:
	rm -rf $(BUILD_DIR)