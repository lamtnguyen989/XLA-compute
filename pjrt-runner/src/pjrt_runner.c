/***
*   This is to create and compile the custom PJRT engine based on the C API (mainly for testing)
***/

#include <stdio.h>
#include <stdlib.h>
#include <dlfcn.h>

#include "xla/pjrt/c/pjrt_c_api.h"
#include "pjrt_runner.h"

// Define the input dimension
#define BATCH_SIZE 32
#define N 88200


int main(int argc, char** argv)
{
    // Basic (very) CLI parsing
    if (argc < 3) {
        fprintf(stderr, "usage: <MLIR> <path/to/libpjrt_c_api_{cpu|gpu}.so>\n", argv[0]);
        return 1;
    }
    const char* mlir_path = argv[1];
    const char* pjrt_path = argv[2];

    // Load the PJRT plugin
}