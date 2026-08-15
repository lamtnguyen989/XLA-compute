/***
*   This is to create and compile the custom PJRT engine based on the C API (mainly for testing)
***/

#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <dlfcn.h>

#include "xla/pjrt/c/pjrt_c_api.h"
#include "pjrt_runner.h"

// Define the input dimension
#define BATCH_SIZE 32
#define N 88200

// TODO: Error checking macros

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
    void* pjrt_handle = dlopen(pjrt_path, RTLD_NOW | RTLD_LOCAL);
    if (pjrt_handle == NULL) {
        fprintf(stderr, "Failed to open the PJRT C API: %s\n", dlerror());
        return 2;
    }

    typedef const PJRT_Api *(*GetPjrtApiFn)(void);
    GetPjrtApiFn get_api = (GetPjrtApiFn)dlsym(pjrt_handle, "GetPjrtApi");
    if (!get_api) {
        fprintf(stderr, "Could not resolve GetPjrtApi: %s\n", dlerror());
        return 3;
    }
    const PJRT_Api *api = get_api();
    printf("Loaded PJRT plugin, API version %d.%d\n", 
            api->pjrt_api_version.major_version,
            api->pjrt_api_version.minor_version);

    // Create a client
    PJRT_Client_Create_Args client_args;
    memset(&client_args, 0, sizeof(client_args));
    client_args.struct_size = PJRT_Client_Create_Args_STRUCT_SIZE;
    PJRT_Client* client = client_args.client;

    // Read and load MLIR/StableHLO module
}