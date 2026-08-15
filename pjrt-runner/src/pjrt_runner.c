/***
*   This is to create and compile the custom PJRT engine based on the C API (mainly for testing)
***/

#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <dlfcn.h>
#include <errno.h>

#include "xla/pjrt/c/pjrt_c_api.h"
#include "pjrt_runner.h"

// Define the input dimension
#define BATCH_SIZE 32
#define N 88200

// Error checks
#define CHECK_PJRT(api, expr, where)                                           \
    do {                                                                       \
        PJRT_Error *_err = (expr);                                             \
        if (_err) {                                                            \
            const PJRT_Api *_api = (api);                                      \
            const char *_where = (where);                                      \
            if (!_api || !_api->PJRT_Error_Message) {                          \
                fprintf(stderr, "%s: unknown error (no PJRT_Error_Message)\n", \
                        _where);                                               \
                exit(1);                                                       \
            }                                                                  \
            PJRT_Error_Message_Args msg_args;                                  \
            memset(&msg_args, 0, sizeof(msg_args));                            \
            msg_args.struct_size = PJRT_Error_Message_Args_STRUCT_SIZE;        \
            msg_args.error = _err;                                             \
            _api->PJRT_Error_Message(&msg_args);                               \
            fprintf(stderr, "%s: %.*s\n", _where, (int)msg_args.message_size,  \
                    msg_args.message);                                         \
            exit(1);                                                           \
        }                                                                      \
    } while (0)

// Read binary handling function
static char* read_file(const char* path, size_t* out_size)
{
    FILE *f = fopen(path, "rb");
    if (!f) {
        fprintf(stderr, "fopen(%s): %s\n", path, strerror(errno));
        exit(1);
    }
    fseek(f, 0, SEEK_END);
    long size = ftell(f);
    fseek(f, 0, SEEK_SET);
    char *buf = (char *)malloc((size_t)size);
    if (fread(buf, 1, (size_t)size, f) != (size_t)size) {
        fprintf(stderr, "short read on %s\n", path);
        exit(1);
    }
    fclose(f);
    *out_size = (size_t)size;
    return buf;

}

// main()
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
    CHECK_PJRT(api, api->PJRT_Client_Create(&client_args), "PJRT_Client_Create");
    PJRT_Client* client = client_args.client;

    // Read MLIR/StableHLO module IR
    size_t mlir_size = 0;
    char *mlir_text = read_file(mlir_path, &mlir_size);

    // Compile the IR

    // Pick Runtime Device

    // Build test input buffer
    size_t n_in = (size_t)(BATCH_SIZE * N);

    // Execute

}