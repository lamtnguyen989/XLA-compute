#ifndef PJRT_SHIM_H_
#define PJRT_SHIM_H_

#include "xla/pjrt/c/pjrt_c_api.h"

// Error checking macros
#define CHECK(cond, ...)                                                       \
    do {                                                                       \
        if (!(cond)) {                                                         \
            fprintf(stderr, __VA_ARGS__);                                      \
            exit(1);                                                           \
        }                                                                      \
    } while (0)


// Struct representing PJRT Runner
typedef struct  {   
    void *dl_handle;
    const PJRT_Api *api;
    PJRT_Client *client;
    PJRT_LoadedExecutable *executable;
    char last_error[1024];
} PjrtRunner;

// Function declerations
PjrtRunner* pjrt_runner_create(const char* plugin_path, const char* allocator);
int pjrt_runner_compile(PjrtRunner* runner, const char* mlir_path, const char* compile_options_path);
void pjrt_runner_destroy(PjrtRunner* r);


#endif  // PJRT_SHIM_H_