/***
*   This file is primarily the source for creating PJRT plugin to be exposed in Rust
***/

#include <dlfcn.h>
#include <errno.h>
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
 
#include "xla/pjrt/c/pjrt_c_api.h"
#include "pjrt_shim.h"

// Define the input dimension
#define BATCH_SIZE 32
#define N (1 << 16)


PjrtRunner* pjrt_runner_create(const char* plugin_path, const char* allocator)
{
    return NULL; // TODO
}

int main(int argc, char** argv)
{
    return 0;
}