/***
*   This is to create and compile the custom PJRT engine based on the C API (mainly for testing)
***/

#include <stdio.h>
#include <stdint.h>
#include <string.h>
#include <stdlib.h>
#include <dlfcn.h>
#include <errno.h>

#include "xla/pjrt/c/pjrt_c_api.h"

// Define the input dimension
#define BATCH_SIZE 32
#define N (1 << 16)
#define INPUT_DIM 2
#define OUTPUT_DIM 3

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

#define CHECK(cond, ...)                                                       \
    do {                                                                       \
        if (!(cond)) {                                                         \
            fprintf(stderr, __VA_ARGS__);                                      \
            exit(1);                                                           \
        }                                                                      \
    } while (0)

// Read binary handling function
static char *read_file(const char *path, size_t *out_size) 
{
    // Open the file
    FILE *f = fopen(path, "rb");
    CHECK(f != NULL, "fopen(%s): %s\n", path, strerror(errno));

    // Read data into a buffer
    CHECK(fseek(f, 0, SEEK_END) == 0, "fseek(%s): %s\n", path, strerror(errno));

    long size = ftell(f);
    CHECK(size >= 0, "ftell(%s): %s\n", path, strerror(errno));

    CHECK(fseek(f, 0, SEEK_SET) == 0, "fseek(%s) rewind: %s\n", path, strerror(errno));

    char *buf = (char *)malloc((size_t)size);
    CHECK(buf != NULL, "malloc(%ld) failed for %s\n", size, path);
    CHECK(fread(buf, 1, (size_t)size, f) == (size_t)size, "short read on %s\n", path);

    // Clean up
    fclose(f);

    // Return 
    *out_size = (size_t)size;
    return buf;
}

char* load_proto_blob(const char* filename, size_t* out_size) {
    FILE* f = fopen(filename, "rb");
    if (!f) return NULL;
    
    CHECK(fseek(f, 0, SEEK_END) == 0, "fseek(%s): %s\n", filename, strerror(errno));
    *out_size = ftell(f);
    CHECK(fseek(f, 0, SEEK_SET) == 0, "fseek(%s) rewind: %s\n", filename, strerror(errno));
    
    char* buffer = malloc(*out_size);
    fread(buffer, 1, *out_size, f);
    fclose(f);
    
    return buffer;
}


// main()
int main(int argc, char** argv)
{
    // Basic (very) CLI parsing
    if (argc < 3) {
        fprintf(stderr, "usage: <MLIR> <path-to-libpjrt_c_api_{cpu|gpu}.so>\n");
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

    // Getthe API
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
    const char* allocator_key = "allocator";
    const char* allocator_val = "address"; // BFC allocator is overkill for small model we are running
    PJRT_NamedValue client_options[1] = {
        {
            .struct_size = PJRT_NamedValue_STRUCT_SIZE,
            .extension_start = NULL,
            .name = allocator_key,
            .name_size = strlen(allocator_key),
            .type= PJRT_NamedValue_kString,
            .string_value = allocator_val,
            .value_size = strlen(allocator_val),
        },
    };

    PJRT_Client_Create_Args client_args = {
        .struct_size = PJRT_Client_Create_Args_STRUCT_SIZE,
        .create_options = client_options,
        .num_options = 1,
        .kv_get_callback = NULL,
        .kv_put_callback = NULL,
        .kv_put_user_arg = NULL,
        .client = NULL
    };
    CHECK_PJRT(api, api->PJRT_Client_Create(&client_args), "PJRT_Client_Create");
    PJRT_Client* client = client_args.client;

    // Read MLIR/StableHLO module IR
    size_t mlir_size = 0;
    char *mlir_text = read_file(mlir_path, &mlir_size);

    // Compile the IR
    static const char format[] = "mlir";
    PJRT_Program program = {
        .struct_size = PJRT_Program_STRUCT_SIZE,
        .code = mlir_text,
        .code_size = mlir_size,
        .format = format,
        .format_size = sizeof(format) - 1,
    };

    size_t proto_size = 0;
    char* proto_bytes = load_proto_blob("build/compile_options.bin", &proto_size);
    if (!proto_bytes) {
        fprintf(stderr, "Failed to load build/compile_options.bin\n");
        exit(1);
    }

    PJRT_Client_Compile_Args compile_args = {
        .struct_size = PJRT_Client_Compile_Args_STRUCT_SIZE,
        .extension_start = NULL,
        .client = client,
        .program = &program,
        .compile_options = proto_bytes,    // TODO: Optimize the compilation
        .compile_options_size = proto_size,
        .executable = NULL
    };

    CHECK_PJRT(api, api->PJRT_Client_Compile(&compile_args), "PJRT_Client_Compile");
    PJRT_LoadedExecutable* executable = compile_args.executable;

    printf("IR compiled.\n");
    free(mlir_text);
    free(proto_bytes);

    // Pick a client Device
    PJRT_Client_AddressableDevices_Args device_args = {
        .struct_size = PJRT_Client_AddressableDevices_Args_STRUCT_SIZE,
        .extension_start = NULL,
        .client = client,
        .addressable_devices = NULL,
        .num_addressable_devices = 0,
    };
    CHECK_PJRT(api, api->PJRT_Client_AddressableDevices(&device_args), "PJRT_Client_AddressableDevices");
    
    if (device_args.num_addressable_devices == 0) {
        fprintf(stderr, "No addressable devices\n");
        return 4;
    }
    PJRT_Device *device = device_args.addressable_devices[0];


    // Build random test input buffer for compilation
    size_t n_in = BATCH_SIZE * N;
    float* host_input = (float*) malloc(n_in * sizeof(float));
    srand(0);
    for (size_t i = 0; i < n_in; i++) {
        host_input[i] = (float)(rand() % 2000 - 1000) / 1000.0f;
    }

    // Copy data from host
    int64_t dims[INPUT_DIM] = {BATCH_SIZE, N};
    PJRT_Client_BufferFromHostBuffer_Args buf_args = {
        .struct_size = PJRT_Client_BufferFromHostBuffer_Args_STRUCT_SIZE,
        .extension_start = NULL,
        .client = client,
        .data = host_input,
        .type = PJRT_Buffer_Type_F32,
        .dims = dims,
        .num_dims = INPUT_DIM,
        .byte_strides = NULL,
        .num_byte_strides = 0,
        .host_buffer_semantics = PJRT_HostBufferSemantics_kImmutableUntilTransferCompletes,
        .device = device,
        .device_layout = NULL,
        .memory = NULL,

        .done_with_host_buffer = NULL,   // out
        .buffer = NULL,    // out
    };
    CHECK_PJRT(api, api->PJRT_Client_BufferFromHostBuffer(&buf_args), "PJRT_Client_BufferFromHostBuffer");
    PJRT_Buffer *input_buffer = buf_args.buffer;
    
    PJRT_Event *input_done_event = buf_args.done_with_host_buffer;


    // Execute options
    PJRT_ExecuteOptions exec_options = {
        .struct_size = PJRT_ExecuteOptions_STRUCT_SIZE,
        .extension_start = NULL,
        .send_callbacks = NULL,
        .recv_callbacks = NULL,
        .num_send_ops = 0,
        .num_recv_ops = 0,
        .launch_id = 0,
        .non_donatable_input_indices = NULL,
        .num_non_donatable_input_indices = 0,
        .context = NULL,
        .call_location = NULL,
        .num_tasks = 0,
        .task_ids = NULL,
        .incarnation_ids = NULL,
        .multi_slice_config = NULL,
        .use_major_to_minor_data_layout_for_callbacks = false,
        .hlo_output_callbacks = NULL,
        .num_hlo_output_callbacks = 0,
    };
    // Arguments to execution
    PJRT_Buffer *arg_list[1] = {input_buffer};
    PJRT_Buffer* const* argument_lists[1] = {arg_list};

    // Execution output
    PJRT_Buffer *output_storage[OUTPUT_DIM] = {0};
    PJRT_Buffer **output_lists[1] = {output_storage};

    // Execute model
    PJRT_LoadedExecutable_Execute_Args exec_args = {
        .struct_size = PJRT_LoadedExecutable_Execute_Args_STRUCT_SIZE,
        .extension_start = NULL,
        .executable = executable,
        .options = &exec_options,
        .argument_lists = argument_lists,
        .num_devices = 1,
        .num_args = 1,
        .output_lists = output_lists,
        .device_complete_events = NULL,
        .execute_device = device,
    };
    CHECK_PJRT(api, api->PJRT_LoadedExecutable_Execute(&exec_args), "PJRT_LoadedExecutable_Execute");

    // Examining output
    size_t output_num_elems[OUTPUT_DIM] = {0};
    float *host_output[OUTPUT_DIM] = {0};
    PJRT_Event *output_done_event[OUTPUT_DIM] = {0};

    for (int i = 0; i < OUTPUT_DIM; i++) {
        PJRT_Buffer_Dimensions_Args dim_args = {
            .struct_size = PJRT_Buffer_Dimensions_Args_STRUCT_SIZE,
            .extension_start = NULL,
            .buffer = output_storage[i],
        };
        CHECK_PJRT(api, api->PJRT_Buffer_Dimensions(&dim_args), "PJRT_Buffer_Dimensions");

        size_t num_elems = 1;
        for (size_t d = 0; d < dim_args.num_dims; d++) {
            num_elems *= (size_t)dim_args.dims[d];
        }

        output_num_elems[i] = num_elems;
        host_output[i] = (float*) malloc(num_elems * sizeof(float));

        PJRT_Buffer_ToHostBuffer_Args to_host_args = {
            .struct_size = PJRT_Buffer_ToHostBuffer_Args_STRUCT_SIZE,
            .extension_start = NULL,
            .src = output_storage[i],
            .dst = host_output[i],
            .dst_size = num_elems * sizeof(float),
            .event = NULL,  // out
        };

        CHECK_PJRT(api, api->PJRT_Buffer_ToHostBuffer(&to_host_args), "PJRT_Buffer_ToHostBuffer");
        output_done_event[i] = to_host_args.event;
    }
    
    // Await output transfers
    for (int i = 0; i < OUTPUT_DIM; i++) {
        if (!output_done_event[i]) 
            continue;

        PJRT_Event_Await_Args await_args = {
            .struct_size = PJRT_Event_Await_Args_STRUCT_SIZE,
            .extension_start = NULL,
            .event = output_done_event[i],
        };
        CHECK_PJRT(api, api->PJRT_Event_Await(&await_args), "PJRT_Event_Await(output)");
    }

    const char *names[OUTPUT_DIM] = {"S0", "S1", "S2"};
    for (int i = 0; i < OUTPUT_DIM; i++) {
        printf("%s: %zu elements, first values:", names[i], output_num_elems[i]);
        for (size_t k = 0; k < 5 && k < output_num_elems[i]; k++) {
            printf(" %f", host_output[i][k]);
        }
        printf("\n");
    }


    // Cleanup
    for (int i = 0; i < OUTPUT_DIM; i++) {
        free(host_output[i]);
        PJRT_Buffer_Destroy_Args d = {
            .struct_size = PJRT_Buffer_Destroy_Args_STRUCT_SIZE,
            .extension_start = NULL,
            .buffer = output_storage[i],
        };
        CHECK_PJRT(api, api->PJRT_Buffer_Destroy(&d), "PJRT_Buffer_Destroy(output)");
    }

    PJRT_Buffer_Destroy_Args in_destroy_args = {
        .struct_size = PJRT_Buffer_Destroy_Args_STRUCT_SIZE,
        .extension_start = NULL,
        .buffer = input_buffer,
    };
    CHECK_PJRT(api, api->PJRT_Buffer_Destroy(&in_destroy_args), "PJRT_Buffer_Destroy(input)");

    if (input_done_event) {
        PJRT_Event_Await_Args await_args = {
            .struct_size = PJRT_Event_Await_Args_STRUCT_SIZE,
            .extension_start = NULL,
            .event = input_done_event,
        };
        CHECK_PJRT(api, api->PJRT_Event_Await(&await_args), "PJRT_Event_Await(input)");
    }
    free(host_input);
 
    PJRT_LoadedExecutable_Destroy_Args exec_destroy_args = {
        .struct_size = PJRT_LoadedExecutable_Destroy_Args_STRUCT_SIZE,
        .extension_start = NULL,
        .executable = executable,
    };
    CHECK_PJRT(api, api->PJRT_LoadedExecutable_Destroy(&exec_destroy_args), "PJRT_LoadedExecutable_Destroy");
 
    PJRT_Client_Destroy_Args client_destroy_args = {
        .struct_size = PJRT_Client_Destroy_Args_STRUCT_SIZE,
        .extension_start = NULL,
        .client = client,
    };
    CHECK_PJRT(api, api->PJRT_Client_Destroy(&client_destroy_args), "PJRT_Client_Destroy");
 
    dlclose(pjrt_handle);
    return 0;

}