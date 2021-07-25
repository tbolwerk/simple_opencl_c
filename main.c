#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <sys/types.h>
#ifdef __APPLE__
#include <OpenCL/opencl.h>
#include <unistd.h>
#else
#include <CL/cl.h>
#endif
#include <string.h>

#define ADD "addition"
#define MULT "multiplication"
#define MAX_SOURCE_SIZE (0x100000)

//pick up device type from compiler command line or from
//the default type
#ifndef DEVICE
#define DEVICE CL_DEVICE_TYPE_ALL
#endif

// Configure these params, to your liking
#define SHOW_OUTPUT false
#define KERNEL_NAME ADD


extern int output_device_info(cl_device_id);

void checkError(int err, char *msg)
{
}

int output_device_info(cl_device_id id)
{
    size_t valueSize;
    clGetDeviceInfo(id, CL_DEVICE_NAME, 0, NULL, &valueSize);
    char *value = malloc(valueSize);
    clGetDeviceInfo(id, CL_DEVICE_NAME, valueSize, value, NULL);
    printf("%s", value);
    free(value);
    return 0;
}

char *err_code(int err)
{
    return "error";
}

typedef const struct Kernel
{
    const char *name;
    char *source;
    size_t size;
} Kernel;

char **split(const char *source, char *delim)
{
    char *token, *string, *tofree;
    char **result = malloc(sizeof(char) * 120);
    tofree = string = strdup(source);

    while ((token = strsep(&string, delim)) != NULL)
    {
        *result = malloc(sizeof(char) * 120);
        *result = token;
    }
    free(tofree);
    return result;
}

// Provide kernel path with specific naming scheme, such that
// prefix_name.cl
// must contain prefix underscore name and extension .cl

const char *format(const char *kernelPath)
{
    char **unprefixed = split(kernelPath, "_");
    const char *name = strsep(unprefixed, ".");
    return name;
}

Kernel loadKernel(const char *kernelPath)
{
    FILE *fp;
    char *sourceString;
    size_t sourceSize;
    fp = fopen(kernelPath, "r");
    if (!fp)
    {
        fprintf(stderr, "Failed to load kernel.cl\n");
        exit(1);
    }
    sourceString = (char *)malloc(MAX_SOURCE_SIZE);
    sourceSize = fread(sourceString, 1, MAX_SOURCE_SIZE, fp);
    fclose(fp);
    const char *name = format(kernelPath);
    Kernel kernel = {name, sourceString, sourceSize};
    return kernel;
}

cl_kernel createKernel(cl_program program, const char *kernel_name, cl_int errcode_ret)
{
    return clCreateKernel(program, KERNEL_NAME, &errcode_ret);
}

int main(int argc, char **argv)
{

    uint size;

    if (argc == 2)
    {
        size = atoi(argv[1]);
    }
    else
    {
        fprintf(stderr, "needs 1 argument, but got: %d", argc);
        exit(1);
    }
    // Declarations
    uint *A = (uint *)calloc(sizeof(uint), size);
    uint *B = (uint *)calloc(sizeof(uint), size);
    uint *C = (uint *)calloc(sizeof(uint), size);
    // Initalizations
    for (int i = 0; i < size; i++)
    {
        A[i] = i + 1;
        B[i] = i + 1;
    }
    char prefix[100] = "kernel_";
    char *withPrefix = malloc(sizeof(char) * 200);
    withPrefix = strcat(prefix, KERNEL_NAME);
    char extension[10] = ".cl";
    char *kernelPath = malloc(sizeof(char) * 300);
    kernelPath = strcat(withPrefix, extension);
    printf("Loading kernel: %s\n", kernelPath);
    Kernel kernel = loadKernel(kernelPath);
    printf("Succesfully loaded kernel: %s\n", kernel.name);
    int err; // error code returned from OpenCL calls

    size_t global; // global domain size

    cl_device_id device_id;    // compute device id
    cl_context context;        // compute context
    cl_command_queue commands; // compute command queue
    cl_program program;        // compute program
    cl_kernel ko_vadd;         // compute kernel

    cl_mem d_a; // device memory used for the input  a vector
    cl_mem d_b; // device memory used for the input  b vector
    cl_mem d_c; // device memory used for the output c vector

    // Set up platform and GPU device
    cl_uint numPlatforms;

    // Find number of platforms
    err = clGetPlatformIDs(0, NULL, &numPlatforms);
    checkError(err, "Finding platforms");
    if (numPlatforms == 0)
    {
        printf("Found 0 platforms!\n");
        return EXIT_FAILURE;
    }

    // Get all platforms
    cl_platform_id Platform[numPlatforms];
    cl_uint deviceCount;
    cl_device_id *devices;
    err = clGetPlatformIDs(numPlatforms, Platform, NULL);
    checkError(err, "Getting platforms");

    // Secure a GPU
    for (int i = 0; i < numPlatforms; i++)
    {
        // get all devices
        clGetDeviceIDs(Platform[i], DEVICE, 0, NULL, &deviceCount);
        devices = (cl_device_id *)malloc(sizeof(cl_device_id) * deviceCount);
        clGetDeviceIDs(Platform[i], DEVICE, deviceCount, devices, NULL);

        for (int j = 0; j < deviceCount; j++)
        {
            device_id = devices[j];
            if (device_id == NULL)
                checkError(err, "Finding a device");

            checkError(err, "Printing device output");

            // Create a compute context
            context = clCreateContext(0, 1, &device_id, NULL, NULL, &err);
            checkError(err, "Creating context");

            // Create a command queue
            commands = clCreateCommandQueue(context, device_id, 0, &err);
            checkError(err, "Creating command queue");

            // Create the compute program from the source buffer
            program = clCreateProgramWithSource(context, 1, (const char **)&kernel.source, NULL, &err);
            checkError(err, "Creating program");

            // Build the program
            err = clBuildProgram(program, 0, NULL, NULL, NULL, NULL);
            if (err != CL_SUCCESS)
            {
                size_t len;
                char buffer[2048];

                printf("Error: Failed to build program executable!\n%s\n", err_code(err));
                clGetProgramBuildInfo(program, device_id, CL_PROGRAM_BUILD_LOG, sizeof(buffer), buffer, &len);
                printf("%s\n", buffer);
                return EXIT_FAILURE;
            }
            // Create the compute kernel from the program
            const char *kernel_name = strdup(kernel.name);
            ko_vadd = createKernel(program, kernel_name, err);
            checkError(err, "Creating kernel");

            // Create the input (a, b) and output (c) arrays in device memory
            d_a = clCreateBuffer(context, CL_MEM_READ_ONLY, sizeof(uint) * size, NULL, &err);
            checkError(err, "Creating buffer d_a");

            d_b = clCreateBuffer(context, CL_MEM_READ_ONLY, sizeof(uint) * size, NULL, &err);
            checkError(err, "Creating buffer d_b");

            d_c = clCreateBuffer(context, CL_MEM_WRITE_ONLY, sizeof(uint) * size, NULL, &err);
            checkError(err, "Creating buffer d_c");

            // Write a and b vectors into compute device memory
            err = clEnqueueWriteBuffer(commands, d_a, CL_TRUE, 0, sizeof(float) * size, A, 0, NULL, NULL);
            checkError(err, "Copying h_a to device at d_a");

            err = clEnqueueWriteBuffer(commands, d_b, CL_TRUE, 0, sizeof(float) * size, B, 0, NULL, NULL);
            checkError(err, "Copying h_b to device at d_b");

            // Set the arguments to our compute kernel
            err = clSetKernelArg(ko_vadd, 0, sizeof(cl_mem), &d_a);
            err |= clSetKernelArg(ko_vadd, 1, sizeof(cl_mem), &d_b);
            err |= clSetKernelArg(ko_vadd, 2, sizeof(cl_mem), &d_c);
            err |= clSetKernelArg(ko_vadd, 3, sizeof(unsigned int), &size);
            checkError(err, "Setting kernel arguments");

            // Start timer
            clock_t begin = clock();

            // Execute the kernel over the entire range of our 1d input data set
            // letting the OpenCL runtime choose the work-group size
            global = size;
            err = clEnqueueNDRangeKernel(commands, ko_vadd, 1, NULL, &global, NULL, 0, NULL, NULL);
            checkError(err, "Enqueueing kernel");

            // Wait for the commands to complete before stopping the timer
            err = clFinish(commands);
            checkError(err, "Waiting for kernel to finish");

            // Stop timer
            clock_t end = clock();
            double time_spent = (double)(end - begin) / CLOCKS_PER_SEC;

            // Read back the results from the compute device
            err = clEnqueueReadBuffer(commands, d_c, CL_TRUE, 0, sizeof(uint) * size, C, 0, NULL, NULL);
            if (err != CL_SUCCESS)
            {
                printf("Error: Failed to read output array!\n%s\n", err_code(err));
                exit(1);
            }
            if (SHOW_OUTPUT)
            {
                for (int i = 0; i < size; i++)
                {
                    output_device_info(device_id);
                    printf("\t");
                    printf("%d+%d=%d\n", A[i], B[i], C[i]);
                }
            }

            printf("%f in seconds\t", time_spent);
            err = output_device_info(device_id);
            printf("\n");
        }
    }

    // cleanup then shutdown
    clReleaseMemObject(d_a);
    clReleaseMemObject(d_b);
    clReleaseMemObject(d_c);
    clReleaseProgram(program);
    clReleaseKernel(ko_vadd);
    clReleaseCommandQueue(commands);
    clReleaseContext(context);

    free(A);
    free(B);
    free(C);

    return 0;
}