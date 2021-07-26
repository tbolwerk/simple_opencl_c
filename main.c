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

#define MAX_SOURCE_SIZE (0x100000)

//pick up device type from compiler command line or from
//the default type
#ifndef DEVICE
#define DEVICE CL_DEVICE_TYPE_ALL
#endif

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
//a
char *err_code(int err)
{
    return "error";
}

typedef const struct Kernel
{
    char *source;
    size_t size;
} Kernel;

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
    Kernel kernel = {sourceString, sourceSize};
    return kernel;
}


int executeWithOpenCL(int size, int * A, int * B, int * C, const char * kernelPath, const char * kernelName){

    Kernel kernel = loadKernel(kernelPath);

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
    cl_int numPlatforms;

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
    cl_int deviceCount;
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

            ko_vadd = clCreateKernel(program, kernelName, &err);
            checkError(err, "Creating kernel");

            // Create the input (a, b) and output (c) arrays in device memory
            d_a = clCreateBuffer(context, CL_MEM_READ_ONLY, sizeof(int) * size, NULL, &err);
            checkError(err, "Creating buffer d_a");

            d_b = clCreateBuffer(context, CL_MEM_READ_ONLY, sizeof(int) * size, NULL, &err);
            checkError(err, "Creating buffer d_b");

            d_c = clCreateBuffer(context, CL_MEM_WRITE_ONLY, sizeof(int) * size, NULL, &err);
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
            err = clEnqueueReadBuffer(commands, d_c, CL_TRUE, 0, sizeof(int) * size, C, 0, NULL, NULL);
            if (err != CL_SUCCESS)
            {
                printf("Error: Failed to read output array!\n%s\n", err_code(err));
                exit(1);
            }
            for (int i = 0; i < size; i++)
            {
                output_device_info(device_id);
                printf("\t");
                printf("%d+%d=%d\n", A[i], B[i], C[i]);
            }

            printf("%f in seconds\t", time_spent);
            err = output_device_info(device_id);
            printf("\n");
            // cleanup then shutdown
            clReleaseContext(context);
            clReleaseMemObject(d_a);
            clReleaseMemObject(d_b);
            clReleaseMemObject(d_c);
            clReleaseProgram(program);
            clReleaseKernel(ko_vadd);
            clReleaseCommandQueue(commands);
        }
    }
    return err;
}


int main(int argc, char **argv)
{

    if ((argc - 1) % 2 != 0)
    {
        fprintf(stderr, "Number of input params needs to be even");
        exit(1);
    }

    int size = (argc) / 2;

    // Declarations
    int *A = (int *)calloc(sizeof(int), size);
    int *B = (int *)calloc(sizeof(int), size);
    int *C = (int *)calloc(sizeof(int), size);

    // Initalization from even input
    for (int i = 0; i < size; i++)
    {
        int a = atoi(argv[i + 1]);
        A[i] = a;

        int b = atoi(argv[i + size + 1]);
        B[i] = b;
    }

    int err = executeWithOpenCL(size, A, B, C,"kernel_addition.cl","addition");

    if(err != 0){
        fprintf(stderr, "Something went wrong while executing openCL");
        exit(1);
    }

    free(A);
    free(B);
    free(C);

    return 0;
}
