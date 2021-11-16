# simple_opencl_c

## Prerequisite

Install OpenCL for you specific OS



## Ubuntu install 

```
sudo apt update
sudo apt install make
sudo apt install ocl-icd-opencl-dev
```

## Mac os

```
brew install make
```

Reference: [OpenCL documentation Apple](https://developer.apple.com/library/archive/documentation/Performance/Conceptual/OpenCL_MacProgGuide/Introduction/Introduction.html)



# How to run

```
make
./main $(cat input.txt)
```

# How to clean

```
make clean
```