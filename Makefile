ifndef CC
	CC = gcc
endif

CCFLAGS=-O3 -std=c99 -Wall

LIBS = -lOpenCL -fopenmp -lm

COMMON_DIR = .

# Check our platform and make sure we define the APPLE variable
# and set up the right compiler flags and libraries
PLATFORM = $(shell uname -s)
ifeq ($(PLATFORM), Darwin)
	LIBS = -framework OpenCL -lm
endif


main: main.c
	$(CC) $^ $(CCFLAGS) $(LIBS) -I $(COMMON_DIR) -o $@


clean:
	rm -f main
