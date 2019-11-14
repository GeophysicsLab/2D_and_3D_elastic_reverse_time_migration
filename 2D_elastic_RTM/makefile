#! /bin/sh

INCLUDES = /usr/local/cuda-5.5/samples/common/inc
 INC=${CWPROOT}/include
 LIK=${CWPROOT}/lib
 LIB=-lsu -lpar -lcwp -lm


ALL:1


1:1.o
	nvcc -o 1_gpu -g 1.o -lm -O3 -I$(INC) -L$(LIK) $(LIB)

1.o:1.cu
	nvcc -c 1.cu -I$(INCLUDES) -O3 -I$(INC) -L$(LIK) $(LIB)

clean:
	rm *.o

#b:
#	nvcc -o b -c b.cu -lm -I$(INCLUDES)

