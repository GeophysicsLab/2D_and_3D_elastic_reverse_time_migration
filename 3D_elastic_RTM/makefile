#! /bin/sh

INCLUDES = /usr/local/cuda-5.5/samples/common/inc
 INC=${CWPROOT}/include
 LIK=${CWPROOT}/lib
 LIB=-lsu -lpar -lcwp -lm -lcublas


ALL:3D_elastic_modeling


3D_elastic_modeling:3D_elastic_modeling.o
	nvcc -o 3D_elastic_modeling_gpu -g 3D_elastic_modeling.o -lm -O3 -I$(INC) -L$(LIK) $(LIB)

3D_elastic_modeling.o:3D_elastic_modeling.cu
	nvcc -c 3D_elastic_modeling.cu -I$(INCLUDES) -O3 -I$(INC) -L$(LIK) $(LIB)

clean:
	rm *.o

#b:
#	nvcc -o b -c b.cu -lm -I$(INCLUDES)

