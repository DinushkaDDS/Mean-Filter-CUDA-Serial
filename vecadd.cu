#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <iostream>
#include <cuda.h>

float *a, *b;  // host data
float *c, *c2;  // results


//GPU Vector Adding Function
__global__ void vecAdd(float *A,float *B,float *C, int N){ 
   if(blockIdx.x*blockDim.x + threadIdx.x < N){
      float value = A[blockIdx.x*blockDim.x + threadIdx.x] + B[blockIdx.x*blockDim.x + threadIdx.x];
      C[blockIdx.x*blockDim.x + threadIdx.x] = value;
   }  
}

//CPU Vector Adding Function
void vecAdd_h(float *A1,float *B1, float *C1, int N){
   for(int i=0;i<N;i++){
      C1[i] = A1[i] + B1[i];
   }
}

int main(int argc,char **argv){

   printf("Begin \n");
   //Declaring number of elements in the Vector
   long int n=10000000  ;     //100, 10000,1000000 and 10000000
   long int nBytes = n*sizeof(float);
   int block_size, block_no;

   //Memory allocating for the vector arrays
   a = (float *)malloc(nBytes);
   b = (float *)malloc(nBytes);
   c = (float *)malloc(nBytes);
   c2 = (float *)malloc(nBytes);

   //Memory pointers and other required parameters for GPU function
   float *a_d,*b_d,*c_d;
   block_size=1024;
   block_no = (n/block_size) + 1;

   dim3 dimBlock(block_size,1,1);
   dim3 dimGrid(block_no,1,1);
   

//Assigning values to the created matrices
   for(int i = 0; i < n; i++ ){
        a[i] = sin(i)*sin(i);
        b[i] = cos(i)*cos(i);
   }

   printf("Allocating device memory on host..\n");
   cudaMalloc((void **)&a_d, n*sizeof(float));
   cudaMalloc((void **)&b_d, n*sizeof(float));
   cudaMalloc((void **)&c_d, n*sizeof(float));

   printf("Copying to device..\n");
   cudaMemcpy(a_d, a, n*sizeof(int), cudaMemcpyHostToDevice);
   cudaMemcpy(b_d, b, n*sizeof(int), cudaMemcpyHostToDevice);

   printf("Doing GPU Vector add\n");
   clock_t start_d=clock();
   vecAdd<<<dimGrid, dimBlock>>>(a_d, b_d, c_d, n);
   cudaDeviceSynchronize();
   clock_t end_d = clock();
   cudaMemcpy(c, c_d, n*sizeof(float), cudaMemcpyDeviceToHost);
   
   // printf("I m in GPU vector matrix\n");
   // for (int i = 0; i <n;i++){
   //    printf("%f ", c[i]);
   // }

   printf("\nDoing CPU Vector add\n");
   clock_t start_h = clock();
   vecAdd_h(a,b,c2,n);
   clock_t end_h = clock();
   
   // printf("I m in cpu vector matrix\n");
   // for (int i = 0; i <n; i++){
   //    printf("%f ", c2[i]);
   // }

   double time_d = (double)(end_d-start_d)/CLOCKS_PER_SEC;
   double time_h = (double)(end_h-start_h)/CLOCKS_PER_SEC;

   printf("Number of elements: %li GPU Time: %f CPU Time: %f\n",n, time_d, time_h);
   cudaFree(a_d);
   cudaFree(b_d);
   cudaFree(c_d);
   
   return 0;

}