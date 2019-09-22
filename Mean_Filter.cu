#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <iostream>
#include <cuda.h>

//Operations to be done by threads
__global__ void mean_filter_apply(int *image, int *filtered_image, int imWidth, int imHeight, int kernalSize){

    int kernal = 0;
    int windowSize = (2*kernalSize+1)*(2*kernalSize+1);

    int i = threadIdx.x + blockDim.x*blockIdx.x;
    int j = threadIdx.y + blockDim.y*blockIdx.y;

    if(i < imWidth && j < imHeight){

        if (i < kernalSize|| i >= imWidth - kernalSize || j < kernalSize || j >= imHeight - kernalSize){
            filtered_image[j*imWidth + i] = 0;
        }
        else{
            for (int kernalH = -kernalSize; kernalH <= kernalSize; kernalH++){
                for (int kernalW = -kernalSize; kernalW <= kernalSize; kernalW++){
                    kernal = kernal + image[(j+kernalH)*imWidth + i + kernalW];
                }
            }
            filtered_image[j*imWidth + i] = kernal/windowSize;
        }
    }
}
//Function to run the GPU mean filtering process
void mean_filter_GPU(int *image, int *filteredImage, int imWidth, int imHeight, int kernalSize, double *time){

    int * image_in_gpu;
    int * filtered_image_in_gpu;

    int sizeofImage = imHeight*imWidth;

    //
    cudaMalloc((void **) &image_in_gpu, sizeofImage*sizeof(int));
    cudaMalloc((void **) &filtered_image_in_gpu, sizeofImage*sizeof(int));

    // printf("Copying images to device..\n");
    cudaMemcpy(image_in_gpu, image, sizeofImage*sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(filtered_image_in_gpu, filteredImage, sizeofImage*sizeof(int), cudaMemcpyHostToDevice);

    //Memory pointers and other required parameters for GPU function    
    dim3 dimBlock(40,20);
    int w = (imWidth/40) + 1;
    int h = (imHeight/20) + 1;
    dim3 dimGrid(w,h);

    // printf("Doing GPU Filtering\n");
    clock_t start=clock();
    mean_filter_apply<<<dimGrid, dimBlock>>>(image_in_gpu, filtered_image_in_gpu, imWidth, imHeight ,kernalSize);
    cudaDeviceSynchronize();
    clock_t end = clock();

    cudaMemcpy(filteredImage, filtered_image_in_gpu, sizeofImage*sizeof(int), cudaMemcpyDeviceToHost);
    *time = (double)(end-start)/CLOCKS_PER_SEC;

    cudaFree(image_in_gpu);
    cudaFree(filtered_image_in_gpu);
    
}
//Function to run the CPU mean filtering process
void mean_filter_CPU(int * image, int *filteredImage, int imWidth, int imHeight, int kernalSize, double *time){

    int kernal;
    long int windowSize = (2*kernalSize+1)*(2*kernalSize+1);

    clock_t start_h = clock();
    for (int i=0; i < imHeight; i++){
        for (int j=0;j < imWidth;j++){
            kernal = 0;
            if(i < kernalSize || i >= imHeight - kernalSize || j < kernalSize || j >= imWidth - kernalSize){
                filteredImage[i*imWidth + j] = 0;
            }
            else {
                for (int kernalH = -kernalSize; kernalH <= kernalSize; kernalH++){
                    for (int kernalW = -kernalSize; kernalW <= kernalSize; kernalW++){
                        kernal = kernal + image[(i+kernalH)*imWidth + j + kernalW];
                    }
                }
                filteredImage[i*imWidth + j] = kernal/windowSize;
            }
        }
    }
    clock_t end_h = clock();
    *time = (double)(end_h-start_h)/CLOCKS_PER_SEC;
}

int main(){

    int imWidth = 1280;
    int imHeight = 1280;

    int *imageTest;
    int *filteredTest_CPU;
    int *filteredTest_GPU;

    //Allocating memory to the image and filtered image
    imageTest = (int *) malloc(imWidth*imHeight*sizeof(int));
    filteredTest_CPU = (int *) malloc(imWidth*imHeight*sizeof(int));
    filteredTest_GPU = (int *) malloc(imWidth*imHeight*sizeof(int));

    //Filling the Test image with values
    for (int i=0; i < imHeight; i++){
        for (int j=0; j < imWidth; j++){
            imageTest[(i*imWidth) + j]  = i*j;
        }
    }
    
    double time_cpu;
    double time_gpu;

    for (int k = 0; k< 5; k++){
        mean_filter_CPU( (int *)imageTest, (int *)filteredTest_CPU, imWidth, imHeight, 2, &time_cpu);
        printf("CPU Processing Time : %f \n", time_cpu);

        mean_filter_GPU( (int *)imageTest, (int *)filteredTest_GPU, imWidth, imHeight, 2, &time_gpu);
        printf("GPU Processing Time %f \n", time_gpu);
    }
    
    return 1;
}