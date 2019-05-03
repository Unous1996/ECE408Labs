#include <wb.h>

#define BLOCK_SIZE 256 //@@ You can change this
#define SEGMENT_SIZE 512

__constant__ float SUMS[2*BLOCK_SIZE];

#define wbCheck(stmt)                                                     \
  do {                                                                    \
    cudaError_t err = stmt;                                               \
    if (err != cudaSuccess) {                                             \
      wbLog(ERROR, "Failed to run stmt ", #stmt);                         \
      wbLog(ERROR, "Got CUDA error ...  ", cudaGetErrorString(err));      \
      return -1;                                                          \
    }                                                                     \
  } while (0)

__global__ void scan(float *input, float *output, int len){
  //@@ Modify the body of this function to complete the functionality of
  //@@ the scan on the device
  //@@ You may need multiple kernel calls; write your kernels before this
  //@@ function and call them from the host
  __shared__ float T[2*BLOCK_SIZE];
  int stride = 1;
  int tx = threadIdx.x, bx = blockIdx.x;
  int i =  2*bx*blockDim.x + tx;
  //Load input into the shared memory
  //Step 1
  if(i < len){
    T[tx] = input[i];
  }
  else{
    T[tx] = 0;
  }
  
  if(i + blockDim.x < len){
    T[tx + blockDim.x] = input[i + blockDim.x];
  }
  else{
    T[tx + blockDim.x] = 0;
  }
  
  while(stride < 2*BLOCK_SIZE){
    __syncthreads();
    int index = (tx + 1) * stride * 2 - 1;
    if(index < 2*BLOCK_SIZE && index - stride >= 0){
      T[index] += T[index - stride];
    }
    stride = stride * 2;
  }
  
  stride = BLOCK_SIZE / 2;
  while(stride > 0){
    __syncthreads();
    int index = (tx + 1)*stride*2 - 1;
    if((index + stride) < 2 * BLOCK_SIZE)
      T[index + stride] += T[index];
    stride = stride / 2;
  }
  
  __syncthreads();
  
  if(i < len){
    //printf("bx = %d, i = %d\n",bx,i);
    output[i] = T[tx];
    //printf("output [%d] = %f\n", i, output[i]);
  }
  
  if(i + blockDim.x < len){
    //output[i + blockDim.x] = 0; 
    //printf("bx = %d, i = %d\n",bx, i + blockDim.x);
    output[i + blockDim.x] = T[tx + blockDim.x];
    //printf("i + blockDim.x = %d\n",i + blockDim.x);
    //printf("output [%d] = %f\n",i + blockDim.x, output[i + blockDim.x]);
  }
  
}
 
__global__ void scan3(float *input, float *output, int data_len){
  //Kernel for single block
  //You can use blockDim.x to reference number_of_blocks in the previous kernel
  
  __shared__ float T[2*BLOCK_SIZE];
  int stride = 1;
  int tx = threadIdx.x;
  int i = tx*2*BLOCK_SIZE-1;
 
  if(i >= 0 && i < data_len){
    T[tx] = input[i];
  }
  else{
    T[tx] = 0;
  }
  
  while(stride < 2*BLOCK_SIZE){
    __syncthreads();
    int index = (tx + 1) * stride * 2 - 1;
    if(index < 2*BLOCK_SIZE && index - stride >= 0){
      T[index] += T[index - stride];
    }
    stride = stride * 2;
  }
  
  __syncthreads();
  
  stride = BLOCK_SIZE / 2;
  while(stride > 0){
    __syncthreads();
    int index = (tx + 1)*stride*2 - 1;
    if((index + stride) < 2 * BLOCK_SIZE)
      T[index + stride] += T[index];
    stride = stride / 2;
  }
  
  output[tx] = T[tx];
  __syncthreads();
  
}

__global__ void merge(float* sums, float *output, int len){
  int bx = blockIdx.x;
  int tx = threadIdx.x;
  int i =  bx * blockDim.x + tx;
  int debug = 768;
  if(i < len){
    if(i == debug){
      printf("output = %f\n", output[debug]);
    }
    int temp = output[i];
    output[i] = temp + SUMS[bx];
    if(i == debug){
      printf("output = %f\n", output[debug]);
    }
  }
}

int main(int argc, char **argv){
  wbArg_t args;
  float *hostInput;  // The input 1D list
  float *hostOutput; // The output list
  float *hostSums;

  float *deviceInput;
  float *deviceOutput;
  float *deviceSums;

  int numElements; // number of elements in the list

  args = wbArg_read(argc, argv);

  wbTime_start(Generic, "Importing data and creating memory on host");
  hostInput = (float *)wbImport(wbArg_getInputFile(args, 0), &numElements);
  hostOutput = (float *)malloc(numElements * sizeof(float));
  wbTime_stop(Generic, "Importing data and creating memory on host");

  wbLog(TRACE, "The number of input elements in the input is ",
        numElements);

  int number_of_blocks = numElements / (2 * BLOCK_SIZE);
  if(numElements % (2 * BLOCK_SIZE) != 0){
    number_of_blocks += 1;
  }
  
  hostSums = (float *)malloc(number_of_blocks * sizeof(float));
  wbLog(TRACE, "The number of blocks in the first stage is :", number_of_blocks);
  wbTime_start(GPU, "Allocating GPU memory.");
  wbCheck(cudaMalloc((void **)&deviceInput, numElements * sizeof(float)));
  wbCheck(cudaMalloc((void **)&deviceOutput, numElements * sizeof(float)));
  wbCheck(cudaMalloc((void **)&deviceSums, number_of_blocks * sizeof(float)));
  
  wbTime_stop(GPU, "Allocating GPU memory.");

  wbTime_start(GPU, "Clearing output memory.");
  wbCheck(cudaMemset(deviceOutput, 0, numElements * sizeof(float)));
  wbCheck(cudaMemset(deviceSums, 0, number_of_blocks * sizeof(float)));
  wbTime_stop(GPU, "Clearing output memory.");

  wbTime_start(GPU, "Copying input memory to the GPU.");
  wbCheck(cudaMemcpy(deviceInput, hostInput, numElements * sizeof(float),
                     cudaMemcpyHostToDevice));
  wbTime_stop(GPU, "Copying input memory to the GPU.");
  //@@ Initialize the grid and block dimensions here

  scan<<<number_of_blocks, BLOCK_SIZE>>>(deviceInput, deviceOutput, numElements);
  scan3<<<1, number_of_blocks>>>(deviceOutput, deviceSums, numElements);
  wbCheck(cudaMemcpy(hostSums, deviceSums, number_of_blocks * sizeof(float), cudaMemcpyDeviceToHost));
  wbCheck(cudaMemcpyToSymbol(SUMS, hostSums, number_of_blocks*sizeof(float)));
  /*
  int number_of_blocks_for_merge = numElements/(BLOCK_SIZE);
  if(numElements % BLOCK_SIZE != 0){
    number_of_blocks_for_merge += 1;
  }
  wbLog(TRACE, "The number of the blocks in the second stage is:", number_of_blocks_for_merge);
  */
  merge<<<number_of_blocks, BLOCK_SIZE*2>>>(deviceSums, deviceOutput, numElements);
  
  wbTime_start(Compute, "Performing CUDA computation");
  //@@ Modify this to complete the functionality of the scan
  //@@ on the deivce

  cudaDeviceSynchronize();
  wbTime_stop(Compute, "Performing CUDA computation");

  wbTime_start(Copy, "Copying output memory to the CPU");
  wbCheck(cudaMemcpy(hostOutput, deviceOutput, numElements * sizeof(float),
                     cudaMemcpyDeviceToHost));
  wbTime_stop(Copy, "Copying output memory to the CPU");

  wbTime_start(GPU, "Freeing GPU Memory");
  cudaFree(deviceInput);
  cudaFree(deviceOutput);
  cudaFree(deviceSums);
  wbTime_stop(GPU, "Freeing GPU Memory");

  wbSolution(args, hostOutput, numElements);

  free(hostInput);
  free(hostOutput);
  free(hostSums);

  return 0;
}