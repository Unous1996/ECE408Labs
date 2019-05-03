#define wbCheck(stmt)                                                     \
  do {                                                                    \
    cudaError_t err = stmt;                                               \
    if (err != cudaSuccess) {                                             \
      wbLog(ERROR, "Failed to run stmt ", #stmt);                         \
      wbLog(ERROR, "Got CUDA error ...  ", cudaGetErrorString(err));      \
      return -1;                                                          \
    }                                                                     \
  } while (0)

// Compute C = A * B
/*
__global__ void matrixMultiply(float *A, float *B, float *C, int numARows,
                               int numAColumns, int numBRows,
                               int numBColumns, int numCRows,
                               int numCColumns) {
  //@@ Insert code to implement matrix multiplication here
  int Row = blockIdx.y * blockDim.y + threadIdx.y;
  int Col = blockIdx.x * blockDim.x + threadIdx.x;
  if(Row < numARows && Col < numBColumns){
    float pValue = 0;
    for(int k=0; k<numAColumns; k++){
      pValue += A[Row*numAColumns + k] * B[k*numBColumns + Col];
    }
    C[Row*numCColumns + Col] = pValue;
  }
}
*/

__global__ void matrixMultiplyShared(float *A, float *B, float *C,
                                     int numARows, int numAColumns,
                                     int numBRows, int numBColumns,
                                     int numCRows, int numCColumns){
  //@@ Insert code to implement matrix multiplication here
  //@@ You have to use shared memory for this MP
  const int TILE_WIDTH = 32;
  __shared__ float MdA[TILE_WIDTH][TILE_WIDTH];
  __shared__ float MdB[TILE_WIDTH][TILE_WIDTH];
  int bx = blockIdx.x, by = blockIdx.y, tx = threadIdx.x, ty = threadIdx.y;
  int Row = by * TILE_WIDTH + ty;
  int Col = bx * TILE_WIDTH + tx;
  
  float pValue = 0;
  for(int ph=0; ph < ceil(numAColumns*1.0/TILE_WIDTH); ph++){
      //Collaborate Loading SubPhase
    if(Row < numARows){
       MdA[ty][tx] = A[Row*numAColumns + ph*TILE_WIDTH + tx];
    }
    else{
      MdA[ty][tx] = 0;
    }
    
    if(ph*TILE_WIDTH+ty < numBRows && Col < numBColumns){
       MdB[ty][tx] = B[(ph*TILE_WIDTH+ty)*numBColumns + Col];
    }
    else{
       MdB[ty][tx] = 0;
    }
    __syncthreads();
    for(int k=0; k<TILE_WIDTH; k++){
      pValue += MdA[ty][k] * MdB[k][tx];
    }
    __syncthreads();
  }
    
  if(Row < numCRows && Col < numCColumns){
    C[Row*numCColumns + Col] = pValue;
  }
}

int main(int argc, char **argv) {
  wbArg_t args;
  float *hostA; // The A matrix
  float *hostB; // The B matrix
  float *hostC; // The output C matrix
  float *deviceA;
  float *deviceB;
  float *deviceC;
  int numARows;    // number of rows in the matrix A
  int numAColumns; // number of columns in the matrix A
  int numBRows;    // number of rows in the matrix B
  int numBColumns; // number of columns in the matrix B
  int numCRows;    // number of rows in the matrix C (you have to set this)
  int numCColumns; // number of columns in the matrix C (you have to set
                   // this)
  
  args = wbArg_read(argc, argv);
  
  wbTime_start(Generic, "Importing data and creating memory on host");
  hostA = (float *)wbImport(wbArg_getInputFile(args, 0), &numARows,
                            &numAColumns);
  hostB = (float *)wbImport(wbArg_getInputFile(args, 1), &numBRows,
                            &numBColumns);
  //@@ Set numCRows and numCColumns
  numCRows = numARows;
  numCColumns = numBColumns;
  //@@ Allocate the hostC matrix
  hostC = (float*)malloc(numCRows * numCColumns * sizeof(float));
  wbTime_stop(Generic, "Importing data and creating memory on host");

  wbLog(TRACE, "The dimensions of A are ", numARows, " x ", numAColumns);
  wbLog(TRACE, "The dimensions of B are ", numBRows, " x ", numBColumns);

  wbTime_start(GPU, "Allocating GPU memory.");
  //@@ Allocate GPU memory here
  cudaMalloc((void**)&deviceA, numARows * numAColumns * sizeof(float));
  cudaMalloc((void**)&deviceB, numBRows * numBColumns * sizeof(float));
  cudaMalloc((void**)&deviceC, numCRows * numCColumns * sizeof(float));
  wbTime_stop(GPU, "Allocating GPU memory.");

  wbTime_start(GPU, "Copying input memory to the GPU.");
  //@@ Copy memory to the GPU here
  cudaMemcpy(deviceA, hostA, numARows * numAColumns * sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(deviceB, hostB, numBRows * numBColumns * sizeof(float), cudaMemcpyHostToDevice);
  
  wbTime_stop(GPU, "Copying input memory to the GPU.");

  //@@ Initialize the grid and block dimensions here
  //int maxInRows = max(numARows, numBRows);
  //int maxInColumns = max(numAColumns, numBColumns);
  int BLOCK_SIZE = 32;
  dim3 DimGrid(ceil(numBColumns*1.0/BLOCK_SIZE),ceil(numARows*1.0/BLOCK_SIZE),1);
  dim3 DimBlock(BLOCK_SIZE,BLOCK_SIZE,1);
  
  wbLog(TRACE,"DimGridX=",DimGrid.x);
  wbLog(TRACE,"DimGridY=",DimGrid.y);
  
  wbTime_start(Compute, "Performing CUDA computation");
   //@@ Launch the GPU Kernel here
  //matrixMultiply<<<DimGrid,DimBlock>>>(deviceA, deviceB, deviceC, numARows, numAColumns, numBRows, numBColumns, numCRows, numCColumns);
  matrixMultiplyShared<<<DimGrid, DimBlock>>>(deviceA, deviceB, deviceC, numARows, numAColumns, numBRows, numBColumns, numCRows, numCColumns);
  
  cudaDeviceSynchronize();
  wbTime_stop(Compute, "Performing CUDA computation");

  wbTime_start(Copy, "Copying output memory to the CPU");
  //@@ Copy the GPU memory back to the CPU here
  cudaMemcpy(hostC, deviceC, numCRows * numCColumns * sizeof(float), cudaMemcpyDeviceToHost);
  wbTime_stop(Copy, "Copying output memory to the CPU");

  wbTime_start(GPU, "Freeing GPU Memory");
  //@@ Free the GPU memory here
  cudaFree(deviceA);
  cudaFree(deviceB);
  cudaFree(deviceC);
  wbTime_stop(GPU, "Freeing GPU Memory");
  
  wbSolution(args, hostC, numCRows, numCColumns);

  free(hostA);
  free(hostB);
  free(hostC);

  return 0;
}
