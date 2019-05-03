#include <wb.h>

#define HISTOGRAM_LENGTH 256
#define BLOCK_SIZE_1 16
#define BLOCK_SIZE_2 512

__constant__ float HIST_CDF[HISTOGRAM_LENGTH];

#define wbCheck(stmt)                                                     \
  do {                                                                    \
    cudaError_t err = stmt;                                               \
    if (err != cudaSuccess) {                                             \
      wbLog(ERROR, "Failed to run stmt ", #stmt);                         \
      wbLog(ERROR, "Got CUDA error ...  ", cudaGetErrorString(err));      \
      return -1;                                                          \
    }                                                                     \
  } while (0)

__device__ float clamp(float x, float start, float end){
    return min(max(x, start), end);  
}

__device__ float correct_color(unsigned char val,float cdfmin){
    return clamp(255*(HIST_CDF[val] - cdfmin)/(1.0-cdfmin),0.0,255.0);
}

__global__ void float2uchar(float* input, unsigned char* output, int z_size, int y_size, int x_size){
    
    int tx = threadIdx.x, ty = threadIdx.y, tz = threadIdx.z;
    int bx = blockIdx.x, by = blockIdx.y, bz = blockIdx.z;
    int i = bz * blockDim.z + tz;
    int j = by * blockDim.y + ty;
    int k = bx * blockDim.x + tx;
    float temp;
    unsigned char assigned_val;
    if(i < z_size && j < y_size && k < x_size){
      temp = input[i*y_size*x_size + j*x_size + k];
      assigned_val = (unsigned char)(255 * temp);
      output[i*y_size*x_size + j*x_size + k] = assigned_val;
    }
}

__global__ void RGB2Gray(unsigned char* input, unsigned char *output, int height, int width){
    int tx = threadIdx.x, ty = threadIdx.y;
    int bx = blockIdx.x, by = blockIdx.y;
    int i = by * blockDim.y + ty;
    int j = bx * blockDim.x + tx;
    int idx;
    unsigned char r,g,b;
    unsigned char assigned_value;
    if(i < height && j < width){
       idx = i * width + j;
       r = input[3*idx];
       g = input[3*idx + 1];
       b = input[3*idx + 2];
       assigned_value = (unsigned char)(0.21*r + 0.71*g + 0.07*b);
       output[idx] = (unsigned char)(assigned_value);
    }
}

__global__ void getHistgram(unsigned char *buffer,
            long size, unsigned int *histo){
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    int stride = blockDim.x * gridDim.x;
  
    while (i < size){
       atomicAdd( &(histo[buffer[i]]), 1);
       i += stride;
    }
}

__global__ void scan(unsigned int *input, float *output, int len, int width_height){
  //@@ Modify the body of this function to complete the functionality of
  //@@ the scan on the device
  //@@ You may need multiple kernel calls; write your kernels before this
  //@@ function and call them from the host
  __shared__ unsigned int T[HISTOGRAM_LENGTH];
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
  
  //printf("T[%d] = %d\n", tx, T[tx]);
  //printf("T[%d] = %d\n", tx + blockDim.x, T[tx + blockDim.x]);
  
  while(stride < HISTOGRAM_LENGTH){
    __syncthreads();
    int index = (tx + 1) * stride * 2 - 1;
    if(index < HISTOGRAM_LENGTH && index - stride >= 0){
      T[index] += T[index - stride];
    }
    stride = stride * 2;
  }

  stride = HISTOGRAM_LENGTH / 4;
  while(stride > 0){
    __syncthreads();
    int index = (tx + 1)*stride*2 - 1;
    if((index + stride) < HISTOGRAM_LENGTH)
      T[index + stride] += T[index];
    stride = stride / 2;
  }
  
  __syncthreads();
  
  float assigned_value;
  if(i < len){
    assigned_value = T[tx] / (width_height*1.0);
    //printf("assigned_value = %f\n", assigned_value);
    output[i] = assigned_value;
  }
  
  if(i + blockDim.x < len){
    assigned_value = T[tx + blockDim.x] / (width_height*1.0);
    //printf("assigned_value = %f\n", assigned_value);
    output[i + blockDim.x] = assigned_value;
  } 
}
 
__global__ void applyEqualization(unsigned char* input, unsigned char* output, int z_size, int y_size, int x_size, float cdfmin){
    int tx = threadIdx.x, ty = threadIdx.y, tz = threadIdx.z;
    int bx = blockIdx.x, by = blockIdx.y, bz = blockIdx.z;
    int i = bz * blockDim.z + tz;
    int j = by * blockDim.y + ty;
    int k = bx * blockDim.x + tx;
    unsigned char temp;
    unsigned char assigned_value;
    if(i < z_size && j < y_size && k < x_size){
      temp = input[i*y_size*x_size + j*x_size + k];
      assigned_value = correct_color(temp, cdfmin);
      output[i*y_size*x_size + j*x_size + k] = (unsigned char)(assigned_value);
    }
}

__global__ void uchar2float(unsigned char* input, float* output, int z_size, int y_size, int x_size){
    int tx = threadIdx.x, ty = threadIdx.y, tz = threadIdx.z;
    int bx = blockIdx.x, by = blockIdx.y, bz = blockIdx.z;
    int i = bz * blockDim.z + tz;
    int j = by * blockDim.y + ty;
    int k = bx * blockDim.x + tx;
    unsigned char temp;
    float assigned_value;
    if(i < z_size && j < y_size && k < x_size){
      temp = input[i*y_size*x_size + j*x_size + k];
      assigned_value = (float)(temp/255.0);
      //printf("temp = %d\n", temp);
      output[i*y_size*x_size + j*x_size + k] = assigned_value;
    }
}

//@@ insert code here
int main(int argc, char **argv) {
  wbArg_t args;
  int imageWidth;
  int imageHeight;
  int imageChannels;
  long width_height;
  long total_elements;
  wbImage_t inputImage;
  wbImage_t outputImage;
  float *hostInputImageData;
  float *hostOutputImageData;
  float *deviceFloatInputImageData;
  float *deviceFloatOutputImageData;
  unsigned int *deviceHist;
  float *deviceHistCdf;
  float *hostHistCdf;
  float mincdf = 1.1, maxcdf = -1.0;
  unsigned char *deviceGrayscaleData;
  unsigned char *deviceUcharInputImageData;
  unsigned char *deviceEqualizedData;
  const char *inputImageFile;

  //@@ Insert more code here

  args = wbArg_read(argc, argv); /* parse the input arguments */

  inputImageFile = wbArg_getInputFile(args, 0);

  wbTime_start(Generic, "Importing data and creating memory on host");
  inputImage = wbImport(inputImageFile);
  hostInputImageData = wbImage_getData(inputImage); 
  imageWidth = wbImage_getWidth(inputImage);
  imageHeight = wbImage_getHeight(inputImage);
  imageChannels = wbImage_getChannels(inputImage);
  width_height = imageWidth * imageHeight;
  total_elements = imageWidth * imageHeight * imageChannels;
  wbLog(TRACE, "inputWidth=", imageWidth);
  wbLog(TRACE, "imageHeight=", imageHeight);
  wbLog(TRACE, "imageChannels=", imageChannels);
  outputImage = wbImage_new(imageWidth, imageHeight, imageChannels);
  hostOutputImageData = wbImage_getData(outputImage); 
  hostHistCdf = (float*)malloc(HISTOGRAM_LENGTH*sizeof(float));
  
  wbTime_stop(Generic, "Importing data and creating memory on host");

  //@@ insert code here
  wbTime_start(GPU, "Doing GPU memory allocation");
  wbCheck(cudaMalloc((void**)&deviceFloatInputImageData, total_elements*sizeof(float));
  );
  wbCheck(cudaMalloc((void**)&deviceFloatOutputImageData, total_elements*sizeof(float));
  );
  wbCheck(cudaMalloc((void**)&deviceUcharInputImageData, total_elements*sizeof(unsigned char));
  );
  wbCheck(cudaMalloc((void**)&deviceGrayscaleData, width_height*sizeof(unsigned char));
  );
  wbCheck(cudaMalloc((void**)&deviceHist, HISTOGRAM_LENGTH*sizeof(unsigned int));
  );
  wbCheck(cudaMalloc((void**)&deviceHistCdf, HISTOGRAM_LENGTH*sizeof(float)));
  wbCheck(cudaMalloc((void**)&deviceEqualizedData, total_elements*sizeof(unsigned char)));
  
  wbTime_stop(GPU, "Doing GPU memory allocation");
  wbTime_start(Copy, "Copying data to the GPU");
  cudaMemcpy(deviceFloatInputImageData, hostInputImageData,total_elements*sizeof(float),cudaMemcpyHostToDevice);
  wbTime_stop(Copy, "Copying data to the GPU");
  
  wbTime_start(Compute, "Doing the computation on the GPU");
  dim3 DimBlock1(3, BLOCK_SIZE_1, BLOCK_SIZE_1);
  dim3 DimGrid1(1, ceil(imageWidth*1.0/BLOCK_SIZE_1), ceil(imageHeight*1.0/BLOCK_SIZE_1));
  float2uchar<<<DimGrid1, DimBlock1>>>(deviceFloatInputImageData, deviceUcharInputImageData, imageHeight, imageWidth, imageChannels);
  wbTime_stop(Compute, "Doing the computation on the GPU");
  
  wbTime_start(Compute, "Doing the computation on the GPU");
  dim3 DimBlock2(BLOCK_SIZE_1, BLOCK_SIZE_1, 1);
  dim3 DimGrid2(ceil(imageWidth*1.0/BLOCK_SIZE_1),ceil(imageHeight*1.0/BLOCK_SIZE_1),1);
  RGB2Gray<<<DimGrid2, DimBlock2>>>(deviceUcharInputImageData, deviceGrayscaleData, imageHeight, imageWidth);
  wbTime_stop(Compute, "Doing the computation on the GPU");
  
  wbTime_start(Compute, "Doing the computation on the GPU");
  dim3 DimBlock3(BLOCK_SIZE_2, 1, 1);
  dim3 DimGrid3(ceil(imageHeight*imageWidth*1.0/BLOCK_SIZE_2),1,1);
  getHistgram<<<DimGrid3, DimBlock3>>>(deviceGrayscaleData, width_height, deviceHist);
  wbTime_stop(Compute, "Doing the computation on the GPU");
  
  wbTime_start(Compute, "Doing the computation on the GPU");
  dim3 DimBlock4(HISTOGRAM_LENGTH/2,1,1);
  dim3 DimGrid4(1,1,1);
  scan<<<DimGrid4, DimBlock4>>>(deviceHist, deviceHistCdf, HISTOGRAM_LENGTH, width_height);
  wbTime_stop(Compute, "Doing the computation on the GPU"); 
  
  wbCheck(cudaMemcpy(hostHistCdf, deviceHistCdf, HISTOGRAM_LENGTH*sizeof(float), cudaMemcpyDeviceToHost));
  for(int i = 0;  i < HISTOGRAM_LENGTH; i++){
    mincdf = min(mincdf, hostHistCdf[i]);
    maxcdf = min(maxcdf, hostHistCdf[i]);
  }
  wbCheck(cudaMemcpyToSymbol(HIST_CDF, hostHistCdf, HISTOGRAM_LENGTH*sizeof(float)));
  
  wbTime_start(Compute, "Doing the computation on the GPU");
  dim3 DimBlock5(3, BLOCK_SIZE_1, BLOCK_SIZE_1);
  dim3 DimGrid5(1, ceil(imageWidth*1.0/BLOCK_SIZE_1), ceil(imageHeight*1.0/BLOCK_SIZE_1));
  applyEqualization<<<DimGrid5,DimBlock5>>>(deviceUcharInputImageData, deviceEqualizedData,imageHeight, imageWidth, imageChannels, mincdf);  
  wbTime_stop(Compute, "Doing the computation on the GPU"); 
  
  wbTime_start(Compute, "Doing the computation on the GPU");
  dim3 DimBlock6(3, BLOCK_SIZE_1, BLOCK_SIZE_1);
  dim3 DimGrid6(1, ceil(imageWidth*1.0/BLOCK_SIZE_1), ceil(imageHeight*1.0/BLOCK_SIZE_1));
  uchar2float<<<DimGrid6, DimBlock6>>>(deviceEqualizedData, deviceFloatOutputImageData, imageHeight, imageWidth, imageChannels);
  wbTime_stop(Compute, "Doing the computation on the GPU"); 
 
  wbTime_start(Copy, "Copying output memory to the CPU");
  wbCheck(cudaMemcpy(hostOutputImageData, deviceFloatOutputImageData, total_elements*sizeof(float),
                     cudaMemcpyDeviceToHost));
  wbTime_stop(Copy, "Copying output memory to the CPU");

  //@@ insert code here
  wbSolution(args, outputImage);
  wbTime_start(GPU, "Freeing GPU Memory");
  cudaFree(deviceFloatInputImageData);
  cudaFree(deviceFloatOutputImageData);
  cudaFree(deviceUcharInputImageData);
  cudaFree(deviceGrayscaleData);
  cudaFree(deviceHist);
  cudaFree(deviceHistCdf);
  cudaFree(deviceEqualizedData);
  wbTime_stop(GPU, "Freeing GPU Memory");
  free(hostHistCdf);
  return 0;
}

