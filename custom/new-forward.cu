
/* 
<--------------------------------------------------------->
            3 Optimizations are implemented                                           
<--------------------------------------------------------->*/

/* 
<----------------------------------------------------------------------------->
                                    No.1
       Using Streams to overlap computation with data transfer (4 points)
<----------------------------------------------------------------------------->
*/

#include <cmath>
#include <iostream>
#include "gpu-new-forward.h"

#define TILE_WIDTH 24


__global__ void conv_forward_kernel(float *output, const float *input, const float *mask, const int Batch, const int Map_out, const int Channel, const int Height, const int Width, const int K)
{
    /*
    Modify this function to implement the forward pass described in Chapter 16.
    We have added an additional dimension to the tensors to support an entire mini-batch
    The goal here is to be correct AND fast.

    Function paramter definitions:
    output - output
    input - input
    mask - convolution kernel
    Batch - batch_size (number of images in x)
    Map_out - number of output feature maps
    Channel - number of input feature maps
    Height - input height dimension
    Width - input width dimension
    K - kernel height and width (K x K)
    */

    const int Height_out = Height - K + 1;
    const int Width_out  = Width - K + 1;
    int tx=threadIdx.x;
    int ty=threadIdx.y;
    int bx=blockIdx.x;
    int by=blockIdx.y;
    int bz=blockIdx.z;

    // We have some nice #defs for you below to simplify indexing. Feel free to use them, or create your own.
    // An example use of these macros:
    // float a = in_4d(0,0,0,0)
    // out_4d(0,0,0,0) = a

    #define out_4d(i3, i2, i1, i0) output[(i3) * (Map_out * Height_out * Width_out) + (i2) * (Height_out * Width_out) + (i1) * (Width_out) + i0]
    #define in_4d(i3, i2, i1, i0) input[(i3) * (Channel * Height * Width) + (i2) * (Height * Width) + (i1) * (Width) + i0]
    #define mask_4d(i3, i2, i1, i0) mask[(i3) * (Channel * K * K) + (i2) * (K * K) + (i1) * (K) + i0]

    // Insert your GPU convolution kernel code here
    // __shared__ float in_4d_s[TILE_WIDTH][TILE_WIDTH][TILE_WIDTH];
    int numBlockEachCol= (Width_out-1)/TILE_WIDTH+1;     //ceil(float(Width)/TILE_WIDTH)

    int b = bz;                                 // batch index
    int m = bx;                                 // out channel index
    int y = TILE_WIDTH * (by/numBlockEachCol) + ty;                   // y coordinates
    int x = TILE_WIDTH * (by%numBlockEachCol) + tx;     // x coordinates

    float result=0;
    if(b<Batch && m<Map_out && y<Height_out && x<Width_out){
        for(int c=0; c<Channel; c++){       // add all channels
            for(int p=0; p<K; p++){         // KxK filter
                for(int q=0; q<K; q++){     // p and q are the index inside tile
                    result += in_4d(b, c, y+p, x+q) * mask_4d(m,c,p,q);
                }
            }
        }
        out_4d(b,m,y,x)=result;
    }

    #undef out_4d
    #undef in_4d
    #undef mask_4d
}


__host__ void GPUInterface::conv_forward_gpu_prolog(const float *host_output, const float *host_input, const float *host_mask, float **device_output_ptr, float **device_input_ptr, float **device_mask_ptr, const int Batch, const int Map_out, const int Channel, const int Height, const int Width, const int K)
{

    // Allocate memory and copy over the relevant data structures to the GPU
    const int H_out = Height - K + 1;
    const int W_out = Width - K + 1;
    int seg_size_in = (Batch * Channel * Height * Width)/2;
    int seg_size_out = (Batch * Map_out * H_out * W_out)/2;
    dim3 blockDim(TILE_WIDTH, TILE_WIDTH, 1);
    dim3 gridDim(Map_out, ceil(float(W_out)/TILE_WIDTH)*ceil(float(H_out)/TILE_WIDTH), Batch/2);
        
    cudaStream_t stream0;
    cudaStream_t stream1;
    cudaStreamCreate(&stream0);
    cudaStreamCreate(&stream1);

    float* host_output_ = (float*)host_output;
    cudaMalloc((void** )device_output_ptr, Batch * Map_out * H_out * W_out * sizeof(float));
    cudaMalloc((void** )device_input_ptr, Batch * Channel * Height * Width * sizeof(float));
    cudaMalloc((void** )device_mask_ptr, Map_out * Channel * K * K * sizeof(float));

    cudaMemcpy(*device_mask_ptr, host_mask, Map_out * Channel * K * K * sizeof(float), cudaMemcpyHostToDevice);
    
    cudaMemcpyAsync(*device_input_ptr, host_input , seg_size_in * sizeof(float), cudaMemcpyHostToDevice, stream0);
    cudaMemcpyAsync((*device_input_ptr)+seg_size_in, host_input + seg_size_in, seg_size_in * sizeof(float), cudaMemcpyHostToDevice, stream1);
    
    conv_forward_kernel<<<gridDim, blockDim, 0, stream0>>>(*device_output_ptr, *device_input_ptr, *device_mask_ptr, Batch, Map_out, Channel, Height, Width, K);
    conv_forward_kernel<<<gridDim, blockDim, 0, stream1>>>((*device_output_ptr) + seg_size_out, (*device_input_ptr) + seg_size_in, *device_mask_ptr, Batch, Map_out, Channel, Height, Width, K);
    
    cudaMemcpyAsync(host_output_, *device_output_ptr, seg_size_out * sizeof(float), cudaMemcpyDeviceToHost, stream0);
    cudaMemcpyAsync(host_output_ + seg_size_out, (*device_output_ptr) + seg_size_out, seg_size_out * sizeof(float), cudaMemcpyDeviceToHost, stream1);
    
    cudaDeviceSynchronize();

    cudaStreamDestroy(stream0);
    cudaStreamDestroy(stream1);

}

__host__ void GPUInterface::conv_forward_gpu(float *device_output, const float *device_input, const float *device_mask, const int Batch, const int Map_out, const int Channel, const int Height, const int Width, const int K)
{
            return;
}


__host__ void GPUInterface::conv_forward_gpu_epilog(float *host_output, float *device_output, float *device_input, float *device_mask, const int Batch, const int Map_out, const int Channel, const int Height, const int Width, const int K)
{

    // Free device memory
    cudaFree(device_output);
    cudaFree(device_input);
    cudaFree(device_mask);
}


__host__ void GPUInterface::get_device_properties()
{
    int deviceCount;
    cudaGetDeviceCount(&deviceCount);

    for(int dev = 0; dev < deviceCount; dev++)
    {
        cudaDeviceProp deviceProp;
        cudaGetDeviceProperties(&deviceProp, dev);

        std::cout<<"Device "<<dev<<" name: "<<deviceProp.name<<std::endl;
        std::cout<<"Computational capabilities: "<<deviceProp.major<<"."<<deviceProp.minor<<std::endl;
        std::cout<<"Max Global memory size: "<<deviceProp.totalGlobalMem<<std::endl;
        std::cout<<"Max Constant memory size: "<<deviceProp.totalConstMem<<std::endl;
        std::cout<<"Max Shared memory size per block: "<<deviceProp.sharedMemPerBlock<<std::endl;
        std::cout<<"Max threads per block: "<<deviceProp.maxThreadsPerBlock<<std::endl;
        std::cout<<"Max block dimensions: "<<deviceProp.maxThreadsDim[0]<<" x, "<<deviceProp.maxThreadsDim[1]<<" y, "<<deviceProp.maxThreadsDim[2]<<" z"<<std::endl;
        std::cout<<"Max grid dimensions: "<<deviceProp.maxGridSize[0]<<" x, "<<deviceProp.maxGridSize[1]<<" y, "<<deviceProp.maxGridSize[2]<<" z"<<std::endl;
        std::cout<<"Warp Size: "<<deviceProp.warpSize<<std::endl;
    }
}


/* 
<--------------------------------------------------------->
                          No.2
        Tiled shared memory convolution (**2 points**)
<--------------------------------------------------------->
*/
// #include <cmath>
// #include <iostream>
// #include "gpu-new-forward.h"

// #define BLOCK_WIDTH 16

// __global__ void conv_forward_kernel(float *output, const float *input, const float *mask, const int Batch, const int Map_out, const int Channel, const int Height, const int Width, const int K)
// {
//     /*
//     Modify this function to implement the forward pass described in Chapter 16.
//     We have added an additional dimension to the tensors to support an entire mini-batch
//     The goal here is to be correct AND fast.

//     Function paramter definitions:
//     output - output
//     input - input
//     mask - convolution kernel
//     Batch - batch_size (number of images in x)
//     Map_out - number of output feature maps
//     Channel - number of input feature maps
//     Height - input height dimension
//     Width - input width dimension
//     K - kernel height and width (K x K)
//     */
//     __shared__ float in_4d_s[BLOCK_WIDTH][BLOCK_WIDTH];

//     const int Height_out = Height - K + 1;
//     const int Width_out  = Width - K + 1;
//     int TILE_WIDTH= BLOCK_WIDTH-K+1;
//     int tx=threadIdx.x;
//     int ty=threadIdx.y;
//     int bx=blockIdx.x;
//     int by=blockIdx.y;
//     int bz=blockIdx.z;

//     // We have some nice #defs for you below to simplify indexing. Feel free to use them, or create your own.
//     // An example use of these macros:
//     // float a = in_4d(0,0,0,0)
//     // out_4d(0,0,0,0) = a

//     #define out_4d(i3, i2, i1, i0) output[(i3) * (Map_out * Height_out * Width_out) + (i2) * (Height_out * Width_out) + (i1) * (Width_out) + i0]
//     #define in_4d(i3, i2, i1, i0) input[(i3) * (Channel * Height * Width) + (i2) * (Height * Width) + (i1) * (Width) + i0]
//     #define mask_4d(i3, i2, i1, i0) mask[(i3) * (Channel * K * K) + (i2) * (K * K) + (i1) * (K) + i0]
    
//     // Insert your GPU convolution kernel code here
//     // __shared__ float in_4d_s[TILE_WIDTH][TILE_WIDTH][TILE_WIDTH];
//     int numBlockEachRow= (Width_out-1)/TILE_WIDTH+1;     //ceil(float(Width)/TILE_WIDTH)

//     int b = bz;                                 // batch index
//     int m = bx;                                 // out channel index
//     int y = TILE_WIDTH * (by/numBlockEachRow) + ty;                   // y coordinates
//     int x = TILE_WIDTH * (by%numBlockEachRow) + tx;     // x coordinates
//     float result=0;
//     for(int c=0; c<Channel  && m<Map_out; c++){       // add all channels
//         // copy data from global memory to shared memory
//         in_4d_s[ty][tx] = 0;
//         if(y<Height && x<Width)  
//             in_4d_s[ty][tx] = in_4d(b, c, y, x);
//         __syncthreads();

//         if(ty<TILE_WIDTH && tx < TILE_WIDTH && y<Height_out && x<Width_out){
//             for(int p=0; p<K; p++){         // KxK filter
//                 for(int q=0; q<K; q++){     // p and q are the index inside tile
//                     result += in_4d_s[ty+p][tx+q] * mask_4d(m,c,p,q);
//                 }
//             }
//         }
//         __syncthreads();
//     }
//     if(ty<TILE_WIDTH && tx < TILE_WIDTH && y<Height_out && x<Width_out)
//         out_4d(b,m,y,x)=result;

//     #undef out_4d
//     #undef in_4d
//     #undef mask_4d
// }

	
// __host__ void GPUInterface::conv_forward_gpu_prolog(const float *host_output, const float *host_input, const float *host_mask, float **device_output_ptr, float **device_input_ptr, float **device_mask_ptr, const int Batch, const int Map_out, const int Channel, const int Height, const int Width, const int K)
// {
//     /* Allocate memory and copy over the relevant data structures to the GPU */
    
//     // Calc dimension of output
//     int Height_out = Height - K + 1;
//     int Width_out  = Width - K + 1;
//     int numInElements   = Batch * Channel * Height * Width; 
//     int numMaskElements = Map_out * Channel * K * K; 
//     int numOutElements  = Batch * Map_out * Height_out * Width_out; 
    
//     // Allocate and copy memory
//     cudaMalloc((void **)device_input_ptr,  numInElements*sizeof(float));
//     cudaMalloc((void **)device_mask_ptr, numMaskElements*sizeof(float));
//     cudaMalloc((void **)device_output_ptr, numOutElements*sizeof(float));

//     cudaMemcpy(*device_input_ptr, host_input, numInElements*sizeof(float), cudaMemcpyHostToDevice);
//     cudaMemcpy(*device_mask_ptr, host_mask, numMaskElements*sizeof(float), cudaMemcpyHostToDevice);
//     cudaMemcpy(*device_output_ptr, host_output, numOutElements*sizeof(float), cudaMemcpyHostToDevice);
    
    
//     // We pass double pointers for you to initialize the relevant device pointers,
//     //  which are passed to the other two functions.

//     // Useful snippet for error checking
//     // cudaError_t error = cudaGetLastError();
//     // if(error != cudaSuccess)
//     // {
//     //     std::cout<<"CUDA error: "<<cudaGetErrorString(error)<<std::endl;
//     //     exit(-1);
//     // }
// }


// __host__ void GPUInterface::conv_forward_gpu(float *device_output, const float *device_input, const float *device_mask, const int Batch, const int Map_out, const int Channel, const int Height, const int Width, const int K)
// {
//     /* Set the kernel dimensions and call the kernel */

//     // Calc dimension of output
//     int Height_out = Height - K + 1;
//     int Width_out  = Width - K + 1;
//     int TILE_WIDTH=BLOCK_WIDTH - K + 1;
    
//     // Set the kernel dimensions
//     int gridDim_x = Map_out;
//     int gridDim_y = ceil(float(Width_out)/TILE_WIDTH)*ceil(float(Height_out)/TILE_WIDTH);   
//     int gridDim_z = Batch;           //z dimension is for batches


//     dim3 gridDim(gridDim_x, gridDim_y, gridDim_z);
//     dim3 blockDim(BLOCK_WIDTH, BLOCK_WIDTH, 1);
//     // dim3 blockDim(TILE_WIDTH, TILE_WIDTH, 1);
//     // Call the kernel
//     conv_forward_kernel<<<gridDim, blockDim>>>(device_output, device_input, device_mask, Batch, Map_out, Channel, Height, Width, K);
//     cudaDeviceSynchronize();
// }


// __host__ void GPUInterface::conv_forward_gpu_epilog(float *host_output, float *device_output, float *device_input, float *device_mask, const int Batch, const int Map_out, const int Channel, const int Height, const int Width, const int K)
// {
//     /* Copy the output back to host and free device memory*/

//     // Calc dimension of output
//     int Height_out = Height - K + 1;
//     int Width_out  = Width - K + 1;
//     int numElements = Batch * Map_out * Height_out * Width_out; 

//     // Copy the output back to host
//     cudaMemcpy(host_output, device_output, numElements*sizeof(float), cudaMemcpyDeviceToHost);

//     // Free device memory
//     cudaFree(device_output);
//     cudaFree(device_input);
//     cudaFree(device_mask);
// }


// __host__ void GPUInterface::get_device_properties()
// {
//     int deviceCount;
//     cudaGetDeviceCount(&deviceCount);

//     for(int dev = 0; dev < deviceCount; dev++)
//     {
//         cudaDeviceProp deviceProp;
//         cudaGetDeviceProperties(&deviceProp, dev);

//         std::cout<<"Device "<<dev<<" name: "<<deviceProp.name<<std::endl;
//         std::cout<<"Computational capabilities: "<<deviceProp.major<<"."<<deviceProp.minor<<std::endl;
//         std::cout<<"Max Global memory size: "<<deviceProp.totalGlobalMem<<std::endl;
//         std::cout<<"Max Constant memory size: "<<deviceProp.totalConstMem<<std::endl;
//         std::cout<<"Max Shared memory size per block: "<<deviceProp.sharedMemPerBlock<<std::endl;
//         std::cout<<"Max threads per block: "<<deviceProp.maxThreadsPerBlock<<std::endl;
//         std::cout<<"Max block dimensions: "<<deviceProp.maxThreadsDim[0]<<" x, "<<deviceProp.maxThreadsDim[1]<<" y, "<<deviceProp.maxThreadsDim[2]<<" z"<<std::endl;
//         std::cout<<"Max grid dimensions: "<<deviceProp.maxGridSize[0]<<" x, "<<deviceProp.maxGridSize[1]<<" y, "<<deviceProp.maxGridSize[2]<<" z"<<std::endl;
//         std::cout<<"Warp Size: "<<deviceProp.warpSize<<std::endl;
//     }
// }

// __host__ void show_matrix(float* m, int w, int h){
//     for(int y=0;y<h;y++){
//         for(int x=0;x<w;x++){
//             printf("%f ",m[y*w+x]);
//         }
//         printf("\n");
//     }
// }
// __host__ void show_matrix(const float* m, int w, int h){
//     for(int y=0;y<h;y++){
//         for(int x=0;x<w;x++){
//             printf("%f ",m[y*w+x]);
//         }
//         printf("\n");
//     }
// }


/* 
<----------------------------------------------------------------------------->
                                    No.3
   Shared memory matrix multiplication and input matrix unrolling (3 points)
<----------------------------------------------------------------------------->
*/
// #include <cmath>
// #include <iostream>
// #include "gpu-new-forward.h"

// #define TILE_WIDTH 16
// // #define out_4d(i3, i2, i1, i0) output[(i3) * (Map_out * Height_out * Width_out) + (i2) * (Height_out * Width_out) + (i1) * (Width_out) + i0]
// // #define mask_4d(i3, i2, i1, i0) mask[(i3) * (Channel * K * K) + (i2) * (K * K) + (i1) * (K) + i0]


// void unroll(int B, int C, int H, int W, int K, const float* X, float* X_unroll){
// int H_out = H - K + 1;
// int W_out  = W - K + 1;
// #define in_unroll(i2, i1, i0) X_unroll[(i2) * (K*K*C * H_out*W_out) + (i1) * (H_out*W_out) + i0]
// #define in_4d(i3, i2, i1, i0) X[(i3) * (C * H * W) + (i2) * (H * W) + (i1) * (W) + i0]
// for (int b = 0; b < B; ++b) {// for each image
// for (int c = 0; c < C; ++c) { // for each input channel
// int w_base = c * (K*K); // per-channel offset for smallest X_unroll index
// for (int p = 0; p < K; ++p) // for each element of KxK filter (two loops)
// for (int q = 0; q < K; ++q) {
// for (int h = 0; h < H_out; ++h) // for each thread (each output value, two loops)
// for (int w = 0; w < W_out; ++w) {
// int h_unroll = w_base + p * K + q; // data needed by one thread
// int w_unroll = h * W_out + w; // smallest index--across threads (output values)
// in_unroll(b, h_unroll, w_unroll) = in_4d(b, c, h + p, w + q); // copy input pixels
// }
// }
// }
// }
// #undef in_4d
// #undef in_unroll
// }



// __global__ void conv_forward_kernel(float *output, const float *input, const float *mask, const int Batch, const int Map_out, const int Channel, const int Height, const int Width, const int K)
// {
//     /*
//     Modify this function to implement the forward pass described in Chapter 16.
//     We have added an additional dimension to the tensors to support an entire mini-batch
//     The goal here is to be correct AND fast.

//     Function paramter definitions:
//     output - output
//     input - input
//     mask - convolution kernel
//     Batch - batch_size (number of images in x)
//     Map_out - number of output feature maps
//     Channel - number of input feature maps
//     Height - input height dimension
//     Width - input width dimension
//     K - kernel height and width (K x K)
//     */

   
//     __shared__ float subtileA[TILE_WIDTH][TILE_WIDTH]; //__shared__ instead of __share__
//     __shared__ float subtileB[TILE_WIDTH][TILE_WIDTH];

//     int tx=threadIdx.x;
//     int ty=threadIdx.y;
//     int bx=blockIdx.x;
//     int by=blockIdx.y;
//     int bz=blockIdx.z;
//     int H_out = Height - K + 1;
//     int W_out = Width - K + 1;

//     int numARows=Map_out;
//     int numAColumns=Channel*K*K;
//     int numBRows=K*K*Channel;
//     int numBColumns=H_out*W_out;
//     int numCRows=numARows;
//     int numCColumns=numBColumns;
//     #define out_4d(i3, i2, i1, i0) output[(i3) * (Map_out * H_out * W_out) + (i2) * (H_out * W_out) + (i1) * (W_out) + i0]
//     #define B(i2, i1, i0) input[ (i2) * (H_out*W_out*K*K*Channel) + (i1) * (H_out*W_out) + i0]
//     #define A(i1, i0) mask[(i1) * (Channel*K*K) + i0]

//     // #define in_4d(i3, i2, i1, i0) input[(i3) * (Map_out * Height_out * Width_out) + (i2) * (Height_out * Width_out) + (i1) * (Width_out) + i0]
//     //@@ Insert code to implement matrix multiplication here
//     int b = bz;                                 // batch index
//     int h = by*TILE_WIDTH+ty;                   // y coordinates
//     int w = bx*TILE_WIDTH+tx;                   // x coordinates
//     float pValue=0;
//     for(int q=0;q<ceil(numAColumns/float(TILE_WIDTH));++q){
//         //load the data to the subtileA and subtileB in shared memory.
//         subtileA[ty][tx]=0;
//         subtileB[ty][tx]=0;
//         if(h<numARows && tx+q*TILE_WIDTH<numAColumns)  //deal with the overflow threads when loading data
//             subtileA[ty][tx]=A(h,tx+q*TILE_WIDTH);       //subtileA[ty][tx]=A[Row][tx+q*TILE_WIDTH]
//         if(ty+q*TILE_WIDTH<numBRows && w<numBColumns)
//             subtileB[ty][tx]=B(b,ty+q*TILE_WIDTH,w);     //subtileB[ty][tx]=B[ty+q*TILE_WIDTH][Col]
//         __syncthreads();                                        //wait untill all the threads finish loading data to the subtiles
//         for(int k=0;k<TILE_WIDTH;++k)
//             pValue+=subtileA[ty][k]*subtileB[k][tx];              //subtileA[ty][k]+subtile[k][tx]
//         __syncthreads();              //wait untill all the threads finish calc, or the data in subtiles will be changed during calc
//     }
//     if(h<numCRows && w<numCColumns && b<Batch){
//         out_4d(b,h,w/W_out,w%W_out)=pValue;
//     }

//     #undef out_4d
// }

	
// __host__ void GPUInterface::conv_forward_gpu_prolog(const float *host_output, const float *host_input, const float *host_mask, float **device_output_ptr, float **device_input_ptr, float **device_mask_ptr, const int Batch, const int Map_out, const int Channel, const int Height, const int Width, const int K)
// {
//     /* Allocate memory and copy over the relevant data structures to the GPU */
//     #define mask_4d(i3, i2, i1, i0) host_mask[(i3) * (Channel * K * K) + (i2) * (K * K) + (i1) * (K) + i0]
//     // Calc dimension of output
//     int Height_out = Height - K + 1;
//     int Width_out  = Width - K + 1;
//     int numInElements   = Batch * Channel * Height * Width; 
//     int numMaskElements = Map_out * Channel * K * K; 
//     int numOutElements  = Batch * Map_out * Height_out * Width_out; 
    
//     float* host_input_unroll;
//     float* host_kernel_unroll;
//     host_input_unroll=(float *)malloc(Batch*Height_out*Width_out*K*K*Channel*sizeof(float));
//     host_kernel_unroll=(float *)malloc(K*K*Channel*Map_out*sizeof(float));

//     unroll(Batch, Channel, Height, Width, K, host_input, host_input_unroll);
//     for(int m=0;m<Map_out;m++){
//         for(int c=0;c<Channel;c++){
//             for(int p=0;p<K;p++){
//                 for(int q=0;q<K;q++){
//                     host_kernel_unroll[m*Channel*K*K + c*K*K + p*K + q]=mask_4d(m,c,p,q);
//                 }
//             }
//         }
//     }
//     // Allocate and copy memo ry
//     cudaMalloc((void **)device_input_ptr, Batch*Height_out*Width_out*K*K*Channel*sizeof(float));
//     cudaMalloc((void **)device_mask_ptr, K*K*Channel*Map_out*sizeof(float));
//     cudaMalloc((void **)device_output_ptr, numOutElements*sizeof(float));

//     cudaMemcpy(*device_input_ptr, host_input_unroll, Batch*Height_out*Width_out*K*K*Channel*sizeof(float), cudaMemcpyHostToDevice);
//     cudaMemcpy(*device_mask_ptr, host_kernel_unroll, K*K*Channel*Map_out*sizeof(float), cudaMemcpyHostToDevice);
//     cudaMemcpy(*device_output_ptr, host_output, numOutElements*sizeof(float), cudaMemcpyHostToDevice);
    
    
//     // We pass double pointers for you to initialize the relevant device pointers,
//     //  which are passed to the other two functions.

//     // Useful snippet for error checking
//     // cudaError_t error = cudaGetLastError();
//     // if(error != cudaSuccess)
//     // {
//     //     std::cout<<"CUDA error: "<<cudaGetErrorString(error)<<std::endl;
//     //     exit(-1);
//     // }
//     #undef mask_4d
// }


// __host__ void GPUInterface::conv_forward_gpu(float *device_output, const float *device_input, const float *device_mask, const int Batch, const int Map_out, const int Channel, const int Height, const int Width, const int K)
// {
//     /* Set the kernel dimensions and call the kernel */

//     // Calc dimension of output
//     int Height_out = Height - K + 1;
//     int Width_out  = Width - K + 1;
//     // Set the kernel dimensions
//     int gridDim_x = ceil(Height_out*Width_out*1.0/TILE_WIDTH);
//     int gridDim_y = ceil(Map_out*1.0/TILE_WIDTH);  
//     int gridDim_z = Batch;           //z dimension is for batches

//     dim3 gridDim(gridDim_x, gridDim_y, gridDim_z);
//     dim3 blockDim(TILE_WIDTH, TILE_WIDTH, 1);

//     // Call the kernel
//     conv_forward_kernel<<<gridDim, blockDim>>>(device_output, device_input, device_mask, Batch, Map_out, Channel, Height, Width, K);
//     cudaDeviceSynchronize();
// }

// __host__ void GPUInterface::conv_forward_gpu_epilog(float *host_output, float *device_output, float *device_input, float *device_mask, const int Batch, const int Map_out, const int Channel, const int Height, const int Width, const int K)
// {
//     /* Copy the output back to host and free device memory*/

//     // Calc dimension of output
//     int Height_out = Height - K + 1;
//     int Width_out  = Width - K + 1;
//     int numElements = Batch * Map_out * Height_out * Width_out; 

//     // Copy the output back to host
//     cudaMemcpy(host_output, device_output, numElements*sizeof(float), cudaMemcpyDeviceToHost);

//     // Free device memory
//     cudaFree(device_output);
//     cudaFree(device_input);
//     cudaFree(device_mask);
// }


// __host__ void GPUInterface::get_device_properties()
// {
//     int deviceCount;
//     cudaGetDeviceCount(&deviceCount);

//     for(int dev = 0; dev < deviceCount; dev++)
//     {
//         cudaDeviceProp deviceProp;
//         cudaGetDeviceProperties(&deviceProp, dev);

//         std::cout<<"Device "<<dev<<" name: "<<deviceProp.name<<std::endl;
//         std::cout<<"Computational capabilities: "<<deviceProp.major<<"."<<deviceProp.minor<<std::endl;
//         std::cout<<"Max Global memory size: "<<deviceProp.totalGlobalMem<<std::endl;
//         std::cout<<"Max Constant memory size: "<<deviceProp.totalConstMem<<std::endl;
//         std::cout<<"Max Shared memory size per block: "<<deviceProp.sharedMemPerBlock<<std::endl;
//         std::cout<<"Max threads per block: "<<deviceProp.maxThreadsPerBlock<<std::endl;
//         std::cout<<"Max block dimensions: "<<deviceProp.maxThreadsDim[0]<<" x, "<<deviceProp.maxThreadsDim[1]<<" y, "<<deviceProp.maxThreadsDim[2]<<" z"<<std::endl;
//         std::cout<<"Max grid dimensions: "<<deviceProp.maxGridSize[0]<<" x, "<<deviceProp.maxGridSize[1]<<" y, "<<deviceProp.maxGridSize[2]<<" z"<<std::endl;
//         std::cout<<"Warp Size: "<<deviceProp.warpSize<<std::endl;
//     }
// }
