#include <iostream>
using namespace std;
//nvcc ass4_1.cu -o ass4_1  -lcuda
//./ass4_1
// Kernel function to add two vectors
__global__ void vectorAdd(int *a, int *b, int *c, int size) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < size) {
        c[tid] = a[tid] + b[tid];
    }
}

int main() {
    // Size of vectors
    int size = 10;

    // Host vectors
    int *h_a, *h_b, *h_c;
    h_a = new int[size];
    h_b = new int[size];
    h_c = new int[size];

    // Initialize input vectors
    for (int i = 0; i < size; ++i) {
        // int a;
        // std::cin>>a;
        // int b;
        // std::cin>>b;
        h_a[i] = i;
        h_b[i] = i*2;
    }
    std::cout << "1st vector" << std::endl;
    for (int i = 0; i < size; ++i) {
        std::cout << h_a[i] << " ";
    }
    std::cout << std::endl;
    std::cout << "2nd vector" << std::endl;
    for (int i = 0; i < size; ++i) {
        std::cout << h_b[i] << " ";
    }
    std::cout << std::endl;

    // Device vectors
    int *d_a, *d_b, *d_c;
    cudaMalloc((void **)&d_a, size * sizeof(int));
    cudaMalloc((void **)&d_b, size * sizeof(int));
    cudaMalloc((void **)&d_c, size * sizeof(int));

    // Copy input vectors from host to device memory
    cudaMemcpy(d_a, h_a, size * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, size * sizeof(int), cudaMemcpyHostToDevice);

    // Define grid and block size
    int blockSize = 256;
    int numBlocks = (size + blockSize - 1) / blockSize;

    // Print kernel launch configuration
    std::cout << "Grid size: " << numBlocks << ", Block size: " << blockSize << std::endl;
    cudaEvent_t start,stop;
    float elapsedTime;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start,0);
    // Launch kernel
    vectorAdd<<<numBlocks, blockSize>>>(d_a, d_b, d_c, size);

    cudaEventRecord(stop,0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&elapsedTime,start,stop);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    // Copy result from device to host memory
    cudaMemcpy(h_c, d_c, size * sizeof(int), cudaMemcpyDeviceToHost);
    // Verify result
    for (int i = 0; i < size; ++i) {
        if (h_c[i] != h_a[i] + h_b[i]) {
            std::cerr << "Error: Incorrect result at index " << i << std::endl;
            break;
        }
    }
        // Print the resultant array
        std::cout << "Resultant array after vector addition:" << std::endl;
        for (int i = 0; i < size; ++i) {
            std::cout << h_c[i] << " ";
        }
        std::cout << std::endl;
        cout << "GPU result:\n";
        cout<<"Elapsed Time = "<<elapsedTime<<" milliseconds" << endl;
    // Clean up
    delete[] h_a;
    delete[] h_b;
    delete[] h_c;
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);

    std::cout << "Vector addition completed successfully!" << std::endl;

    return 0;
}
