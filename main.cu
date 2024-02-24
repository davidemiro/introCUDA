#include <iostream>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#define N 6

//Define a kernel.
//A KERNEL is a function is a function that can be called N times by N differents CUDA threads.
//The number of threads that have to run the function is specified by the syntax <<1,N>> when you call the function
__global__ void add(float* A, float* B, float* C){
    int i = threadIdx.x;
    C[i] = A[i] + B[i];
}

// the threadIdx.x is a 3D vector (threadIdx.x, threadIdx.y, threadIdx.z) for
__global__ void MatAdd(float A[N][N], float B[N][N],
                       float C[N][N])
{
    int i = threadIdx.x;
    int j = threadIdx.y;
    C[i][j] = A[i][j] + B[i][j];
}

void init(float* X, float seed){
    for(int i = 0; i < N; i++){
        X[i] = seed;
    }


}

using namespace std;

void print(float* X,int num){
    for(int i = 0; i < num; i++){
        cout << X[i];
    }
    cout << endl;
}

int main() {
    float* A = new float[N];
    float* B = new float[N];
    float* C = new float[N];

    init(A,1.0);
    init(B,2.0);
    print(A,N);
    print(B,N);
    add<<<1,N>>>(A,B,C);
    cudaDeviceSynchronize();

    print(A,N);
    print(B,N);
    print(C,N);
    return 0;
}
