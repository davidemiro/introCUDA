#include <iostream>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#define N 3

//Define a kernel.
//A KERNEL is a function is a function that can be called N times by N differents CUDA threads.
//The number of threads that have to run the function is specified by the syntax <<1,N>> when you call the function
__global__ void add(float* A, float* B, float* C){
    int i = threadIdx.x;
    C[i] = A[i] + B[i];
}



// the threadIdx.x is a 3D vector (threadIdx.x, threadIdx.y, threadIdx.z) for
__global__ void matrix_add(float A[N][N], float B[N][N],
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

void matrix_print(float X[N][N]){
    for(int i = 0; i < N; i++){
        for(int j = 0; j < N; j++){
            cout << X[i][j];
        }
        cout << endl;
    }
    cout << endl;
}

int main() {
    float AM[N][N] = {
            {1, 2, 3},
            {4, 5, 6},
            {7, 8, 9}
    };
    float BM[N][N] = {
            {1, 2, 3},
            {4, 5, 6},
            {7, 8, 9}
    };
    float CM[N][N] = {
            {1, 2, 3},
            {4, 5, 6},
            {7, 8, 9}
    };



    int numBlocks = 1;
    dim3 threadsPerBlock(N, N);
    matrix_add<<<numBlocks, threadsPerBlock>>>(AM, BM, CM);



    cudaDeviceSynchronize();

    matrix_print(AM);
    matrix_print(BM);
    matrix_print(CM);
    return 0;

}
