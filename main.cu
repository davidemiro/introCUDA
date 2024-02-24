#include <iostream>

#define N 6

//Define a kernel.
//A KERNEL is a function is a function that can be called N times by N differents CUDA threads.
//The number of threads that have to run the function is specified by the syntax <<1,N>> when you call the function
__global__ void add(float* A, float* B, float* C){
    int i = threadIdx.x;
    C[i] = A[i] + B[i];
}

__global__ void init(float* X, float seed){
    int i = threadIdx.x;
    X[i] = (float) (i) * seed;

}

using namespace std;

void print(float* X,int N){
    for(int i = 0; i < N; i++){
        cout << X[i];
    }
    cout << endl;
}

int main() {
    float A = new float[N];
    float B = new float[N];
    float C = new float[N];

    init<<<1,N>>>(A,1.0)
    init<<<1,N>>>(B,2.0)
    add<<<1,N>>>(A,B,C);

    print(A,N);
    print(B,N);
    print(C,N);
    return 0;
}
