/*******************************************
#include "GALATIC/include/CSR.cuh"
#include "GALATIC/include/dCSR.cuh"
#include "GALATIC/include/SemiRingInterface.h"
#include "GALATIC/source/device/Multiply.cuh"

Your "includes" probably needs to look something like the above, rather than what's below. 
*******************************************/

#include "include/CSR.cuh"
#include "include/dCSR.cuh"
#include "include/SemiRingInterface.h"
#include "include/TestSpGEMM.cuh"

#include "source/device/Multiply.cuh"

struct foo {
    double a;

    double b;
    short c;
};

struct foo2 {
    short h;
    double a;
    double b;
    double c;

    double d;
    short k;
};

struct Arith_SR : SemiRing<foo, foo2, double>
{
  __host__ __device__ double multiply(const foo& a, const foo2& b) const { return a.b * b.d; }
  __host__ __device__ double add(const double& a, const double& b)   const   { return a + b; }
   __host__ __device__  static double AdditiveIdentity()                  { return     0; }
};


int main() 
{
    CSR<Arith_SR::leftInput_t> input_A_CPU;
    CSR<Arith_SR::rightInput_t> input_B_CPU;

    CSR<Arith_SR::output_t> result_mat_CPU;
    
    dCSR<Arith_SR::leftInput_t> input_A_GPU;
    dCSR<Arith_SR::rightInput_t> input_B_GPU;

    dCSR<Arith_SR::output_t> result_mat_GPU;

    input_A_CPU.alloc(2,2,4);
    input_A_CPU.row_offsets[0] = 0;
    input_A_CPU.row_offsets[1] = 2;
    input_A_CPU.row_offsets[2] = 4;

    input_A_CPU.col_ids[0] = 0;
    input_A_CPU.col_ids[1] = 1;

    input_A_CPU.col_ids[2] = 0;
    input_A_CPU.col_ids[3] = 1;


    input_B_CPU.alloc(2,2,4);
    input_B_CPU.row_offsets[0] = 0;
    input_B_CPU.row_offsets[1] = 2;
    input_B_CPU.row_offsets[2] = 4;

    input_B_CPU.col_ids[0] = 0;
    input_B_CPU.col_ids[1] = 1;

    input_B_CPU.col_ids[2] = 0;
    input_B_CPU.col_ids[3] = 1;


    /* ...
       ... load data into input_A_CPU
       ...*/
    
   
    for (int i = 0; i < 4; i++) {
        foo f;
        foo2 g;
        f.b = g.d = i+1;
        input_A_CPU.data[i] = f;
        input_B_CPU.data[i] = g;
    }
     // [ [ 1,  2],
     //   [ 3 4 ] ]
     cudaDeviceSynchronize();

    
    // Transfer input matrices onto GPU
    convert(input_A_GPU, input_A_CPU);
    convert(input_B_GPU, input_B_CPU);

    // load data into semiring struct. For this one, we don't need to do anything
    Arith_SR semiring;
    
    
    // Setup execution options, we'll skip the details for now.
    
    const int Threads = 128;
    const int BlocksPerMP = 1;
    const int NNZPerThread = 2;
    const int InputElementsPerThreads = 2;
    const int RetainElementsPerThreads = 1;
    const int MaxChunksToMerge = 16;
    const int MaxChunksGeneralizedMerge = 256; // MAX: 865
    const int MergePathOptions = 8;
    
    
    GPUMatrixMatrixMultiplyTraits DefaultTraits(Threads, BlocksPerMP, NNZPerThread,
                                                 InputElementsPerThreads, RetainElementsPerThreads,
                                                 MaxChunksToMerge, MaxChunksGeneralizedMerge, MergePathOptions );
    
    const bool Debug_Mode = true;
    // DefaultTraits.preferLoadBalancing = true;
     ExecutionStats stats;
    // stats.measure_all = false;
    
    for (int i =0; i < 1; i++){
    // Actually perform the matrix multiplicaiton
        ACSpGEMM::Multiply<Arith_SR>(input_A_GPU, input_B_GPU, result_mat_GPU, DefaultTraits, stats, Debug_Mode, semiring);
         cudaDeviceSynchronize();
    }

    TestSpGEMM(input_A_GPU, input_B_GPU, semiring, [=] (const Arith_SR::output_t &a, const Arith_SR::output_t &b) { return std::abs(a-b) < 0.01; }, DefaultTraits);

    convert(result_mat_CPU, result_mat_GPU);

    cudaDeviceSynchronize();

    for (int i =0; i < 4; i++) {
        std::cout << "nnz: " << i <<   " val " <<  result_mat_CPU.data[i] << std::endl;
    }
    
}