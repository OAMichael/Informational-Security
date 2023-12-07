#include "Feistel.cu.h"

#include <cmath>

template <typename T1, typename T2>
__device__ __host__ unsigned ceilDiv(T1 Lhs, T2 Rhs) {
  auto LhsF = static_cast<float>(Lhs);
  auto RhsF = static_cast<float>(Rhs);
  return ceil(LhsF / RhsF);
}

void makeRound(unsigned char *Block, unsigned BlockSize) {

}

void swapBlockHalfs(unsigned char *Block, unsigned BlockSize) {

}

__global__
void encryptKernel(EncryptData Data) {
  unsigned BlockSizeInBytes = Data.CipherBlockBitSize / 8u;
  unsigned NumOfBlocks = ceilDiv(Data.Size, BlockSizeInBytes);
  unsigned ThreadID = blockIdx.x * blockDim.x + threadIdx.x;
  for (unsigned i = ThreadID; i < NumOfBlocks; i += Data.NumOfThreads) {
    makeRound(Data.CudaDataPtr +  i * BlockSizeInBytes, BlockSizeInBytes);
    swapBlockHalfs(Data.CudaDataPtr +  i * BlockSizeInBytes, BlockSizeInBytes);
  }
}

void encryptOnCuda(EncryptData Data) {
  dim3 ThrBlockDim{Data.BlockSize};
  dim3 BlockGridDim{ceilDiv(Data.NumOfThreads, Data.BlockSize)};

  encryptKernel<<<BlockGridDim, ThrBlockDim>>>(Data);
}