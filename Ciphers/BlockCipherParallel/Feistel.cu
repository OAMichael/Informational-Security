#include "Feistel.cu.h"

#include <cmath>

template <typename T1, typename T2>
__device__ __host__ unsigned ceilDiv(T1 Lhs, T2 Rhs) {
  auto LhsF = static_cast<float>(Lhs);
  auto RhsF = static_cast<float>(Rhs);
  return ceil(LhsF / RhsF);
}

void f(unsigned char *Block, unsigned char *OutBlock,
       unsigned BlockSize,
       unsigned char *SubKey, unsigned SubKeySize) {
  for (unsigned i = 0; i < BlockSize / 2; i++) {
    uint8_t currHBByte = Block[i];
    uint8_t currKeyByte = SubKey[i % SubKeySize];

    // Some weird stuff is going on here
    OutBlock[i] = (currHBByte ^ currKeyByte & currHBByte ^ 0xD5) & 
                  (currKeyByte >> 3) ^ currHBByte | currKeyByte;
  }
}

void makeRound(unsigned char *Block, unsigned BlockSize, 
               unsigned char *SubKey, unsigned SubKeySize) {
  unsigned char *Lhs = Block;
  unsigned char *Rhs = Block + BlockSize / 2;

  unsigned char *OldRhs = (unsigned char*)malloc(BlockSize / 2);
  for (unsigned i = 0; i < BlockSize / 2; i++)
    OldRhs[i] = Rhs[i];

  f(Rhs, Rhs, BlockSize, SubKey, SubKeySize);

  for (unsigned i = 0; i < BlockSize / 2; i++)
    Rhs[i] = Lhs[i] ^ Rhs[i];

  for (unsigned i = 0; i < BlockSize / 2; i++)
    Lhs[i] = OldRhs[i];
}

void swapBlockHalfs(unsigned char *Block, unsigned BlockSize) {

}

__global__
void encryptKernel(EncryptData Data) {
  unsigned BlockSizeInBytes = Data.CipherBlockBitSize / 8u;
  unsigned NumOfBlocks = ceilDiv(Data.Size, BlockSizeInBytes);
  unsigned ThreadID = blockIdx.x * blockDim.x + threadIdx.x;
  for (unsigned i = ThreadID; i < NumOfBlocks; i += Data.NumOfThreads) {
    for (unsigned i = 0; i < Data.NumberOfRound; ++i)
      makeRound(Data.CudaDataPtr +  i * BlockSizeInBytes, BlockSizeInBytes,
                Data.Subkeys, Data.SubKeySizes);
    swapBlockHalfs(Data.CudaDataPtr +  i * BlockSizeInBytes, BlockSizeInBytes);
  }
}

void encryptOnCuda(EncryptData Data) {
  dim3 ThrBlockDim{Data.BlockSize};
  dim3 BlockGridDim{ceilDiv(Data.NumOfThreads, Data.BlockSize)};

  encryptKernel<<<BlockGridDim, ThrBlockDim>>>(Data);
}