#pragma once

#include <cmath>
#include <cuda.h>
#include <cuda_runtime_api.h>
#include <algorithm>
#include <vector>
#include <cstdio>

#define CUDA_CHECK(expr)                                                       \
  {                                                                            \
    auto MyErr = (expr);                                                       \
    if (MyErr != cudaSuccess) {                                                \
      printf("%s in %s at line %d\n", cudaGetErrorString(MyErr), __FILE__,     \
             __LINE__);                                                        \
      exit(1);                                                                 \
    }                                                                          \
  }

struct EncryptData {
  unsigned BlockSize = 1;
  unsigned NumOfThreads = 1;

  unsigned CipherBlockBitSize = 64;
  unsigned NumberOfRound = 16;
  unsigned char *CudaDataPtr;
  unsigned Size = 0;

  unsigned char *SubKey;
  unsigned SubKeySize;
};

static constexpr unsigned KeyByteSize = 8;

using Key = unsigned char[KeyByteSize];

void encryptOnCuda(EncryptData Data);

template <int BlockSize, int NumOfThreads>
class FeistelCudaEncryptor {
  static constexpr size_t BlockBitSize = 64; 
  std::array<unsigned char> MasterKey = {14, 48, 228, 13, 37, 42, 69, 223};
  std::vector<Key> SubKey;
  
public:
  FeistelCudaEncryptor() {
    
  }

  template <typename Cont_t, typename Insert>
  void encrypt(const Cont_t &Cont, Insert Inserter) {
    static_assert(sizeof(typename Cont_t::value_type) == 1);
    unsigned char *CudaDataPtr = nullptr;
    CUDA_CHECK(cudaMalloc((void **)&CudaDataPtr, Cont.size()));
    CUDA_CHECK(cudaMemcpy(CudaDataPtr, Cont.data(), Cont.size(),
                          cudaMemcpyHostToDevice));

    EncryptData Data;
    Data.BlockSize = BlockSize;
    Data.NumOfThreads = NumOfThreads;
    Data.CipherBlockBitSize = BlockBitSize;
    Data.CudaDataPtr = CudaDataPtr;
    Data.Size = Cont.size();
    CUDA_CHECK(cudaMalloc((void **)&Data.SubKey, SubKey.size()));
    CUDA_CHECK(cudaMemcpy(Data.SubKey, SubKey.data(), SubKey.size(),
                          cudaMemcpyHostToDevice));
    encryptOnCuda(Data);

    std::vector<typename Cont_t::value_type> ResData(Cont.size());
    CUDA_CHECK(cudaMemcpy(ResData.data(), CudaDataPtr, Cont.size(),
                          cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaFree(CudaDataPtr));
    std::copy(ResData.begin(), ResData.end(), Inserter);
  }

};