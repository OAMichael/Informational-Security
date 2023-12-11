#include "Elephant.cu.h"

#include <cuda.h>
#include <cuda_runtime_api.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cassert>
#include <algorithm>

typedef unsigned char BYTE; // 8 bit
typedef unsigned long long LEN;

#define CAST_TO_BYTE(val) reinterpret_cast<BYTE *>(val)
#define CAST_TO_CONST_BYTE(val) reinterpret_cast<const BYTE *>(val)

// Macro to perform a cyclic left shift on eight bits
#define ROTATE_LEFT_8_BIT(a, offset) ((offset != 0) ? ((((BYTE)a) << offset) ^ (((BYTE)a) >> (sizeof(BYTE) * 8 - offset))) : a)

#define CUDA_CHECK(expr)                                                       \
  {                                                                            \
    auto MyErr = (expr);                                                       \
    if (MyErr != cudaSuccess) {                                                \
      printf("%s in %s at line %d\n", cudaGetErrorString(MyErr), __FILE__,     \
             __LINE__);                                                        \
      exit(1);                                                                 \
    }                                                                          \
  }

namespace {

template <typename T1, typename T2>
__device__ __host__ unsigned ceilDiv(T1 Lhs, T2 Rhs) {
  auto LhsF = static_cast<float>(Lhs);
  auto RhsF = static_cast<float>(Rhs);
  return ceil(LhsF / RhsF);
}

__device__
void theta(BYTE *TRIAL_ARR) {
  unsigned int a, b;
  BYTE ARR[5], ARR2[5];

  for (a = 0; a < 5; a++) {
    ARR[a] = 0;
    for (b = 0; b < 5; b++)
      ARR[a] ^= TRIAL_ARR[INDEX(a, b)];
  }

  for (a = 0; a < 5; a++)
    ARR2[a] = ROTATE_LEFT_8_BIT(ARR[(a + 1) % 5], 1) ^ ARR[(a + 4) % 5];

  for (a = 0; a < 5; a++)
    for (b = 0; b < 5; b++)
      TRIAL_ARR[INDEX(a, b)] ^= ARR2[a];
}

__device__
void rho(BYTE *TRIAL_ARR) {
  const unsigned int kessak_rho_offsets[25] = {0, 1, 6, 4, 3, 4, 4, 6, 7, 4, 3, 2, 3, 1, 7, 1, 5, 7, 5, 0, 2, 2, 5, 0, 6};
  for (unsigned int a = 0; a < 5; a++)
    for (unsigned int b = 0; b < 5; b++)
      TRIAL_ARR[INDEX(a, b)] = ROTATE_LEFT_8_BIT(TRIAL_ARR[INDEX(a, b)], kessak_rho_offsets[INDEX(a, b)]);
}

__device__
void pi(BYTE *TRIAL_ARR) {
  BYTE fix_mask[25];

  for (unsigned int a = 0; a < 5; a++)
    for (unsigned int b = 0; b < 5; b++)
      fix_mask[INDEX(a, b)] = TRIAL_ARR[INDEX(a, b)];

  for (unsigned int a = 0; a < 5; a++)
    for (unsigned int b = 0; b < 5; b++)
      TRIAL_ARR[INDEX(0 * a + 1 * b, 2 * a + 3 * b)] = fix_mask[INDEX(a, b)];
}

__device__
void hi(BYTE *TRIAL_ARR) {
  unsigned int a, b;
  BYTE ARR[5];

  for (b = 0; b < 5; b++) {
    for (a = 0; a < 5; a++)
      ARR[a] = TRIAL_ARR[INDEX(a, b)] ^ ((~TRIAL_ARR[INDEX(a + 1, b)]) & TRIAL_ARR[INDEX(a + 2, b)]);
    for (a = 0; a < 5; a++)
      TRIAL_ARR[INDEX(a, b)] = ARR[a];
  }
}

__device__
void io(BYTE *TRIAL_ARR, unsigned int ind) {
  const BYTE kessak_round_constants[MAX_NUMBER_OF_KESSAK_ROUNDS] = {
    0x01, 0x82, 0x8a, 0x00, 0x8b, 0x01, 0x81, 0x09, 0x8a,
    0x88, 0x09, 0x0a, 0x8b, 0x8b, 0x89, 0x03, 0x02, 0x80};
  
  TRIAL_ARR[INDEX(0, 0)] ^= kessak_round_constants[ind];
}

__device__
void Kessak200UnionForOneRound(BYTE *tmp_state, unsigned int tmp_round_idx) {
  theta(tmp_state);
  rho(tmp_state);
  pi(tmp_state);
  hi(tmp_state);
  io(tmp_state, tmp_round_idx);
}

__device__
void permutation(BYTE *param) {
  for (unsigned int t = 0; t < MAX_NUMBER_OF_KESSAK_ROUNDS; t++)
    Kessak200UnionForOneRound(param, t);
}

__device__
BYTE LeftShift(BYTE byte) {
  return (byte << 1) | (byte >> 7);
}

__device__
void lfsr(BYTE *out, BYTE *in) {
  BYTE cur = LeftShift(in[0]) ^ LeftShift(in[2]) ^ (in[13] << 1);
  for (LEN t = 0; t < BLOCK_SIZE - 1; ++t)
    out[t] = in[t + 1];
  out[BLOCK_SIZE - 1] = cur;
}

__device__
void xorOfBlock(BYTE *tmp_state, const BYTE *block, LEN length) {
  for (LEN t = 0; t < length; ++t)
    tmp_state[t] ^= block[t];
}

__device__
void get_ciphertext_block(BYTE *out, const BYTE *c, LEN cipher_len, LEN t) {
  const LEN offset = t * BLOCK_SIZE;

  if (offset == cipher_len) {
    memset(out, 0x00, BLOCK_SIZE);
    out[0] = 0x01;
    return;
  }
  const LEN r_text = cipher_len - offset;

  if (BLOCK_SIZE <= r_text) {
    memcpy(out, c + offset, BLOCK_SIZE);
  } else {
    if (r_text > 0) // c might be nullptr
      memcpy(out, c + offset, r_text);
    memset(out + r_text, 0x00, BLOCK_SIZE - r_text);
    out[r_text] = 0x01;
  }
}

__device__
void get_associated_data_block(BYTE *out, const BYTE *aData, LEN len_aData, 
                               const BYTE *nonce, LEN t) {
  const LEN offset = t * BLOCK_SIZE - (t != 0) * NONCE_NUM_BYTES;
  LEN len = 0;

  if (t == 0){
    memcpy(out, nonce, NONCE_NUM_BYTES);
    len += NONCE_NUM_BYTES;
  }

  if (t != 0 && offset == len_aData) {
    memset(out, 0x00, BLOCK_SIZE);
    out[0] = 0x01;
    return;
  }
  const LEN r_out = BLOCK_SIZE - len;
  const LEN r_data = len_aData - offset;

  if (r_out <= r_data) { // enough AD
    memcpy(out + len, aData + offset, r_out);
  }
  else {              // not enough AD, need to pad
    if (r_data > 0) // ad might be nullptr
      memcpy(out + len, aData + offset, r_data);
    memset(out + len + r_data, 0x00, r_out - r_data);
    out[len + r_data] = 0x01;
  }
}

__global__
void implementation_of_crypto(BYTE *c, BYTE *tag, 
                         const BYTE *m, LEN len_message,
                         const BYTE *aData, LEN len_aData,
                         const BYTE *nonce, const BYTE *k, bool encrypt) {
  const LEN num_of_blocks_cipher = 1 + len_message / BLOCK_SIZE;
  const LEN num_of_blocks_message = (len_message % BLOCK_SIZE) ? num_of_blocks_cipher : num_of_blocks_cipher - 1;
  const LEN num_of_blocks_adata = 1 + (NONCE_NUM_BYTES + len_aData) / BLOCK_SIZE;
  const LEN num_of_blocks_it = (num_of_blocks_cipher > num_of_blocks_adata) ? num_of_blocks_cipher : num_of_blocks_adata + 1;

  BYTE expanded_k[BLOCK_SIZE] = {0};
  memcpy(expanded_k, k, KEY_NUMBER_OF_BYTES);
  permutation(expanded_k);

  BYTE buffer_back[BLOCK_SIZE] = {0};
  BYTE buffer_current[BLOCK_SIZE] = {0};
  BYTE buffer_forward[BLOCK_SIZE] = {0};
  memcpy(buffer_current, expanded_k, BLOCK_SIZE);

  BYTE *mask_prev = buffer_back;
  BYTE *mask_tmp = buffer_current;
  BYTE *mask_next = buffer_forward;
  BYTE elephant_buf[BLOCK_SIZE];
  BYTE tag_buf[BLOCK_SIZE] = {0};
  memset(tag, 0, TAG_SIZE);

  LEN offset = 0;

  for (LEN t = 0; t < num_of_blocks_it; ++t) {
    lfsr(mask_next, mask_tmp);
    
    if (t % blockDim.x == threadIdx.x) {
      if (t < num_of_blocks_message) {
        memcpy(elephant_buf, nonce, NONCE_NUM_BYTES);
        memset(elephant_buf + NONCE_NUM_BYTES, 0, BLOCK_SIZE - NONCE_NUM_BYTES);
        xorOfBlock(elephant_buf, mask_tmp, BLOCK_SIZE);
        xorOfBlock(elephant_buf, mask_next, BLOCK_SIZE);
        permutation(elephant_buf);
        xorOfBlock(elephant_buf, mask_tmp, BLOCK_SIZE);
        xorOfBlock(elephant_buf, mask_next, BLOCK_SIZE);
        const LEN r_len = (t == num_of_blocks_message - 1) ? len_message - offset : BLOCK_SIZE;
        xorOfBlock(elephant_buf, m + offset, r_len);
        memcpy(c + offset, elephant_buf, r_len);
      }

      if (t > 0 && t <= num_of_blocks_cipher) {
        get_ciphertext_block(tag_buf, encrypt ? c : m, len_message, t - 1);
        xorOfBlock(tag_buf, mask_prev, BLOCK_SIZE);
        xorOfBlock(tag_buf, mask_next, BLOCK_SIZE);
        permutation(tag_buf);
        xorOfBlock(tag_buf, mask_prev, BLOCK_SIZE);
        xorOfBlock(tag_buf, mask_next, BLOCK_SIZE);
        xorOfBlock(tag, tag_buf, TAG_SIZE);
      }

      // If there is any AD left, compute tag for AD block 
      if (t + 1 < num_of_blocks_adata) {
        get_associated_data_block(tag_buf, aData, len_aData, nonce, t + 1);
        xorOfBlock(tag_buf, mask_next, BLOCK_SIZE);
        permutation(tag_buf);
        xorOfBlock(tag_buf, mask_next, BLOCK_SIZE);
        xorOfBlock(tag, tag_buf, TAG_SIZE);
      }
    }
    // Cyclically shift the mask buffers
    // Value of next_mask will be computed in the next iteration
    BYTE *const fix_mask = mask_prev;
    mask_prev = mask_tmp;
    mask_tmp = mask_next;
    mask_next = fix_mask;

    offset += BLOCK_SIZE;
  }
}

} // anonymous namespace

__host__
void encryptCuda(std::string &CipherText,
            const std::string &PlainText,
            const std::string &aData,
            const unsigned char *nonce,
            const unsigned char *k,
            unsigned BlockSize,
            unsigned NumOfThreads) {
  BYTE *tag_cuda;
  CUDA_CHECK(cudaMalloc((void **)&tag_cuda, TAG_SIZE));

  BYTE *CipherText_cuda;
  CUDA_CHECK(cudaMalloc((void **)&CipherText_cuda, PlainText.size() + TAG_SIZE));

  BYTE *PlainText_cuda;  
  CUDA_CHECK(cudaMalloc((void **)&PlainText_cuda, PlainText.size()));
  CUDA_CHECK(cudaMemcpy(PlainText_cuda, PlainText.data(), PlainText.size(),
                        cudaMemcpyHostToDevice));

  BYTE *aData_cuda;
  CUDA_CHECK(cudaMalloc((void **)&aData_cuda, aData.size()));
  CUDA_CHECK(cudaMemcpy(aData_cuda, aData.data(), aData.size(),
                        cudaMemcpyHostToDevice));

  BYTE *nonce_cuda;
  CUDA_CHECK(cudaMalloc((void **)&nonce_cuda, NONCE_NUM_BYTES));
  CUDA_CHECK(cudaMemcpy(nonce_cuda, nonce, NONCE_NUM_BYTES,
                        cudaMemcpyHostToDevice));
  
  BYTE *k_cuda;
  CUDA_CHECK(cudaMalloc((void **)&k_cuda, KEY_NUMBER_OF_BYTES));
  CUDA_CHECK(cudaMemcpy(k_cuda, k, KEY_NUMBER_OF_BYTES,
                        cudaMemcpyHostToDevice));

  dim3 ThrBlockDim{BlockSize};
  dim3 BlockGridDim{ceilDiv(NumOfThreads, BlockSize)};

  implementation_of_crypto<<<BlockGridDim, ThrBlockDim>>>(
                              CipherText_cuda, tag_cuda,
                              PlainText_cuda, PlainText.size(), 
                              aData_cuda, aData.size(), 
                              nonce_cuda, k_cuda, true);

  CipherText.clear();
  auto Buf = new char[PlainText.size()];
  CUDA_CHECK(cudaMemcpy(Buf, CipherText_cuda, 
                        PlainText.size(),
                        cudaMemcpyDeviceToHost));
  CipherText.append(Buf, PlainText.size());

  CUDA_CHECK(cudaMemcpy(Buf, tag_cuda, TAG_SIZE,
                        cudaMemcpyDeviceToHost));
  CipherText.append(Buf, TAG_SIZE);
  delete[] Buf;

  CUDA_CHECK(cudaFree(tag_cuda));
  CUDA_CHECK(cudaFree(CipherText_cuda));
  CUDA_CHECK(cudaFree(PlainText_cuda));
  CUDA_CHECK(cudaFree(aData_cuda));
  CUDA_CHECK(cudaFree(nonce_cuda));
  CUDA_CHECK(cudaFree(k_cuda));
}

__host__
void decryptCuda(std::string &PlainText,
             const std::string &CipherText,
             const std::string &aData,
             const unsigned char *nonce,
             const unsigned char *k,
             unsigned BlockSize,
             unsigned NumOfThreads) {
  assert(CipherText.size() >= TAG_SIZE);
  
  BYTE *tag_cuda;
  CUDA_CHECK(cudaMalloc((void **)&tag_cuda, TAG_SIZE));

  BYTE *PlainText_cuda;
  CUDA_CHECK(cudaMalloc((void **)&PlainText_cuda, CipherText.size() + TAG_SIZE));

  BYTE *CipherText_cuda;  
  CUDA_CHECK(cudaMalloc((void **)&CipherText_cuda, CipherText.size()));
  CUDA_CHECK(cudaMemcpy(CipherText_cuda, CipherText.data(), CipherText.size(),
                        cudaMemcpyHostToDevice));

  BYTE *aData_cuda;
  CUDA_CHECK(cudaMalloc((void **)&aData_cuda, aData.size()));
  CUDA_CHECK(cudaMemcpy(aData_cuda, aData.data(), aData.size(),
                        cudaMemcpyHostToDevice));

  BYTE *nonce_cuda;
  CUDA_CHECK(cudaMalloc((void **)&nonce_cuda, NONCE_NUM_BYTES));
  CUDA_CHECK(cudaMemcpy(nonce_cuda, nonce, NONCE_NUM_BYTES,
                        cudaMemcpyHostToDevice));
  
  BYTE *k_cuda;
  CUDA_CHECK(cudaMalloc((void **)&k_cuda, KEY_NUMBER_OF_BYTES));
  CUDA_CHECK(cudaMemcpy(k_cuda, k, KEY_NUMBER_OF_BYTES,
                        cudaMemcpyHostToDevice));

  dim3 ThrBlockDim{BlockSize};
  dim3 BlockGridDim{ceilDiv(NumOfThreads, BlockSize)};

  implementation_of_crypto<<<BlockGridDim, ThrBlockDim>>>(
                              PlainText_cuda, tag_cuda, 
                              CipherText_cuda, CipherText.size(),
                              aData_cuda, aData.size(), 
                              nonce_cuda, k_cuda, false);
  
  PlainText.clear();
  auto Buf = new char[CipherText.size()];
  CUDA_CHECK(cudaMemcpy(Buf, PlainText_cuda, 
                        CipherText.size() - TAG_SIZE,
                        cudaMemcpyDeviceToHost));
  PlainText.append(Buf, CipherText.size() - TAG_SIZE);
  delete[] Buf;

  CUDA_CHECK(cudaFree(tag_cuda));
  CUDA_CHECK(cudaFree(CipherText_cuda));
  CUDA_CHECK(cudaFree(PlainText_cuda));
  CUDA_CHECK(cudaFree(aData_cuda));
  CUDA_CHECK(cudaFree(nonce_cuda));
  CUDA_CHECK(cudaFree(k_cuda));
}