#pragma once

#include "Executor.h"
#include "Elephant.cu.h"

namespace elephantCuda {

template <int BlockSize, int NumOfThreads>
class Cipher : public Executor {
  ElephantCudadEncryptor<BlockSize, NumOfThreads> Cipherer;

public:
  std::string encrypt(const std::string &Plaintext) override {
    return Cipherer.encrypt(Plaintext);
  } 

  std::string decrypt(const std::string &Ciphertext) override {
    return Cipherer.decrypt(Ciphertext);
  } 

  std::string getName() const override {
    return "Elephant cuda {" + std::to_string(BlockSize) + 
            ", " + std::to_string(NumOfThreads) + "}";
  }
};

};