#pragma once

#include "Executor.h"
#include "Elephant.h"

namespace elephant {

class Cipher : public Executor {
  ElephantEncryptor Cipherer;

public:
  std::string encrypt(const std::string &Plaintext) override {
    return Cipherer.encrypt(Plaintext);
  } 

  std::string decrypt(const std::string &Ciphertext) override {
    return Cipherer.decrypt(Ciphertext);
  } 

  std::string getName() const override {
    return "Elephant cipher";
  }
};

};