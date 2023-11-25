#pragma once

#include "Executor.h"
#include "Feistel.h"

namespace feistel {

class Cipher : public Executor {
  BlockCipher::FeistelCipherEncryptor Cipherer;

public:
  Cipher() {
    auto MasterKey = BlockCipher::Key{ 14, 48, 228, 13, 37, 42, 69, 223 };
    Cipherer.setMasterKey(MasterKey);
  }

  std::string encrypt(const std::string &Plaintext) override {
    auto Ciphertext = std::string{};
    Cipherer.encrypt(Plaintext, Ciphertext);
    return Ciphertext;
  } 

  std::string decrypt(const std::string &Ciphertext) override {
    auto Plaintext = std::string{};
    Cipherer.decrypt(Ciphertext, Plaintext);
    return Plaintext;
  } 

  std::string getName() const override {
    return "Feistel cipher";
  }
};

} // namespace feistel