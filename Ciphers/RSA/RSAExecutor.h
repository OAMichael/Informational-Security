#pragma once

#include "Executor.h"
#include "RSA.h"

namespace RSA {

class Cipher : public Executor {
  RSACipher::RSAEncryptor Cipherer;

public:
  Cipher() {}

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
    return "RSA cipher";
  }
};

} // namespace RSA