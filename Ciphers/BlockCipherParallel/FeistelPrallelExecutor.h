#include "Executor.h"
#include "Feistel.cu.h"

namespace feistelCuda {

class Cipher : public Executor {
  FeistelCudaEncryptor<16, 16> Cipherer;

public:
  Cipher() {}

  std::string encrypt(const std::string &Plaintext) override {
    auto Ciphertext = std::string{};
    Cipherer.encrypt(Plaintext, std::inserter(Ciphertext, Ciphertext.end()));
    return Ciphertext;
  } 

  std::string decrypt(const std::string &Ciphertext) override {
    return Ciphertext;
  } 

  std::string getName() const override {
    return "Feistel cuda";
  }
};

} // namespace feistelCuda
