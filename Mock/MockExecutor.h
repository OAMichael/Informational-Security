#pragma once

#include "Executor.h"

#include <string>

namespace mock {

std::string encrypt(const std::string &S);
std::string decrypt(const std::string &S);

struct Cipher : public Executor {
  std::string encrypt(const std::string &S) override {
    return mock::encrypt(S);
  } 

  std::string decrypt(const std::string &S) override {
    return mock::decrypt(S);
  } 

  std::string getName() const override {
    return "Test cipher";
  }
};

} // namespace mock