#pragma once

#include <string>

struct Executor {
  virtual std::string encrypt(const std::string &) = 0;  
  virtual std::string decrypt(const std::string &) = 0;  
  virtual std::string getName() const = 0;

  virtual ~Executor() {}
};
