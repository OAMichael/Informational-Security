#pragma once

#include "Executor.h"
#include "Generator.h"

#include <vector>
#include <map>
#include <string> 
#include <chrono>
#include <algorithm>
#include <iostream>
#include <cassert>
#include <memory>
#include <iomanip>

struct BenchConfig {
  size_t TextSize = 10000000;
  size_t IterNum = 2;
  bool Validate = true;

  using Time = std::chrono::milliseconds;
};

class Benchmark {
  std::vector<Executor *> Ciphers;
  BenchConfig Cfg;

  void reportFatalError(std::string Msg) {
    std::cerr << Msg << std::endl;
    exit(-1);
  } 

  template <typename Functor>
  size_t measureTime(Functor Func) {
    auto Sum = 0ull;
    for (size_t i = 0; i < Cfg.IterNum; ++i) {
      auto Start = std::chrono::steady_clock::now();
      Func();
      auto End = std::chrono::steady_clock::now();
      Sum += std::chrono::duration_cast<BenchConfig::Time>(End - Start).count();
    }
    return Sum / Cfg.IterNum;
  }

public:
  // encrypt time, decrypt time
  using CipherResult = std::pair<size_t, size_t>;
  // name -> speed
  using Result = std::map<std::string, CipherResult>;

  template <typename It>
  Benchmark(It Beg, It End, BenchConfig Cfg = BenchConfig{}) : Ciphers{Beg, End}, 
                                                     Cfg{Cfg} {}

  Result run() {
    auto Res = Result{};
    auto M = Generator::generate(Cfg.TextSize);
    for (auto &Exec : Ciphers) {
      assert(Exec);
      auto C = Exec->encrypt(M);
      auto EncryptTime = measureTime([&Exec, &M]() {
                                       Exec->encrypt(M);
                                     });
      auto DecryptTime = measureTime([&Exec, &C]() {
                                        Exec->decrypt(C);
                                     });
      Res[Exec->getName()] = CipherResult{EncryptTime, DecryptTime};

      if (Cfg.Validate) {
        auto DecrM = Exec->decrypt(C);
        if (M.size() != DecrM.size() ||
            !std::equal(M.begin(), M.end(), DecrM.begin()))
          reportFatalError("Wrong cipher: " + Exec->getName());
      }
    }

    return Res;  
  }

  static void printResults(const Result &Res, std::ostream &S) {
    S << std::setw(15) << "Name" 
      << std::setw(15) << "Exncryption ms" 
      << std::setw(15) << "Decryption ms" 
      << "\n" << std::endl;
    for (auto &[Name, Time] : Res)
      S << std::setw(15) << Name 
        << std::setw(15) << Time.first 
        << std::setw(15) << Time.second 
        << std::endl;
  }
};