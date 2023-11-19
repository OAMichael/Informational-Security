#pragma once

#include <string>
#include <vector>
#include <cassert>
#include <algorithm>
#include <random>
#include <map>

struct Generator {
  struct Config {
    double DeviationsPercent = 0.1;
    double DeviationSize = 3.0;
    double DeviationDeviation = 0.1;
    size_t AlphabetSize = 128;
    size_t Seed = 1;
    size_t FirstSymbol = 30;
  };

  using Histogram = std::map<char, size_t>;

  static std::string generate(size_t Size, Config Cfg = Config{}) {
    assert(Cfg.AlphabetSize <= 256);
    assert(Cfg.DeviationSize >= 0);
    assert(Cfg.DeviationsPercent >= 0 && Cfg.DeviationsPercent <= 1);
    auto Gen = std::mt19937{Cfg.Seed};
    auto Weights = std::vector<double>(Cfg.AlphabetSize);
    std::fill(Weights.begin(), Weights.end(), 1.0);

    for (size_t i = 0; i < Cfg.DeviationsPercent * Cfg.DeviationSize; ++i) {
      auto DeviationDistr = 
        std::normal_distribution{Cfg.DeviationSize, 
                                 Cfg.DeviationSize * Cfg.DeviationDeviation};
      Weights[i] += std::abs(DeviationDistr(Gen));
    }
    std::shuffle(Weights.begin(), Weights.end(), Gen);

    auto Symbols = std::vector<char>(Cfg.AlphabetSize);
    // тут переполнение законно
    std::iota(Symbols.begin(), Symbols.end(), Cfg.FirstSymbol);
    
    auto SymbolDistr = std::discrete_distribution<size_t>{Weights.begin(), Weights.end()};
    auto Res = std::string(Size, '!');
    std::generate(Res.begin(), Res.end(), 
      [&]() {
        return Symbols[SymbolDistr(Gen)];
      });
    return Res;
  }

  static Histogram getHistogram(std::string M) {
    auto Res = Histogram{};
    for (auto C : M)
      ++Res[C];
    return Res;
  }
};