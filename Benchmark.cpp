#include "Benchmark.h"
#include "Mock/MockExecutor.h"
#include "BlockCipher/FeistelExecutor.h"
#include "RSA/RSAExecutor.h"
#include "BlockCipherParallel/FeistelPrallelExecutor.h"

#include <fstream>
#include <iostream>

void printHist(const Generator::Histogram &Hist, std::ostream &S) {
  for (auto [C, Count] : Hist)
    S << static_cast<int>(C) << " " 
      << static_cast<double>(Count) << std::endl;
}

int main() {
  auto Mock = std::make_unique<mock::Cipher>(); 
  auto Feistel = std::make_unique<feistel::Cipher>();
  auto FeistelPar4 = std::make_unique<feistelParallel::Cipher<4>>();
  auto FeistelPar10 = std::make_unique<feistelParallel::Cipher<10>>();
  auto RSA = std::make_unique<RSA::Cipher>();
  auto FeistelCuda = std::make_unique<feistelCuda::Cipher>();
  auto Ciphers = std::vector<Executor *>{Mock.get(), Feistel.get(), 
                                         RSA.get()};

  auto Cfg = BenchConfig{};
  Cfg.TextSize = 512;
  auto Res = Benchmark{Ciphers.begin(), Ciphers.end(), Cfg}.run();
  std::cout << "Text size: " << Cfg.TextSize << std::endl;
  Benchmark::printResults(Res, std::cout);

  Cfg.TextSize = 512;
  Ciphers = std::vector<Executor *>{Feistel.get(), FeistelPar4.get(), 
                                    FeistelPar10.get(), FeistelCuda.get()};
  Res = Benchmark{Ciphers.begin(), Ciphers.end(), Cfg}.run();
  std::cout << "\nParallel version on text: " << Cfg.TextSize << std::endl; 
  Benchmark::printResults(Res, std::cout);

  auto GeneratorCfg = GenConfig{};
  auto PlainText = Generator::generate(500000, GeneratorCfg);
  auto Hist = Generator::getHistogram(PlainText);

  auto InputHistFile = std::ofstream{"input-hist"};
  auto OutputHistFile = std::ofstream{"output-hist"}; 
  printHist(Hist, InputHistFile);
  Hist = Generator::getHistogram(Feistel->encrypt(PlainText)); 
  printHist(Hist, OutputHistFile);
}