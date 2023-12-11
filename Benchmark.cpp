#include "Benchmark.h"
#include "Mock/MockExecutor.h"
#include "BlockCipher/FeistelExecutor.h"
#include "RSA/RSAExecutor.h"
#include "Elephant/ElephantExecutor.h"
#include "ElephantParallel/ElephantCudaExecutor.h"

#include <fstream>
#include <iostream>

void printHist(const Generator::Histogram &Hist, std::ostream &S) {
  for (auto [C, Count] : Hist)
    S << static_cast<int>(C) << " " 
      << static_cast<double>(Count) << std::endl;
}

int main(int Argc, char **Argv) {
  auto Mock = std::make_unique<mock::Cipher>(); 
  auto Feistel = std::make_unique<feistel::Cipher>();
  auto FeistelPar4 = std::make_unique<feistelParallel::Cipher<4>>();
  auto FeistelPar10 = std::make_unique<feistelParallel::Cipher<10>>();
  auto RSA = std::make_unique<RSA::Cipher>();
  auto Elephant = std::make_unique<elephant::Cipher>();
  auto ElephantCuda = std::make_unique<elephantCuda::Cipher<1, 1>>();
  auto Ciphers = std::vector<Executor *>{Mock.get(), Feistel.get(), 
                                         Elephant.get(), RSA.get(),
                                         ElephantCuda.get()};

  auto Cfg = BenchConfig{};
  Cfg.TextSize = 5120;
  auto Res = Benchmark{Ciphers.begin(), Ciphers.end(), Cfg}.run();
  std::cout << "Text size: " << Cfg.TextSize << std::endl;
  Benchmark::printResults(Res, std::cout);

return 0;

  Cfg.TextSize = 51200000;
  Ciphers = std::vector<Executor *>{Feistel.get(), FeistelPar4.get(), 
                                    FeistelPar10.get(), Elephant.get()};
  Res = Benchmark{Ciphers.begin(), Ciphers.end(), Cfg}.run();
  std::cout << "\nParallel version on text: " << Cfg.TextSize << std::endl; 
  Benchmark::printResults(Res, std::cout);

  if (Argc == 1 || Argv[1] != "--histogram")
    return 0;
  assert(Argc == 3);

  auto GeneratorCfg = GenConfig{};
  auto PlainText = Generator::generate(500000, GeneratorCfg);
  auto Hist = Generator::getHistogram(PlainText);

  auto InputHistFile = std::ofstream{"input-hist"};
  auto OutputHistFile = std::ofstream{"output-hist"}; 
  printHist(Hist, InputHistFile);
  
  if (Argv[2] == "Feistel") {
    Hist = Generator::getHistogram(Feistel->encrypt(PlainText)); 
  } else if (Argv[2] == "Elephant") {
    Hist = Generator::getHistogram(Elephant->encrypt(PlainText)); 
  } else if (Argv[2] == "RSA") {
    Hist = Generator::getHistogram(RSA->encrypt(PlainText)); 
  } else {
    assert("Unknown cipher\n");
  }
  printHist(Hist, OutputHistFile);
}