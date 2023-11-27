#include "Benchmark.h"
#include "Mock/MockExecutor.h"
#include "BlockCipher/FeistelExecutor.h"
#include "RSA/RSAExecutor.h"

int main() {
  auto Mock = std::make_unique<mock::Cipher>(); 
  auto Feistel = std::make_unique<feistel::Cipher>();
  auto FeistelPar4 = std::make_unique<feistelParallel::Cipher<4>>();
  auto FeistelPar10 = std::make_unique<feistelParallel::Cipher<10>>();
  auto RSA = std::make_unique<RSA::Cipher>();
  auto Ciphers = std::vector<Executor *>{Mock.get(), Feistel.get(), 
                                         RSA.get()};

  auto Cfg = BenchConfig{};
  Cfg.TextSize = 5000;
  auto Res = Benchmark{Ciphers.begin(), Ciphers.end(), Cfg}.run();
  std::cout << "Text size: " << Cfg.TextSize << std::endl;
  Benchmark::printResults(Res, std::cout);

  Cfg.TextSize = 10000000;
  Ciphers = std::vector<Executor *>{Feistel.get(), FeistelPar4.get(), 
                                    FeistelPar10.get()};
  Res = Benchmark{Ciphers.begin(), Ciphers.end(), Cfg}.run();
  std::cout << "\nParallel version on text: " << Cfg.TextSize << std::endl; 
  Benchmark::printResults(Res, std::cout);
}