#include "Benchmark.h"
#include "Mock/MockExecutor.h"
#include "BlockCipher/FeistelExecutor.h"
#include "RSA/RSAExecutor.h"

int main() {
  auto Mock = std::make_unique<mock::Cipher>(); 
  auto Feistel = std::make_unique<feistel::Cipher>();
  auto RSA = std::make_unique<RSA::Cipher>();
  auto Ciphers = std::vector<Executor *>{Mock.get(), Feistel.get(), 
                                         RSA.get()};

  auto Bench = Benchmark{Ciphers.begin(), Ciphers.end()};
  auto Res = Bench.run();
  Benchmark::printResults(Res, std::cout);
}