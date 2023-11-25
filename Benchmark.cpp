#include "Benchmark.h"
#include "Mock/MockExecutor.h"
#include "BlockCipher/FeistelExecutor.h"

int main() {
  auto Mock = std::make_unique<mock::Cipher>(); 
  auto Feistel = std::make_unique<feistel::Cipher>();
  auto Ciphers = std::vector<Executor *>{Mock.get(), Feistel.get()};

  auto Bench = Benchmark{Ciphers.begin(), Ciphers.end()};
  auto Res = Bench.run();
  Benchmark::printResults(Res, std::cout);
}