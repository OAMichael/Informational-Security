#ifndef RSA_H
#define RSA_H

#include <iostream>
#include <cmath>
#include <algorithm>
#include <random>
#include <gmp.h>

namespace RSACipher {


class RSAEncryptor {
private:
    mpz_t m_p, m_q, m_n, m_phi, m_e, m_d;

    // Used multiple times to produce random big integer number
    std::mt19937_64 m_randEngine{228};

    // Generate random prime number
    void generateBigRandomNumber(mpz_t numOut, const mpz_t max);
    void generateBigRandomPrime(mpz_t primeOut);

public:
    void setSeed(const uint64_t seed);
    void generateEncryptionNumbers();
    void encrypt(std::vector<uint8_t> &bytesIn, std::vector<uint8_t> &bytesOut);
    void decrypt(std::vector<uint8_t> &bytesIn, std::vector<uint8_t> &bytesOut);

    RSAEncryptor() {
        generateEncryptionNumbers();
    }

    ~RSAEncryptor() {
        mpz_clears(m_p, m_q, m_n, m_phi, m_e, m_d, nullptr);
    }
};

}

#endif  // RSA_H
