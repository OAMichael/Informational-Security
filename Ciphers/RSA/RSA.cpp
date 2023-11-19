#include <cstring>

#include "RSA.h"

namespace RSACipher {


void RSAEncryptor::generateBigRandomNumber(mpz_t numOut, const mpz_t max = nullptr) {
    std::uniform_int_distribution<uint64_t> dist(1ULL << 48, 1ULL << 63);
    constexpr size_t iterationCount = 48;

    mpz_set_ui(numOut, 1);
    if (max) {
        mpz_t tmp;
        mpz_init_set_ui(tmp, 1);
        // After this loop: 2^2304 <= numOut <= max
        for (size_t i = 0; i < iterationCount; ++i) {
            mpz_mul_ui(tmp, numOut, dist(m_randEngine));
            if (mpz_cmp(tmp, max) > 0) {
                break;
            }
            mpz_set(numOut, tmp);
        }
        mpz_clear(tmp);
    }
    else {
        // After this loop: 2^2304 <= numOut <= 2^3024
        for (size_t i = 0; i < iterationCount; ++i) {
            mpz_mul_ui(numOut, numOut, dist(m_randEngine));
        }
    }
}


void RSAEncryptor::generateBigRandomPrime(mpz_t primeOut) {
    mpz_t bigRandomNum;
    mpz_init(bigRandomNum);

    generateBigRandomNumber(bigRandomNum);
    mpz_nextprime(primeOut, bigRandomNum);
    mpz_clear(bigRandomNum);
}


void RSAEncryptor::setSeed(const uint64_t seed) {
    m_randEngine.seed(seed);
}


void RSAEncryptor::generateEncryptionNumbers() {
    mpz_inits(m_p, m_q, m_n, m_phi, m_e, m_d, nullptr);

    generateBigRandomPrime(m_p);
    generateBigRandomPrime(m_q);
    mpz_mul(m_n, m_p, m_q);

    // phi = (p - 1)(q - 1) = pq - p - q + 1 = n - p - q + 1
    mpz_sub(m_phi, m_n, m_p);
    mpz_sub(m_phi, m_phi, m_q);
    mpz_add_ui(m_phi, m_phi, 1);

    generateBigRandomNumber(m_e, m_phi);

    mpz_t gcd;
    mpz_init_set_ui(gcd, 0);
    while (mpz_cmp_ui(gcd, 1) != 0) {
        mpz_add_ui(m_e, m_e, 1);
        mpz_gcd(gcd, m_e, m_phi);
    }
    mpz_clear(gcd);

    mpz_invert(m_d, m_e, m_phi);
}

}
