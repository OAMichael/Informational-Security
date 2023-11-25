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


void RSAEncryptor::encrypt(std::vector<uint8_t> &bytesIn, std::vector<uint8_t> &bytesOut) {
    mpz_t currNum;
    mpz_init_set_ui(currNum, 0);

    constexpr size_t RSABlockBytesize = 512;
    const size_t RSABlockCount = bytesIn.size() / RSABlockBytesize;
    const size_t RSABlockRemainder = bytesIn.size() % RSABlockBytesize;

    // This is never less than needed space but almost always too much
    // Shrink it later. sizeof(uint64_t) is for size serialization
    bytesOut.resize(2 * sizeof(uint64_t) + (RSABlockCount + 1) * (mpz_sizeinbase(m_n, 256) + sizeof(uint64_t)));

    size_t outIdx = 0;
    *reinterpret_cast<uint64_t*>(bytesOut.data() + outIdx) = RSABlockCount;
    outIdx += sizeof(uint64_t);

    *reinterpret_cast<uint64_t*>(bytesOut.data() + outIdx) = RSABlockRemainder;
    outIdx += sizeof(uint64_t);

    for (size_t i = 0; i < RSABlockCount; ++i) {
        mpz_import(currNum, RSABlockBytesize, 1, 1, 1, 0, &bytesIn[i * RSABlockBytesize]);
        mpz_powm(currNum, currNum, m_e, m_n);

        uint64_t currNumBytesize = mpz_sizeinbase(currNum, 256);
        *reinterpret_cast<uint64_t*>(bytesOut.data() + outIdx) = currNumBytesize;
        outIdx += sizeof(uint64_t);

        mpz_export(&bytesOut[outIdx], nullptr, 1, 1, 1, 0, currNum);
        outIdx += currNumBytesize;
    }

    if (RSABlockRemainder > 0) {
        mpz_import(currNum, RSABlockRemainder, 1, 1, 1, 0, &bytesIn[RSABlockCount * RSABlockBytesize]);
        mpz_powm(currNum, currNum, m_e, m_n);

        uint64_t currNumBytesize = mpz_sizeinbase(currNum, 256);
        *reinterpret_cast<uint64_t*>(bytesOut.data() + outIdx) = currNumBytesize;
        outIdx += sizeof(uint64_t);

        mpz_export(&bytesOut[outIdx], nullptr, 1, 1, 1, 0, currNum);
        outIdx += currNumBytesize;
    }

    bytesOut.resize(outIdx);
    bytesOut.shrink_to_fit();
    mpz_clear(currNum);
}


void RSAEncryptor::decrypt(std::vector<uint8_t> &bytesIn, std::vector<uint8_t> &bytesOut) {
    mpz_t currNum;
    mpz_init_set_ui(currNum, 0);

    constexpr size_t RSABlockBytesize = 512;

    size_t inIdx = 0;
    const size_t RSABlockCount = *reinterpret_cast<uint64_t*>(bytesIn.data() + inIdx);
    inIdx += sizeof(uint64_t);

    const size_t RSABlockRemainder = *reinterpret_cast<uint64_t*>(bytesIn.data() + inIdx);
    inIdx += sizeof(uint64_t);

    // This is never less than needed space but almost always too much
    // Shrink it later
    bytesOut.resize((RSABlockCount + 1) * mpz_sizeinbase(m_n, 256));

    for (size_t i = 0; i < RSABlockCount; ++i) {
        uint64_t currNumBytesize = *reinterpret_cast<uint64_t*>(bytesIn.data() + inIdx);
        inIdx += sizeof(uint64_t);

        mpz_import(currNum, currNumBytesize, 1, 1, 1, 0, &bytesIn[inIdx]);
        mpz_powm(currNum, currNum, m_d, m_n);
        inIdx += currNumBytesize;

        mpz_export(&bytesOut[i * RSABlockBytesize], nullptr, 1, 1, 1, 0, currNum);
    }

    if (RSABlockRemainder > 0) {
        uint64_t currNumBytesize = *reinterpret_cast<uint64_t*>(bytesIn.data() + inIdx);
        inIdx += sizeof(uint64_t);

        mpz_import(currNum, currNumBytesize, 1, 1, 1, 0, &bytesIn[inIdx]);
        mpz_powm(currNum, currNum, m_d, m_n);
        inIdx += currNumBytesize;

        mpz_export(&bytesOut[RSABlockCount * RSABlockBytesize], nullptr, 1, 1, 1, 0, currNum);
    }

    bytesOut.resize(RSABlockCount * RSABlockBytesize + RSABlockRemainder);
    bytesOut.shrink_to_fit();
    mpz_clear(currNum);
}

}
