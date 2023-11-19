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

    template <typename Cont_t>
    void encrypt(const Cont_t &bytesIn, Cont_t &bytesOut) {
        static_assert(sizeof(typename Cont_t::value_type) == 1);
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
    
    template <typename Cont_t>
    void decrypt(const Cont_t &bytesIn, Cont_t &bytesOut) {
        static_assert(sizeof(typename Cont_t::value_type) == 1);
        mpz_t currNum;
        mpz_init_set_ui(currNum, 0);

        constexpr size_t RSABlockBytesize = 512;

        size_t inIdx = 0;
        const size_t RSABlockCount = *reinterpret_cast<const uint64_t*>(bytesIn.data() + inIdx);
        inIdx += sizeof(uint64_t);

        const size_t RSABlockRemainder = *reinterpret_cast<const uint64_t*>(bytesIn.data() + inIdx);
        inIdx += sizeof(uint64_t);

        // This is never less than needed space but almost always too much
        // Shrink it later
        bytesOut.resize((RSABlockCount + 1) * mpz_sizeinbase(m_n, 256));

        for (size_t i = 0; i < RSABlockCount; ++i) {
            uint64_t currNumBytesize = *reinterpret_cast<const uint64_t*>(bytesIn.data() + inIdx);
            inIdx += sizeof(uint64_t);

            mpz_import(currNum, currNumBytesize, 1, 1, 1, 0, &bytesIn[inIdx]);
            mpz_powm(currNum, currNum, m_d, m_n);
            inIdx += currNumBytesize;

            mpz_export(&bytesOut[i * RSABlockBytesize], nullptr, 1, 1, 1, 0, currNum);
        }

        if (RSABlockRemainder > 0) {
            uint64_t currNumBytesize = *reinterpret_cast<const uint64_t*>(bytesIn.data() + inIdx);
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

    RSAEncryptor() {
        generateEncryptionNumbers();
    }

    ~RSAEncryptor() {
        mpz_clears(m_p, m_q, m_n, m_phi, m_e, m_d, nullptr);
    }
};

}

#endif  // RSA_H
