#include <cstring>

#define STB_IMAGE_IMPLEMENTATION
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "Feistel.h"
#include "Utils.h"

namespace BlockCipher {

void FeistelCipherEncryptor::generateKeys() {
    // Simple cyclic shift to the left
    for (int j = 0; j < KeyByteSize; ++j) {
        m_subkeys[0][j] = m_masterKey[j];
    }

    for (int i = 1; i < NumberOfRound; ++i) {
        uint8_t tmp = m_subkeys[i - 1][0];
        for (int j = 0; j < KeyByteSize - 1; ++j) {
            m_subkeys[i][j] = m_subkeys[i - 1][j + 1];
        }
        m_subkeys[i][KeyByteSize - 1] = tmp;
    }
}


void FeistelCipherEncryptor::makeRound(std::vector<Block> &blocks, const int roundIdx) const {
    const size_t blocksCount = blocks.size();

    #pragma omp parallel for num_threads(m_numWorkers)
    for (size_t i = 0; i < blocksCount; ++i) {
        HalfBlock L, R;
        std::memcpy(&L, &blocks[i], sizeof(HalfBlock));
        std::memcpy(&R, (char *)&blocks[i] + sizeof(HalfBlock), sizeof(HalfBlock));

        HalfBlock tmp;
        functionF(R, m_subkeys[roundIdx], tmp);

        HalfBlock newR = L ^ tmp;

        std::memcpy(&blocks[i], &R, sizeof(HalfBlock));
        std::memcpy((char *)&blocks[i] + sizeof(HalfBlock), &newR, sizeof(HalfBlock));
    }
}


void FeistelCipherEncryptor::functionF(const HalfBlock &inHBlock, const Key &subkey, HalfBlock &outHBlock) const {
    for (int i = 0; i < BlockByteSize / 2; ++i) {
        const uint8_t currHBByte = inHBlock[i];
        const uint8_t currKeyByte = subkey[i % KeyByteSize];

        // Some weird stuff is going on here
        outHBlock[i] = (currHBByte ^ currKeyByte & currHBByte ^ 0xD5) & (currKeyByte >> 3) ^ currHBByte | currKeyByte;
    }
}


void FeistelCipherEncryptor::swapHalfBlockes(std::vector<Block> &blocks) const {
    const size_t blocksCount = blocks.size();

    #pragma omp parallel for num_threads(m_numWorkers)
    for (size_t i = 0; i < blocksCount; ++i) {
        HalfBlock tmp;
        std::memcpy(&tmp, &blocks[i], sizeof(HalfBlock));
        std::memcpy(&blocks[i], (char *)&blocks[i] + sizeof(HalfBlock), sizeof(HalfBlock));
        std::memcpy((char *)&blocks[i] + sizeof(HalfBlock), &tmp, sizeof(HalfBlock));
    }
}


void FeistelCipherEncryptor::setMasterKey(const Key& inKey) {
    m_masterKey = inKey;
}

void FeistelCipherEncryptor::setNumWorkers(const size_t numWorkers) {
    m_numWorkers = numWorkers;
}

}   // BlockCipher
