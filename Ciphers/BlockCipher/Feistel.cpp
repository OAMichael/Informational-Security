#include <cstring>

#include "Feistel.h"
#include "Utils.h"

namespace BlockCipher {


void FeistelCipherEncryptor::splitBytesIntoBlocks(const std::vector<uint8_t> &bytesIn, std::vector<Block> &outBlocks) const {
    const size_t inputTextBitSize = bytesIn.size() * 8;
    const size_t numOutBlocks = inputTextBitSize / BlockBitSize + (inputTextBitSize % BlockBitSize != 0); 

    outBlocks.resize(numOutBlocks);
    std::memcpy(outBlocks.data(), bytesIn.data(), bytesIn.size());
}


void FeistelCipherEncryptor::mergeBlocksIntoBytes(const std::vector<Block> &inBlocks, std::vector<uint8_t> &bytesOut) const {
    // Perform copying
    const size_t totalSize = inBlocks.size() * sizeof(Block);
    bytesOut.resize(totalSize);
    for (int i = 0; i < totalSize; ++i) {
        bytesOut[i] = inBlocks[i / BlockByteSize][i % BlockByteSize];
    }
}


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
    for (auto &block : blocks) {
        HalfBlock L, R;
        std::memcpy(&L, &block, sizeof(HalfBlock));
        std::memcpy(&R, (char *)&block + sizeof(HalfBlock), sizeof(HalfBlock));

        HalfBlock tmp;
        functionF(R, m_subkeys[roundIdx], tmp);

        HalfBlock newR = L ^ tmp;

        std::memcpy(&block, &R, sizeof(HalfBlock));
        std::memcpy((char *)&block + sizeof(HalfBlock), &newR, sizeof(HalfBlock));
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
    for (auto &block : blocks) {
        HalfBlock tmp;
        std::memcpy(&tmp, &block, sizeof(HalfBlock));
        std::memcpy(&block, (char *)&block + sizeof(HalfBlock), sizeof(HalfBlock));
        std::memcpy((char *)&block + sizeof(HalfBlock), &tmp, sizeof(HalfBlock));
    }
}


void FeistelCipherEncryptor::setMasterKey(const Key& inKey) {
    m_masterKey = inKey;
}


void FeistelCipherEncryptor::encrypt(std::vector<uint8_t> &bytesIn, std::vector<uint8_t> &bytesOut) {
    
    std::vector<Block> blocks;
    splitBytesIntoBlocks(bytesIn, blocks);

    for (int i = 0; i < NumberOfRound; ++i) {
        makeRound(blocks, i);
    }

    swapHalfBlockes(blocks);
    mergeBlocksIntoBytes(blocks, bytesOut);

    Utils::shuffleBytes(bytesOut, false);
}


void FeistelCipherEncryptor::decrypt(std::vector<uint8_t> &bytesIn, std::vector<uint8_t> &bytesOut) {

    Utils::shuffleBytes(bytesIn, true);

    std::vector<Block> blocks;
    splitBytesIntoBlocks(bytesIn, blocks);

    // A little inorder. But it allows to use same function for decryption
    for (int i = NumberOfRound - 1; i >= 0; --i) {
        makeRound(blocks, i);
    }

    swapHalfBlockes(blocks);
    mergeBlocksIntoBytes(blocks, bytesOut);
}

}   // BlockCipher
