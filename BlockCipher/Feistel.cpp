#include <cstring>

#include "Feistel.h"


namespace BlockCipher {


void FeistelCipherEncryptor::splitIntoBlocks(const std::string &text, std::vector<Block> &outBlocks) const {
    const size_t inputTextBitSize = text.size() * 8;
    const size_t numOutBlocks = inputTextBitSize / BlockBitSize + (inputTextBitSize % BlockBitSize != 0); 

    outBlocks.resize(numOutBlocks);
    std::memcpy(outBlocks.data(), text.data(), text.size());
}


void FeistelCipherEncryptor::mergeBlocks(const std::vector<Block> &inBlocks, std::string &text) const {
    // Perform copying
    const size_t totalSize = inBlocks.size() * sizeof(Block);
    text.resize(totalSize);
    for (int i = 0; i < totalSize; ++i) {
        text[i] = inBlocks[i / BlockByteSize][i % BlockByteSize];
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


void FeistelCipherEncryptor::functionF(const HalfBlock& inHBlock, const Key& subkey, HalfBlock& outHBlock) const {
    for (int i = 0; i < BlockByteSize / 2; ++i) {
        const uint8_t currHBByte = inHBlock[i];
        const uint8_t currKeyByte = subkey[i % KeyByteSize];

        // Some weird stuff is going on here
        outHBlock[i] = (currHBByte ^ currKeyByte & currHBByte ^ 0xD5) & (currKeyByte >> 3);
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


void FeistelCipherEncryptor::encrypt(const std::string &plaintext, std::string &ciphertext) {
    
    generateKeys();

    std::vector<Block> blocks;
    splitIntoBlocks(plaintext, blocks);

    for (int i = 0; i < NumberOfRound; ++i)
        makeRound(blocks, i);

    swapHalfBlockes(blocks);

    mergeBlocks(blocks, ciphertext);
}


void FeistelCipherEncryptor::decrypt(const std::string &ciphertext, std::string &plaintext) {

    std::vector<Block> blocks;
    splitIntoBlocks(ciphertext, blocks);

    // A little inorder. But it allows to use same function for decryption
    for (int i = NumberOfRound - 1; i >= 0; --i)
        makeRound(blocks, i);

    swapHalfBlockes(blocks);

    mergeBlocks(blocks, plaintext);
}

}   // BlockCipher
