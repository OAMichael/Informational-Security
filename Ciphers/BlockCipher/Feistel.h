#ifndef FEISTEL_H
#define FEISTEL_H

#include "Utils.h"

#include <cstdint>
#include <vector>
#include <string>
#include <cstring>

namespace BlockCipher {

static constexpr size_t BlockBitSize  = 64; 
static constexpr size_t KeyBitSize    = 64; 
static constexpr size_t BlockByteSize = BlockBitSize / 8; 
static constexpr size_t KeyByteSize   = KeyBitSize / 8; 
static constexpr size_t NumberOfRound = 16;
static constexpr size_t KeysInBlock   = BlockBitSize / KeyBitSize;


struct Block {
    uint8_t data[BlockByteSize] = {};

    uint8_t& operator[](const int idx) { return data[idx]; }
    const uint8_t& operator[](const int idx) const { return data[idx]; }
};


struct HalfBlock {
    uint8_t data[BlockByteSize / 2] = {};

    uint8_t& operator[](const int idx) { return data[idx]; }
    const uint8_t& operator[](const int idx) const { return data[idx]; }

    HalfBlock operator^(const HalfBlock& other) {
        HalfBlock ret;
        for (int i = 0; i < BlockByteSize / 2; ++i) {
            ret[i] = data[i] ^ other.data[i];
        }
        return ret;
    }
};


struct Key {
    uint8_t data[KeyByteSize] = {};

    uint8_t& operator[](const int idx) { return data[idx]; }
    const uint8_t& operator[](const int idx) const { return data[idx]; }
};



class FeistelCipherEncryptor {
private:
    Key m_masterKey = { 14, 48, 228, 13, 37, 42, 69, 223 };
    Key m_subkeys[NumberOfRound];

    size_t m_numWorkers = 1;

    template <typename Cont_t>
    void splitBytesIntoBlocks(Cont_t &bytesIn, std::vector<Block> &outBlocks) const {
        const size_t inputTextBitSize = bytesIn.size() * 8;
        const size_t numOutBlocks = inputTextBitSize / BlockBitSize + (inputTextBitSize % BlockBitSize != 0); 

        outBlocks.resize(numOutBlocks);
        std::memcpy(outBlocks.data(), bytesIn.data(), bytesIn.size());
    }

    template <typename Cont_t>
    void mergeBlocksIntoBytes(const std::vector<Block> &inBlocks, Cont_t &bytesOut) const {
        // Perform copying
        const size_t totalSize = inBlocks.size() * sizeof(Block);
        bytesOut.resize(totalSize);

        #pragma omp parallel for num_threads(m_numWorkers)
        for (int i = 0; i < totalSize; ++i) {
            bytesOut[i] = inBlocks[i / BlockByteSize][i % BlockByteSize];
        }
    }

    void generateKeys();

    void makeRound(std::vector<Block> &blocks, const int roundIdx) const;
    void functionF(const HalfBlock &inHBlock, const Key &subkey, HalfBlock &outHBlock) const;
    void swapHalfBlockes(std::vector<Block> &blocks) const;

public:
    void setNumWorkers(const size_t numWorkers);
    void setMasterKey(const Key &inKey);
    
    template <typename Cont_t>
    void encrypt(const Cont_t &bytesIn, Cont_t &bytesOut) {
        static_assert(sizeof(typename Cont_t::value_type) == 1);
        std::vector<Block> blocks;
        splitBytesIntoBlocks(bytesIn, blocks);

        for (int i = 0; i < NumberOfRound; ++i) {
            makeRound(blocks, i);
        }

        swapHalfBlockes(blocks);
        mergeBlocksIntoBytes(blocks, bytesOut);

        Utils::shuffleBytes(bytesOut, false);
    }

    template <typename Cont_t>
    void decrypt(const Cont_t &bytesIn, Cont_t &bytesOut) {
        static_assert(sizeof(typename Cont_t::value_type) == 1);
        Cont_t tmpIn(bytesIn.begin(), bytesIn.end());
        Utils::shuffleBytes(tmpIn, true);

        std::vector<Block> blocks;
        splitBytesIntoBlocks(tmpIn, blocks);

        // A little inorder. But it allows to use same function for decryption
        for (int i = NumberOfRound - 1; i >= 0; --i) {
            makeRound(blocks, i);
        }

        swapHalfBlockes(blocks);
        mergeBlocksIntoBytes(blocks, bytesOut);
    }

    FeistelCipherEncryptor() {
        generateKeys();
    };
};

}   // BlockCipher


#endif  // FEISTEL_H
