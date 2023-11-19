#ifndef FEISTEL_H
#define FEISTEL_H

#include <cstdint>
#include <vector>
#include <string>


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

    void splitBytesIntoBlocks(const std::vector<uint8_t> &bytesIn, std::vector<Block> &outBlocks) const;
    void mergeBlocksIntoBytes(const std::vector<Block> &inBlocks, std::vector<uint8_t> &bytesOut) const;

    void generateKeys();

    void makeRound(std::vector<Block> &blocks, const int roundIdx) const;
    void functionF(const HalfBlock &inHBlock, const Key &subkey, HalfBlock &outHBlock) const;
    void swapHalfBlockes(std::vector<Block> &blocks) const;

public:
    void setMasterKey(const Key &inKey);
    void encrypt(std::vector<uint8_t> &bytesIn, std::vector<uint8_t> &bytesOut);
    void decrypt(std::vector<uint8_t> &bytesIn, std::vector<uint8_t> &bytesOut);

    FeistelCipherEncryptor() {
        generateKeys();
    };
};

}   // BlockCipher


#endif  // FEISTEL_H
