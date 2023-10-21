#ifndef FIESTEL_H
#define FIESTEL_H

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
    Key m_masterKey;
    Key m_subkeys[NumberOfRound];

    void splitIntoBlocks(const std::string &text, std::vector<Block> &outBlocks) const;
    void mergeBlocks(const std::vector<Block> &inBlocks, std::string &text) const;
    void generateKeys();
    void makeRound(std::vector<Block> &blocks, const int roundIdx) const;
    void functionF(const HalfBlock& inHBlock, const Key& subkey, HalfBlock& outHBlock) const;
    void swapHalfBlockes(std::vector<Block> &blocks) const;

public:
    void setMasterKey(const Key& inKey);
    void encrypt(const std::string &plaintext, std::string &ciphertext);
    void decrypt(const std::string &ciphertext, std::string &plaintext);
};

}   // BlockCipher


#endif  // FIESTEL_H
