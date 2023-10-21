#include <iostream>
#include <cstdint>
#include <cstring>
#include <vector>
#include <string>

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



class FiestelCipherEncryptor {
private:
    Key m_masterKey;
    Key m_subkeys[NumberOfRound];

    void splitIntoBlocks(const std::string &text, std::vector<Block> &outBlocks) const {
        const size_t inputTextBitSize = text.size() * 8;
        const size_t numOutBlocks = inputTextBitSize / BlockBitSize + (inputTextBitSize % BlockBitSize != 0); 

        outBlocks.resize(numOutBlocks);
        std::memcpy(outBlocks.data(), text.data(), text.size());
    }

    void mergeBlocks(const std::vector<Block> &inBlocks, std::string &text) const {
        // Perform copying
        const size_t totalSize = inBlocks.size() * sizeof(Block);
        text.resize(totalSize);
        for (int i = 0; i < totalSize; ++i) {
            text[i] = inBlocks[i / BlockByteSize][i % BlockByteSize];
        }
    }

    void generateKeys() {
        for (int i = 0; i < NumberOfRound; ++i) {
            for (int j = 0; j < KeyByteSize; ++j) {
                m_subkeys[i][j] = std::rand() % 256;
            }
        }
    }

    void makeRound(std::vector<Block> &blocks, const int roundIdx) const {
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

    void functionF(const HalfBlock& inHBlock, const Key& subkey, HalfBlock& outHBlock) const {
        for (int i = 0; i < BlockByteSize / 2; ++i) {
            const uint8_t currHBByte = inHBlock[i];
            const uint8_t currKeyByte = subkey[i % KeyByteSize];

            outHBlock[i] = currHBByte ^ currKeyByte;
        }
    }

    void swapHalfBlockes(std::vector<Block> &blocks) {
        for (auto &block : blocks) {
            HalfBlock tmp;
            std::memcpy(&tmp, &block, sizeof(HalfBlock));
            std::memcpy(&block, (char *)&block + sizeof(HalfBlock), sizeof(HalfBlock));
            std::memcpy((char *)&block + sizeof(HalfBlock), &tmp, sizeof(HalfBlock));
        }
    }


public:
    void setMasterKey(const Key& inKey) {
        m_masterKey = inKey;
    }

    void encrypt(const std::string &plaintext, std::string &ciphertext) {
        
        generateKeys();

        std::vector<Block> blocks;
        splitIntoBlocks(plaintext, blocks);

        for (int i = 0; i < NumberOfRound; ++i)
            makeRound(blocks, i);

        swapHalfBlockes(blocks);

        mergeBlocks(blocks, ciphertext);
    }

    void decrypt(const std::string &ciphertext, std::string &plaintext) {

        std::vector<Block> blocks;
        splitIntoBlocks(ciphertext, blocks);

        // A little inorder. But it allows to use same function for decryption
        for (int i = NumberOfRound - 1; i >= 0; --i)
            makeRound(blocks, i);

        swapHalfBlockes(blocks);

        mergeBlocks(blocks, plaintext);
    }
};


int main() {
    FiestelCipherEncryptor cipherer;
    
    std::string plaintext;
    std::string ciphertext;
    std::string decrypted;
        
    std::getline(std::cin, plaintext);

    std::cout << "\nPlaintext: " << plaintext.size() << "\n" << plaintext << std::endl;

    cipherer.encrypt(plaintext, ciphertext);

    std::cout << "\nCiphertext: " << ciphertext.size() << "\n" << ciphertext << std::endl;

    cipherer.decrypt(ciphertext, decrypted);

    std::cout << "\nDecrytped: " << decrypted.size() << "\n" << decrypted << std::endl;

    return 0;
}
