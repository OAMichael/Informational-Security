#include <iostream>
#include <string>

#include "Feistel.h"


int main() {
    BlockCipher::FeistelCipherEncryptor cipherer;
    BlockCipher::Key masterKey = { 14, 48, 228, 13, 37, 42, 69, 223 };
    cipherer.setMasterKey(masterKey);

    std::string plaintext;
    std::string ciphertext;
    std::string decrypted;
        
    std::getline(std::cin, plaintext);

    std::cout << "\nPlaintext: " << plaintext.size() << " symbols\n" << plaintext << std::endl;

    cipherer.encrypt(plaintext, ciphertext);

    std::cout << "\nCiphertext: " << ciphertext.size() << " symbols\n" << ciphertext << std::endl;

    cipherer.decrypt(ciphertext, decrypted);

    std::cout << "\nDecrypted: " << decrypted.size() << " symbols\n" << decrypted << std::endl;

    return 0;
}
