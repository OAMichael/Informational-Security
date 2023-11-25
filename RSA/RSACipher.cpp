#include <iostream>
#include <string>
#include <fstream>
#include <sstream>
#include <chrono>

#include "RSA.h"

#define STB_IMAGE_IMPLEMENTATION
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "Utils.h"

using namespace Utils;


int main(int argc, char *argv[]) {

    std::string filenameIn;
    std::string filenameOut;
    WorkingMode workMode;

    if (!parseCommandLineArgs(argc, argv, filenameIn, filenameOut, workMode)) {
        return -1;
    }

    std::cout << "Using config: ";
    if (workMode & WorkingMode::ENCRYPT) {
        std::cout << "encrypting ";
    }
    else if (workMode & WorkingMode::DECRYPT) {
        std::cout << "decrypting ";
    }

    if (workMode & WorkingMode::INPUT_TEXT) {
        std::cout << "text";
    }
    else if (workMode & WorkingMode::INPUT_IMAGE) {
        std::cout << "image";
    }
    std::cout << std::endl;

    std::vector<uint8_t> bytesIn;
    std::vector<uint8_t> bytesOut;
    Image imageIn;
    Image imageOut;

    if (workMode & WorkingMode::INPUT_TEXT) {
        std::ifstream fileIn;
        fileIn.open(filenameIn);
        if (!fileIn.is_open()) {
            std::cerr << "Could not open file: " << filenameIn << std::endl;
            return -1;
        }
        std::stringstream bufferIn;
        bufferIn << fileIn.rdbuf();
        fileIn.close();

        convertStringToBytes(bufferIn.str(), bytesIn);
    }
    else if (workMode & WorkingMode::INPUT_IMAGE) {
        if (!loadImage(filenameIn, imageIn)) {
            std::cerr << "Could not load image: " << filenameIn << std::endl;
            return -1;
        }
        convertImageToBytes(imageIn, bytesIn);
    }



    RSACipher::RSAEncryptor cipherer;

    auto start = std::chrono::high_resolution_clock::now();
    if (workMode & WorkingMode::ENCRYPT) {
        std::cout << "Encrypting..." << std::endl;
        cipherer.encrypt(bytesIn, bytesOut);
    }
    else if (workMode & WorkingMode::DECRYPT) {
        std::cout << "Decrypting..." << std::endl;
        cipherer.decrypt(bytesIn, bytesOut);
    }
    auto end = std::chrono::high_resolution_clock::now();

    const uint64_t microseconds = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
    std::cout << "Done! Elapsed time: " << microseconds / 1000000.0 << " seconds" << std::endl;


    if (workMode & WorkingMode::INPUT_TEXT) {
        std::ofstream fileOut;
        fileOut.open(filenameOut);
        if (!fileOut.is_open()) {
            std::cerr << "Could not open file: " << filenameOut << std::endl;
            return -1;
        }
        fileOut.write((const char *)bytesOut.data(), bytesOut.size());
        fileOut.close();
    }
    else if (workMode & WorkingMode::INPUT_IMAGE) {
        imageOut.width = imageIn.width;
        imageOut.height = imageIn.height;
        convertBytesToImage(bytesOut, imageOut);
        saveImage(filenameOut, imageOut);
        stbi_image_free(imageIn.data);
        stbi_image_free(imageOut.data);
    }

    return 0;
}
