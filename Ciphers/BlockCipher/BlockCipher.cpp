#include <iostream>
#include <string>
#include <fstream>
#include <sstream>
#include <chrono>

#include "Feistel.h"

#define STB_IMAGE_IMPLEMENTATION
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "Utils.h"

using namespace Utils;


int main(int argc, char *argv[]) {

    CommandLineArgsInfo cmdArgs;
    if (!parseCommandLineArgs(argc, argv, cmdArgs)) {
        return -1;
    }

    std::cout << "Using config: ";
    if (cmdArgs.workMode & WorkingMode::ENCRYPT) {
        std::cout << "encrypting ";
    }
    else if (cmdArgs.workMode & WorkingMode::DECRYPT) {
        std::cout << "decrypting ";
    }

    if (cmdArgs.workMode & WorkingMode::INPUT_TEXT) {
        std::cout << "text";
    }
    else if (cmdArgs.workMode & WorkingMode::INPUT_IMAGE) {
        std::cout << "image";
    }
    std::cout << std::endl;

    std::vector<uint8_t> bytesIn;
    std::vector<uint8_t> bytesOut;
    Image imageIn;
    Image imageOut;

    if (cmdArgs.workMode & WorkingMode::INPUT_TEXT) {
        std::ifstream fileIn;
        fileIn.open(cmdArgs.filenameIn);
        if (!fileIn.is_open()) {
            std::cerr << "Could not open file: " << cmdArgs.filenameIn << std::endl;
            return -1;
        }
        std::stringstream bufferIn;
        bufferIn << fileIn.rdbuf();
        fileIn.close();

        convertStringToBytes(bufferIn.str(), bytesIn);
    }
    else if (cmdArgs.workMode & WorkingMode::INPUT_IMAGE) {
        if (!loadImage(cmdArgs.filenameIn, imageIn)) {
            std::cerr << "Could not load image: " << cmdArgs.filenameIn << std::endl;
            return -1;
        }
        convertImageToBytes(imageIn, bytesIn);
    }



    BlockCipher::FeistelCipherEncryptor cipherer;
    cipherer.setNumWorkers(cmdArgs.numWorkers);

    auto start = std::chrono::high_resolution_clock::now();
    if (cmdArgs.workMode & WorkingMode::ENCRYPT) {
        std::cout << "Encrypting..." << std::endl;
        cipherer.encrypt(bytesIn, bytesOut);
    }
    else if (cmdArgs.workMode & WorkingMode::DECRYPT) {
        std::cout << "Decrypting..." << std::endl;
        cipherer.decrypt(bytesIn, bytesOut);
    }
    auto end = std::chrono::high_resolution_clock::now();

    const uint64_t microseconds = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
    std::cout << "Done! Elapsed time: " << microseconds / 1000000.0 << " seconds" << std::endl;


    if (cmdArgs.workMode & WorkingMode::INPUT_TEXT) {
        std::ofstream fileOut;
        fileOut.open(cmdArgs.filenameOut);
        if (!fileOut.is_open()) {
            std::cerr << "Could not open file: " << cmdArgs.filenameOut << std::endl;
            return -1;
        }
        fileOut.write((const char *)bytesOut.data(), bytesOut.size());
        fileOut.close();
    }
    else if (cmdArgs.workMode & WorkingMode::INPUT_IMAGE) {
        imageOut.width = imageIn.width;
        imageOut.height = imageIn.height;
        convertBytesToImage(bytesOut, imageOut);
        saveImage(cmdArgs.filenameOut, imageOut);
        stbi_image_free(imageIn.data);
        stbi_image_free(imageOut.data);
    }

    return 0;
}
