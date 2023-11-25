#ifndef UTILS_H
#define UTILS_H

#include <string>
#include <iostream>
#include <vector>
#include <cstring>
#include <random>
#include <algorithm>

#include "stb_image.h"
#include "stb_image_write.h"

namespace Utils {

enum WorkingMode : uint16_t {
    ENCRYPT     = 1 << 0,
    DECRYPT     = 1 << 1,
    INPUT_TEXT  = 1 << 2,
    INPUT_IMAGE = 1 << 3
};

inline WorkingMode operator|(WorkingMode lhs, WorkingMode rhs) {
    return static_cast<WorkingMode>(static_cast<uint16_t>(lhs) | static_cast<uint16_t>(rhs));
}
    
inline WorkingMode& operator|=(WorkingMode& lhs, WorkingMode rhs) {
    lhs = lhs | rhs;
    return lhs;
}


struct Image {
    int width;
    int height;
    uint8_t *data = nullptr;
};

static bool loadImage(const std::string &filename, Image &imageOut) {
    int comp = 4;
    imageOut.data = stbi_load(filename.c_str(), &imageOut.width, &imageOut.height, &comp, comp);
    return imageOut.data;
}

static void saveImage(const std::string &filename, const Image &imageIn) {
    stbi_write_png(filename.c_str(), imageIn.width, imageIn.height, 4, imageIn.data, 4 * imageIn.width);
}



static inline void convertStringToBytes(const std::string &strIn, std::vector<uint8_t> &bytesOut) {
    bytesOut.resize(strIn.size());
    std::memcpy(bytesOut.data(), strIn.data(), strIn.size());
}

static inline void convertBytesToString(const std::vector<uint8_t> &bytesIn, std::string &strOut) {
    strOut.resize(bytesIn.size());
    std::memcpy(&strOut[0], bytesIn.data(), bytesIn.size());
}

static inline void convertImageToBytes(const Image &image, std::vector<uint8_t> &bytesOut) {
    const size_t imageSize = 4 * image.width * image.height;
    bytesOut.resize(imageSize);
    std::memcpy(bytesOut.data(), image.data, imageSize);
}

static inline void convertBytesToImage(const std::vector<uint8_t> &bytesIn, Image &image) {
    if (!image.data) {
        image.data = new uint8_t[bytesIn.size()];
    }
    std::memcpy(image.data, bytesIn.data(), bytesIn.size());
}



static void shuffleBytes(std::vector<uint8_t> &bytes, bool reverse = false, const uint64_t seed = 228) {
    static std::mt19937_64 randEngine{0};
    randEngine.seed(seed);
    std::vector<size_t> randIndices;
    randIndices.resize(bytes.size());

    std::vector<uint8_t> tmpBytes;
    tmpBytes.resize(bytes.size());
    std::memcpy(tmpBytes.data(), bytes.data(), bytes.size());

    std::iota(std::begin(randIndices), std::end(randIndices), 0);
    std::shuffle(randIndices.begin(), randIndices.end(), randEngine);

    if (reverse) {
        for (size_t i = 0; i < randIndices.size(); ++i) {
            bytes[randIndices[i]] = tmpBytes[i];
        }
    }
    else {
        for (size_t i = 0; i < randIndices.size(); ++i) {
            bytes[i] = tmpBytes[randIndices[i]];
        }
    }
}


static inline bool parseCommandLineArgs(const int argc, char *argv[], std::string &filenameIn, std::string &filenameOut, WorkingMode &workMode) {
    if (argc < 2) {
        std::cout << "Usage: " << argv[0] << " <filenameIn> [-o <filenameOut> ...]" << std::endl;
        return false;
    }

    filenameIn = argv[1];
    filenameOut = "a.out";
    workMode = static_cast<WorkingMode>(0);

    WorkingMode encryptType = static_cast<WorkingMode>(0);
    WorkingMode inputType = static_cast<WorkingMode>(0);

    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "-o") {
            if (i + 1 < argc) {
                filenameOut = argv[i + 1];
                ++i;
            }
            continue;
        }
        if (arg == "-e" || arg == "--encrypt") {
            if (encryptType & WorkingMode::DECRYPT) {
                std::cerr << "Cannot specify both encryption and decryption working modes" << std::endl;
                return false;
            }
            encryptType |= WorkingMode::ENCRYPT;
            continue;
        }
        if (arg == "-d" || arg == "--decrypt") {
            if (encryptType & WorkingMode::ENCRYPT) {
                std::cerr << "Cannot specify both encryption and decryption working modes" << std::endl;
                return false;
            }
            encryptType |= WorkingMode::DECRYPT;
            continue;
        }
        if (arg == "-t" || arg == "--text") {
            if (inputType & WorkingMode::INPUT_IMAGE) {
                std::cerr << "Cannot specify both text and image input working modes" << std::endl;                
                return false;
            }
            inputType |= WorkingMode::INPUT_TEXT;
            continue;
        }
        if (arg == "-i" || arg == "--image") {
            if (inputType & WorkingMode::INPUT_TEXT) {
                std::cerr << "Cannot specify both text and image input working modes" << std::endl;                
                return false;
            }
            inputType |= WorkingMode::INPUT_IMAGE;
            continue;
        }
    }

    if (encryptType > 0) {
        workMode |= encryptType;
    }
    else {
        workMode |= WorkingMode::ENCRYPT;
    }

    if (inputType > 0) {
        workMode |= inputType;
    }
    else {
        workMode |= WorkingMode::INPUT_TEXT;
    }

    return true;
}

}

#endif  // UTILS_H
