#pragma once

#include <string>

#define TAG_SIZE 16
#define BLOCK_SIZE 25
#define ELEPHANT_NUMBER_OF_BYTES 64
#define NONCE_NUM_BYTES 12
#define KEY_NUMBER_OF_BYTES 16
#define MAX_NUMBER_OF_KESSAK_ROUNDS 18
#define INDEX(a, b) (((a) % 5) + 5 * ((b) % 5)) 

class ElephantEncryptor {
  unsigned long long lenMessage;
  unsigned long long len;
  unsigned char plaintext[ELEPHANT_NUMBER_OF_BYTES];
  unsigned char cipher[ELEPHANT_NUMBER_OF_BYTES];
  unsigned char nonce[NONCE_NUM_BYTES] = "";
  std::string aData;
  unsigned char ns[TAG_SIZE] = "";
  unsigned char key[KEY_NUMBER_OF_BYTES];

  char pl[ELEPHANT_NUMBER_OF_BYTES] = "jopa";
  char hex[ELEPHANT_NUMBER_OF_BYTES] = "";
  char key_hex[2 * KEY_NUMBER_OF_BYTES + 1] = "0123456789ABCDEF0123456789ABCDEF";
  char nonce_hex[2 * NONCE_NUM_BYTES + 1] = "000000000000111111111111";
  char additional[TAG_SIZE] = "kek";

public:
  ElephantEncryptor() {
    aData.reserve(1);
  }

  std::string encrypt(const std::string &PlainText);
  std::string decrypt(const std::string &CipherText);
};