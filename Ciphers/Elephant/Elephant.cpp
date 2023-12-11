#include "Elephant.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cassert>

typedef unsigned char BYTE; // 8 bit
typedef unsigned long long LEN;
const unsigned int kessak_rho_offsets[25] = {0, 1, 6, 4, 3, 4, 4, 6, 7, 4, 3, 2, 3, 1, 7, 1, 5, 7, 5, 0, 2, 2, 5, 0, 6};

const BYTE kessak_round_constants[MAX_NUMBER_OF_KESSAK_ROUNDS] = {
    0x01, 0x82, 0x8a, 0x00, 0x8b, 0x01, 0x81, 0x09, 0x8a,
    0x88, 0x09, 0x0a, 0x8b, 0x8b, 0x89, 0x03, 0x02, 0x80};

// Macro to perform a cyclic left shift on eight bits
#define ROTATE_LEFT_8_BIT(a, offset) ((offset != 0) ? ((((BYTE)a) << offset) ^ (((BYTE)a) >> (sizeof(BYTE) * 8 - offset))) : a)

void theta(BYTE *TRIAL_ARR) {
  unsigned int a, b;
  BYTE ARR[5], ARR2[5];

  for (a = 0; a < 5; a++) {
    ARR[a] = 0;
    for (b = 0; b < 5; b++)
      ARR[a] ^= TRIAL_ARR[INDEX(a, b)];
  }

  for (a = 0; a < 5; a++)
    ARR2[a] = ROTATE_LEFT_8_BIT(ARR[(a + 1) % 5], 1) ^ ARR[(a + 4) % 5];

  for (a = 0; a < 5; a++)
    for (b = 0; b < 5; b++)
      TRIAL_ARR[INDEX(a, b)] ^= ARR2[a];
}

void rho(BYTE *TRIAL_ARR) {
  for (unsigned int a = 0; a < 5; a++)
    for (unsigned int b = 0; b < 5; b++)
      TRIAL_ARR[INDEX(a, b)] = ROTATE_LEFT_8_BIT(TRIAL_ARR[INDEX(a, b)], kessak_rho_offsets[INDEX(a, b)]);
}

void pi(BYTE *TRIAL_ARR) {
  BYTE fix_mask[25];

  for (unsigned int a = 0; a < 5; a++)
    for (unsigned int b = 0; b < 5; b++)
      fix_mask[INDEX(a, b)] = TRIAL_ARR[INDEX(a, b)];

  for (unsigned int a = 0; a < 5; a++)
    for (unsigned int b = 0; b < 5; b++)
      TRIAL_ARR[INDEX(0 * a + 1 * b, 2 * a + 3 * b)] = fix_mask[INDEX(a, b)];
}

void hi(BYTE *TRIAL_ARR) {
  unsigned int a, b;
  BYTE ARR[5];

  for (b = 0; b < 5; b++) {
    for (a = 0; a < 5; a++)
      ARR[a] = TRIAL_ARR[INDEX(a, b)] ^ ((~TRIAL_ARR[INDEX(a + 1, b)]) & TRIAL_ARR[INDEX(a + 2, b)]);
    for (a = 0; a < 5; a++)
      TRIAL_ARR[INDEX(a, b)] = ARR[a];
  }
}

void io(BYTE *TRIAL_ARR, unsigned int ind) {
  TRIAL_ARR[INDEX(0, 0)] ^= kessak_round_constants[ind];
}

void Kessak200UnionForOneRound(BYTE *tmp_state, unsigned int tmp_round_idx) {
  theta(tmp_state);
  rho(tmp_state);
  pi(tmp_state);
  hi(tmp_state);
  io(tmp_state, tmp_round_idx);
}

void permutation(BYTE *param) {
  for (unsigned int t = 0; t < MAX_NUMBER_OF_KESSAK_ROUNDS; t++)
    Kessak200UnionForOneRound(param, t);
}

BYTE LeftShift(BYTE byte) {
  return (byte << 1) | (byte >> 7);
}

// Cmp function compares two blocks of data byte by byte and returns 0 if the blocks are identical, and 1 otherwise
int cmp(const BYTE *a, const BYTE *b, LEN length) {
  BYTE r = 0;
  for (LEN i = 0; i < length; ++i)
    r |= a[i] ^ b[i];
  return r;
}

// Implements a single step linear feedback shifter (LFSR). Used to generate a mask for data encryption
void lfsr(BYTE *out, BYTE *in) {
  BYTE cur = LeftShift(in[0]) ^ LeftShift(in[2]) ^ (in[13] << 1);
  for (LEN t = 0; t < BLOCK_SIZE - 1; ++t)
    out[t] = in[t + 1];
  out[BLOCK_SIZE - 1] = cur;
}

void xorOfBlock(BYTE *tmp_state, const BYTE *block, LEN length) {
  for (LEN t = 0; t < length; ++t)
    tmp_state[t] ^= block[t];
}

void get_ciphertext_block(BYTE *out, const char *c, LEN cipher_len, LEN t) {
  const LEN offset = t * BLOCK_SIZE;

  if (offset == cipher_len) {
    memset(out, 0x00, BLOCK_SIZE);
    out[0] = 0x01;
    return;
  }
  const LEN r_text = cipher_len - offset;

  if (BLOCK_SIZE <= r_text) {
    memcpy(out, c + offset, BLOCK_SIZE);
  } else {
    if (r_text > 0) // c might be nullptr
      memcpy(out, c + offset, r_text);
    memset(out + r_text, 0x00, BLOCK_SIZE - r_text);
    out[r_text] = 0x01;
  }
}

void get_associated_data_block(BYTE *out, const std::string &aData, const BYTE *nonce, LEN t) {
  const LEN offset = t * BLOCK_SIZE - (t != 0) * NONCE_NUM_BYTES;
  LEN len = 0;

  if (t == 0){
    memcpy(out, nonce, NONCE_NUM_BYTES);
    len += NONCE_NUM_BYTES;
  }

  if (t != 0 && offset == aData.size()) {
    memset(out, 0x00, BLOCK_SIZE);
    out[0] = 0x01;
    return;
  }
  const LEN r_out = BLOCK_SIZE - len;
  const LEN r_data = aData.size() - offset;

  if (r_out <= r_data) { // enough AD
    memcpy(out + len, aData.data() + offset, r_out);
  }
  else {              // not enough AD, need to pad
    if (r_data > 0) // ad might be nullptr
      memcpy(out + len, aData.data() + offset, r_data);
    memset(out + len + r_data, 0x00, r_out - r_data);
    out[len + r_data] = 0x01;
  }
}

void implementation_of_crypto(std::string &c, BYTE *tag, 
                         const std::string &m, const std::string &aData,
                         const BYTE *nonce, const BYTE *k, bool encrypt) {
  const LEN num_of_blocks_cipher = 1 + m.size() / BLOCK_SIZE;
  const LEN num_of_blocks_message = (m.size() % BLOCK_SIZE) ? num_of_blocks_cipher : num_of_blocks_cipher - 1;
  const LEN num_of_blocks_adata = 1 + (NONCE_NUM_BYTES + aData.size()) / BLOCK_SIZE;
  const LEN num_of_blocks_it = (num_of_blocks_cipher > num_of_blocks_adata) ? num_of_blocks_cipher : num_of_blocks_adata + 1;

  BYTE expanded_k[BLOCK_SIZE] = {0};
  memcpy(expanded_k, k, KEY_NUMBER_OF_BYTES);
  permutation(expanded_k);

  BYTE buffer_back[BLOCK_SIZE] = {0};
  BYTE buffer_current[BLOCK_SIZE] = {0};
  BYTE buffer_forward[BLOCK_SIZE] = {0};
  memcpy(buffer_current, expanded_k, BLOCK_SIZE);

  BYTE *mask_prev = buffer_back;
  BYTE *mask_tmp = buffer_current;
  BYTE *mask_next = buffer_forward;
  BYTE elephant_buf[BLOCK_SIZE];
  BYTE tag_buf[BLOCK_SIZE] = {0};
  memset(tag, 0, TAG_SIZE);

  LEN offset = 0;

  for (LEN t = 0; t < num_of_blocks_it; ++t) {
    lfsr(mask_next, mask_tmp);

    if (t < num_of_blocks_message) {
        memcpy(elephant_buf, nonce, NONCE_NUM_BYTES);
        memset(elephant_buf + NONCE_NUM_BYTES, 0, BLOCK_SIZE - NONCE_NUM_BYTES);
        xorOfBlock(elephant_buf, mask_tmp, BLOCK_SIZE);
        xorOfBlock(elephant_buf, mask_next, BLOCK_SIZE);
        permutation(elephant_buf);
        xorOfBlock(elephant_buf, mask_tmp, BLOCK_SIZE);
        xorOfBlock(elephant_buf, mask_next, BLOCK_SIZE);
        const LEN r_len = (t == num_of_blocks_message - 1) ? m.size() - offset : BLOCK_SIZE;
        xorOfBlock(elephant_buf, reinterpret_cast<const BYTE *>(m.data()) + offset, r_len);
        memcpy(c.data() + offset, elephant_buf, r_len);
    }

    if (t > 0 && t <= num_of_blocks_cipher) {
        // Compute tag for ciphertext block
        get_ciphertext_block(tag_buf, encrypt ? c.data() : m.data(), m.size(), t - 1);
        xorOfBlock(tag_buf, mask_prev, BLOCK_SIZE);
        xorOfBlock(tag_buf, mask_next, BLOCK_SIZE);
        permutation(tag_buf);
        xorOfBlock(tag_buf, mask_prev, BLOCK_SIZE);
        xorOfBlock(tag_buf, mask_next, BLOCK_SIZE);
        xorOfBlock(tag, tag_buf, TAG_SIZE);
    }

    // If there is any AD left, compute tag for AD block 
    if (t + 1 < num_of_blocks_adata) {
        get_associated_data_block(tag_buf, aData, nonce, t + 1);
        xorOfBlock(tag_buf, mask_next, BLOCK_SIZE);
        permutation(tag_buf);
        xorOfBlock(tag_buf, mask_next, BLOCK_SIZE);
        xorOfBlock(tag, tag_buf, TAG_SIZE);
    }

    // Cyclically shift the mask buffers
    // Value of next_mask will be computed in the next iteration
    BYTE *const fix_mask = mask_prev;
    mask_prev = mask_tmp;
    mask_tmp = mask_next;
    mask_next = fix_mask;

    offset += BLOCK_SIZE;
  }
}

void encrypt(std::string &CipherText,
            const std::string &PlainText,
            const std::string &aData,
            const unsigned char *nonce,
            const unsigned char *k) {
  CipherText.resize(PlainText.size() + TAG_SIZE);
  BYTE tag[1];
  implementation_of_crypto(CipherText, tag, PlainText, aData, nonce, k, true);
  memcpy(CipherText.data() + PlainText.size(), tag, TAG_SIZE);
}

void decrypt(std::string &PlainText,
             const std::string &CipherText,
             const std::string &aData,
             const unsigned char *nonce,
             const unsigned char *k) {
  assert(CipherText.size() >= TAG_SIZE);
  PlainText.resize(CipherText.size() - TAG_SIZE);
  BYTE tag[1];
  implementation_of_crypto(PlainText, tag, CipherText, aData, nonce, k, false);
}

std::string ElephantEncryptor::encrypt(const std::string &PlainText) {
  auto CipherText = std::string{};
  ::encrypt(CipherText, PlainText, aData, nonce, key);
  return CipherText;
}

std::string ElephantEncryptor::decrypt(const std::string &CipherText) {
  auto PlainText = std::string{};
  ::decrypt(PlainText, CipherText, aData, nonce, key);
  return PlainText;
}