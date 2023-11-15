#include <iostream>
#include <cmath>
#include <algorithm>
#include <random>

static std::random_device dev;
static std::mt19937_64 rng(dev());

// Check if number is prime
bool isPrime(const uint64_t n) {
    const uint64_t upperBound = std::ceil(std::sqrt(n));
    for (uint64_t i = 2; i <= upperBound; ++i){
        if (n % i == 0) {
            return false;
        }
    }
    return true;
}

// Generate random prime number
uint64_t getRandomPrime(const uint64_t min = (1ULL << 10), const uint64_t max = (1ULL << 15)) {
    static std::uniform_int_distribution<uint64_t> dist(min, max);

    uint64_t num = dist(rng);

    // Increase num until it is a prime number
    while (!isPrime(num)) {
        ++num;
    }
    return num;
}

// Function to find GCD (Greatest Common Divisor) of a and b
inline uint64_t GCD(const uint64_t a, const uint64_t b) {
    return std::__gcd(a, b);
}


// Function which is looking for a random number e
// from (2, phi-1) such that GCD(e, phi) = 1
uint64_t lookingForE(uint64_t phi) {
    std::uniform_int_distribution<uint64_t> dist(2, phi);
    
    uint64_t e = dist(rng);
    while ((GCD(phi, e) != 1) && (!isPrime(e))){
        ++e;
    }
    return e;
}


// Function to find inverse element for e in the ring of remainders of phi
uint64_t inverseE(const uint64_t phi, const uint64_t e) {
    uint64_t r[3] = {};
    uint64_t q = 0;
    uint64_t v[3] = {};

    r[0] = phi;
    r[1] = e;

    v[0] = 0;
    v[1] = 1;

    while (r[2] != 1) {
        q = r[0] / r[1];
        r[2] = r[0] % r[1];
        v[2] = v[0] + (phi * phi - q * v[1]);
        v[2] = v[0] + (phi - q) * v[1];
        v[2] %= phi;
        
        v[0] = v[1];
        v[1] = v[2];
        r[0] = r[1];
        r[1] = r[2];
    }
    return v[2];
}


// Returns a^b (mod m)
uint64_t modpow(uint64_t a, uint64_t b, const uint64_t m) {
    a %= m;
    uint64_t result = 1;
    while (b > 0) {
        if (b & 1) {
            result = (result * a) % m;
        }
        a = (a * a) % m;
        b >>= 1;
    }
    return result;
}


int main() {

    std::cout << "Generated parameters: " << std::endl;
    // Generate random numbers p and q
    const uint64_t p = getRandomPrime();
    const uint64_t q = getRandomPrime();

    std::cout << "p = " << p << std::endl;
    std::cout << "q = " << q << std::endl;

    // Calculate n
    const uint64_t n = p * q;
    std::cout << "n = " << n << std::endl;

    // Calculate phi-function
    const uint64_t phi = (p - 1) * (q - 1);
    std::cout << "phi = " << phi << std::endl;

    // Generate random coprime with phi
    const uint64_t e = lookingForE(phi);
    std::cout << "e = " << e << std::endl;

    // Find inverse element for e
    const uint64_t d = inverseE(phi, e);
    std::cout << "d = " << d << std::endl << std::endl;


    std::cout << "Enter message:" << std::endl;
	uint64_t message = 0;
	std::cin >> message;

    std::cout << "Plaintext:" << std::endl;
    std::cout << message << std::endl << std::endl;

    // Encrypt
	uint64_t cipher = modpow(message, e, n);

    std::cout << "Ciphertext:" << std::endl;
    std::cout << cipher << std::endl << std::endl;

    // Decrypt
    uint64_t decrypted = modpow(cipher, d, n);

    std::cout << "Decrypted:" << std::endl;
    std::cout << decrypted << std::endl;

    return 0;
}

