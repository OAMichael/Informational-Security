# Informational Security project

### Our goal is to implement encryption algorithm called Elephant. Preferably using GPU parallelism

### How to build & run

#### First of all, clone repository:

```
git clone https://github.com/OAMichael/Informational-Security.git
```

#### Then 
```
cd Informational-Security
cmake -B build -DCMAKE_BUILD_TYPE=Release && cd build
cmake --build .
```

#### Now you can go to binary and execute it. For example, for BlockCipher:
```
cd BlockCipher
./BlockCipher
```
