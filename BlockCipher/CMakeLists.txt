set(SOURCES
    BlockCipher.cpp
    Feistel.cpp
)

add_executable(BlockCipher ${SOURCES})
target_sources(Benchmark PRIVATE Feistel.cpp)
