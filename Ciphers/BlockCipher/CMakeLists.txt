set(SOURCES
    BlockCipher.cpp
    Feistel.cpp
)

add_executable(BlockCipher ${SOURCES})
target_link_libraries(BlockCipher -fopenmp)
target_compile_options(BlockCipher PRIVATE -fopenmp)

target_sources(Benchmark PRIVATE Feistel.cpp)
