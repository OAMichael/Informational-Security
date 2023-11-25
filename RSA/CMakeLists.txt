set(SOURCES
    RSACipher.cpp
    RSA.cpp
)

add_executable(RSA ${SOURCES})
target_link_libraries(RSA -lgmp)
