# Makefile

# Compiler and flags
CC = clang
CFLAGS = -march=native -mavx512f -fopenmp

# Source and target
SRC = avx512_16.c
TARGET = avx512_16

.PHONY: all clean

# Default rule
all: $(TARGET)

# Build target
$(TARGET): $(SRC)
	$(CC) $(CFLAGS) $< -o $@

# Clean up
clean:
	rm -f $(TARGET)
