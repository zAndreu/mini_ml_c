CC = gcc

CFLAGS = -Wall -Wextra -Wpedantic -std=c11 -Iinclude -O3 -march=native -ffast-math -fopenmp
LDFLAGS = -fopenmp -lm

# Flags para depuracion
DEBUG_FLAGS = -Wall -Wextra -Wpedantic -std=c11 -Iinclude -g -O0 -fsanitize=address,undefined
DLDFLAGS = -fsanitize=address,undefined -lm

TARGET = bin/test
SRCS = src/main.c src/matrix.c src/mini_ml.c
HEADERS = include/matrix.h include/mini_ml.h

.PHONY: all run clean

all: $(TARGET)

$(TARGET): $(SRCS) $(HEADERS)
	mkdir -p bin
	$(CC) $(CFLAGS) $(SRCS) -o $@ $(LDFLAGS)

run: $(TARGET)
	./$(TARGET)

clean:
	rm -rf bin