CC = gcc
CFLAGS = -Wall -Wextra -std=c11 -Iinclude -O2
TARGET = bin/test
LDFLAGS =
SRCS = src/main.c src/matrix.c

all: $(TARGET)

$(TARGET): $(SRCS)
	mkdir -p bin
	$(CC) $(CFLAGS) $^ -o $@ $(LDFLAGS)

run: $(TARGET)
	./$(TARGET)

clean:
	rm -rf bin