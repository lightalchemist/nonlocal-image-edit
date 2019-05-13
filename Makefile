UNAME_S:=$(shell uname -s)
ifeq ($(UNAME_S),Linux)
	CFLAGS=-I/usr/local/include -I/usr/include/eigen3 -I/usr/local/include/opencv4 -Wall -Wextra -std=c++1z
else ifeq ($(UNAME_S),Darwin)
	CFLAGS=-I/usr/local/include -I/usr/local/include/eigen3 -I/usr/local/include/opencv4 -Wall -Wextra -std=c++1z
else
	echo "Unknown platform"
endif

# ifeq ($(BUILD), debug)
CFLAGS += -ggdb3 -O0
# else
	# CFLAGS += -O3 -DNDEBUG
# endif

LDFLAGS = -L/usr/local/lib -lopencv_core -lopencv_highgui -lopencv_imgproc -lopencv_imgcodecs

CC = clang++
EXE = "image_filter"

all: main.cpp | output_directory
	$(CC) $(CFLAGS) $(LDFLAGS) main.cpp -o "bin/$(EXE)"

output_directory:
	@mkdir -p bin

clean:
	rm -rf bin/
