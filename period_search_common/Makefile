# Define compiler and flags
CXX = g++
CXXFLAGS = -Wall -Wextra -O2

# Define the target library
TARGET = build/libps_commonc.a

# Define the source files
SRC = arrayHelpers.cpp dot_product.cpp LcHelpers.cpp

# Define the object files
OBJ = $(SRC:%.cpp=build/%.o)

# Default target to build the library
all: $(TARGET)

# Rule to build the static library
$(TARGET): $(OBJ)
	ar rcs $@ $^

# Rule to build object files
build/%.o: %.cpp
	@mkdir -p build
	$(CXX) $(CXXFLAGS) -c $< -o $@

# Clean rule to remove compiled files
clean:
	rm -f $(OBJ) $(TARGET)
	rm -rf build

.PHONY: all clean
