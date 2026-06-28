# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
# Compiler and language standard
CXX      := g++
CXXSTD   := -std=c++20

# Preprocessor macros, warnings, and optimisation flags
CXXFLAGS := $(CXXSTD) -DNDEBUG -fcoroutines -I./ -g -march=native \
            -Wall -Wextra

# Libraries to link against
LIBS     := -lzmq -lpthread

# Final executable name
TARGET   := ox

# Source files (add more .cpp files here as needed)
SRCS     := ox.cpp

# Derive object file names automatically
OBJS     := $(SRCS:.cpp=.o)

# Default target
all: $(TARGET)

# Link object files into the executable
# After linking, remove .o files
$(TARGET): $(OBJS)
	$(CXX) $(CXXFLAGS) $^ -o $@ $(LIBS)
	@echo "Removing object files..."
	@rm -f $(OBJS)

# Pattern rule to compile .cpp → .o
%.o: %.cpp
	$(CXX) $(CXXFLAGS) -c $< -o $@

# Remove build artefacts
clean:
	rm -f $(OBJS) $(TARGET)

# Clean and rebuild everything from scratch
rebuild: clean all

# Declare phony targets
.PHONY: all clean rebuild
