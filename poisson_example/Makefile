CXX ?= g++
MKDIR=mkdir -p
RM=rm -f

BIN_DIR=../bin

APP_NAME=poisson_example

EXE=$(BIN_DIR)/$(APP_NAME)

HALIDE_LIB=$(HALIDE_DIR)/bin/libHalide.a

CPP := $(wildcard *.cpp)

OBJ := _build/$(addprefix ,$(notdir $(CPP:.cpp=.o)))

all: $(OBJ) $(EXE)

_build/%.o: %.cpp
	@$(MKDIR) _build
	$(CXX) -c $< -o $@ -std=c++11 -I$(HALIDE_DIR)/include/ -I$(HALIDE_DIR)/apps/support/ -I../ # we can include stuff from up one level, by default

$(EXE): $(OBJ) $(HALIDE_LIB)
	@$(MKDIR) $(BIN_DIR)
	$(CXX) $(OBJ) $(HALIDE_LIB) -o $(EXE) -lpthread -lpng -ldl -g -Wall

clean:
	$(RM) $(OBJ) $(EXE)
