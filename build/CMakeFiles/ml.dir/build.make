# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.22

# Delete rule output on recipe failure.
.DELETE_ON_ERROR:

#=============================================================================
# Special targets provided by cmake.

# Disable implicit rules so canonical targets will work.
.SUFFIXES:

# Disable VCS-based implicit rules.
% : %,v

# Disable VCS-based implicit rules.
% : RCS/%

# Disable VCS-based implicit rules.
% : RCS/%,v

# Disable VCS-based implicit rules.
% : SCCS/s.%

# Disable VCS-based implicit rules.
% : s.%

.SUFFIXES: .hpux_make_needs_suffix_list

# Command-line flag to silence nested $(MAKE).
$(VERBOSE)MAKESILENT = -s

#Suppress display of executed commands.
$(VERBOSE).SILENT:

# A target that is always out of date.
cmake_force:
.PHONY : cmake_force

#=============================================================================
# Set environment variables for the build.

# The shell in which to execute make rules.
SHELL = /bin/sh

# The CMake executable.
CMAKE_COMMAND = /usr/bin/cmake

# The command to remove a file.
RM = /usr/bin/cmake -E rm -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = "/home/george/Desktop/Modular_Neural_Networks (copy)"

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = "/home/george/Desktop/Modular_Neural_Networks (copy)/build"

# Include any dependencies generated for this target.
include CMakeFiles/ml.dir/depend.make
# Include any dependencies generated by the compiler for this target.
include CMakeFiles/ml.dir/compiler_depend.make

# Include the progress variables for this target.
include CMakeFiles/ml.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/ml.dir/flags.make

CMakeFiles/ml.dir/main.cpp.o: CMakeFiles/ml.dir/flags.make
CMakeFiles/ml.dir/main.cpp.o: ../main.cpp
CMakeFiles/ml.dir/main.cpp.o: CMakeFiles/ml.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir="/home/george/Desktop/Modular_Neural_Networks (copy)/build/CMakeFiles" --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object CMakeFiles/ml.dir/main.cpp.o"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT CMakeFiles/ml.dir/main.cpp.o -MF CMakeFiles/ml.dir/main.cpp.o.d -o CMakeFiles/ml.dir/main.cpp.o -c "/home/george/Desktop/Modular_Neural_Networks (copy)/main.cpp"

CMakeFiles/ml.dir/main.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/ml.dir/main.cpp.i"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E "/home/george/Desktop/Modular_Neural_Networks (copy)/main.cpp" > CMakeFiles/ml.dir/main.cpp.i

CMakeFiles/ml.dir/main.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/ml.dir/main.cpp.s"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S "/home/george/Desktop/Modular_Neural_Networks (copy)/main.cpp" -o CMakeFiles/ml.dir/main.cpp.s

CMakeFiles/ml.dir/lib/src/activations.cpp.o: CMakeFiles/ml.dir/flags.make
CMakeFiles/ml.dir/lib/src/activations.cpp.o: ../lib/src/activations.cpp
CMakeFiles/ml.dir/lib/src/activations.cpp.o: CMakeFiles/ml.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir="/home/george/Desktop/Modular_Neural_Networks (copy)/build/CMakeFiles" --progress-num=$(CMAKE_PROGRESS_2) "Building CXX object CMakeFiles/ml.dir/lib/src/activations.cpp.o"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT CMakeFiles/ml.dir/lib/src/activations.cpp.o -MF CMakeFiles/ml.dir/lib/src/activations.cpp.o.d -o CMakeFiles/ml.dir/lib/src/activations.cpp.o -c "/home/george/Desktop/Modular_Neural_Networks (copy)/lib/src/activations.cpp"

CMakeFiles/ml.dir/lib/src/activations.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/ml.dir/lib/src/activations.cpp.i"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E "/home/george/Desktop/Modular_Neural_Networks (copy)/lib/src/activations.cpp" > CMakeFiles/ml.dir/lib/src/activations.cpp.i

CMakeFiles/ml.dir/lib/src/activations.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/ml.dir/lib/src/activations.cpp.s"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S "/home/george/Desktop/Modular_Neural_Networks (copy)/lib/src/activations.cpp" -o CMakeFiles/ml.dir/lib/src/activations.cpp.s

CMakeFiles/ml.dir/lib/src/cost.cpp.o: CMakeFiles/ml.dir/flags.make
CMakeFiles/ml.dir/lib/src/cost.cpp.o: ../lib/src/cost.cpp
CMakeFiles/ml.dir/lib/src/cost.cpp.o: CMakeFiles/ml.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir="/home/george/Desktop/Modular_Neural_Networks (copy)/build/CMakeFiles" --progress-num=$(CMAKE_PROGRESS_3) "Building CXX object CMakeFiles/ml.dir/lib/src/cost.cpp.o"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT CMakeFiles/ml.dir/lib/src/cost.cpp.o -MF CMakeFiles/ml.dir/lib/src/cost.cpp.o.d -o CMakeFiles/ml.dir/lib/src/cost.cpp.o -c "/home/george/Desktop/Modular_Neural_Networks (copy)/lib/src/cost.cpp"

CMakeFiles/ml.dir/lib/src/cost.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/ml.dir/lib/src/cost.cpp.i"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E "/home/george/Desktop/Modular_Neural_Networks (copy)/lib/src/cost.cpp" > CMakeFiles/ml.dir/lib/src/cost.cpp.i

CMakeFiles/ml.dir/lib/src/cost.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/ml.dir/lib/src/cost.cpp.s"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S "/home/george/Desktop/Modular_Neural_Networks (copy)/lib/src/cost.cpp" -o CMakeFiles/ml.dir/lib/src/cost.cpp.s

CMakeFiles/ml.dir/lib/src/dense.cpp.o: CMakeFiles/ml.dir/flags.make
CMakeFiles/ml.dir/lib/src/dense.cpp.o: ../lib/src/dense.cpp
CMakeFiles/ml.dir/lib/src/dense.cpp.o: CMakeFiles/ml.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir="/home/george/Desktop/Modular_Neural_Networks (copy)/build/CMakeFiles" --progress-num=$(CMAKE_PROGRESS_4) "Building CXX object CMakeFiles/ml.dir/lib/src/dense.cpp.o"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT CMakeFiles/ml.dir/lib/src/dense.cpp.o -MF CMakeFiles/ml.dir/lib/src/dense.cpp.o.d -o CMakeFiles/ml.dir/lib/src/dense.cpp.o -c "/home/george/Desktop/Modular_Neural_Networks (copy)/lib/src/dense.cpp"

CMakeFiles/ml.dir/lib/src/dense.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/ml.dir/lib/src/dense.cpp.i"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E "/home/george/Desktop/Modular_Neural_Networks (copy)/lib/src/dense.cpp" > CMakeFiles/ml.dir/lib/src/dense.cpp.i

CMakeFiles/ml.dir/lib/src/dense.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/ml.dir/lib/src/dense.cpp.s"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S "/home/george/Desktop/Modular_Neural_Networks (copy)/lib/src/dense.cpp" -o CMakeFiles/ml.dir/lib/src/dense.cpp.s

CMakeFiles/ml.dir/lib/src/layer.cpp.o: CMakeFiles/ml.dir/flags.make
CMakeFiles/ml.dir/lib/src/layer.cpp.o: ../lib/src/layer.cpp
CMakeFiles/ml.dir/lib/src/layer.cpp.o: CMakeFiles/ml.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir="/home/george/Desktop/Modular_Neural_Networks (copy)/build/CMakeFiles" --progress-num=$(CMAKE_PROGRESS_5) "Building CXX object CMakeFiles/ml.dir/lib/src/layer.cpp.o"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT CMakeFiles/ml.dir/lib/src/layer.cpp.o -MF CMakeFiles/ml.dir/lib/src/layer.cpp.o.d -o CMakeFiles/ml.dir/lib/src/layer.cpp.o -c "/home/george/Desktop/Modular_Neural_Networks (copy)/lib/src/layer.cpp"

CMakeFiles/ml.dir/lib/src/layer.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/ml.dir/lib/src/layer.cpp.i"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E "/home/george/Desktop/Modular_Neural_Networks (copy)/lib/src/layer.cpp" > CMakeFiles/ml.dir/lib/src/layer.cpp.i

CMakeFiles/ml.dir/lib/src/layer.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/ml.dir/lib/src/layer.cpp.s"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S "/home/george/Desktop/Modular_Neural_Networks (copy)/lib/src/layer.cpp" -o CMakeFiles/ml.dir/lib/src/layer.cpp.s

CMakeFiles/ml.dir/lib/src/matrixOperations.cpp.o: CMakeFiles/ml.dir/flags.make
CMakeFiles/ml.dir/lib/src/matrixOperations.cpp.o: ../lib/src/matrixOperations.cpp
CMakeFiles/ml.dir/lib/src/matrixOperations.cpp.o: CMakeFiles/ml.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir="/home/george/Desktop/Modular_Neural_Networks (copy)/build/CMakeFiles" --progress-num=$(CMAKE_PROGRESS_6) "Building CXX object CMakeFiles/ml.dir/lib/src/matrixOperations.cpp.o"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT CMakeFiles/ml.dir/lib/src/matrixOperations.cpp.o -MF CMakeFiles/ml.dir/lib/src/matrixOperations.cpp.o.d -o CMakeFiles/ml.dir/lib/src/matrixOperations.cpp.o -c "/home/george/Desktop/Modular_Neural_Networks (copy)/lib/src/matrixOperations.cpp"

CMakeFiles/ml.dir/lib/src/matrixOperations.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/ml.dir/lib/src/matrixOperations.cpp.i"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E "/home/george/Desktop/Modular_Neural_Networks (copy)/lib/src/matrixOperations.cpp" > CMakeFiles/ml.dir/lib/src/matrixOperations.cpp.i

CMakeFiles/ml.dir/lib/src/matrixOperations.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/ml.dir/lib/src/matrixOperations.cpp.s"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S "/home/george/Desktop/Modular_Neural_Networks (copy)/lib/src/matrixOperations.cpp" -o CMakeFiles/ml.dir/lib/src/matrixOperations.cpp.s

CMakeFiles/ml.dir/lib/src/mnist.cpp.o: CMakeFiles/ml.dir/flags.make
CMakeFiles/ml.dir/lib/src/mnist.cpp.o: ../lib/src/mnist.cpp
CMakeFiles/ml.dir/lib/src/mnist.cpp.o: CMakeFiles/ml.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir="/home/george/Desktop/Modular_Neural_Networks (copy)/build/CMakeFiles" --progress-num=$(CMAKE_PROGRESS_7) "Building CXX object CMakeFiles/ml.dir/lib/src/mnist.cpp.o"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT CMakeFiles/ml.dir/lib/src/mnist.cpp.o -MF CMakeFiles/ml.dir/lib/src/mnist.cpp.o.d -o CMakeFiles/ml.dir/lib/src/mnist.cpp.o -c "/home/george/Desktop/Modular_Neural_Networks (copy)/lib/src/mnist.cpp"

CMakeFiles/ml.dir/lib/src/mnist.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/ml.dir/lib/src/mnist.cpp.i"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E "/home/george/Desktop/Modular_Neural_Networks (copy)/lib/src/mnist.cpp" > CMakeFiles/ml.dir/lib/src/mnist.cpp.i

CMakeFiles/ml.dir/lib/src/mnist.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/ml.dir/lib/src/mnist.cpp.s"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S "/home/george/Desktop/Modular_Neural_Networks (copy)/lib/src/mnist.cpp" -o CMakeFiles/ml.dir/lib/src/mnist.cpp.s

CMakeFiles/ml.dir/lib/src/network.cpp.o: CMakeFiles/ml.dir/flags.make
CMakeFiles/ml.dir/lib/src/network.cpp.o: ../lib/src/network.cpp
CMakeFiles/ml.dir/lib/src/network.cpp.o: CMakeFiles/ml.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir="/home/george/Desktop/Modular_Neural_Networks (copy)/build/CMakeFiles" --progress-num=$(CMAKE_PROGRESS_8) "Building CXX object CMakeFiles/ml.dir/lib/src/network.cpp.o"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT CMakeFiles/ml.dir/lib/src/network.cpp.o -MF CMakeFiles/ml.dir/lib/src/network.cpp.o.d -o CMakeFiles/ml.dir/lib/src/network.cpp.o -c "/home/george/Desktop/Modular_Neural_Networks (copy)/lib/src/network.cpp"

CMakeFiles/ml.dir/lib/src/network.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/ml.dir/lib/src/network.cpp.i"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E "/home/george/Desktop/Modular_Neural_Networks (copy)/lib/src/network.cpp" > CMakeFiles/ml.dir/lib/src/network.cpp.i

CMakeFiles/ml.dir/lib/src/network.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/ml.dir/lib/src/network.cpp.s"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S "/home/george/Desktop/Modular_Neural_Networks (copy)/lib/src/network.cpp" -o CMakeFiles/ml.dir/lib/src/network.cpp.s

# Object files for target ml
ml_OBJECTS = \
"CMakeFiles/ml.dir/main.cpp.o" \
"CMakeFiles/ml.dir/lib/src/activations.cpp.o" \
"CMakeFiles/ml.dir/lib/src/cost.cpp.o" \
"CMakeFiles/ml.dir/lib/src/dense.cpp.o" \
"CMakeFiles/ml.dir/lib/src/layer.cpp.o" \
"CMakeFiles/ml.dir/lib/src/matrixOperations.cpp.o" \
"CMakeFiles/ml.dir/lib/src/mnist.cpp.o" \
"CMakeFiles/ml.dir/lib/src/network.cpp.o"

# External object files for target ml
ml_EXTERNAL_OBJECTS =

ml: CMakeFiles/ml.dir/main.cpp.o
ml: CMakeFiles/ml.dir/lib/src/activations.cpp.o
ml: CMakeFiles/ml.dir/lib/src/cost.cpp.o
ml: CMakeFiles/ml.dir/lib/src/dense.cpp.o
ml: CMakeFiles/ml.dir/lib/src/layer.cpp.o
ml: CMakeFiles/ml.dir/lib/src/matrixOperations.cpp.o
ml: CMakeFiles/ml.dir/lib/src/mnist.cpp.o
ml: CMakeFiles/ml.dir/lib/src/network.cpp.o
ml: CMakeFiles/ml.dir/build.make
ml: /usr/lib/gcc/x86_64-linux-gnu/11/libgomp.so
ml: /usr/lib/x86_64-linux-gnu/libpthread.a
ml: CMakeFiles/ml.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir="/home/george/Desktop/Modular_Neural_Networks (copy)/build/CMakeFiles" --progress-num=$(CMAKE_PROGRESS_9) "Linking CXX executable ml"
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/ml.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/ml.dir/build: ml
.PHONY : CMakeFiles/ml.dir/build

CMakeFiles/ml.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/ml.dir/cmake_clean.cmake
.PHONY : CMakeFiles/ml.dir/clean

CMakeFiles/ml.dir/depend:
	cd "/home/george/Desktop/Modular_Neural_Networks (copy)/build" && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" "/home/george/Desktop/Modular_Neural_Networks (copy)" "/home/george/Desktop/Modular_Neural_Networks (copy)" "/home/george/Desktop/Modular_Neural_Networks (copy)/build" "/home/george/Desktop/Modular_Neural_Networks (copy)/build" "/home/george/Desktop/Modular_Neural_Networks (copy)/build/CMakeFiles/ml.dir/DependInfo.cmake" --color=$(COLOR)
.PHONY : CMakeFiles/ml.dir/depend
