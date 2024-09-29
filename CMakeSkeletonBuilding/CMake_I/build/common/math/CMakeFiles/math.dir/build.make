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
CMAKE_SOURCE_DIR = /home/yoda/25-Vision-LiuXiang/CMakeSkeletonBuilding/CMake_I

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /home/yoda/25-Vision-LiuXiang/CMakeSkeletonBuilding/CMake_I/build

# Include any dependencies generated for this target.
include common/math/CMakeFiles/math.dir/depend.make
# Include any dependencies generated by the compiler for this target.
include common/math/CMakeFiles/math.dir/compiler_depend.make

# Include the progress variables for this target.
include common/math/CMakeFiles/math.dir/progress.make

# Include the compile flags for this target's objects.
include common/math/CMakeFiles/math.dir/flags.make

common/math/CMakeFiles/math.dir/src/Math.cpp.o: common/math/CMakeFiles/math.dir/flags.make
common/math/CMakeFiles/math.dir/src/Math.cpp.o: ../common/math/src/Math.cpp
common/math/CMakeFiles/math.dir/src/Math.cpp.o: common/math/CMakeFiles/math.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/yoda/25-Vision-LiuXiang/CMakeSkeletonBuilding/CMake_I/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object common/math/CMakeFiles/math.dir/src/Math.cpp.o"
	cd /home/yoda/25-Vision-LiuXiang/CMakeSkeletonBuilding/CMake_I/build/common/math && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT common/math/CMakeFiles/math.dir/src/Math.cpp.o -MF CMakeFiles/math.dir/src/Math.cpp.o.d -o CMakeFiles/math.dir/src/Math.cpp.o -c /home/yoda/25-Vision-LiuXiang/CMakeSkeletonBuilding/CMake_I/common/math/src/Math.cpp

common/math/CMakeFiles/math.dir/src/Math.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/math.dir/src/Math.cpp.i"
	cd /home/yoda/25-Vision-LiuXiang/CMakeSkeletonBuilding/CMake_I/build/common/math && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/yoda/25-Vision-LiuXiang/CMakeSkeletonBuilding/CMake_I/common/math/src/Math.cpp > CMakeFiles/math.dir/src/Math.cpp.i

common/math/CMakeFiles/math.dir/src/Math.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/math.dir/src/Math.cpp.s"
	cd /home/yoda/25-Vision-LiuXiang/CMakeSkeletonBuilding/CMake_I/build/common/math && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/yoda/25-Vision-LiuXiang/CMakeSkeletonBuilding/CMake_I/common/math/src/Math.cpp -o CMakeFiles/math.dir/src/Math.cpp.s

# Object files for target math
math_OBJECTS = \
"CMakeFiles/math.dir/src/Math.cpp.o"

# External object files for target math
math_EXTERNAL_OBJECTS =

common/math/libmath.a: common/math/CMakeFiles/math.dir/src/Math.cpp.o
common/math/libmath.a: common/math/CMakeFiles/math.dir/build.make
common/math/libmath.a: common/math/CMakeFiles/math.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/home/yoda/25-Vision-LiuXiang/CMakeSkeletonBuilding/CMake_I/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Linking CXX static library libmath.a"
	cd /home/yoda/25-Vision-LiuXiang/CMakeSkeletonBuilding/CMake_I/build/common/math && $(CMAKE_COMMAND) -P CMakeFiles/math.dir/cmake_clean_target.cmake
	cd /home/yoda/25-Vision-LiuXiang/CMakeSkeletonBuilding/CMake_I/build/common/math && $(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/math.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
common/math/CMakeFiles/math.dir/build: common/math/libmath.a
.PHONY : common/math/CMakeFiles/math.dir/build

common/math/CMakeFiles/math.dir/clean:
	cd /home/yoda/25-Vision-LiuXiang/CMakeSkeletonBuilding/CMake_I/build/common/math && $(CMAKE_COMMAND) -P CMakeFiles/math.dir/cmake_clean.cmake
.PHONY : common/math/CMakeFiles/math.dir/clean

common/math/CMakeFiles/math.dir/depend:
	cd /home/yoda/25-Vision-LiuXiang/CMakeSkeletonBuilding/CMake_I/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/yoda/25-Vision-LiuXiang/CMakeSkeletonBuilding/CMake_I /home/yoda/25-Vision-LiuXiang/CMakeSkeletonBuilding/CMake_I/common/math /home/yoda/25-Vision-LiuXiang/CMakeSkeletonBuilding/CMake_I/build /home/yoda/25-Vision-LiuXiang/CMakeSkeletonBuilding/CMake_I/build/common/math /home/yoda/25-Vision-LiuXiang/CMakeSkeletonBuilding/CMake_I/build/common/math/CMakeFiles/math.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : common/math/CMakeFiles/math.dir/depend

