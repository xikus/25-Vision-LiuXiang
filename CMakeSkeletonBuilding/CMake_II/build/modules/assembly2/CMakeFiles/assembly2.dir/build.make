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
CMAKE_SOURCE_DIR = /home/yoda/25-Vision-LiuXiang/CMakeSkeletonBuilding/CMake_II

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /home/yoda/25-Vision-LiuXiang/CMakeSkeletonBuilding/CMake_II/build

# Include any dependencies generated for this target.
include modules/assembly2/CMakeFiles/assembly2.dir/depend.make
# Include any dependencies generated by the compiler for this target.
include modules/assembly2/CMakeFiles/assembly2.dir/compiler_depend.make

# Include the progress variables for this target.
include modules/assembly2/CMakeFiles/assembly2.dir/progress.make

# Include the compile flags for this target's objects.
include modules/assembly2/CMakeFiles/assembly2.dir/flags.make

modules/assembly2/CMakeFiles/assembly2.dir/src/assembly2.cpp.o: modules/assembly2/CMakeFiles/assembly2.dir/flags.make
modules/assembly2/CMakeFiles/assembly2.dir/src/assembly2.cpp.o: ../modules/assembly2/src/assembly2.cpp
modules/assembly2/CMakeFiles/assembly2.dir/src/assembly2.cpp.o: modules/assembly2/CMakeFiles/assembly2.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/yoda/25-Vision-LiuXiang/CMakeSkeletonBuilding/CMake_II/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object modules/assembly2/CMakeFiles/assembly2.dir/src/assembly2.cpp.o"
	cd /home/yoda/25-Vision-LiuXiang/CMakeSkeletonBuilding/CMake_II/build/modules/assembly2 && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT modules/assembly2/CMakeFiles/assembly2.dir/src/assembly2.cpp.o -MF CMakeFiles/assembly2.dir/src/assembly2.cpp.o.d -o CMakeFiles/assembly2.dir/src/assembly2.cpp.o -c /home/yoda/25-Vision-LiuXiang/CMakeSkeletonBuilding/CMake_II/modules/assembly2/src/assembly2.cpp

modules/assembly2/CMakeFiles/assembly2.dir/src/assembly2.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/assembly2.dir/src/assembly2.cpp.i"
	cd /home/yoda/25-Vision-LiuXiang/CMakeSkeletonBuilding/CMake_II/build/modules/assembly2 && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/yoda/25-Vision-LiuXiang/CMakeSkeletonBuilding/CMake_II/modules/assembly2/src/assembly2.cpp > CMakeFiles/assembly2.dir/src/assembly2.cpp.i

modules/assembly2/CMakeFiles/assembly2.dir/src/assembly2.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/assembly2.dir/src/assembly2.cpp.s"
	cd /home/yoda/25-Vision-LiuXiang/CMakeSkeletonBuilding/CMake_II/build/modules/assembly2 && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/yoda/25-Vision-LiuXiang/CMakeSkeletonBuilding/CMake_II/modules/assembly2/src/assembly2.cpp -o CMakeFiles/assembly2.dir/src/assembly2.cpp.s

# Object files for target assembly2
assembly2_OBJECTS = \
"CMakeFiles/assembly2.dir/src/assembly2.cpp.o"

# External object files for target assembly2
assembly2_EXTERNAL_OBJECTS =

modules/assembly2/libassembly2.a: modules/assembly2/CMakeFiles/assembly2.dir/src/assembly2.cpp.o
modules/assembly2/libassembly2.a: modules/assembly2/CMakeFiles/assembly2.dir/build.make
modules/assembly2/libassembly2.a: modules/assembly2/CMakeFiles/assembly2.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/home/yoda/25-Vision-LiuXiang/CMakeSkeletonBuilding/CMake_II/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Linking CXX static library libassembly2.a"
	cd /home/yoda/25-Vision-LiuXiang/CMakeSkeletonBuilding/CMake_II/build/modules/assembly2 && $(CMAKE_COMMAND) -P CMakeFiles/assembly2.dir/cmake_clean_target.cmake
	cd /home/yoda/25-Vision-LiuXiang/CMakeSkeletonBuilding/CMake_II/build/modules/assembly2 && $(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/assembly2.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
modules/assembly2/CMakeFiles/assembly2.dir/build: modules/assembly2/libassembly2.a
.PHONY : modules/assembly2/CMakeFiles/assembly2.dir/build

modules/assembly2/CMakeFiles/assembly2.dir/clean:
	cd /home/yoda/25-Vision-LiuXiang/CMakeSkeletonBuilding/CMake_II/build/modules/assembly2 && $(CMAKE_COMMAND) -P CMakeFiles/assembly2.dir/cmake_clean.cmake
.PHONY : modules/assembly2/CMakeFiles/assembly2.dir/clean

modules/assembly2/CMakeFiles/assembly2.dir/depend:
	cd /home/yoda/25-Vision-LiuXiang/CMakeSkeletonBuilding/CMake_II/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/yoda/25-Vision-LiuXiang/CMakeSkeletonBuilding/CMake_II /home/yoda/25-Vision-LiuXiang/CMakeSkeletonBuilding/CMake_II/modules/assembly2 /home/yoda/25-Vision-LiuXiang/CMakeSkeletonBuilding/CMake_II/build /home/yoda/25-Vision-LiuXiang/CMakeSkeletonBuilding/CMake_II/build/modules/assembly2 /home/yoda/25-Vision-LiuXiang/CMakeSkeletonBuilding/CMake_II/build/modules/assembly2/CMakeFiles/assembly2.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : modules/assembly2/CMakeFiles/assembly2.dir/depend

