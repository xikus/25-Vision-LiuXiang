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
CMAKE_SOURCE_DIR = /home/yoda/25-Vision-LiuXiang/CMakeSkeletonBuilding/CMake_III

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /home/yoda/25-Vision-LiuXiang/CMakeSkeletonBuilding/CMake_III/build

# Include any dependencies generated for this target.
include modules/ml/CMakeFiles/ml.dir/depend.make
# Include any dependencies generated by the compiler for this target.
include modules/ml/CMakeFiles/ml.dir/compiler_depend.make

# Include the progress variables for this target.
include modules/ml/CMakeFiles/ml.dir/progress.make

# Include the compile flags for this target's objects.
include modules/ml/CMakeFiles/ml.dir/flags.make

modules/ml/CMakeFiles/ml.dir/src/ort/ort.cpp.o: modules/ml/CMakeFiles/ml.dir/flags.make
modules/ml/CMakeFiles/ml.dir/src/ort/ort.cpp.o: ../modules/ml/src/ort/ort.cpp
modules/ml/CMakeFiles/ml.dir/src/ort/ort.cpp.o: modules/ml/CMakeFiles/ml.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/yoda/25-Vision-LiuXiang/CMakeSkeletonBuilding/CMake_III/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object modules/ml/CMakeFiles/ml.dir/src/ort/ort.cpp.o"
	cd /home/yoda/25-Vision-LiuXiang/CMakeSkeletonBuilding/CMake_III/build/modules/ml && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT modules/ml/CMakeFiles/ml.dir/src/ort/ort.cpp.o -MF CMakeFiles/ml.dir/src/ort/ort.cpp.o.d -o CMakeFiles/ml.dir/src/ort/ort.cpp.o -c /home/yoda/25-Vision-LiuXiang/CMakeSkeletonBuilding/CMake_III/modules/ml/src/ort/ort.cpp

modules/ml/CMakeFiles/ml.dir/src/ort/ort.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/ml.dir/src/ort/ort.cpp.i"
	cd /home/yoda/25-Vision-LiuXiang/CMakeSkeletonBuilding/CMake_III/build/modules/ml && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/yoda/25-Vision-LiuXiang/CMakeSkeletonBuilding/CMake_III/modules/ml/src/ort/ort.cpp > CMakeFiles/ml.dir/src/ort/ort.cpp.i

modules/ml/CMakeFiles/ml.dir/src/ort/ort.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/ml.dir/src/ort/ort.cpp.s"
	cd /home/yoda/25-Vision-LiuXiang/CMakeSkeletonBuilding/CMake_III/build/modules/ml && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/yoda/25-Vision-LiuXiang/CMakeSkeletonBuilding/CMake_III/modules/ml/src/ort/ort.cpp -o CMakeFiles/ml.dir/src/ort/ort.cpp.s

modules/ml/CMakeFiles/ml.dir/src/ort/ort_print.cpp.o: modules/ml/CMakeFiles/ml.dir/flags.make
modules/ml/CMakeFiles/ml.dir/src/ort/ort_print.cpp.o: ../modules/ml/src/ort/ort_print.cpp
modules/ml/CMakeFiles/ml.dir/src/ort/ort_print.cpp.o: modules/ml/CMakeFiles/ml.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/yoda/25-Vision-LiuXiang/CMakeSkeletonBuilding/CMake_III/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Building CXX object modules/ml/CMakeFiles/ml.dir/src/ort/ort_print.cpp.o"
	cd /home/yoda/25-Vision-LiuXiang/CMakeSkeletonBuilding/CMake_III/build/modules/ml && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT modules/ml/CMakeFiles/ml.dir/src/ort/ort_print.cpp.o -MF CMakeFiles/ml.dir/src/ort/ort_print.cpp.o.d -o CMakeFiles/ml.dir/src/ort/ort_print.cpp.o -c /home/yoda/25-Vision-LiuXiang/CMakeSkeletonBuilding/CMake_III/modules/ml/src/ort/ort_print.cpp

modules/ml/CMakeFiles/ml.dir/src/ort/ort_print.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/ml.dir/src/ort/ort_print.cpp.i"
	cd /home/yoda/25-Vision-LiuXiang/CMakeSkeletonBuilding/CMake_III/build/modules/ml && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/yoda/25-Vision-LiuXiang/CMakeSkeletonBuilding/CMake_III/modules/ml/src/ort/ort_print.cpp > CMakeFiles/ml.dir/src/ort/ort_print.cpp.i

modules/ml/CMakeFiles/ml.dir/src/ort/ort_print.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/ml.dir/src/ort/ort_print.cpp.s"
	cd /home/yoda/25-Vision-LiuXiang/CMakeSkeletonBuilding/CMake_III/build/modules/ml && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/yoda/25-Vision-LiuXiang/CMakeSkeletonBuilding/CMake_III/modules/ml/src/ort/ort_print.cpp -o CMakeFiles/ml.dir/src/ort/ort_print.cpp.s

# Object files for target ml
ml_OBJECTS = \
"CMakeFiles/ml.dir/src/ort/ort.cpp.o" \
"CMakeFiles/ml.dir/src/ort/ort_print.cpp.o"

# External object files for target ml
ml_EXTERNAL_OBJECTS =

modules/ml/libml.a: modules/ml/CMakeFiles/ml.dir/src/ort/ort.cpp.o
modules/ml/libml.a: modules/ml/CMakeFiles/ml.dir/src/ort/ort_print.cpp.o
modules/ml/libml.a: modules/ml/CMakeFiles/ml.dir/build.make
modules/ml/libml.a: modules/ml/CMakeFiles/ml.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/home/yoda/25-Vision-LiuXiang/CMakeSkeletonBuilding/CMake_III/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_3) "Linking CXX static library libml.a"
	cd /home/yoda/25-Vision-LiuXiang/CMakeSkeletonBuilding/CMake_III/build/modules/ml && $(CMAKE_COMMAND) -P CMakeFiles/ml.dir/cmake_clean_target.cmake
	cd /home/yoda/25-Vision-LiuXiang/CMakeSkeletonBuilding/CMake_III/build/modules/ml && $(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/ml.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
modules/ml/CMakeFiles/ml.dir/build: modules/ml/libml.a
.PHONY : modules/ml/CMakeFiles/ml.dir/build

modules/ml/CMakeFiles/ml.dir/clean:
	cd /home/yoda/25-Vision-LiuXiang/CMakeSkeletonBuilding/CMake_III/build/modules/ml && $(CMAKE_COMMAND) -P CMakeFiles/ml.dir/cmake_clean.cmake
.PHONY : modules/ml/CMakeFiles/ml.dir/clean

modules/ml/CMakeFiles/ml.dir/depend:
	cd /home/yoda/25-Vision-LiuXiang/CMakeSkeletonBuilding/CMake_III/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/yoda/25-Vision-LiuXiang/CMakeSkeletonBuilding/CMake_III /home/yoda/25-Vision-LiuXiang/CMakeSkeletonBuilding/CMake_III/modules/ml /home/yoda/25-Vision-LiuXiang/CMakeSkeletonBuilding/CMake_III/build /home/yoda/25-Vision-LiuXiang/CMakeSkeletonBuilding/CMake_III/build/modules/ml /home/yoda/25-Vision-LiuXiang/CMakeSkeletonBuilding/CMake_III/build/modules/ml/CMakeFiles/ml.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : modules/ml/CMakeFiles/ml.dir/depend

