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
CMAKE_SOURCE_DIR = /home/yoda/25-Vision-LiuXiang/ArmorDetectingAndCal/detect

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /home/yoda/25-Vision-LiuXiang/ArmorDetectingAndCal/detect/build

# Include any dependencies generated for this target.
include LightDescriptor/CMakeFiles/LightDescriptor.dir/depend.make
# Include any dependencies generated by the compiler for this target.
include LightDescriptor/CMakeFiles/LightDescriptor.dir/compiler_depend.make

# Include the progress variables for this target.
include LightDescriptor/CMakeFiles/LightDescriptor.dir/progress.make

# Include the compile flags for this target's objects.
include LightDescriptor/CMakeFiles/LightDescriptor.dir/flags.make

LightDescriptor/CMakeFiles/LightDescriptor.dir/src/LightDescriptor.cpp.o: LightDescriptor/CMakeFiles/LightDescriptor.dir/flags.make
LightDescriptor/CMakeFiles/LightDescriptor.dir/src/LightDescriptor.cpp.o: ../LightDescriptor/src/LightDescriptor.cpp
LightDescriptor/CMakeFiles/LightDescriptor.dir/src/LightDescriptor.cpp.o: LightDescriptor/CMakeFiles/LightDescriptor.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/yoda/25-Vision-LiuXiang/ArmorDetectingAndCal/detect/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object LightDescriptor/CMakeFiles/LightDescriptor.dir/src/LightDescriptor.cpp.o"
	cd /home/yoda/25-Vision-LiuXiang/ArmorDetectingAndCal/detect/build/LightDescriptor && g++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT LightDescriptor/CMakeFiles/LightDescriptor.dir/src/LightDescriptor.cpp.o -MF CMakeFiles/LightDescriptor.dir/src/LightDescriptor.cpp.o.d -o CMakeFiles/LightDescriptor.dir/src/LightDescriptor.cpp.o -c /home/yoda/25-Vision-LiuXiang/ArmorDetectingAndCal/detect/LightDescriptor/src/LightDescriptor.cpp

LightDescriptor/CMakeFiles/LightDescriptor.dir/src/LightDescriptor.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/LightDescriptor.dir/src/LightDescriptor.cpp.i"
	cd /home/yoda/25-Vision-LiuXiang/ArmorDetectingAndCal/detect/build/LightDescriptor && g++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/yoda/25-Vision-LiuXiang/ArmorDetectingAndCal/detect/LightDescriptor/src/LightDescriptor.cpp > CMakeFiles/LightDescriptor.dir/src/LightDescriptor.cpp.i

LightDescriptor/CMakeFiles/LightDescriptor.dir/src/LightDescriptor.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/LightDescriptor.dir/src/LightDescriptor.cpp.s"
	cd /home/yoda/25-Vision-LiuXiang/ArmorDetectingAndCal/detect/build/LightDescriptor && g++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/yoda/25-Vision-LiuXiang/ArmorDetectingAndCal/detect/LightDescriptor/src/LightDescriptor.cpp -o CMakeFiles/LightDescriptor.dir/src/LightDescriptor.cpp.s

# Object files for target LightDescriptor
LightDescriptor_OBJECTS = \
"CMakeFiles/LightDescriptor.dir/src/LightDescriptor.cpp.o"

# External object files for target LightDescriptor
LightDescriptor_EXTERNAL_OBJECTS =

LightDescriptor/libLightDescriptor.a: LightDescriptor/CMakeFiles/LightDescriptor.dir/src/LightDescriptor.cpp.o
LightDescriptor/libLightDescriptor.a: LightDescriptor/CMakeFiles/LightDescriptor.dir/build.make
LightDescriptor/libLightDescriptor.a: LightDescriptor/CMakeFiles/LightDescriptor.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/home/yoda/25-Vision-LiuXiang/ArmorDetectingAndCal/detect/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Linking CXX static library libLightDescriptor.a"
	cd /home/yoda/25-Vision-LiuXiang/ArmorDetectingAndCal/detect/build/LightDescriptor && $(CMAKE_COMMAND) -P CMakeFiles/LightDescriptor.dir/cmake_clean_target.cmake
	cd /home/yoda/25-Vision-LiuXiang/ArmorDetectingAndCal/detect/build/LightDescriptor && $(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/LightDescriptor.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
LightDescriptor/CMakeFiles/LightDescriptor.dir/build: LightDescriptor/libLightDescriptor.a
.PHONY : LightDescriptor/CMakeFiles/LightDescriptor.dir/build

LightDescriptor/CMakeFiles/LightDescriptor.dir/clean:
	cd /home/yoda/25-Vision-LiuXiang/ArmorDetectingAndCal/detect/build/LightDescriptor && $(CMAKE_COMMAND) -P CMakeFiles/LightDescriptor.dir/cmake_clean.cmake
.PHONY : LightDescriptor/CMakeFiles/LightDescriptor.dir/clean

LightDescriptor/CMakeFiles/LightDescriptor.dir/depend:
	cd /home/yoda/25-Vision-LiuXiang/ArmorDetectingAndCal/detect/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/yoda/25-Vision-LiuXiang/ArmorDetectingAndCal/detect /home/yoda/25-Vision-LiuXiang/ArmorDetectingAndCal/detect/LightDescriptor /home/yoda/25-Vision-LiuXiang/ArmorDetectingAndCal/detect/build /home/yoda/25-Vision-LiuXiang/ArmorDetectingAndCal/detect/build/LightDescriptor /home/yoda/25-Vision-LiuXiang/ArmorDetectingAndCal/detect/build/LightDescriptor/CMakeFiles/LightDescriptor.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : LightDescriptor/CMakeFiles/LightDescriptor.dir/depend

