## Parallel processing
#NPROCS = $(shell grep -c 'processor' /proc/cpuinfo)
#MAKEFLAGS += -j$(NPROCS)
MAKEFLAGS += -j16

## Set project layout

# Where to find the code?
SOURCE_DIR := OpenSimRoot/src/

# Where to store objects as we build them?
RELEASE_BUILD_DIR := release_build/
NOPYTHON_BUILD_DIR := nopython_build/
DEBUG_BUILD_DIR := debug_build/
WIN_BUILD_DIR := windows_build/

# Names and paths for the compiled binaries
# Why save these in build instead of root dir? Mostly because if we put a
# release binary named `OpenSimRoot` into the project root,
# its name would conflict with the existing `OpenSimRoot/` subdir
RELEASE_BIN := $(RELEASE_BUILD_DIR)OpenSimRoot
NOPYTHON_BIN := $(NOPYTHON_BUILD_DIR)OpenSimRoot_noPython
DEBUG_BIN := $(DEBUG_BUILD_DIR)OpenSimRoot_debug
WIN_BIN := $(WIN_BUILD_DIR)OpenSimRoot.exe


## Set compiler options

# C++ compiler to use
CXX := g++
CXXWIN := x86_64-w64-mingw32-g++

# compilation flags
# -std: Which version of the C++ standard to enforce?
# -DGITHASH: hash of current commit, so we can compile it into the binary for user information
# -O: set optimization level
# -c: compile only; we'll link in a separate step
# -W: enable compiler warnings
# -fmessage-length=0: Don't insert linebreaks into error messages
CXXFLAGS =  -I/usr/include/python3.12 -std=c++23 -DGITHASH='$(GIT_HASH)' -O3 -Wall -c -fmessage-length=0 
CXXFLAGSWIN =  -I/usr/include/python3.12 -std=c++23 -DGITHASH='$(GIT_HASH)' -O3 -Wall -c -fmessage-length=0 
# -g: include debug symbols
# -Og: include some speed optimizations but none that make debugger output confusing
CXXFLAGS_DEBUG = $(subst -O3, -Og -g, $(CXXFLAGS))

# options controlling how GCC generates header dependency files
# -MMD: Generate Make rules listing all the (non-system) headers included in each cpp file.
#	These are needed so that when a header changes, Make knows it needs to recompile the objects that use it.
# -MP: Include phony targets for files that have no dependencies
#	This isn't strictly needed, but it keeps Make from complaining when you remove a header.
# -MF, MT: Set filename of the resulting *.d file (-MT) and name of target *.o file in the generated rule (-MT)
#	We need these because otherwise directories would be stripped off the name by default:
#	"Mesh.o" instead of "src/modules/Soil/Mesh.o"
DEPFLAGS = -MMD -MP -MF"$(@:%.o=%.d)" -MT"$(@)"
DEPFLAGSWIN = -Wa,-mbig-obj -MMD -MP -MF"$(@:%.o=%.d)" -MT"$(@)"


## Expand user settings into lists of files to be processed

# Look up what Git revision we're building from
# We use this below to compile the hash into the binary for ease of debugging
GIT_HASH := $(shell git rev-parse --short=10 HEAD; git diff-index --quiet HEAD || echo " plus uncommitted changes")

# Find all the source files to be compiled,
# i.e. any cpp file anywhere in the src directory
# Make doesn't provide a build-in recursive wildcard match, so we define one ourselves
recursive_wildcard = $(foreach d, $(wildcard $1*), \
	$(call recursive_wildcard, $d/, $2)$(filter $(subst *, %, $2), $d))
SOURCES := $(call recursive_wildcard, $(SOURCE_DIR), *.cpp)

# For each source file, we will generate a corresponding object file and
# 	dependency list in the build dir. These files may or may not exist already,
# 	so we construct the list by copying SOURCES and altering its file extensions,
# 	and make sure to explicitly create any needed subdirectories
# Yes, it looks redundant to specify these separately for release and debug,
#	but necessary to keep from mixing compiled objects with and without debug symbols
DEPENDS_RELEASE := $(patsubst $(SOURCE_DIR)%.cpp, $(RELEASE_BUILD_DIR)%.d, $(SOURCES))
DEPENDS_NOPYTHON := $(patsubst $(SOURCE_DIR)%.cpp, $(NOPYTHON_BUILD_DIR)%.d, $(SOURCES))
DEPENDS_WIN := $(patsubst $(SOURCE_DIR)%.cpp, $(WIN_BUILD_DIR)%.d, $(SOURCES))
DEPENDS_DEBUG := $(patsubst $(SOURCE_DIR)%.cpp, $(DEBUG_BUILD_DIR)%.d, $(SOURCES))

OBJECTS_RELEASE := $(patsubst $(SOURCE_DIR)%.cpp, $(RELEASE_BUILD_DIR)%.o, $(SOURCES))
OBJECTS_NOPYTHON := $(patsubst $(SOURCE_DIR)%.cpp, $(NOPYTHON_BUILD_DIR)%.o, $(SOURCES))
OBJECTS_WIN := $(patsubst $(SOURCE_DIR)%.cpp, $(WIN_BUILD_DIR)%.o, $(SOURCES))
OBJECTS_DEBUG := $(patsubst $(SOURCE_DIR)%.cpp, $(DEBUG_BUILD_DIR)%.o, $(SOURCES))

BUILD_SUBDIRS := $(patsubst $(SOURCE_DIR)%, %, $(SOURCES))
BUILD_SUBDIRS := $(sort $(dir $(BUILD_SUBDIRS))) # sort is to drop duplicates

BUILD_TREE_RELEASE := $(addprefix $(RELEASE_BUILD_DIR), $(BUILD_SUBDIRS))
BUILD_TREE_NOPYTHON := $(addprefix $(NOPYTHON_BUILD_DIR), $(BUILD_SUBDIRS))
BUILD_TREE_WIN := $(addprefix $(WIN_BUILD_DIR), $(BUILD_SUBDIRS))
BUILD_TREE_DEBUG := $(addprefix $(DEBUG_BUILD_DIR), $(BUILD_SUBDIRS))


## End of variable definitions, begin target recipes (AKA instructions for building specific files)

# first-listed target is used as the default,
# so typing `make` with no arguments produces a release build
release: $(RELEASE_BIN)

nopython: $(NOPYTHON_BIN)

debug: $(DEBUG_BIN)

win: $(WIN_BIN)

all: release nopython debug win

# Create output directories if they don't yet exist
# NB creates whole subtree as soon as it notices we need any of it
$(BUILD_TREE_RELEASE):
	mkdir -p $@
$(BUILD_TREE_NOPYTHON):
	mkdir -p $@
$(BUILD_TREE_WIN):
	mkdir -p $@
$(BUILD_TREE_DEBUG):
	mkdir -p $@

# Compile individual objects in a separate directory from their matching source files,
# ensuring intermediate subdirs exist before writing to them
# `$@` is name of the target (in this case any .o) to be created)
# `$<` is the first prerequisite (in this case the .cpp file with the same name as the target)
# `| <dir>` is an order-only prerequisite:
#	it ensures that the build subdirectories _exist_ before we compile *.o, but doesn't waste time rebuilding
#	*.o when the build directory's modification time changes
$(RELEASE_BUILD_DIR)%.o: $(SOURCE_DIR)%.cpp | $(BUILD_TREE_RELEASE)
	$(CXX) $(CXXFLAGS) $(DEPFLAGS) -o "$@" "$<"
$(NOPYTHON_BUILD_DIR)%.o: $(SOURCE_DIR)%.cpp | $(BUILD_TREE_NOPYTHON)
	$(CXX) $(CXXFLAGS) -DNOPYTHON $(DEPFLAGS)  -o "$@" "$<"
$(WIN_BUILD_DIR)%.o: $(SOURCE_DIR)%.cpp | $(BUILD_TREE_WIN)
	$(CXXWIN) $(CXXFLAGSWIN) $(DEPFLAGSWIN) -o "$@" "$<"
$(DEBUG_BUILD_DIR)%.o: $(SOURCE_DIR)%.cpp | $(BUILD_TREE_DEBUG)
	$(CXX) $(CXXFLAGS_DEBUG) $(DEPFLAGS) -o "$@" "$<"

# Link objects into one binary
# `$@` and `| <dir>` as above
# `$+` is a space-separated list of all the prerequisites, including any duplicates
#	(we probably don't have duplicates, but that's sometimes useful when linking libraries)
$(RELEASE_BIN): $(OBJECTS_RELEASE) | $(RELEASE_BUILD_DIR)
	$(CXX) -o $@ $+ -lpython3.12
$(NOPYTHON_BIN): $(OBJECTS_NOPYTHON) | $(NOPYTHON_BUILD_DIR)
	$(CXX) -o $@ $+ 
$(WIN_BIN): $(OBJECTS_WIN) | $(WIN_BUILD_DIR)
	$(CXXWIN) -o $@ $+ -lpsapi -lcomdlg32
$(DEBUG_BIN): $(OBJECTS_DEBUG) | $(DEBUG_BUILD_DIR)
	$(CXX) -g -o $@ $+ -lpython3.12 

# Each dependency file contains Make directives that specify which headers are included in which objects.
# We include them here as if we'd typed them out for ourselves.
-include $(DEPENDS_RELEASE)
-include $(DEPENDS_NOPYTHON)
-include $(DEPENDS_WIN)
-include $(DEPENDS_DEBUG)

# Remove all generated files for a clean start.
# Runs only when called explicitly.
clean: clean-release clean-nopython clean-win clean-debug clean-dirs 
clean-release:
	rm -rf $(OBJECTS_RELEASE) $(DEPENDS_RELEASE) $(RELEASE_BIN)
clean-nopython:
	rm -rf $(OBJECTS_NOPYTHON) $(DEPENDS_NOPYTHON) $(NOPYTHON_BIN)
clean-win:
	rm -rf $(OBJECTS_WIN) $(DEPENDS_WIN) $(WIN_BIN)
clean-debug:
	rm -rf $(OBJECTS_DEBUG) $(DEPENDS_DEBUG) $(DEBUG_BIN)
clean-dirs:
	rm -rf $(RELEASE_BUILD_DIR) $(NOPYTHON_BUILD_DIR) $(WIN_BUILD_DIR)  $(DEBUG_BUILD_DIR) 

# Declare phony targets
# Normal (non-phony) targets are expected to generate files with the same name they have:
# 	`make my_binary` means "if the file `my_binary` doesn't exist or is out of date,
# 	please regenerate it by running the recipe for `my_binary`"
# Phony targets, on the other hand, are simply names for groups of actions,
# 	and they *do not* have to generate a file with their own name:
# 	`make clean` means "Please run the recipe named `clean`, and it is OK
# 	that no file with the literal name `clean` will be created in the process"
.PHONY: all release nopython win debug clean clean-release clean-nopython clean-win clean-debug clean-dirs
