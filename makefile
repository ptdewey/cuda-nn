# Makefile

# Define a variable for the directory
DIR :=

# Default target
all: build

# Build target
build:
	@echo "Building in directory $(DIR)"
	@$(MAKE) -C $(DIR)

# Clean target
clean:
	@echo "Cleaning in directory $(DIR)"
	@$(MAKE) -C $(DIR) clean

# Run target
run:
	@echo "Running in directory $(DIR)"
	@$(MAKE) -C $(DIR) run

.PHONY: all build clean run
