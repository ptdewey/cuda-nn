# Makefile

# Define a variable for the directory
DIR :=

all: build

build:
	@echo "Building in directory $(DIR)"
	@$(MAKE) -C $(DIR)

clean:
	@echo "Cleaning in directory $(DIR)"
	@$(MAKE) -C $(DIR) clean

run:
	@echo "Running in directory $(DIR)"
	@$(MAKE) -C $(DIR) run

profile: 
	@echo "Building for profiling in directory $(DIR)"
	@$(MAKE) -C $(DIR) profile

.PHONY: all build clean run profile
