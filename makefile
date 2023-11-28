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

.PHONY: all build clean run
