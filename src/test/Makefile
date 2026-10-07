CXX = g++
CXXFLAGS = -std=c++17 -Wall -Wextra -g
INCLUDES = -I..
TARGETS = test_yolo_best_box

.PHONY: all clean test

all: $(TARGETS)

test_yolo_best_box: test_yolo_best_box.o helpers.o
	$(CXX) $(CXXFLAGS) -o $@ $^

test_yolo_best_box.o: test_yolo_best_box.cpp
	$(CXX) $(CXXFLAGS) $(INCLUDES) -c $< -o $@

helpers.o: ../de_common/helpers/helpers.cpp
	$(CXX) $(CXXFLAGS) $(INCLUDES) -c $< -o $@

test: $(TARGETS)
	./test_yolo_best_box

clean:
	rm -f *.o $(TARGETS)

help:
	@echo "Available targets:"
	@echo "  all    - Build the test executables"
	@echo "  test   - Build and run the tests"
	@echo "  clean  - Remove built files"
	@echo "  help   - Show this help message"
