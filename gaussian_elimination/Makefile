TEST_FILES = ../thirdparty/gtest/gtest-all.cc ../thirdparty/gtest/gtest_main.cc
SRC = gaussian_elimination_test.cpp
HDR = gaussian_elimination.hpp

all: tests.done

gaussian_elimination_test: $(SRC) $(HDR)
	g++ -O1 -g -std=c++11 -o gaussian_elimination_test $(SRC) -I../thirdparty $(TEST_FILES) -lpthread

tests.done: gaussian_elimination_test
	./gaussian_elimination_test
	touch tests.done
