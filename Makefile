# *****************************************************
# Variables to control Makefile operation

CXX = g++
CXXFLAGS = -Wall -g

# ****************************************************
# Targets needed to bring the executable up to date

# main: main.o
# 	$(CXX) $(CXXFLAGS) -o main main.o

# The main.o target can be written more simply

main.o: main.cpp hand_detection.h
	$(CXX) $(CXXFLAGS) -c main.cpp

hand_detection.o: hand_detection.h hand_calculator.h

hand_calculator.o: hand_calculator.h
