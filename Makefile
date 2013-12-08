all:
	nvcc -arch sm_13 canny.cu -o canny

test:
	./canny test/input/small.pgm 30 60
	./canny test/input/medium.pgm 30 50
	./canny test/input/big.pgm 60 70
