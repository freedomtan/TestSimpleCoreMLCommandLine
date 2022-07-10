CFLAGS = -framework CoreML -framework Foundation -std=c++14
CC=clang++

test_cmd: test_cmd.mm
	$(CC) -o $@ ${CFLAGS} $<

run: test_cmd
	./$< MobileNet_EdgeTPU_multi_arrays.mlmodel
clean:
	rm -f test_cmd
