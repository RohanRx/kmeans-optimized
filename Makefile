all:
	nvcc mmm.cu -o mmm.x --resource-usage

clean:
	rm -f *.o *~ *.x *.nsys-rep *.sqlite
