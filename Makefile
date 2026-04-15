all:
	nvcc kmeans.cu -o kmeans.x --resource-usage

clean:
	rm -f *.o *~ *.x *.nsys-rep *.sqlite
