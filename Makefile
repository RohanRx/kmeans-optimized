all:
	nvcc kmeans.cu -o kmeans.x --resource-usage
	nvcc kmeans512.cu -o kmeans512.x --resource-usage
clean:
	rm -f *.o *~ *.x *.nsys-rep *.sqlite
