all: compile

compile: main.o actor.o SchellingActor.o lodepng.o
	/usr/local/cuda/bin/nvcc -arch sm_20 main.o actor.o SchellingActor.o lodepng.o -o ActorSim

main.o: main.cu
	/usr/local/cuda/bin/nvcc -arch sm_20 -c -dc main.cu

actor.o: actor.cu
	/usr/local/cuda/bin/nvcc -arch sm_20 -c -dc actor.cu

SchellingActor.o: SchellingActor.cu
	/usr/local/cuda/bin/nvcc -arch sm_20 -c -dc SchellingActor.cu

lodepng.o: lodepng.cpp
	gcc -c lodepng.cpp
  
clean:
	rm -rf *o
