all: compile

compile: main.o actor.o SchellingActor.o
	/usr/local/cuda/bin/nvcc main.o actor.o SchellingActor.o -o ActorSim

main.o: main.cu
	/usr/local/cuda/bin/nvcc -c -dc main.cu

actor.o: actor.cu
	/usr/local/cuda/bin/nvcc -c -dc actor.cu

SchellingActor.o: SchellingActor.cu
	/usr/local/cuda/bin/nvcc -c -dc SchellingActor.cu

clean:
	rm -rf *o
