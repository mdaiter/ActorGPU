all: compile

compile: main.o actor.o SchellingActor.o 
	/usr/local/cuda/bin/nvcc main.o actor.o SchellingActor.o -o ActorSim

main.o: main.cu
	/usr/local/cuda/bin/nvcc -c main.cu

actor.o: actor.cu
	/usr/local/cuda/bin/nvcc -c actor.cu

ScellingActor.o: SchellingActor.cu
	/usr/local/cuda/bin/nvcc -c SchellingActor.cu

clean:
	rm *o ActorSim
