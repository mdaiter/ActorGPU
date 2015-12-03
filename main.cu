/*
 Main testing file
 Making sure that our tests run, that things work, etc.
 */

#include <cuda.h>
#include <stdio.h>
#include "SchellingActor.hxx"
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>

#define NUM_ACTORS 10
#define BLOCK_SIZE 1024

__global__ void init(Actor** actor_array_d, int size) {
	int idx = threadIdx.x + blockDim.x * blockIdx.x;
	actor_array_d[idx] = new SchellingActor(0);
	__syncthreads();
}

__global__ void sim(Actor** actor_array_d, int size) {
	int idx = threadIdx.x + blockDim.x * blockIdx.x;
	if (idx < NUM_ACTORS){
		actor_array_d[idx]->react();
		actor_array_d[idx]->send(NULL, 0);
	}
	__syncthreads();
}

int main() {
	Actor** actor_array_d;
	//SchellingActor* schelling_actor_h = new SchellingActor();
	cudaMalloc((void**)&actor_array_d, NUM_ACTORS * sizeof(Actor*));
  dim3 DimGrid(ceil((float)NUM_ACTORS/(float)BLOCK_SIZE), 1, 1);
  dim3 DimBlock(BLOCK_SIZE, 1, 1);
  init<<<DimGrid, DimBlock>>>(actor_array_d, BLOCK_SIZE);
  sim<<<DimGrid, DimBlock>>>(actor_array_d, BLOCK_SIZE);
	//cudaMemcpy(schelling_actor_h, actor_array_d, sizeof(SchellingActor), cudaMemcpyDeviceToHost);
	//printf("schelling_actor_h: %c\n", schelling_actor_h->type());
	cudaFree(actor_array_d);
}

/*
int main(){
	Actor* wit_actor = new WitActor();
	Actor* mic_actor = new MicActor();

	ActorSystem* system = new ActorSystem();

	system.addActor(mic_actor);
	system.addActor(wit_actor);

	system.simulate();
}*/
