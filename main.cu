/*
 Main testing file
 Making sure that our tests run, that things work, etc.
 */

#include <cuda.h>
#include <stdio.h>
#include "actor.h"
#include "SchellingActor.hxx"

__global__ void init(Actor* input){
	int idx = threadIdx.x + blockDim.x * blockIdx.x;
	input[idx] = new SchellingActor();
	__syncthreads();
}

__global__ void sim(Actor* input){
	int idx = threadIdx.x;
	if (idx < 1){
		//input[idx].test = 'a';
		input[idx].react();
	}
	__syncthreads();
}

int main() {
	SchellingActor* schelling_actor_d;
	SchellingActor* schelling_actor_h = new SchellingActor();
	cudaMalloc((void**)&schelling_actor_d, sizeof(SchellingActor));
	sim<<<1, 1>>>(schelling_actor_d);
	cudaMemcpy(schelling_actor_h, schelling_actor_d, sizeof(SchellingActor), cudaMemcpyDeviceToHost);
	printf("schelling_actor_h: %c\n", schelling_actor_h->type());
	cudaFree(schelling_actor_d);
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
