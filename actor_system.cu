#include "actor_system.h"

ActorSystem::ActorSystem(){
	//Do nothing. No parameters specified
}

ActorSystem::ActorSystem(unsigned int width, Actor* actor_array) {
	//Make an actor system with width defined
	m_width = width;
	m_height = 1;
	m_depth = 1;
	//Need to instantiate this on the GPU
	m_actor_array = actor_array;
}

ActorSystem::ActorSystem(unsigned int width, unsigned int height, unsigned int depth, Actor* actor_array){
	m_width = width;
	m_height = height;
	m_depth = depth;
	//Need to instantiate this on the GPU
	m_actor_array = actor_array;
}

ActorSystem::~ActorSystem(){
	cudaFree(m_actor_array);
}

// ! This is the function that should be run in parallel to send types of actors to all other actors
__global__ void ActorSystem::actorRun(unsigned int size){
	int idx = threadIdx.x + blockIdx.x * blockDim.x;
	if (idx < size)
		m_actor_array[idx]->react();
}

void ActorSystem::simulate(){
	actor_run<<<m_width, m_height, m_depth>>>(m_width * m_height * m_depth);
}
