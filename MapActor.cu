#include <cuda.h>
#include <stdlib.h>
#include "MapActor.hxx"

#define UPPER_LEFT(x) x - width - 1
#define UPPER_MID(x) x - width
#define UPPER_RIGHT(x) x - width + 1
#define LEFT(X) x - 1
#define RIGHT(X) x + 1
#define LOWER_LEFT(x) x + width - 1
#define LOWER_MID(x) x + width
#define LOWER_RIGHT(x) + width + 1

__host__ __device__ MapActor::MapActor(unsigned int width, unsigned int height) {
	m_width = width;
	m_height = height;
	
}

__host__ __device__ MapActor::~MapActor() {
	
}

__global__ void compute_free(SchellingActor* actorsGlobal, unsigned int width, unsigned int height, unsigned int* positionIndexMap) {
	int idx = threadIdx.x + blockDim.x * blockIdx.x;
	int idy = threadIdx.y + blockDim.y * blockIdx.y;

	__shared__ Actor previousActors[width * height];
	if (idy * width + idx < width * height) {
		previousActors[idy * width + idx] = actorsGlobal[idy * width + idx];
		if (previousActors[idy * width + idx].type() == 'f')
			//Position map index needs to equal 1 if free; 0 if not free. Use this data structure as a representation of a 
			positionIndexMap[idy * width + idx] = 1;
	}
}

SchellingActor* compute_free_synchronous(SchellingActor* actorsGlobal, unsigned int width, unsigned int height) {
	std::vector<SchellingActor> freeVector = 
}

__global__ void compute_shifts(SchellingActor* actorsGlobal, unsigned int width, unsigned int height, unsigned int threshold, unsigned int* 
		actorsThatNeedShifting){
	int idx = threadIdx.x + blockDim.x * blockIdx.x;
	int idy = threadIdx.y + blockDim.y * blockIdx.y;
	
	 Actor adjacentActors[8];
	__shared__ Actor previousActors[width * height];

	previousActors[idy * width + idx] = actorsGlobal[idy * width + idx];
	
	__syncthreads();
	// TODO: Must implement bounds checking for this
	adjacentActors[0] = actorsGlobal[UPPER_LEFT(idy * width + idx)];
	adjacentActors[1] = actorsGlobal[UPPER_MID(idy * width + idx)];
	adjacentActors[2] = actorsGlobal[UPPER_RIGHT(idy * width + idx)];
	adjacentActors[3] = actorsGlobal[LEFT(idy * width + idx)];
	adjacentActors[4] = actorsGlobal[RIGHT(idy * width + idx)];
	adjacentActors[5] = actorsGlobal[LOWER_LEFT(idy * width + idx)];
	adjacentActors[6] = actorsGlobal[LOWER_MID(idy * width + idx)];
	adjacentActors[7] = actorsGlobal[LOWER_RIGHT(idy * width + idx)];
	
	for(int j = 0; j < 7; j++)
		adjacentActors[idy * width + idx].send(adjacentActors[j], adjacentActors[j].type());
	__syncthreads();

	// If you're above the threshold, you MUST move. To do this, we need to create a histogram of what we're going to move
	if (actorsGlobal[idy * width + idx].numberAdjacent() > threshold)
		actorsThatNeedShifting[idy * width + idx]++;
	__syncthreads();
}

__device__ void MapActor::react() {
	
}

void MapActor::send(Actor* actor, char* message) {
	
}
