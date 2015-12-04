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

#define NUM_THREADS 256
#define NUM_THRESHOLD 2

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

bool zip_actors(unsigned int* freeActors, unsigned int* actorsThatNeedMoving, unsigned int width, unsigned int height) {
	// Both vectors have been allocated with the width * height
	unsigned int prev_free = width * height + 1;
	unsigned int prev_actorThatNeedsMoving = width * height + 1;
	for (size_t i = 0; i < width * height; i++){
		if (freeActors[i] == 1){
			prev_free = i;
		if (actorsThatNeedsMoving[i] == 1 )
			prev_actorThatNeedsMoving = i;

		//Now we determine whether we can make a shift
		if (prev_actorThatNeedsMoving != width * height + 1 && 
				prev_free != width * height + 1){
			// Shift
			
		}	
	}
}

// You MUST pass in a device-vector of the 
__host__ void MapActor::moveActorsAround() {
	unsigned int* freePositions_h = (unsigned int*) malloc(sizeof(unsigned int) * m_width * m_height);
	unsigned int* freePositions_d;
	unsigned int* actorsThatNeedMoving_h = (unsigned int*) malloc(sizeof(unsigned int) * m_width * m_height);
	unsigned int* actorsThatNeedMoving_d;
	cudaMalloc((void**)&freePositions_d, sizeof(unsigned int) * m_width * m_height);
	cudaMemset(freePosiitons_d, 0, sizeof(unsigned int) * m_width * m_height);
	cudaMalloc((void**)&actorsThatNeedMoving_d, sizeof(unsigned int) * m_width * m_height);
	cudaMemset(actorsThatNeedMoving_d, 0, sizeof(unsigned int) * m_width * m_height);
	dim3 blockDim((m_width - 1) / NUM_THREADS + 1, (m_height - 1) / NUM_THREADS + 1, 1);
	dim3 threadDim(NUM_THREADS, NUM_THREADS, 0);
	compute_free<<<blockDim, threadDim>>>(m_map_d, m_width, m_height, freePositions_d);
	compute_shifts<<<blockDim, threadDim>>>(m_map_d, m_width, m_height, NUM_THRESHOLD, actorsThatNeedMoving_d);
	
	//TODO: Compute shifts synchronously
	cudaMemcpy(freePositions_h, freePositions_d, sizeof(unsigned int) * m_width * m_height, cudaMemcpyDeviceToHost);
	cudaMemcpy(actorsThatNeedMoving_h, actorsThatNeedMoving_d, sizeof(unsigned int) * m_width * m_height, cudaMemcpyDeviceToHost);
	
	zip_actors(freePositions_h, actorsThatNeedMoving_h, m_width, m_height);

	free(freePositions_h);
	free(actorsThatNeedMoving_h);
	cudaFree(actorsThatNeedMoving_d);
	cudaFree(freePositions_d);
}

__device__ void MapActor::react() {
	
}

void MapActor::send(Actor* actor, char* message) {
	
}
