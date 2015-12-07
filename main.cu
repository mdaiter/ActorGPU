/*
 Main testing file
 Making sure that our tests run, that things work, etc.
 */

#include <cuda.h>
#include <stdio.h>
#include "SchellingActor.hxx"
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include "loadpng.h"
#include <vector>

const int X_SIZE = 50;
const int Y_SIZE = 20;
const int NUM_ACTORS = X_SIZE * Y_SIZE;
const int BLOCK_SIZE = 1024;

using std::vector;

__device__ Actor* getActor(Actor** actor_array_d, int x, int y, int x_size, int y_size) {
  if (x < x_size && y < y_size) {
    int idx = y * x_size + x;
    return actor_array_d[idx];
  }
  return NULL;
}

__global__ void init(Actor** actor_array_d, int x_size, int y_size) {
	int xi = threadIdx.x + blockDim.x * blockIdx.x;
	int yi = threadIdx.y + blockDim.y * blockIdx.y;
  int idx = yi * x_size + xi;
  if (xi < x_size && yi < y_size) {
	  actor_array_d[idx] = new SchellingActor((idx % 3) ? 'b' : 'w');
  }
	__syncthreads();
}

__global__ void sim(Actor** actor_array_d, int x_size, int y_size, int num_it, unsigned char* system_state_d) {
	int xi = threadIdx.x + blockDim.x * blockIdx.x;
	int yi = threadIdx.y + blockDim.y * blockIdx.y;
  int idx = yi * x_size + xi;
  if (xi < x_size && yi < y_size) {
    Actor* curr_actor = actor_array_d[idx];
    Actor* e_actor = getActor(actor_array_d, xi + 1, yi, X_SIZE, Y_SIZE);
		curr_actor->send(e_actor, 0);
    //printf("%d\t%c\t%d\n", idx, static_cast<SchellingActor*>(curr_actor)->type(), static_cast<SchellingActor*>(curr_actor)->numberAdjacent());
	  __syncthreads();
    system_state_d[idx] = static_cast<SchellingActor*>(curr_actor)->type();
	  __syncthreads();
  }
}

void draw_state(unsigned char* system_state_h) {
  // Draw PNG
  const char* filename = "test.png";
  vector<unsigned char> image;
  image.resize(X_SIZE * Y_SIZE * 4);
  for (int y = 0; y < Y_SIZE; ++y) {
    for (int x = 0; x < X_SIZE; ++x) {
      int idx = X_SIZE * y + x;
      image[4 * X_SIZE * y + 4 * x + 0] = (system_state_h[idx] == 'b') ? 255 : 0;
      image[4 * X_SIZE * y + 4 * x + 1] = (system_state_h[idx] == 'w') ? 255 : 0;
      image[4 * X_SIZE * y + 4 * x + 2] = (system_state_h[idx] == 'f') ? 255 : 0;
      image[4 * X_SIZE * y + 4 * x + 3] = 255;
    }
  }
  unsigned error = lodepng::encode(filename, image, X_SIZE, Y_SIZE);
  //if there's an error, display it
  if(error) std::cout << "encoder error " << error << ": "<< lodepng_error_text(error) << std::endl;
}

int main() {
  unsigned char* system_state_h = (unsigned char*)malloc(NUM_ACTORS * sizeof(unsigned char));
  unsigned char* system_state_d;
	cudaMalloc((void**)&system_state_d, NUM_ACTORS * sizeof(unsigned char));
	Actor** actor_array_d;
	cudaMalloc((void**)&actor_array_d, NUM_ACTORS * sizeof(Actor*));
  dim3 DimGrid(ceil((float)X_SIZE/(float)BLOCK_SIZE), ceil((float)Y_SIZE/(float)BLOCK_SIZE), 1);
  dim3 DimBlock(X_SIZE, Y_SIZE, 1);
  init<<<DimGrid, DimBlock>>>(actor_array_d, X_SIZE, Y_SIZE);
  sim<<<DimGrid, DimBlock>>>(actor_array_d, X_SIZE, Y_SIZE, 1, system_state_d);
  cudaMemcpy(system_state_h, system_state_d, NUM_ACTORS * sizeof(unsigned char), cudaMemcpyDeviceToHost);
  for (int i = 0; i < NUM_ACTORS; ++i) {
    printf("%c ", system_state_h[i]);
  }
  printf("\n");

  draw_state(system_state_h);

	cudaFree(actor_array_d);
  cudaFree(system_state_d);
  free(system_state_h);
}
