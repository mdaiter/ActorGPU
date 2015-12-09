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
#include <string>
#include <sstream>
#include <curand.h>
#include <curand_kernel.h>

const int X_SIZE = 20;
const int Y_SIZE = 20;
const int NUM_ACTORS = X_SIZE * Y_SIZE;
const int BLOCK_SIZE = 1024;
const int NUM_RUNS = 20;
const int IT_PER_RUN = 10;

using std::vector;
using std::string;
using std::stringstream;

__device__ Actor* getActor(Actor** actor_array_d, int x, int y, int x_size, int y_size) {
  if (x >= 0 && x < x_size && y >= 0 && y < y_size) {
    int idx = y * x_size + x;
    return actor_array_d[idx];
  }
  return NULL;
}

__global__ void init(Actor** actor_array_d, int x_size, int y_size, unsigned int* system_state_d) {
	int xi = threadIdx.x + blockDim.x * blockIdx.x;
	int yi = threadIdx.y + blockDim.y * blockIdx.y;
  int idx = yi * x_size + xi;
  if (xi < x_size && yi < y_size) {
    // Random number generator.
    curandState_t state;
    curand_init(0, idx, 0, &state);
    //int type = (idx * 10) + (curand(&state) % 3);
    int type = (idx * 10) + (idx % 3);
	  actor_array_d[idx] = new SchellingActor(type);
	  __syncthreads();
    // Save the state of the system.
    system_state_d[idx] = (type % 10) ? type : 0;
  }
}

__global__ void sim(Actor** actor_array_d, int x_size, int y_size, int num_it, unsigned int* system_state_d) {
	int xi = threadIdx.x + blockDim.x * blockIdx.x;
	int yi = threadIdx.y + blockDim.y * blockIdx.y;
  int idx = yi * x_size + xi;
  if (xi < x_size && yi < y_size) {
    // Random number generator.
    curandState_t state;
    curand_init(0, idx, 0, &state);
    for (int it = 0; it < num_it; ++it) {
      Actor* curr_actor = actor_array_d[idx];
      Actor* neighbors[8];
      neighbors[0] = getActor(actor_array_d, xi    , yi + 1, X_SIZE, Y_SIZE);
      neighbors[1] = getActor(actor_array_d, xi + 1, yi + 1, X_SIZE, Y_SIZE);
      neighbors[2] = getActor(actor_array_d, xi + 1, yi    , X_SIZE, Y_SIZE);
      neighbors[3] = getActor(actor_array_d, xi + 1, yi - 1, X_SIZE, Y_SIZE);
      neighbors[4] = getActor(actor_array_d, xi    , yi - 1, X_SIZE, Y_SIZE);
      neighbors[5] = getActor(actor_array_d, xi - 1, yi - 1, X_SIZE, Y_SIZE);
      neighbors[6] = getActor(actor_array_d, xi - 1, yi    , X_SIZE, Y_SIZE);
      neighbors[7] = getActor(actor_array_d, xi - 1, yi + 1, X_SIZE, Y_SIZE);

      for (int i = 0; i < 8; ++i) {
        curr_actor->send(neighbors[i]);
      }

      // Moving phase involves swapping actors around.
      Actor* swap_neighbor = ((SchellingActor*)curr_actor)->findFreeNeighbor((SchellingActor**)neighbors, curand(&state) % 8);
      curr_actor->react(swap_neighbor);

      //printf("%d\t%c\t%d\n", idx, static_cast<SchellingActor*>(curr_actor)->type(), static_cast<SchellingActor*>(curr_actor)->numberAdjacent());
      __syncthreads();
      // Save the state of the system.
      system_state_d[idx] = static_cast<SchellingActor*>(curr_actor)->m_type;
      __syncthreads();
    }
  }
}

void draw_state(unsigned int* system_state_h, string filename) {
  /*
  // Terminal output.
  for (int i = 0; i < Y_SIZE; ++i) {
    for (int j = 0; j < X_SIZE; ++j) {
      printf("%d\t", system_state_h[i * X_SIZE + j]);
    }
    printf("\n");
  }
  printf("\n");
  */

  // Draw PNG
  vector<unsigned char> image;
  image.resize(X_SIZE * Y_SIZE * 4);
  for (int y = 0; y < Y_SIZE; ++y) {
    for (int x = 0; x < X_SIZE; ++x) {
      int idx = X_SIZE * y + x;
      image[4 * X_SIZE * y + 4 * x + 0] = (system_state_h[idx] % 10 == 0) ? 255 : 0;
      image[4 * X_SIZE * y + 4 * x + 1] = (system_state_h[idx] % 10 == 1) ? 255 : 0;
      image[4 * X_SIZE * y + 4 * x + 2] = (system_state_h[idx] % 10 == 2) ? 255 : 0;
      image[4 * X_SIZE * y + 4 * x + 3] = 255;
    }
  }
  unsigned error = lodepng::encode(filename, image, X_SIZE, Y_SIZE);
  //if there's an error, display it
  if(error) std::cout << "encoder error " << error << ": "<< lodepng_error_text(error) << std::endl;
}

int main() {
  unsigned int* system_state_h = (unsigned int*)malloc(NUM_ACTORS * sizeof(unsigned int));
  unsigned int* system_state_d;
	cudaMalloc((void**)&system_state_d, NUM_ACTORS * sizeof(unsigned int));
	Actor** actor_array_d;
	cudaMalloc((void**)&actor_array_d, NUM_ACTORS * sizeof(Actor*));
  // Initialize the actors.
  printf("INIT\n");
  dim3 DimGrid(ceil((float)X_SIZE/(float)BLOCK_SIZE), ceil((float)Y_SIZE/(float)BLOCK_SIZE), 1);
  dim3 DimBlock(X_SIZE, Y_SIZE, 1);
  init<<<DimGrid, DimBlock>>>(actor_array_d, X_SIZE, Y_SIZE, system_state_d);
  cudaMemcpy(system_state_h, system_state_d, NUM_ACTORS * sizeof(unsigned int), cudaMemcpyDeviceToHost);
  draw_state(system_state_h, "init.png");
  // Run simulation.
  printf("SIM\n");
  for (int i = 0; i < NUM_RUNS; ++i) {
    sim<<<DimGrid, DimBlock>>>(actor_array_d, X_SIZE, Y_SIZE, IT_PER_RUN, system_state_d);
    cudaMemcpy(system_state_h, system_state_d, NUM_ACTORS * sizeof(unsigned int), cudaMemcpyDeviceToHost);
    stringstream ss;
    ss << "sim" << i << ".png";
    draw_state(system_state_h, ss.str());
  }

	cudaFree(actor_array_d);
  cudaFree(system_state_d);
  free(system_state_h);
}
