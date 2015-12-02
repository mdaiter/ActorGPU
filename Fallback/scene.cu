#include <stdlib.h>
#include "scene.h"

__host__ void init_scene(scene_t* scene, unsigned int width, unsigned int height){
	scene = (scene_t*)malloc(sizeof(scene_t));
	scene->scene_h = (char*)malloc(sizeof(char) * width * height);
	cudaMalloc((void**)&scene->scene_d, sizeof(char) * width * height);
	scene->width = width;
	scene->height = height;
}

__host__ void modify_scene_host(scene_t* scene, unsigned int x, unsigned int y, char new_val){
	scene->scene_h[y * scene->width + x] = new_val;
}

__host__ void transfer_scene_to_host(scene_t* scene){
	cudaMemcpy(scene->scene_h, scene->scene_d, sizeof(char) * scene->width * scene->height, cudaMemcpyDeviceToHost);
}

__host__ void transfer_scene_to_device(scene_t* scene){
	cudaMemcpy(scene->scene_d, scene->scene_h, sizeof(char) * scene->width * scene->height, cudaMemcpyDeviceToHost);
}
