#ifndef SCENE_H
#define SCENE_H

#include "scelling_actor.h"

typedef struct scene_s{
	schelling_actor* scene_h;
	schelling_actor* scene_d;
	unsigned int width;
	unsigned int height;
} scene_t;

__host__ void init_scene(scene_t*, unsigned int, unsigned int);

__device__ void modify_scene(scene_t*, unsigned int, unsigned int, char);

__host__ void transfer_scene_to_host(scene_t*);

__host__ void transfer_scene_to_device(scene_t*);

#endif
