#ifndef SCHELLING_ACTOR_H
#define SCHELLING_ACTOR_H

typedef struct schelling_actor_s {
	unsigned int id;
	unsigned char data;
	unsigned int x;
	unsigned int y;
} schelling_actor_t;

__host__ void schelling_actor_init(schelling_actor_t*, unsigned int, unsigned int, unsigned int);

__host__ __device__ void schelling_actor_modify_adjacent(schelling_actor_t*, unsigned char);

#endif
