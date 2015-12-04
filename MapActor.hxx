#ifndef MAP_ACTOR_H
#define MAP_ACTOR_H

class MapActor : public Actor {
	public:
		__host__ __device__ MapActor(unsigned int, unsigned int);
		__host__ __device__ ~MapActor();
		// ! Needs to be able to react to messages sent by SchellingActors within our system, move SchellingActors around within its grid through atomicCAS.
		// !!!!!! TODO: determine whether we swap by pointer or by value between cells
		__device__ virtual void react();
		__device__ virtual void send(Actor*, char*);
		__host__ void moveActorsAround();
	private:
		Actor* m_map_h;
		Actor* m_map_d;
		unsigned int m_width;
		unsigned int m_height;
};

#endif
