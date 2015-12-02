#ifndef MAP_ACTOR_H
#define MAP_ACTOR_H

class MapActor : public Actor {
	public:
		__host__ __device__ Scene(unsigned int, unsigned int);
		__host__ __device__ ~Scene();
		// ! Needs to be able to react to messages sent by SchellingActors within our system, move SchellingActors around within its grid through atomicCAS.
		// !!!!!! TODO: determine whether we swap by pointer or by value between cells
		__device__ virtual void react(); 
	private:
		Actor* m_map;
		unsigned int m_width;
		unsigned int m_height;
};

#endif
