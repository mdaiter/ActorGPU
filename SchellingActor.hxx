#ifndef SCHELLING_ACTOR_H
#define SCHELLING_ACTOR_H

#include "actor.h"

class SchellingActor : public Actor {
	public:
		__device__ __host__ SchellingActor();
		__device__ __host__ ~SchellingActor();
		__device__ __host__ char type() { return m_type; }
		char test;
	protected:
		__device__ virtual void react();
	private:
		char m_type;
};

#endif
