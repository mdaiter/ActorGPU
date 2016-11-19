/*
 * This class is meant to be subclassed
 * Supposed to define some sort of structure to the system. We need to have
 * systems be able to be subclassed and run. This system contains the actors
 * we use to run each said system. So we can run each system with different actors underneath each system.
 * */

#ifndef ACTOR_SYSTEM_H
#define ACTOR_SYSTEM_H

#include "actor.h"

class ActorSystem {
	public:
		ActorSystem();
		ActorSystem(unsigned int);
		ActorSystem(unsigned int, unsigned int, unsigned int);
		~ActorSystem();
		// ! This is meant to simulate each actor's sending and receiving of a certain type on the GPU. 
		// This function should launch off a kernel, which runs each actor.
		// In the case of the SchellingActor, each actor sends its type in parallel to the actors around it.
		void simulate();
	private:
		Actor* m_actor_array;
		//Make these the three default dimensions. We can specify more, but for now, just do three
		unsigned int m_width;
		unsigned int m_height;
		unsigned int m_depth;
};

#endif
