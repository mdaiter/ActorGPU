#ifndef SCHELLING_ACTOR_H
#define SCHELLING_ACTOR_H

#include <cuda.h>
#include "actor.h"

class SchellingActor : public Actor {
	public:
		__device__ __host__ SchellingActor(unsigned char);
		__device__ __host__ ~SchellingActor();
		// ! Used to receive each message. Should react to messages with an 'w' or 'b' sent. If the message is different from the
		// m_type of the object, then increaseNumberAdjacent()
		__device__ void receive(Actor*, unsigned char);
		// ! Used to run after the actor receives all of its data. Reacts by sending the MapActor its coordinates and type.
		__device__ virtual void react();
    	
		__device__ __host__ unsigned char type() { return m_type; }

		__device__ __host__ unsigned int numberAdjacent() { return m_numberAdjacent; }

    	__device__ virtual void send(Actor* receiver, char message);
		
		__device__ void increaseNumberAdjacent() { atomicAdd(&m_numberAdjacent, 1); }
		// Used to increase the number of adjacent actors
		// Used to reset the number of adjacent actors
		__device__ void resetNumberAdjacent() { atomicCAS(&m_numberAdjacent, m_numberAdjacent, 0); }

	private:
		// This is used for the color of the Schelling Actor
		unsigned char m_type;
		// This is used to temporarily store the number of adjacent actors
		unsigned int m_numberAdjacent;
};

#endif
