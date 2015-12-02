#ifndef SCHELLING_ACTOR_H
#define SCHELLING_ACTOR_H

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
		__device__ void increaseNumberAdjacent() { m_numberAdjacent++; }
		// Used to increase the number of adjacent actors
		__device__ unsigned char getType() { return m_type; }
		// Used to reset the number of adjacent actors
		__device__ void resetNumberAdjacent() { m_numberAdjacent = 0; }

	private:
		// This is used for the color of the Schelling Actor
		unsigned char m_type;
		// This is used to temporarily store the number of adjacent actors
		unsigned int m_numberAdjacent;
};

#endif
