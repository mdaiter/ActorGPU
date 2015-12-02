#ifndef ACTOR_H
#define ACTOR_H

class Actor{
	public:
		Actor();
		Actor(unsigned int);
		~Actor();
		// This pops off messages and reacts to each one 
		__device__ virtual void react() = 0;
		// This sends a message to an actor
		__device__ virtual void send(Actor*, char* message) = 0;
	protected:
		unsigned int m_id;
		// This should be able to push in and pop off messages
		MessageBox* message_box;
};

#endif
