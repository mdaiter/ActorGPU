#ifndef ACTOR_H
#define ACTOR_H

class Actor{
	public:
		Actor();
		Actor(unsigned int);
		~Actor();
	protected:
		__device__ virtual void react() = 0;
		unsigned int m_id;
};

#endif
