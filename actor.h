#ifndef ACTOR_H
#define ACTOR_H

class Actor{
	public:
		Actor();
		Actor(unsigned int);
		~Actor();
		virtual void react() = 0;

	private:
		// We make this -1 to use this as a check that the actor at hand has been initialized
		unsigned int m_id;
};

#endif
