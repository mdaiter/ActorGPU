/*
   Boilerplate code for more actor stuff. For now, we have an actor model just be an abstraction away
   to make sure we must MUST MUST implement the react function
 */

#include "actor.h"

Actor::Actor(){
	m_id = 0;
}

Actor::Actor(unsigned int id){
	m_id = id;
}

Actor::~Actor(){

}
