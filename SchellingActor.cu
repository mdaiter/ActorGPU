#include "SchellingActor.hxx"
#include <cstdio>

__device__ __host__ SchellingActor::SchellingActor(unsigned int type)
  : m_type((type % 10) ? type : 0), m_numberAdjacent(0) {};

__device__ __host__ SchellingActor::~SchellingActor(){
}

__device__ Actor* SchellingActor::findFreeNeighbor(SchellingActor** neighbors, int start) {
  for (int i = 0; i < 8; ++i) {
    int idx = (start + i) % 8;
    SchellingActor* curr_neighbor = (SchellingActor*)neighbors[idx];
    if (curr_neighbor != NULL && curr_neighbor->m_type == 0) {
      return curr_neighbor;
    }
  }
  return NULL;
}

__device__ void SchellingActor::react(Actor* actor) {
  if (m_numberAdjacent >= THRESHOLD && actor != NULL) {
    SchellingActor* neighbor = static_cast<SchellingActor*>(actor);
    atomicCAS(&(neighbor->m_type), 0, this->m_type);
    if (neighbor->m_type == this->m_type) {
      this->m_type = 0;
    }
  }
  m_numberAdjacent = 0;
}

__device__ void SchellingActor::send(Actor* receiver) {
  //printf("SchellingActor send()\n");
  if (receiver == NULL) {
    return;
  }
  unsigned int this_type = this->m_type % 10;
  unsigned int receiver_type = ((SchellingActor*)receiver)->m_type % 10;
  if (this_type != 0 && receiver_type != 0 && this_type != receiver_type) {
    ++m_numberAdjacent;
  }
}
