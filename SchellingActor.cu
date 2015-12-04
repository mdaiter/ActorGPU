#include "SchellingActor.hxx"
#include <cstdio>

__device__ __host__ SchellingActor::SchellingActor(unsigned char type){
	m_type = type;
}

__device__ __host__ SchellingActor::~SchellingActor(){
  printf("SchellingActor constructor\n");
}

__device__ void SchellingActor::react() {
  printf("SchellingActor react()\n");
	for (int i = 1; i < 257; i <<= 1) {
		if (atomicXor((int*)&m_type, i) >> i - 1 == 0)
			increaseNumberAdjacent();
	}
}

__device__ void SchellingActor::send(Actor* receiver, char message) {
  printf("SchellingActor send()\n");
  if (message != 'f' && message != static_cast<SchellingActor*>(receiver)->type())
  	  static_cast<SchellingActor*>(receiver)->increaseNumberAdjacent();
}

