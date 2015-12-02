#include "SchellingActor.hxx"

__device__ __host__ SchellingActor::SchellingActor(unsigned char type){
	m_type = type;
}

__device__ __host__ SchellingActor::~SchellingActor(){

}

__device__ void SchellingActor::react() {
	for (int i = 1; i < 257; i <<= 1) {
		if (atomicXOR(&m_type, i) >> i - 1 == 0)
			increaseNumberAdjacent();
	}
}
