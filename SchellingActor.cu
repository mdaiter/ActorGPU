#include "SchellingActor.hxx"

__device__ __host__ SchellingActor::SchellingActor(){
	m_type = 'a';
}

__device__ __host__ SchellingActor::~SchellingActor(){

}

__device__ void SchellingActor::react() {
	m_type = 'b';
}
