#ifndef SCENE_H
#define SCENE_H

class Scene {
	public:
		__host__ __device__ Scene(unsigned int, unsigned int);
		__host__ __device__ ~Scene();
		
	private:
		unsigned Actor* m_map;
		unsigned int m_width;
		unsigned int m_height;
};

#endif
