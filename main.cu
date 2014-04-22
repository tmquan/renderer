#include <iostream>
#include <fstream>
#include <string>
#include <assert.h>

#include <cuda.h>
#include <opencv2\opencv.hpp>

using namespace std;
using namespace cv;

class Volume
{
public:
	Volume();
	Volume(string, string, int , int, int);
	~Volume();
private:
	string type;
	string filename;
	int dimx;
	int dimy;
	int dimz;
	float *data;
};
Volume::Volume()
{
	this->dimx = 0; 
	this->dimy = 0; 
	this->dimz = 0;
	this->data = NULL;
}

Volume::Volume(string filename, string type, int dimx, int dimy, int dimz)
{
	this->filename =  filename;
	this->dimx = dimx; 
	this->dimy = dimy; 
	this->dimz = dimz;

	if(type=="float")
		this->data = new float[dimx*dimy*dimz];

	fstream fs;
	fs.open(filename, ios::in|ios::binary);								
	assert(!fs.is_open());
																
	fs.read(reinterpret_cast<char*>(data), dimx*dimy*dimz);					
	fs.close();															
}

Volume::~Volume()
{
	delete(this->data);
}

class Renderer
{
public:
	Renderer();
	~Renderer();

	void draw();
	void redraw();
private:
	fstream fs;
	float* data;
	int dimx;
	int dimy;
	int dimz;
};

Renderer::Renderer()
{
}

Renderer::~Renderer()
{
}
void Renderer::draw()
{
	namedWindow("Volume Rendering Viewer");
	namedWindow("Control Panel Viewer");
	namedWindow("Transfer Function Viewer");
}

void Renderer::redraw()
{
	
}
int main(int argc, char** argv)
{
	Renderer renderer;
	Volume volume("bucky.raw", "float", 32, 32, 32);
	renderer.draw();

	waitKey(0);
	return 0;
}