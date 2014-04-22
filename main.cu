#include <iostream>
#include <fstream>
#include <string>
#include <assert.h>

#include <cuda.h>
#include <opencv2/opencv.hpp>

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
	fs.open(filename.c_str(), ios::in|ios::binary);								
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

struct RenderPara
{
	//camera parameters 
	unsigned int width;
	unsigned int height;
	float focal;

	//view matrix
	float viewMatrix[4 * 4];
	
	//model matrix
	float modelMatrix[4 * 4];
	
	//box parameters, in local coordinate system
	float boxMin[3];
	float boxSize[3];


};

void updateAllPara(struct RenderPara* renderPara, struct RayCastAllPara* allPara);

void initRenderParas(struct RenderPara* renderPara);
void rotateModel(struct RenderPara* renderPara);

const unsigned int win_width = 512;
const unsigned int win_height = 512;
const float focal = (float)win_width* 8.0f; 



struct RayCastAllPara
{
	/*window's width and height, focus is the center*/
	unsigned int width;
	unsigned int height;
	float focal;

	/*eye position in world coordinate system*/
	float eyeo[3];

	/*modelview matrix and its invert*/
	float MVmatrix[4 * 4];
	float c_invMatrix[4 * 4];
	

	/*bounding box: min and max, and size = max - min, in local coordinate system*/
	float boundingBoxMin[3];
	float boundingBoxMax[3];
	float boxSize[3];

	
};

struct VolumeData
{
	unsigned int xsize;
	unsigned int ysize;
	unsigned int zsize;
	float *data;
	float density;
	float opacityThreshold;
};

struct Ray
{
	float ori[3];
	float dir[3];
};

struct IntersectRes
{
	int flag;
	float tnear;
	float tfar;
};

/*return 0 correct*/
/*return 1 not hit
/*return 2 hit tfar < 0
/*return 3 hit tnear < 0
/*out of steps
/**/
int computeRayCast(struct RayCastAllPara* allPara,struct VolumeData* vdata, unsigned int w, unsigned int h, float* r, float* g, float* b, float* a);

/*0.0f <= fx,fy,fz <= 1.0f */
int getVolumeData(struct VolumeData* vdata, float fx, float fy, float fz, float* elmt);

/*0.0f < input < 1.0f. ouput is a color*/
int getTransferTex(float input, float output[4]);

struct IntersectRes intersectBox(struct Ray r, float boxmin[], float boxmax[]);

int normalizef3(float n[]); 

float _min(float x,float y);
float _max(float x,float y);
float _mid(float x,float y,float z);

bool initVolumeData(struct VolumeData* vdata,char file[])
{
	char filename[] = "../bucky.raw";
	const int size = 32;
	vdata->xsize = size;
	vdata->ysize = size;
	vdata->zsize = size;
	vdata->density = 0.02f;
	vdata->opacityThreshold = 0.95f;
	int m_size = vdata->xsize * vdata->ysize * vdata->zsize;
	vdata->data = new float[m_size];
	
	strcpy(filename,file);
	FILE *fp = fopen(filename, "rb");
    if (!fp) {
        fprintf(stderr, "Error opening file '%s'\n", filename);
        return false;
    }

	unsigned char* data = (unsigned char*)malloc(m_size);
	size_t read = fread(data, 1, m_size, fp);
	for(int i = 0;i < m_size;i++)
		vdata->data[i] = (float)(data[i]) / 255.0f;

	free(data);
	fclose(fp);
	return true;

}






int main(int argc, char** argv)
{
	Renderer renderer;
	Volume volume("bucky.raw", "float", 32, 32, 32);
	
	struct VolumeData vdata;
	Mat screen =  Mat(Size(512, 512), CV_8UC3);
	
	struct RayCastAllPara allPara;
	struct RenderPara renderPara;
	
	initVolumeData(&vdata,"../bucky.raw");
	initRenderParas(&renderPara);
	
	
	while(1)
	{
		rotateModel(&renderPara);
		updateAllPara(&renderPara,&allPara);
		
		renderer.draw();
		
		// IplImage* rc_im = cvCreateImage(cvSize(win_width,win_height),IPL_DEPTH_8U,3);
		for(int h = 0;h < win_height;h++)
		{
			for(int w = 0;w < win_width;w++)
			{	
				float r,g,b,a;
				computeRayCast(&allPara,&vdata,w,h,&r,&g,&b,&a);
				// cvSet2D(rc_im,h,w,cvScalar((double)b * 255,(double)g * 255,(double)r * 255));
				// cvSet2D(screen,h,w,cvScalar((double)b * 255,(double)g * 255,(double)r * 255));
				screen.at<Vec3b>(h, w)[0] = 255*b;
				screen.at<Vec3b>(h, w)[1] = 255*g;
				screen.at<Vec3b>(h, w)[2] = 255*r;
			}
		}
		if(cvGetWindowHandle("Volume Rendering Viewer") == 0)
			cvNamedWindow("Volume Rendering Viewer");
		// cvShowImage("Volume Rendering Viewer",rc_im);
		// cvShowImage("Volume Rendering Viewer", screen);
		imshow("Volume Rendering Viewer", screen);
		// cvWaitKey(1);
		// cvReleaseImage(&rc_im);
		
		char key = waitKey(1);
		if( key == 27 || key == 'Q' || key == 'q' )	break;
  
	}


	return 0;
}



void initRenderParas(struct RenderPara* renderPara)
{
	float modelMatrix[16] = {   1.0f, 0.0f, 0.0f, 0.0f,
								0.0f, 1.0f, 0.0f, 0.0f,
								0.0f, 0.0f, 1.0f, 0.0f,
								0.0f, 0.0f, 0.0f, 1.0f};
	float pi = 3.1415926f;
	float viewMatrix[16] = {   cos(0.0f), 0.0f, sin(0.0f), 0.0f,
								0.0f, 1.0f, 0.0f, 0.0f,
								-sin(0.0f), 0.0f, cos(0.0f), 16.0f,
								0.0f, 0.0f, 0.0f, 1.0f};

	renderPara->width = win_width;
	renderPara->height = win_height;
	renderPara->focal = focal;
	renderPara->boxMin[0] = -1.0f;
	renderPara->boxMin[1] = -1.0f;
	renderPara->boxMin[2] = -1.0f;
	renderPara->boxSize[0] = 2.0f;
	renderPara->boxSize[1] = 2.0f;
	renderPara->boxSize[2] = 2.0f;
	
	memcpy(renderPara->modelMatrix, modelMatrix, sizeof(float) * 16);
	memcpy(renderPara->viewMatrix, viewMatrix, sizeof(float) * 16);
}


bool MatrixMul(float* mat1,int row1,int col1,float* mat2,int row2,int col2,float* out)
{
	if(col1 != row2)
		return false;

	float* tmpMat = new float[row1 * col2];
	for(int i = 0;i < row1;i++)
	{
		for(int j = 0;j < col2;j++)
		{
			float tmp = 0.0f;
			for(int k = 0; k < col1;k++)
				tmp += mat1[i * col1 + k] * mat2[k * col2 + j];
			tmpMat[i * col2 + j] = tmp;
		}
	}
	for(int i = 0;i < row1;i++)
	{
		for(int j = 0;j < col2;j++)
			out[i * col2 + j] = tmpMat[i * col2 + j];
	}
	delete []tmpMat;
	return true;
}


bool MatrixInverse(float * src,int dims,float * dst)
{
	float * a = new float[dims * dims];
	int n = dims;
	for(int i = 0;i < dims;i++)
	{
		for(int j = 0;j < dims;j++)
			a[i * dims + j] = src[i * dims + j];
	}


	int *is, *js, i, j, k, l, u, v;
    float d, p;
    is=(int *)malloc(n * sizeof(int));
    js=(int *)malloc(n * sizeof(int));
    for(k = 0; k <= n - 1;k++)
	{ 
		d = 0.0;
        for(i = k;i <= n - 1;i++)
		{
			for(j = k;j <= n - 1;j++)
			{
				l = i * n + j;
				p = fabs(a[l]);
				if(p > d)
				{
					d = p;
					is[k] = i;
					js[k] = j;
				}
			}
		}
		if(d+1.0==1.0)
		{
			free(is);
			free(js);
			//printf("err**not inv\n");
            return (0);
        }
        if(is[k] != k)
		{
			for(j = 0;j <= n - 1;j++)
			{
				u = k * n + j;
				v = is[k] * n + j;
				p = a[u];
				a[u] = a[v];
				a[v] = p;
			}
		}
        if (js[k]!=k)
		{
			for(i = 0;i <= n - 1;i++)
			{
				u = i * n + k;
				v = i * n + js[k];
				p = a[u];
				a[u] = a[v];
				a[v] = p;
            }
		}
        l = k * n + k;
        a[l] = 1.0f / a[l];
        for(j = 0;j <= n - 1;j++)
		{
			if(j != k)
			{
				u = k * n + j;
				a[u] = a[u] * a[l];
			}
		}
        for(i = 0;i <= n - 1;i++)
		{
			if(i != k)
			{
				for(j = 0;j <= n - 1;j++)
				{
					if (j != k)
					{ 
						u = i * n + j;
						a[u] = a[u] - a[i * n + k] * a[k * n + j];
					}
                }
			}
		}
        for(i = 0;i <= n - 1;i++)
		{
			if(i != k)
			{
				u = i * n + k;
				a[u] = -a[u] * a[l];
			}
		}
	}
    for(k = n - 1;k >= 0;k--)
	{
		if(js[k] != k)
		{
			for(j = 0;j <= n - 1;j++)
			{
				u = k * n + j;
				v = js[k] * n + j;
				p = a[u];
				a[u] = a[v];
				a[v] = p;
            }
		}
        if(is[k] != k)
		{
			for(i = 0;i <= n-1;i++)
			{ 
				u = i * n + k;
				v = i * n + is[k];
				p = a[u];
				a[u] = a[v];
				a[v] = p;
			}
		}
	}
    free(is);
	free(js);
	for(int i = 0;i < dims;i++)
	{
		for(int j = 0;j < dims;j++)
			dst[i * dims + j] = a[i * dims + j];
	}
	delete []a;
    return (1);
}

int computeRayCast(struct RayCastAllPara* allpara,struct VolumeData* vdata, unsigned int w, unsigned int h, float* r, float* g, float* b,float* a)
{
	int i,istep;

	float volumeElmt;

	float boxMin[3],boxMax[3];
	float x,y,u,v,focal;

	struct Ray eyeRay;
	float eyedir4[4];

	struct IntersectRes hit;
	float tnear,tfar,t;
	
	int maxSteps = 200;
	float density = vdata->density;
	float opacityThreshold = vdata->opacityThreshold;
	float tstep = _max(_max(allpara->boxSize[0],allpara->boxSize[1]),_max(allpara->boxSize[0],allpara->boxSize[2])) / (maxSteps * 1.732);

	float pos[3],step[3],m_coord[3];
	float sum[4] ={0},transferCol[4];

	// Reset the color before modifying
	*r = *g = *b = *a =  0.0f;
	
	// Set bounding box min and max
	for(i = 0;i < 3;i++)
	{
		boxMin[i] = allpara->boundingBoxMin[i];
		boxMax[i] = allpara->boundingBoxMax[i];
	}

	// Set current location
    x = w;
    y = h;
	

    u = ((x - allpara->width / 2.0f) / (float)(allpara->height))*2.0f;
    v = (1.0f - y / (float)(allpara->height))*2.0f-1.0f;
	focal = allpara->focal / ((float)(allpara->height));

	for(i = 0;i < 3;i++)
		eyeRay.ori[i] = allpara->eyeo[i];
	
	eyeRay.dir[0] = u;
	eyeRay.dir[1] = v;
	eyeRay.dir[2] = focal;
	normalizef3(eyeRay.dir);
	eyedir4[0] = eyeRay.dir[0]; 
	eyedir4[1] = eyeRay.dir[1]; 
	eyedir4[2] = eyeRay.dir[2]; 
	eyedir4[3] = 0.0f;

	MatrixMul(allpara->c_invMatrix,4,4,eyedir4,4,1,eyedir4);
	eyeRay.dir[0] = eyedir4[0]; eyeRay.dir[1] = eyedir4[1]; eyeRay.dir[2] = eyedir4[2];
  
    // find intersection with box
	hit = intersectBox(eyeRay, boxMin, boxMax);
	
    if (hit.flag == 0) 
	{
	
		*r = 0.0f;
		*g = 0.0f;
		*b = 0.0f;
		return 1;
	}
	else
	{
	
	}

	if(hit.tfar < 0.0f)
	{
	
		*r = 0.0f;
		*g = 0.0f;
		*b = 0.3f;
		return 2;
	}
	if (hit.tnear < 0.0f) {
		hit.tnear = 0.0f;     // clamp to near plane
	
		*r = 0.5f;
		*g = 0.0f;
		*b = 0.0f;
		return 3;
	}
	
	tnear = hit.tnear;
	tfar = hit.tfar;
	
	// march along ray from front to back, accumulating color
	t = tnear;
	
	for(i = 0;i < 3;i++)
	{
	    pos[i] = eyeRay.ori[i] + eyeRay.dir[i]*tnear;
		step[i] = eyeRay.dir[i]*tstep;
	}

    
	for(i = 0;i < 3 ;i++ )
		m_coord[i] = (pos[i] - allpara->boundingBoxMin[i]) / allpara->boxSize[i] ;
	
	

    for(istep = 0; istep < maxSteps; istep++) {
		for(i = 0;i < 3;i++)
			m_coord[i] = (pos[i] - allpara->boundingBoxMin[i]) / allpara->boxSize[i] ;

        volumeElmt = 0.0f;
		getVolumeData(vdata,m_coord[0],m_coord[1],m_coord[2],&volumeElmt);

		//if(volumeElmt <= target)
		//{
		//	*r = 1.0f;
		//	*g = 1.0f;
		//	*b = 1.0f;
		//	return 0;
		//}

		getTransferTex(volumeElmt,transferCol);
		for(i = 0;i < 4;i++)
		{
			sum[i] = sum[i] + transferCol[i]*(1.0f - sum[3]) * density;
		}
			
		if (sum[3] > opacityThreshold)
		{
			*r = sum[0];
			*g = sum[1];
			*b = sum[2];
			*a = sum[3];
            break;
		}

		t += tstep;

        if (t > hit.tfar)
		{
			*r = sum[0];
			*g = sum[1];
			*b = sum[2];
			*a = sum[3];
			break;
		}

		for(i = 0;i < 3;i++)
			pos[i] += step[i];
    }
    
	*r = sum[0];
	*g = sum[1];
	*b = sum[2];
	*a = sum[3];
	return 4;
}

void rotateModel(struct RenderPara *renderPara)
{
	const float angle = 3.1415926f / 10.0f;
	float rotx[16] = {1.0f, 0.0f, 0.0f, 0.0f,
					  0.0f, cos(angle), sin(angle), 0.0f,
					  0.0f,-sin(angle), cos(angle), 0.0f,
					  0.0f, 0.0f, 0.0f, 1.0f};
	float roty[16] = {cos(angle), 0.0f, sin(angle), 0.0f,
					  0.0f, 1.0f, 0.0f, 0.0f,
					  -sin(angle), 0.0f, cos(angle), 0.0f,
					  0.0f, 0.0f, 0.0f, 1.0f};
	float rotz[16] = {cos(angle), sin(angle), 0.0f, 0.0f,
					  -sin(angle), cos(angle), 0.0f, 0.0f,
					  0.0f, 0.0f, 1.0f, 0.0f,
					  0.0f, 0.0f, 0.0f, 1.0f};

	static int step = 10;
	static int m_rand = 0;
	if(step >= 7)
	{
		m_rand = rand()%3;
		step = 0;
	}
	step++;
	if(m_rand == 0)
	{
		MatrixMul(renderPara->modelMatrix,4,4,rotx,4,4,renderPara->modelMatrix);
	}
	else if(m_rand == 1)
	{
		MatrixMul(renderPara->modelMatrix,4,4,roty,4,4,renderPara->modelMatrix);
	}
	else
	{
		MatrixMul(renderPara->modelMatrix,4,4,rotz,4,4,renderPara->modelMatrix);
	}
}


void updateAllPara(struct RenderPara* renderPara, struct RayCastAllPara* allPara)
{
	int i,j;

	allPara->width = renderPara->width;
	allPara->height = renderPara->height;
	allPara->focal = renderPara->focal;

	for(i = 0;i < 3;i++)
		allPara->boundingBoxMin[i] = renderPara->boxMin[i];
	for(i = 0;i < 3;i++)
		allPara->boxSize[i] = renderPara->boxSize[i];
	for(i = 0;i < 3;i++)
		allPara->boundingBoxMax[i] = allPara->boundingBoxMin[i] + allPara->boxSize[i];
	
	MatrixMul(renderPara->viewMatrix, 4, 4, renderPara->modelMatrix, 4, 4, allPara->MVmatrix);
	MatrixInverse(allPara->MVmatrix, 4 , allPara->c_invMatrix);
	
	for(i = 0;i < 3;i++)
		allPara->eyeo[i] = allPara->c_invMatrix[i * 4 + 3];

	
}


float _min(float x, float y)
{
	return x < y ? x : y; 
}

float _max(float x, float y)
{
	return x > y ? x : y;
}

float _mid(float x, float y, float z)
{
	if(x <= y && y <= z)
		return y;
	else if(x <= z && z <= y)
		return z;
	else if(y <= x && x <= z)
		return x;
	else if(y <= z && z <= x)
		return z;
	else if(z <= x && x <= y)
		return x;
	else 
		return y;
}



int getVolumeData(struct VolumeData* vdata, float fx, float fy, float fz, float* elmt)
{
	int xsize = vdata->xsize;
	int ysize = vdata->ysize;
	int zsize = vdata->zsize;
	float fX = fx * (vdata->xsize-1);
	float fY = fy * (vdata->ysize-1);
	float fZ = fz * (vdata->zsize-1);

	int x = (int)fX;
	int y = (int)fY;
	int z = (int)fZ;

	float offx = fX - x;
	float offy = fY - y;
	float offz = fZ - z;

	float w000 = (1.0f - offx) * (1.0f - offy) * (1.0f - offz);
	float w100 =		 offx  * (1.0f - offy) * (1.0f - offz);
	float w010 = (1.0f - offx) *	     offy  * (1.0f - offz);
	float w110 =		 offx  *		 offy  * (1.0f - offz);
	float w001 = (1.0f - offx) * (1.0f - offy) *		 offz;
	float w101 =		 offx  * (1.0f - offy) *		 offz;
	float w011 = (1.0f - offx) *		 offy  *	     offz;
	float w111 =		 offx  *		 offy  *		 offz;

	float d000,d100,d010,d110,d001,d101,d011,d111;
		d000 = vdata->data[ x    * ysize * zsize +  y    * zsize +  z   ];
	
	if(x+1 < xsize)
		d100 = vdata->data[(x+1) * ysize * zsize +  y    * zsize +  z   ];
	else 
		d100 = 0.0f;
	
	if(y+1 < ysize)
		d010 = vdata->data[ x    * ysize * zsize + (y+1) * zsize +  z   ];
	else
		d010 = 0.0f;
	
	if(x+1 < xsize && y+1 < ysize)
		d110 = vdata->data[(x+1) * ysize * zsize + (y+1) * zsize +  z   ];
	else
		d110 = 0.0f;

	if(z+1 < zsize)
		d001 = vdata->data[ x    * ysize * zsize +  y    * zsize + (z+1)];
	else
		d001 = 0.0f;

	if(x+1 < xsize && z+1 < zsize)
		d101 = vdata->data[(x+1) * ysize * zsize +  y    * zsize + (z+1)];
	else
		d101 = 0.0f;

	if(y+1 < ysize && z+1 < zsize)
		d011 = vdata->data[ x    * ysize * zsize + (y+1) * zsize + (z+1)];
	else
		d011 = 0.0f;

	if(x+1 < xsize && y+1 < ysize && z+1 < zsize)	
		d111 = vdata->data[(x+1) * ysize * zsize + (y+1) * zsize + (z+1)];
	else
		d111 = 0.0f;
	
	*elmt = w000 * d000 + w100 * d100 + w010 * d010 + w110 * d110 + w001 * d001 + w101 * d101 + w011 * d011 + w111 * d111; 
	return 1;
}

int getTransferTex(float input, float output[4])
{
	const int transferSize = 9;
	float transferFunc[9][4] = {
        {  0.0, 0.0, 0.0, 0.0, },
        {  1.0, 0.0, 0.0, 1.0, },
        {  1.0, 0.5, 0.0, 1.0, },
        {  1.0, 1.0, 0.0, 1.0, },
        {  0.0, 1.0, 0.0, 1.0, },
        {  0.0, 1.0, 1.0, 1.0, },
        {  0.0, 0.0, 1.0, 1.0, },
        {  1.0, 0.0, 1.0, 1.0, },
        {  0.0, 0.0, 0.0, 0.0, },
    };

	int i;
	float m_input = _mid(0.0f,input,1.0f) * (transferSize - 1);
	
	int m_IN = (int)m_input;
	float offset = m_input - m_IN;

	float w0 = 1.0f - offset;
	float w1 = offset;

	if(m_IN == transferSize - 1)
	{
		for(i = 0;i < 4;i++)
			output[i] = transferFunc[m_IN][i];
	}
	else
	{
		for(i = 0;i < 4;i++)
		{
			output[i] = w0 * transferFunc[m_IN][i] + w1 * transferFunc[m_IN+1][i];
		}
	}
	return 1;
}

	
struct IntersectRes intersectBox(struct Ray r, float boxmin[], float boxmax[])
{
	int i;
	struct IntersectRes res = {0,0,0}; 
	const float maxfloat = 10000000000000000000000.0f;
	const float minfloat = 1.0f / maxfloat;

	float largest_tmin,smallest_tmax;

    // compute intersection of ray with all six bbox planes
	float invR[3],tmin[3],tmax[3];
	for(i = 0;i < 3;i++)
	{
		r.dir[i] = fabs(r.dir[i]) > minfloat ? r.dir[i] : minfloat;
		invR[i] = 1.0f / r.dir[i];
	}

	for(i = 0;i < 3;i++)
	{
		if(invR[i] >= 0)
		{
			tmin[i] = invR[i] * (boxmin[i] - r.ori[i]);
			tmax[i] = invR[i] * (boxmax[i] - r.ori[i]);	
		}
		else
		{
			tmin[i] = invR[i] * (boxmax[i] - r.ori[i]);
			tmax[i] = invR[i] * (boxmin[i] - r.ori[i]);	
		}
	}
	
	 

    // find the largest tmin and the smallest tmax
    largest_tmin = _max(_max(tmin[0], tmin[1]), _max(tmin[0], tmin[2]));
	smallest_tmax = _min(_min(tmax[0], tmax[1]), _min(tmax[0], tmax[2]));

	res.tnear = largest_tmin;
	res.tfar = smallest_tmax;
	res.flag = (smallest_tmax > largest_tmin) ? 1 : 0;
	return res;

}

int normalizef3(float n[])
{
	float len = sqrt(n[0] * n[0] + n[1] * n[1] + n[2] * n[2]);

	const float minfloat = 0.0000001f;
	if(len > minfloat)
	{
		n[0] /= len;
		n[1] /= len;
		n[2] /= len;
		return 1;
	}
	else
	{
		n[0] = 1.0f;
		n[1] = 0.0f;
		n[2] = 0.0f;
		return 0;
	}

	
}