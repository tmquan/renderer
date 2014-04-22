#include <iostream>
#include <fstream>
#include <string>
#include <assert.h>

#include <cuda.h>
#include <helper_math.h>
#include <helper_image.h>
#include <opencv2/opencv.hpp>

using namespace std;
using namespace cv;

#define checkLastError() {                                          				\
	cudaError_t error = cudaGetLastError();                               			\
	int id; 																		\
	cudaGetDevice(&id);																\
	if(error != cudaSuccess) {                                         				\
		printf("Cuda failure error in file '%s' in line %i: '%s' at device %d \n",	\
			__FILE__,__LINE__, cudaGetErrorString(error), id);			      	 	\
		exit(EXIT_FAILURE);  														\
	}                                                               				\
}


#define checkWriteFile(filename, pData, size) {                    				\
		fstream *fs = new fstream;												\
		fs->open(filename, ios::out|ios::binary);								\
		if (!fs->is_open())														\
		{																		\
			fprintf(stderr, "Cannot open file '%s' in file '%s' at line %i\n",	\
			filename, __FILE__, __LINE__);										\
			return 1;															\
		}																		\
		fs->write(reinterpret_cast<char*>(pData), size);						\
		fs->close();															\
		delete fs;																\
	}

	
int width  = 512;
int height = 512;
dim3 blockSize(16, 16);
dim3 gridSize(512/16, 512/16);


typedef unsigned int  uint;
typedef unsigned char uchar;

float density = 0.05f;
float brightness = 1.0f;
float transferOffset = 0.0f;
float transferScale = 1.0f;
bool linearFiltering = true;

cudaExtent volumeSize = make_cudaExtent(32, 32, 32);
cudaArray *d_volumeArray = 0;
cudaArray *d_transferFuncArray;

typedef unsigned char VolumeType;
//typedef unsigned short VolumeType;

texture<VolumeType, 3, cudaReadModeNormalizedFloat> tex;         // 3D texture
texture<float4, 1, cudaReadModeElementType>         transferTex; // 1D transfer function texture




typedef struct
{
    float4 m[3];
} float3x4;

__constant__ float3x4 c_invViewMatrix;  // inverse view matrix

struct Ray
{
    float3 o;   // origin
    float3 d;   // direction
};

// intersect ray with a box
// http://www.siggraph.org/education/materials/HyperGraph/raytrace/rtinter3.htm

__device__
int intersectBox(Ray r, float3 boxmin, float3 boxmax, float *tnear, float *tfar)
{
    // compute intersection of ray with all six bbox planes
    float3 invR = make_float3(1.0f) / r.d;
    float3 tbot = invR * (boxmin - r.o);
    float3 ttop = invR * (boxmax - r.o);

    // re-order intersections to find smallest and largest on each axis
    float3 tmin = fminf(ttop, tbot);
    float3 tmax = fmaxf(ttop, tbot);

    // find the largest tmin and the smallest tmax
    float largest_tmin = fmaxf(fmaxf(tmin.x, tmin.y), fmaxf(tmin.x, tmin.z));
    float smallest_tmax = fminf(fminf(tmax.x, tmax.y), fminf(tmax.x, tmax.z));

    *tnear = largest_tmin;
    *tfar = smallest_tmax;

    return smallest_tmax > largest_tmin;
}

// transform vector by matrix (no translation)
__device__
float3 mul(const float3x4 &M, const float3 &v)
{
    float3 r;
    r.x = dot(v, make_float3(M.m[0]));
    r.y = dot(v, make_float3(M.m[1]));
    r.z = dot(v, make_float3(M.m[2]));
    return r;
}

// transform vector by matrix with translation
__device__
float4 mul(const float3x4 &M, const float4 &v)
{
    float4 r;
    r.x = dot(v, M.m[0]);
    r.y = dot(v, M.m[1]);
    r.z = dot(v, M.m[2]);
    r.w = 1.0f;
    return r;
}

__device__ uint rgbaFloatToInt(float4 rgba)
{
    rgba.x = __saturatef(rgba.x);   // clamp to [0.0, 1.0]
    rgba.y = __saturatef(rgba.y);
    rgba.z = __saturatef(rgba.z);
    rgba.w = __saturatef(rgba.w);
    return (uint(rgba.w*255)<<24) | (uint(rgba.z*255)<<16) | (uint(rgba.y*255)<<8) | uint(rgba.x*255);
}

__global__ void
d_render(uint *d_output, uint imageW, uint imageH,
         float density, float brightness,
         float transferOffset, float transferScale)
{
    const int maxSteps = 500;
    const float tstep = 0.01f;
    const float opacityThreshold = 0.95f;
    const float3 boxMin = make_float3(-1.0f, -1.0f, -1.0f);
    const float3 boxMax = make_float3(1.0f, 1.0f, 1.0f);

    uint x = blockIdx.x*blockDim.x + threadIdx.x;
    uint y = blockIdx.y*blockDim.y + threadIdx.y;

    if ((x >= imageW) || (y >= imageH)) return;

    float u = (x / (float) imageW)*2.0f-1.0f;
    float v = (y / (float) imageH)*2.0f-1.0f;

    // calculate eye ray in world space
    Ray eyeRay;
    eyeRay.o = make_float3(mul(c_invViewMatrix, make_float4(0.0f, 0.0f, 0.0f, 1.0f)));
    eyeRay.d = normalize(make_float3(u, v, -2.0f));
    eyeRay.d = mul(c_invViewMatrix, eyeRay.d);

    // find intersection with box
    float tnear, tfar;
    int hit = intersectBox(eyeRay, boxMin, boxMax, &tnear, &tfar);

    if (!hit) return;

    if (tnear < 0.0f) tnear = 0.0f;     // clamp to near plane

    // march along ray from front to back, accumulating color
    float4 sum = make_float4(0.0f);
    float t = tnear;
    float3 pos = eyeRay.o + eyeRay.d*tnear;
    float3 step = eyeRay.d*tstep;

    for (int i=0; i<maxSteps; i++)
    {
        // read from 3D texture
        // remap position to [0, 1] coordinates
        float sample = tex3D(tex, pos.x*0.5f+0.5f, pos.y*0.5f+0.5f, pos.z*0.5f+0.5f);
        //sample *= 64.0f;    // scale for 10-bit data
		// if(sample > 0.0f) 
		// {
			// printf("Here: %d %d\n", x, y);
			// return;
		// }
		
        // lookup in transfer function texture
        float4 col = tex1D(transferTex, (sample-transferOffset)*transferScale);
        col.w *= density;

        // "under" operator for back-to-front blending
        // sum = lerp(sum, col, col.w);

        // pre-multiply alpha
        col.x *= col.w;
        col.y *= col.w;
        col.z *= col.w;
        // "over" operator for front-to-back blending
        sum = sum + col*(1.0f - sum.w);

        // exit early if opaque
        if (sum.w > opacityThreshold)
            break;

        t += tstep;

        if (t > tfar) break;

        pos += step;
    }

    sum *= brightness;

    // write output color
    d_output[y*imageW + x] = rgbaFloatToInt(sum);
	
	if(rgbaFloatToInt(sum)>0) printf("Here: %d %d\n", x, y);
	// if(sample>0) printf("Here: %d %d\n", x, y);
}




int main(int argc, char** argv)
{
	
	Mat screen =  Mat(Size(width, height), CV_8UC4, Scalar::all(128));
	Mat image =  Mat(Size(width, height), CV_32FC1);
	//////////////////////////////////////////////////////////
	// string filename = "/home/tmquan/renderer/bucky.raw";
	char *filename = argv[1];
	cout << filename <<endl;
	// cout << volumeSize.depth << " " << volumeSize.height << " " << volumeSize.width << endl;
	uchar *h_volume = new uchar[volumeSize.depth*volumeSize.height*volumeSize.width];
	fstream fs;
	fs.open(filename, ios::in|ios::binary);								
	assert(fs.is_open()); 
	// if(!fs.is_open())
	// {
		// cout << "Cannot open the file" << endl;
		// return 1;
	// }

	fs.read(reinterpret_cast<char*>(h_volume), volumeSize.depth*volumeSize.height*volumeSize.width);					
	fs.close();		
	
	checkWriteFile("test.raw", h_volume, volumeSize.depth*volumeSize.height*volumeSize.width);
	//////////////////////////////////////////////////////////
	// initCuda(void *h_volume, cudaExtent volumeSize)
	// create 3D array
    cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc<VolumeType>(); checkLastError();
    cudaMalloc3DArray(&d_volumeArray, &channelDesc, volumeSize); checkLastError();

    // copy data to 3D array
    cudaMemcpy3DParms copyParams = {0};
    copyParams.srcPtr   = make_cudaPitchedPtr(h_volume, volumeSize.width*sizeof(VolumeType), volumeSize.width, volumeSize.height);
    copyParams.dstArray = d_volumeArray;
    copyParams.extent   = volumeSize;
    copyParams.kind     = cudaMemcpyHostToDevice;
    cudaMemcpy3D(&copyParams); checkLastError();

    // set texture parameters
    tex.normalized = true;                      // access with normalized texture coordinates
    tex.filterMode = cudaFilterModeLinear;      // linear interpolation
    tex.addressMode[0] = cudaAddressModeClamp;  // clamp texture coordinates
    tex.addressMode[1] = cudaAddressModeClamp;

    // bind array to 3D texture
    cudaBindTextureToArray(tex, d_volumeArray, channelDesc); checkLastError();

    // create transfer function texture
    float4 transferFunc[] =
    {
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

    cudaChannelFormatDesc channelDesc2 = cudaCreateChannelDesc<float4>();
    cudaArray *d_transferFuncArray;
    cudaMallocArray(&d_transferFuncArray, &channelDesc2, sizeof(transferFunc)/sizeof(float4), 1);
    cudaMemcpyToArray(d_transferFuncArray, 0, 0, transferFunc, sizeof(transferFunc), cudaMemcpyHostToDevice);

	checkLastError();
    transferTex.filterMode = cudaFilterModeLinear;
    transferTex.normalized = true;    // access with normalized texture coordinates
    transferTex.addressMode[0] = cudaAddressModeClamp;   // wrap texture coordinates

    // Bind the array to the texture
    cudaBindTextureToArray(transferTex, d_transferFuncArray, channelDesc2);
	
	checkLastError();
	//////////////////////////////////////////////////////////
	uint *d_output;
    cudaMalloc((void **)&d_output, width*height*sizeof(uint));
    cudaMemset(d_output, 65535, width*height*sizeof(uint));

	// uchar *h_output = (uchar*)malloc(width*height*4);
	unsigned char *h_output = (unsigned char *)malloc(width*height*4);
	// h_output = screen.ptr<uchar4>();
	uchar4 *ptrScreen = screen.ptr<uchar4>();
	
    // float modelView[16] =
    // {
        // 1.0f, 0.0f, 0.0f, 0.0f,
        // 0.0f, 1.0f, 0.0f, 0.0f,
        // 0.0f, 0.0f, 1.0f, 0.0f,
        // 0.0f, 0.0f, 4.0f, 1.0f
    // };

	// float invViewMatrix[12];
    // invViewMatrix[0] = modelView[0];
    // invViewMatrix[1] = modelView[4];
    // invViewMatrix[2] = modelView[8];
    // invViewMatrix[3] = modelView[12];
    // invViewMatrix[4] = modelView[1];
    // invViewMatrix[5] = modelView[5];
    // invViewMatrix[6] = modelView[9];
    // invViewMatrix[7] = modelView[13];
    // invViewMatrix[8] = modelView[2];
    // invViewMatrix[9] = modelView[6];
    // invViewMatrix[10] = modelView[10];
    // invViewMatrix[11] = modelView[14];
	//////////////////////////////////////////////////////////
	imshow("Volume Rendering Viewer", screen);
	waitKey(0);
	while(1)
	{
		// Render here
		// render_kernel(gridSize, blockSize, d_output, width, height, density, brightness, transferOffset, transferScale);
		d_render<<<gridSize, blockSize>>>(d_output, width, height, density,
                                      brightness, transferOffset, transferScale);
		// Copy to host
		cudaMemcpy(h_output, d_output, width*height*4, cudaMemcpyDeviceToHost);
		// Re address pointer
		memcpy(ptrScreen, h_output, width*height*4);
		
		// sdkSavePPM4ub("volume.ppm", h_output, width, height);
		// screen = Mat(width, height, CV_8UC4, h_output, CV_AUTOSTEP); // does not copy
		// screen.ptr<uchar4>() = h_output;
		// for(int y=0; y<512; y++)
		// {
			// for(int x=0; x<512; x++)
			// {
				// screen.at<Vec4b>(x, y)[0] = screen.at<Vec4d>(x, y)[0]*255;
				// screen.at<Vec4b>(x, y)[1] = screen.at<Vec4d>(x, y)[1]*255;
				// screen.at<Vec4b>(x, y)[2] = screen.at<Vec4d>(x, y)[2]*255;
			// }
		// }
		// int size = w * h;
		// unsigned char *ndata = (unsigned char *) malloc(sizeof(unsigned char) * size*3);
		// unsigned char *ptr = (uchar*)reinterpret_cast<uchar4>(screen.ptr<uchar4>());
		// uchar
		// for (int i=0; i<size; i++)
		// {
			// *ptr++ = *data++;
			// *ptr++ = *data++;
			// *ptr++ = *data++;
			// data++;
		// }		
		
		// Display
		imshow("Volume Rendering Viewer", screen);
		
		char key = waitKey(1);
		if( key == 27 || key == 'Q' || key == 'q' )	break;
  
	}

	cudaFree(d_output);
    free(h_output);

	return 0;
}

