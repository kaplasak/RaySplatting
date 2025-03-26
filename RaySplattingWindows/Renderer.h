#pragma once

#include <Windows.h>

// !!! !!! !!!
#include "C3DScene.h"
// !!! !!! !!!

// *** *** *** *** ***

struct SCamera {
	float Ox; float Oy; float Oz;
	float Rx; float Ry; float Rz;
	float Dx; float Dy; float Dz;
	float Fx; float Fy; float Fz;
};

// *** *** *** *** ***

struct SGaussianComponent {
	float mX, mY, mZ;

	float qr, qi, qj, qk;
	float sX, sY, sZ;

	float A11, A12, A13;
	float A21, A22, A23;
	float A31, A32, A33;

	float R, G, B;
	float alpha;
};

// *** *** *** *** ***

struct SLBVHTreeNode {
	char info;
	int lNode, rNode;
	float lB, rB;
	float uB, dB;
	float bB, fB;
};

// *** *** *** *** ***

float RandomFloat();
unsigned RandomInteger();
double RandomDouble();
void RandomNormalFloat(float& Z1, float& Z2);
void RandomNormalDouble(double& Z1, double& Z2);

unsigned long long InterleaveBits(
	unsigned long long x,
	unsigned long long y,
	unsigned long long z
);

void GetGaussianComponentAABB(
	SGaussianComponent& GC,
	float threshold,
	float& lB, float& rB,
	float& uB, float& dB,
	float& bB, float& fB
);

bool BuildLBVHTree(
	SGaussianComponent* GC, int size, int numberOfBits,
	SLBVHTreeNode*& LBVHTree,
	int*& d,
	int& D, int& H
);

bool IntersectRayAABB(
	float Ox, float Oy, float Oz,
	float vx, float vy, float vz,
	float lB, float rB, float uB, float dB, float bB, float fB,
	float& tHit1, float& tHit2
);

bool IntersectRayGaussianComponent(
	float Ox, float Oy, float Oz,
	float vx, float vy, float vz,
	SGaussianComponent& GC,
	float& tHit1, float& tHit2
);

// *************************************************************************************************

struct SRenderParams {
	float Ox; float Oy; float Oz;
	float Rx; float Ry; float Rz;
	float Dx; float Dy; float Dz;
	float Fx; float Fy; float Fz;
	float double_tan_half_fov_x;
	float double_tan_half_fov_y;
	void* bitmap;
	int w; int h;
	SLBVHTreeNode* tree;
	SGaussianComponent* GC;
	int numberOfGaussians;

	int* d;
	int D;
	int H;

	int threadsNum;
	int threadId;
	bool volumetric;

	// !!! !!! !!!
	SCamera *poses;
	unsigned *bitmap_ref;
	int poseNum;
	int epoch;
	int NUMBER_OF_POSES;
	double loss;

	void *dL_dparams_1;
	void *dL_dparams_2;
	void *dL_dparams_3;
	void *dL_dparams_4;

	void *m1, *m2, *m3, *m4;
	void *v1, *v2, *v3, *v4;

	int *dump;
	// !!! !!! !!!

	// *** *** *** *** ***

	typedef int cufftHandle;
	typedef void COMPLEX;
	typedef void REAL;

	// !!! !!! !!!
	cufftHandle planr2c;
	cufftHandle planc2r;

	COMPLEX *dev_F_1;
	COMPLEX *dev_F_2;

	REAL *dev_bitmap_ref_R;
	REAL *dev_bitmap_ref_G;
	REAL *dev_bitmap_ref_B;
	REAL *dev_mu_bitmap_ref_R;
	REAL *dev_mu_bitmap_ref_G;
	REAL *dev_mu_bitmap_ref_B;
	REAL *dev_mu_bitmap_out_bitmap_ref_R;
	REAL *dev_mu_bitmap_out_bitmap_ref_G;
	REAL *dev_mu_bitmap_out_bitmap_ref_B;
	REAL *dev_mu_bitmap_ref_R_square;
	REAL *dev_mu_bitmap_ref_G_square;
	REAL *dev_mu_bitmap_ref_B_square;

	REAL *dev_mu_bitmap_out_R;
	REAL *dev_mu_bitmap_out_G;
	REAL *dev_mu_bitmap_out_B;
	REAL *dev_mu_bitmap_out_R_square;
	REAL *dev_mu_bitmap_out_G_square;
	REAL *dev_mu_bitmap_out_B_square;
	// !!! !!! !!!
};

// *************************************************************************************************

struct SCUDARenderParams {
	float Ox; float Oy; float Oz;
	float Rx; float Ry; float Rz;
	float Dx; float Dy; float Dz;
	float Fx; float Fy; float Fz;
	float FOV;

	void* bitmap;
	int w; int h;
	
	void* tree_part_1; // info (2 most significant bits of lNode), lNode, rNode, lB, rB
	void* tree_part_2; // uB, dB, bB, fB

	void* GC_part_1; // mX, mY, mZ, A11
	void* GC_part_2; // A12, A13, A21, A22
	void* GC_part_3; // A23, A31, A32, A33
	void* GC_part_4; // R, G, B, alpha

	void* d;
	int D;
	int H;

	bool render; // !!! !!! !!!
	bool limit; // !!! !!! !!!

	// !!! !!! !!!
	int* bitmap_ref;
	int poseNum;
	void *dL_dparams_1;
	void *dL_dparams_2;
	void *dL_dparams_3;
	void *dL_dparams_4;
	void *output_params;

	int *dump;
	// !!! !!! !!!

	// *** *** *** *** ***

	typedef int cufftHandle;
	typedef void COMPLEX;
	typedef void REAL;

	// !!! !!! !!!
	REAL *dev_bitmap_out_R, *dev_bitmap_out_G, *dev_bitmap_out_B;
	REAL *dev_mu_bitmap_out_R, *dev_mu_bitmap_out_G, *dev_mu_bitmap_out_B;
	// !!! !!! !!!
};

// *************************************************************************************************

typedef struct {
	float x;
	float y;
	float z;
} float3;

struct SOptiXRenderParams {
	// RENDERER
	void *optixContext;
	void *pipeline;

	void *sbt;
	void *raygenRecordsBuffer;
	void *missRecordsBuffer;
	void *hitgroupRecordsBuffer;

	int numberOfGaussians;
	int maxNumberOfGaussians;

	void *aabbBuffer;
	void *compactedSizeBuffer;
	void *tempBuffer;
	unsigned long long tempBufferSize;
	void *outputBuffer;
	unsigned long long outputBufferSize;
	void *asBuffer;
	unsigned long long asBufferSize;

	unsigned long long asHandle;

	unsigned *bitmap_device;
	unsigned *bitmap_host;
	unsigned width;
	unsigned height;

	float3 O;
	float3 R, D, F;
	float double_tan_half_fov_x;
	float double_tan_half_fov_y;

	void *GC_part_1_1, *GC_part_1_2;
	void *GC_part_2_1, *GC_part_2_2;
	void *GC_part_3_1, *GC_part_3_2;
	void *GC_part_4_1, *GC_part_4_2;

	void *m11, *m12;
	void *m21, *m22;
	void *m31, *m32;
	void *m41, *m42;

	void *v11, *v12;
	void *v21, *v22;
	void *v31, *v32;
	void *v41, *v42;

	// *** *** *** *** ***

	// OPTIMIZER
	unsigned *bitmap_ref;
	void *dL_dparams_1;
	void *dL_dparams_2;
	void *dL_dparams_3;
	void *dL_dparams_4;
	double *loss_device;
	double loss_host;
	int *Gaussians_indices;
	
	typedef int cufftHandle;
	typedef void cufftComplex;
	typedef void cufftReal;

	// !!! !!! !!!
	cufftHandle planr2c;
	cufftHandle planc2r;

	cufftComplex *F_1;
	cufftComplex *F_2;

	cufftReal *bitmap_ref_R;
	cufftReal *bitmap_ref_G;
	cufftReal *bitmap_ref_B;
	cufftReal *mu_bitmap_ref_R;
	cufftReal *mu_bitmap_ref_G;
	cufftReal *mu_bitmap_ref_B;
	cufftReal *mu_bitmap_out_bitmap_ref_R;
	cufftReal *mu_bitmap_out_bitmap_ref_G;
	cufftReal *mu_bitmap_out_bitmap_ref_B;
	cufftReal *mu_bitmap_ref_R_square;
	cufftReal *mu_bitmap_ref_G_square;
	cufftReal *mu_bitmap_ref_B_square;

	cufftReal *bitmap_out_R;
	cufftReal *bitmap_out_G;
	cufftReal *bitmap_out_B;
	cufftReal *mu_bitmap_out_R;
	cufftReal *mu_bitmap_out_G;
	cufftReal *mu_bitmap_out_B;
	cufftReal *mu_bitmap_out_R_square;
	cufftReal *mu_bitmap_out_G_square;
	cufftReal *mu_bitmap_out_B_square;
	// !!! !!! !!!

	// *** *** *** *** ***

	bool copyBitmapToHostMemory;
	unsigned poseNum;
	unsigned epoch;
	void *counter1; // !!! !!! !!!
	void *counter2; // !!! !!! !!!
};

// *************************************************************************************************

struct SOptiXRenderConfig {
	char *learning_phase;
	char *data_path;
	char *pretrained_model_path;
	char *data_format;

	int start_epoch;
	int end_epoch;

	float bg_color_R;
	float bg_color_G;
	float bg_color_B;

	float lr_RGB;
	float lr_RGB_exponential_decay_coefficient;
	float lr_RGB_final;

	float lr_alpha;
	float lr_alpha_exponential_decay_coefficient;
	float lr_alpha_final;

	float lr_m;
	float lr_m_exponential_decay_coefficient;
	float lr_m_final;

	float lr_s;
	float lr_s_exponential_decay_coefficient;
	float lr_s_final;

	float lr_q;
	float lr_q_exponential_decay_coefficient;
	float lr_q_final;

	int densification_frequency;
	int densification_start_epoch;
	int densification_end_epoch;
	float alpha_threshold_for_Gauss_removal;
	float min_s_coefficients_clipping_threshold;
	float max_s_coefficients_clipping_threshold;
	float min_s_norm_threshold_for_Gauss_removal;
	float max_s_norm_threshold_for_Gauss_removal;
	float mu_grad_norm_threshold_for_densification;
	float s_norm_threshold_for_split_strategy;
	float split_ratio;
	float lambda;
	float ray_termination_T_threshold;
	float last_significant_Gauss_alpha_gradient_precision;
	float chi_square_squared_radius;
	int max_Gaussians_per_ray;
	int saving_frequency;
	int evaluation_frequency;
	int evaluation_epoch;
	int max_Gaussians_per_model;
};

// *************************************************************************************************

DWORD WINAPI RenderRoutine(LPVOID paramsPtr);

void Render(
	SRenderParams* params
);

extern bool InitializeCUDARenderer(SRenderParams& params, SCUDARenderParams& dev_params);
extern bool InitializeCUDARendererAS(SRenderParams& params, SCUDARenderParams& dev_params);
extern bool InitializeCUDAGradient(SRenderParams& params, SCUDARenderParams& dev_params);
extern bool ZeroCUDAGradient(SRenderParams& params, SCUDARenderParams& dev_params);
extern bool UpdateCUDAGradient(SRenderParams& params, SCUDARenderParams& dev_params);
extern bool RenderCUDA(SRenderParams& params, SCUDARenderParams& dev_params);
extern bool DestroyCUDARendererAS(SCUDARenderParams& dev_params);
extern bool SetConfigurationOptiX(SOptiXRenderConfig &config_OptiX);
extern void GetSceneBoundsOptiX(float& lB, float& rB, float& uB, float& dB, float& bB, float& fB, float &scene_extent_param);

// *************************************************************************************************

extern bool InitializeOptiXRenderer(
	SRenderParams &params,
	SOptiXRenderParams &params_OptiX,
	bool loadFromFile = false,
	int epoch = 0
);
extern bool InitializeOptiXOptimizer(
	SRenderParams &params,
	SOptiXRenderParams &params_OptiX,
	bool loadFromFile = false,
	int epoch = 0
);
extern bool ZeroGradientOptiX(SOptiXRenderParams &params_OptiX);
extern bool RenderOptiX(SOptiXRenderParams& params_OptiX);
extern bool UpdateGradientOptiX(SOptiXRenderParams& params_OptiX, int &state);

extern bool DumpParameters(SOptiXRenderParams& params_OptiX);
extern bool DumpParametersToPLYFile(SOptiXRenderParams& params_OptiX);


struct SOptiXRenderParamsMesh {
	SLight *light;
	void *TC;
	SMaterial *materials;
	unsigned long long asHandle;
};

extern bool InitializeOptiXRendererMesh(
	SRenderParams &params,
	SOptiXRenderParams &params_OptiX,
	C3DScene *scene,
	SOptiXRenderParamsMesh &params_OptiXMesh,
	bool loadFromFile = false,
	int epoch = 0
);
extern bool RenderOptiXMesh(SOptiXRenderParams& params_OptiX, SOptiXRenderParamsMesh &params_OptiXMesh);