#include "Header.cuh"

#include "optix_stubs.h"
#include "optix_function_table_definition.h"

// *************************************************************************************************

struct SAuxiliaryValues {
	uint3 scene_lower_bound = make_uint3(0xFF800000, 0xFF800000, 0xFF800000);
	uint3 scene_upper_bound = make_uint3(0x007FFFFF, 0x007FFFFF, 0x007FFFFF);
} initial_values;

__device__ struct {
	uint3 scene_lower_bound;
	uint3 scene_upper_bound;
} auxiliary_values;

__constant__ float scene_extent;

// *************************************************************************************************

int densification_frequency_host;
int densification_start_epoch_host;
int densification_end_epoch_host;
float min_s_coefficients_clipping_threshold_host;
float max_s_coefficients_clipping_threshold_host;
float ray_termination_T_threshold_host;
float last_significant_Gauss_alpha_gradient_precision_host;
int max_Gaussians_per_ray_host;

__constant__ float lr_RGB;
__constant__ float lr_RGB_exponential_decay_coefficient;
__constant__ float lr_alpha;
__constant__ float lr_alpha_exponential_decay_coefficient;
__constant__ float lr_m;
__constant__ float lr_m_exponential_decay_coefficient;
__constant__ float lr_s;
__constant__ float lr_s_exponential_decay_coefficient;
__constant__ float lr_q;
__constant__ float lr_q_exponential_decay_coefficient;
__constant__ int densification_frequency;
__constant__ int densification_start_epoch;
__constant__ int densification_end_epoch;
__constant__ float alpha_threshold_for_Gauss_removal;
__constant__ float min_s_coefficients_clipping_threshold;
__constant__ float max_s_coefficients_clipping_threshold;
__constant__ float min_s_norm_threshold_for_Gauss_removal;
__constant__ float max_s_norm_threshold_for_Gauss_removal;
__constant__ float mu_grad_norm_threshold_for_densification;
__constant__ float s_norm_threshold_for_split_strategy;
__constant__ float split_ratio;
__constant__ float lambda;
__constant__ float ray_termination_T_threshold;
__constant__ int max_Gaussians_per_ray;

// *************************************************************************************************

struct SbtRecord {
	__align__(OPTIX_SBT_RECORD_ALIGNMENT) char header[OPTIX_SBT_RECORD_HEADER_SIZE];
};

// *************************************************************************************************

struct LaunchParams {
	unsigned *bitmap;
	unsigned width;

	float3 O;
	float3 R, D, F;
	float FOV;

	OptixTraversableHandle traversable;

	float4 *GC_part_1;
	float4 *GC_part_2;
	float4 *GC_part_3;
	float2 *GC_part_4;

	int *Gaussians_indices;
	REAL_G *bitmap_out_R, *bitmap_out_G, *bitmap_out_B;

	float ray_termination_T_threshold;
	float last_significant_Gauss_alpha_gradient_precision;
	int max_Gaussians_per_ray;
};

// *************************************************************************************************

static void LoadFromFile(const char *fPath, int epochNum, const char *fExtension, void *buf, int size) {
	FILE *f;

	char fName[256];
	sprintf_s(fName, "%s/%d.%s", fPath, epochNum, fExtension);

	fopen_s(&f, fName, "rb");
	fread(buf, size, 1, f);
	fclose(f);
}

// *************************************************************************************************

unsigned Float2SortableUint(float value) {
	unsigned tmp = *((unsigned *)&value);
	return tmp ^ ((tmp >= 0x80000000) ? 0xFFFFFFFF : 0x80000000);
}

// *************************************************************************************************

float SortableUint2Float(unsigned value) {
	unsigned tmp = value ^ ((value < 0x80000000) ? 0xFFFFFFFF : 0x80000000);
	return *((float *)&tmp);
}

// *************************************************************************************************

__device__ unsigned dev_Float2SortableUint(float value) {
	unsigned tmp = __float_as_uint(value);
	return tmp ^ ((tmp >= 0x80000000) ? 0xFFFFFFFF : 0x80000000);
}

// *************************************************************************************************

__device__ float dev_SortableUint2Float(unsigned value) {
	unsigned tmp = value ^ ((value < 0x80000000) ? 0xFFFFFFFF : 0x80000000);
	return __uint_as_float(tmp);
}

//**************************************************************************************************
//* InitializeOptiXRenderer                                                                        *
//**************************************************************************************************

bool InitializeOptiXRenderer(
	SRenderParams &params,
	SOptiXRenderParams &params_OptiX,
	bool loadFromFile = false,
	int epoch = 0
) {
	cudaError_t error_CUDA;
	OptixResult error_OptiX;
	CUresult error_CUDA_Driver_API;

	error_CUDA = cudaFree(0);
	if (error_CUDA != cudaSuccess) goto Error;

	error_OptiX = optixInit();
	if (error_OptiX != OPTIX_SUCCESS) goto Error;

	CUcontext cudaContext;
	error_CUDA_Driver_API = cuCtxGetCurrent(&cudaContext);
	if(error_CUDA_Driver_API != CUDA_SUCCESS) goto Error;

	error_OptiX = optixDeviceContextCreate(cudaContext, 0, &params_OptiX.optixContext);
	if (error_OptiX != OPTIX_SUCCESS) goto Error;

	// *********************************************************************************************

	FILE *f = fopen("C:/Users/pc/source/repos/GaussianRenderingCUDA/GaussianRenderingCUDA/x64/Release/shaders.cu.ptx", "rb");
	fseek(f, 0, SEEK_END);
	int ptxCodeSize = ftell(f);
	fclose(f);

	char *ptxCode = (char *)malloc(sizeof(char) * (ptxCodeSize + 1));
	char *buffer = (char *)malloc(sizeof(char) * (ptxCodeSize + 1));
	ptxCode[0] = 0; // !!! !!! !!!

	f = fopen("C:/Users/pc/source/repos/GaussianRenderingCUDA/GaussianRenderingCUDA/x64/Release/shaders.cu.ptx", "rt");
	while (!feof(f)) {
		fgets(buffer, ptxCodeSize + 1, f);
		ptxCode = strcat(ptxCode, buffer);
	}
	fclose(f);

	free(buffer);

	// *********************************************************************************************

	OptixModuleCompileOptions moduleCompileOptions = {};
	OptixPipelineCompileOptions pipelineCompileOptions = {};

	moduleCompileOptions.maxRegisterCount = 50;
	moduleCompileOptions.optLevel = OPTIX_COMPILE_OPTIMIZATION_DEFAULT;
	moduleCompileOptions.debugLevel = OPTIX_COMPILE_DEBUG_LEVEL_NONE;

	pipelineCompileOptions.traversableGraphFlags = OPTIX_TRAVERSABLE_GRAPH_FLAG_ALLOW_SINGLE_GAS;
	pipelineCompileOptions.usesMotionBlur = false;
#ifndef RENDERER_OPTIX_USE_DOUBLE_PRECISION
	pipelineCompileOptions.numPayloadValues = 2; // (12) !!! !!! !!!
#else
	pipelineCompileOptions.numPayloadValues = 2; // (19) !!! !!! !!!
#endif
	pipelineCompileOptions.numAttributeValues = 0;
	pipelineCompileOptions.exceptionFlags = OPTIX_EXCEPTION_FLAG_NONE;
	pipelineCompileOptions.pipelineLaunchParamsVariableName = "optixLaunchParams";

	OptixModule module;
	error_OptiX = optixModuleCreate(
		params_OptiX.optixContext,
		&moduleCompileOptions,
		&pipelineCompileOptions,
		ptxCode,
		strlen(ptxCode),
		NULL, NULL,
		&module
	);
	if (error_OptiX != OPTIX_SUCCESS) goto Error;

	free(ptxCode);

	// *********************************************************************************************

	OptixProgramGroupOptions pgOptions = {};

	// *********************************************************************************************

	OptixProgramGroupDesc pgDesc_raygen = {};
	pgDesc_raygen.kind = OPTIX_PROGRAM_GROUP_KIND_RAYGEN;
	pgDesc_raygen.raygen.module = module;           
	pgDesc_raygen.raygen.entryFunctionName = "__raygen__renderFrame";

	OptixProgramGroup raygenPG;
	error_OptiX = optixProgramGroupCreate(
		params_OptiX.optixContext,
		&pgDesc_raygen,
		1,
		&pgOptions,
		NULL, NULL,
		&raygenPG
	);
	if (error_OptiX != OPTIX_SUCCESS) goto Error;

	// *********************************************************************************************

	OptixProgramGroupDesc pgDesc_miss = {};
	pgDesc_miss.kind = OPTIX_PROGRAM_GROUP_KIND_MISS;
	
	OptixProgramGroup missPG;
	error_OptiX = optixProgramGroupCreate(
		params_OptiX.optixContext,
		&pgDesc_miss,
		1, 
		&pgOptions,
		NULL, NULL,
		&missPG
	);
	if (error_OptiX != OPTIX_SUCCESS) goto Error;

	// *********************************************************************************************

	OptixProgramGroupDesc pgDesc_hitgroup = {};
	pgDesc_hitgroup.kind = OPTIX_PROGRAM_GROUP_KIND_HITGROUP;
	pgDesc_hitgroup.hitgroup.moduleAH            = module;
	pgDesc_hitgroup.hitgroup.entryFunctionNameAH = "__anyhit__radiance";
	pgDesc_hitgroup.hitgroup.moduleCH            = module;           
	pgDesc_hitgroup.hitgroup.entryFunctionNameCH = "__closesthit__radiance";
	pgDesc_hitgroup.hitgroup.moduleIS            = module;
	pgDesc_hitgroup.hitgroup.entryFunctionNameIS = "__intersection__is";

	OptixProgramGroup hitgroupPG;
	error_OptiX = optixProgramGroupCreate(
		params_OptiX.optixContext,
		&pgDesc_hitgroup,
		1, 
		&pgOptions,
		NULL, NULL,
		&hitgroupPG
	);
	if (error_OptiX != OPTIX_SUCCESS) goto Error;

	// *********************************************************************************************

	OptixPipelineLinkOptions pipelineLinkOptions = {};
	pipelineLinkOptions.maxTraceDepth = 0;

	OptixProgramGroup program_groups[] = { raygenPG, missPG, hitgroupPG };

	error_OptiX = optixPipelineCreate(
		params_OptiX.optixContext,
		&pipelineCompileOptions,
		&pipelineLinkOptions,
		program_groups,
		3,
		NULL, NULL,
		&params_OptiX.pipeline
	);
	if (error_OptiX != OPTIX_SUCCESS) goto Error;

	error_OptiX = optixPipelineSetStackSize(
		params_OptiX.pipeline, 
		2*1024,
		2*1024,
		2*1024,
		1
	);
	if (error_OptiX != OPTIX_SUCCESS) goto Error;

	// *********************************************************************************************

	params_OptiX.sbt = new OptixShaderBindingTable();

	// *********************************************************************************************

	SbtRecord rec_raygen;
	error_OptiX = optixSbtRecordPackHeader(raygenPG, &rec_raygen);
	if (error_OptiX != OPTIX_SUCCESS) goto Error;

	error_CUDA = cudaMalloc(&params_OptiX.raygenRecordsBuffer, sizeof(SbtRecord) * 1);
	if (error_CUDA != cudaSuccess) goto Error;

	error_CUDA = cudaMemcpy(params_OptiX.raygenRecordsBuffer, &rec_raygen, sizeof(SbtRecord) * 1, cudaMemcpyHostToDevice);
	if (error_CUDA != cudaSuccess) goto Error;

	params_OptiX.sbt->raygenRecord = (CUdeviceptr)params_OptiX.raygenRecordsBuffer;

	// *********************************************************************************************

	SbtRecord rec_miss;
	error_OptiX = optixSbtRecordPackHeader(missPG, &rec_miss);
	if (error_OptiX != OPTIX_SUCCESS) goto Error;

	error_CUDA = cudaMalloc(&params_OptiX.missRecordsBuffer, sizeof(SbtRecord) * 1);
	if (error_CUDA != cudaSuccess) goto Error;

	error_CUDA = cudaMemcpy(params_OptiX.missRecordsBuffer, &rec_miss, sizeof(SbtRecord) * 1, cudaMemcpyHostToDevice);
	if (error_CUDA != cudaSuccess) goto Error;

	params_OptiX.sbt->missRecordBase = (CUdeviceptr)params_OptiX.missRecordsBuffer;
	params_OptiX.sbt->missRecordStrideInBytes = sizeof(SbtRecord);
	params_OptiX.sbt->missRecordCount = 1;

	// *********************************************************************************************

	SbtRecord rec_hitgroup;
	error_OptiX = optixSbtRecordPackHeader(hitgroupPG, &rec_hitgroup);
	if (error_OptiX != OPTIX_SUCCESS) goto Error;

	error_CUDA = cudaMalloc(&params_OptiX.hitgroupRecordsBuffer, sizeof(SbtRecord) * 1);
	if (error_CUDA != cudaSuccess) goto Error;

	error_CUDA = cudaMemcpy(params_OptiX.hitgroupRecordsBuffer, &rec_hitgroup, sizeof(SbtRecord) * 1, cudaMemcpyHostToDevice);
	if (error_CUDA != cudaSuccess) goto Error;

	params_OptiX.sbt->hitgroupRecordBase          = (CUdeviceptr)params_OptiX.hitgroupRecordsBuffer;
	params_OptiX.sbt->hitgroupRecordStrideInBytes = sizeof(SbtRecord);
	params_OptiX.sbt->hitgroupRecordCount         = 1;

	// *********************************************************************************************

	if (!loadFromFile) {
		params_OptiX.numberOfGaussians = params.numberOfGaussians; // !!! !!! !!!
		params_OptiX.maxNumberOfGaussians1 = params_OptiX.numberOfGaussians; // !!! !!! !!!
		params_OptiX.maxNumberOfGaussians2 = params_OptiX.numberOfGaussians * 3; // !!! !!! !!!
	} else {
		FILE *f;

		char fName[256];
		sprintf_s(fName, "dump/save/%d.GC1", epoch);

		fopen_s(&f, fName, "rb");
		fseek(f, 0, SEEK_END);
		params_OptiX.numberOfGaussians = ftell(f) / sizeof(float4); // !!! !!! !!!
		fclose(f);

		params_OptiX.maxNumberOfGaussians1 = params_OptiX.numberOfGaussians; // !!! !!! !!!
		params_OptiX.maxNumberOfGaussians2 = params_OptiX.numberOfGaussians * 3; // !!! !!! !!!
	}

	// *********************************************************************************************

	float4 *GC_part_1 = (float4 *)malloc(sizeof(float4) * params_OptiX.numberOfGaussians);
	float4 *GC_part_2 = (float4 *)malloc(sizeof(float4) * params_OptiX.numberOfGaussians);
	float4 *GC_part_3 = (float4 *)malloc(sizeof(float4) * params_OptiX.numberOfGaussians);
	float2 *GC_part_4 = (float2 *)malloc(sizeof(float2) * params_OptiX.numberOfGaussians);

	if (!loadFromFile) {
		for (int i = 0; i < params_OptiX.numberOfGaussians; ++i) {
			GC_part_1[i].x = params.GC[i].R;
			GC_part_1[i].y = params.GC[i].G;
			GC_part_1[i].z = params.GC[i].B;
			GC_part_1[i].w = params.GC[i].alpha;

			GC_part_2[i].x = params.GC[i].mX;
			GC_part_2[i].y = params.GC[i].mY;
			GC_part_2[i].z = params.GC[i].mZ;
			GC_part_2[i].w = params.GC[i].sX;

			GC_part_3[i].x = params.GC[i].sY;
			GC_part_3[i].y = params.GC[i].sZ;
			GC_part_3[i].z = params.GC[i].qr;
			GC_part_3[i].w = params.GC[i].qi;

			GC_part_4[i].x = params.GC[i].qj;
			GC_part_4[i].y = params.GC[i].qk;
		}
	} else {
		LoadFromFile("dump/save", epoch, "GC1", GC_part_1, sizeof(float4) * params_OptiX.numberOfGaussians);
		LoadFromFile("dump/save", epoch, "GC2", GC_part_2, sizeof(float4) * params_OptiX.numberOfGaussians);
		LoadFromFile("dump/save", epoch, "GC3", GC_part_3, sizeof(float4) * params_OptiX.numberOfGaussians);
		LoadFromFile("dump/save", epoch, "GC4", GC_part_4, sizeof(float2) * params_OptiX.numberOfGaussians);
	}

	// *********************************************************************************************

	float *aabbs = (float *)malloc(sizeof(float) * 6 * params_OptiX.numberOfGaussians);

	SAuxiliaryValues auxiliary_values_local;
	auxiliary_values_local.scene_lower_bound = initial_values.scene_lower_bound;
	auxiliary_values_local.scene_upper_bound = initial_values.scene_upper_bound;

	float scene_extent_local;

	for (int i = 0; i < params_OptiX.numberOfGaussians; ++i) {
		float aa = GC_part_3[i].z * GC_part_3[i].z;
		float bb = GC_part_3[i].w * GC_part_3[i].w;
		float cc = GC_part_4[i].x * GC_part_4[i].x;
		float dd = GC_part_4[i].y * GC_part_4[i].y;
		float s = 2.0f / (aa + bb + cc + dd);

		float bs = GC_part_3[i].w * s;  float cs = GC_part_4[i].x * s;  float ds = GC_part_4[i].y * s;
		float ab = GC_part_3[i].z * bs; float ac = GC_part_3[i].z * cs; float ad = GC_part_3[i].z * ds;
		bb = bb * s;                    float bc = GC_part_3[i].w * cs; float bd = GC_part_3[i].w * ds;
		cc = cc * s;                    float cd = GC_part_4[i].x * ds;       dd = dd * s;

		float Q11 = 1.0f - cc - dd;
		float Q12 = bc - ad;
		float Q13 = bd + ac;

		float Q21 = bc + ad;
		float Q22 = 1.0f - bb - dd;
		float Q23 = cd - ab;

		float Q31 = bd - ac;
		float Q32 = cd + ab;
		float Q33 = 1.0f - bb - cc;

		float sX = 1.0f / (1.0f + expf(-GC_part_2[i].w));
		float sY = 1.0f / (1.0f + expf(-GC_part_3[i].x));
		float sZ = 1.0f / (1.0f + expf(-GC_part_3[i].y));

		sX = ((sX < min_s_coefficients_clipping_threshold_host) ? min_s_coefficients_clipping_threshold_host : sX);
		sY = ((sY < min_s_coefficients_clipping_threshold_host) ? min_s_coefficients_clipping_threshold_host : sY);
		sZ = ((sZ < min_s_coefficients_clipping_threshold_host) ? min_s_coefficients_clipping_threshold_host : sZ);

		sX = ((sX > max_s_coefficients_clipping_threshold_host) ? max_s_coefficients_clipping_threshold_host : sX);
		sY = ((sY > max_s_coefficients_clipping_threshold_host) ? max_s_coefficients_clipping_threshold_host : sY);
		sZ = ((sZ > max_s_coefficients_clipping_threshold_host) ? max_s_coefficients_clipping_threshold_host : sZ);

		GC_part_2[i].w = -logf((1.0f / sX) - 1.0f);
		GC_part_3[i].x = -logf((1.0f / sY) - 1.0f);
		GC_part_3[i].y = -logf((1.0f / sZ) - 1.0f);

		float tmpX = sqrtf(11.3449f * ((sX * sX * Q11 * Q11) + (sY * sY * Q12 * Q12) + (sZ * sZ * Q13 * Q13)));
		float tmpY = sqrtf(11.3449f * ((sX * sX * Q21 * Q21) + (sY * sY * Q22 * Q22) + (sZ * sZ * Q23 * Q23)));
		float tmpZ = sqrtf(11.3449f * ((sX * sX * Q31 * Q31) + (sY * sY * Q32 * Q32) + (sZ * sZ * Q33 * Q33)));

		aabbs[(i * 6) + 0] = GC_part_2[i].x - tmpX; // !!! !!! !!!
		aabbs[(i * 6) + 3] = GC_part_2[i].x + tmpX; // !!! !!! !!!

		aabbs[(i * 6) + 1] = GC_part_2[i].y - tmpY; // !!! !!! !!!
		aabbs[(i * 6) + 4] = GC_part_2[i].y + tmpY; // !!! !!! !!!

		aabbs[(i * 6) + 2] = GC_part_2[i].z - tmpZ; // !!! !!! !!!
		aabbs[(i * 6) + 5] = GC_part_2[i].z + tmpZ; // !!! !!! !!!

		auxiliary_values_local.scene_lower_bound.x = (
			(Float2SortableUint(aabbs[(i * 6) + 0]) < auxiliary_values_local.scene_lower_bound.x) ?
			Float2SortableUint(aabbs[(i * 6) + 0]) :
			auxiliary_values_local.scene_lower_bound.x
		);
		auxiliary_values_local.scene_lower_bound.y = (
			(Float2SortableUint(aabbs[(i * 6) + 1]) < auxiliary_values_local.scene_lower_bound.y) ?
			Float2SortableUint(aabbs[(i * 6) + 1]) :
			auxiliary_values_local.scene_lower_bound.y
		);
		auxiliary_values_local.scene_lower_bound.z = (
			(Float2SortableUint(aabbs[(i * 6) + 2]) < auxiliary_values_local.scene_lower_bound.z) ?
			Float2SortableUint(aabbs[(i * 6) + 2]) :
			auxiliary_values_local.scene_lower_bound.z
		);
	
		auxiliary_values_local.scene_upper_bound.x = (
			(Float2SortableUint(aabbs[(i * 6) + 3]) > auxiliary_values_local.scene_upper_bound.x) ?
			Float2SortableUint(aabbs[(i * 6) + 3]) :
			auxiliary_values_local.scene_upper_bound.x
		);
		auxiliary_values_local.scene_upper_bound.y = (
			(Float2SortableUint(aabbs[(i * 6) + 4]) > auxiliary_values_local.scene_upper_bound.y) ?
			Float2SortableUint(aabbs[(i * 6) + 4]) :
			auxiliary_values_local.scene_upper_bound.y
		);
		auxiliary_values_local.scene_upper_bound.z = (
			(Float2SortableUint(aabbs[(i * 6) + 5]) > auxiliary_values_local.scene_upper_bound.z) ?
			Float2SortableUint(aabbs[(i * 6) + 5]) :
			auxiliary_values_local.scene_upper_bound.z
		);
	}

	float dX = SortableUint2Float(auxiliary_values_local.scene_upper_bound.x) - SortableUint2Float(auxiliary_values_local.scene_lower_bound.x);
	float dY = SortableUint2Float(auxiliary_values_local.scene_upper_bound.y) - SortableUint2Float(auxiliary_values_local.scene_lower_bound.y);
	float dZ = SortableUint2Float(auxiliary_values_local.scene_upper_bound.z) - SortableUint2Float(auxiliary_values_local.scene_lower_bound.z);
	
	scene_extent_local = sqrtf((dX * dX) + (dY * dY) + (dZ * dZ));

	// *********************************************************************************************

	error_CUDA = cudaMalloc(&params_OptiX.GC_part_1_1, sizeof(float4) * params_OptiX.maxNumberOfGaussians1);
	if (error_CUDA != cudaSuccess) goto Error;

	error_CUDA = cudaMalloc(&params_OptiX.GC_part_1_2, sizeof(float4) * params_OptiX.maxNumberOfGaussians2);
	if (error_CUDA != cudaSuccess) goto Error;

	error_CUDA = cudaMalloc(&params_OptiX.GC_part_2_1, sizeof(float4) * params_OptiX.maxNumberOfGaussians1);
	if (error_CUDA != cudaSuccess) goto Error;

	error_CUDA = cudaMalloc(&params_OptiX.GC_part_2_2, sizeof(float4) * params_OptiX.maxNumberOfGaussians2);
	if (error_CUDA != cudaSuccess) goto Error;

	error_CUDA = cudaMalloc(&params_OptiX.GC_part_3_1, sizeof(float4) * params_OptiX.maxNumberOfGaussians1);
	if (error_CUDA != cudaSuccess) goto Error;

	error_CUDA = cudaMalloc(&params_OptiX.GC_part_3_2, sizeof(float4) * params_OptiX.maxNumberOfGaussians2);
	if (error_CUDA != cudaSuccess) goto Error;

	error_CUDA = cudaMalloc(&params_OptiX.GC_part_4_1, sizeof(float2) * params_OptiX.maxNumberOfGaussians1);
	if (error_CUDA != cudaSuccess) goto Error;

	error_CUDA = cudaMalloc(&params_OptiX.GC_part_4_2, sizeof(float2) * params_OptiX.maxNumberOfGaussians2);
	if (error_CUDA != cudaSuccess) goto Error;

	error_CUDA = cudaMemcpy(params_OptiX.GC_part_1_1, GC_part_1, sizeof(float4) * params_OptiX.numberOfGaussians, cudaMemcpyHostToDevice);
	if (error_CUDA != cudaSuccess) goto Error;

	error_CUDA = cudaMemcpy(params_OptiX.GC_part_2_1, GC_part_2, sizeof(float4) * params_OptiX.numberOfGaussians, cudaMemcpyHostToDevice);
	if (error_CUDA != cudaSuccess) goto Error;

	error_CUDA = cudaMemcpy(params_OptiX.GC_part_3_1, GC_part_3, sizeof(float4) * params_OptiX.numberOfGaussians, cudaMemcpyHostToDevice);
	if (error_CUDA != cudaSuccess) goto Error;

	error_CUDA = cudaMemcpy(params_OptiX.GC_part_4_1, GC_part_4, sizeof(float2) * params_OptiX.numberOfGaussians, cudaMemcpyHostToDevice);
	if (error_CUDA != cudaSuccess) goto Error;

	free(GC_part_1);
	free(GC_part_2);
	free(GC_part_3);
	free(GC_part_4);

	// *********************************************************************************************

	error_CUDA = cudaMalloc(&params_OptiX.aabbBuffer, sizeof(float) * 6 * params_OptiX.maxNumberOfGaussians2); // !!! !!! !!!
	if (error_CUDA != cudaSuccess) goto Error;

	error_CUDA = cudaMemcpy(params_OptiX.aabbBuffer, aabbs, sizeof(float) * 6 * params_OptiX.numberOfGaussians, cudaMemcpyHostToDevice);
	if (error_CUDA != cudaSuccess) goto Error;

	error_CUDA = cudaMemcpyToSymbol(auxiliary_values, &auxiliary_values_local, sizeof(SAuxiliaryValues) * 1);
	if (error_CUDA != cudaSuccess) goto Error;

	error_CUDA = cudaMemcpyToSymbol(scene_extent, &scene_extent_local, sizeof(float) * 1);
	if (error_CUDA != cudaSuccess) goto Error;

	free(aabbs);

	// *********************************************************************************************

	OptixAccelBuildOptions accel_options = {};
	accel_options.buildFlags = OPTIX_BUILD_FLAG_ALLOW_COMPACTION;
	accel_options.operation  = OPTIX_BUILD_OPERATION_BUILD;

	OptixBuildInput aabb_input = {};
	aabb_input.type                               = OPTIX_BUILD_INPUT_TYPE_CUSTOM_PRIMITIVES;
	aabb_input.customPrimitiveArray.aabbBuffers   = (CUdeviceptr *)&params_OptiX.aabbBuffer;
	aabb_input.customPrimitiveArray.numPrimitives = params_OptiX.numberOfGaussians;

	unsigned aabb_input_flags[1]                  = {OPTIX_GEOMETRY_FLAG_NONE};
	aabb_input.customPrimitiveArray.flags         = (const unsigned int *)aabb_input_flags;
	aabb_input.customPrimitiveArray.numSbtRecords = 1;

	// *********************************************************************************************

	OptixAccelBufferSizes blasBufferSizes;
	error_OptiX = optixAccelComputeMemoryUsage(
		params_OptiX.optixContext,
		&accel_options,
		&aabb_input,
		1,
		&blasBufferSizes
	);
	if (error_OptiX != OPTIX_SUCCESS) goto Error;

	// *********************************************************************************************

	error_CUDA = cudaMalloc(&params_OptiX.compactedSizeBuffer, 8);
	if (error_CUDA != cudaSuccess) goto Error;

	OptixAccelEmitDesc emitDesc;
	emitDesc.type   = OPTIX_PROPERTY_TYPE_COMPACTED_SIZE;
	emitDesc.result = (CUdeviceptr)params_OptiX.compactedSizeBuffer;

	params_OptiX.tempBufferSize = blasBufferSizes.tempSizeInBytes * 2; // !!! !!! !!!
	error_CUDA = cudaMalloc(&params_OptiX.tempBuffer, params_OptiX.tempBufferSize);
	if (error_CUDA != cudaSuccess) goto Error;

	params_OptiX.outputBufferSize = blasBufferSizes.outputSizeInBytes * 2; // !!! !!! !!!
	error_CUDA = cudaMalloc(&params_OptiX.outputBuffer, params_OptiX.outputBufferSize);
	if (error_CUDA != cudaSuccess) goto Error;

	// *********************************************************************************************

	error_OptiX = optixAccelBuild(
		params_OptiX.optixContext,
		0,
		&accel_options,
		&aabb_input,
		1,  
		(CUdeviceptr)params_OptiX.tempBuffer,
		blasBufferSizes.tempSizeInBytes,
		(CUdeviceptr)params_OptiX.outputBuffer,
		blasBufferSizes.outputSizeInBytes,
		&params_OptiX.asHandle,
		&emitDesc,
		1
	);
	if (error_OptiX != OPTIX_SUCCESS) goto Error;

	cudaDeviceSynchronize();
	error_CUDA = cudaGetLastError();
	if (error_CUDA != cudaSuccess) goto Error;

	unsigned long long compactedSize;
	error_CUDA = cudaMemcpy(&compactedSize, params_OptiX.compactedSizeBuffer, 8, cudaMemcpyDeviceToHost);
	if (error_CUDA != cudaSuccess) goto Error;

	params_OptiX.asBufferSize = compactedSize * 2; // !!! !!! !!! 
	error_CUDA = cudaMalloc(&params_OptiX.asBuffer, params_OptiX.asBufferSize);
	if (error_CUDA != cudaSuccess) goto Error;

	error_OptiX = optixAccelCompact(
		params_OptiX.optixContext,
		0,
		params_OptiX.asHandle,
		(CUdeviceptr)params_OptiX.asBuffer,
		compactedSize,
		&params_OptiX.asHandle
	);
	if (error_OptiX != OPTIX_SUCCESS) goto Error;

	cudaDeviceSynchronize();
	error_CUDA = cudaGetLastError();
	if (error_CUDA != cudaSuccess) goto Error;

	// *********************************************************************************************

	params_OptiX.width = params.w; // !!! !!! !!!

	error_CUDA = cudaMalloc(&params_OptiX.bitmap_out_device, sizeof(unsigned) * params_OptiX.width * params_OptiX.width);
	if (error_CUDA != cudaSuccess) goto Error;

	params_OptiX.bitmap_out_host = (unsigned *)params.bitmap; // !!! !!! !!!

	// *********************************************************************************************

	return true;
Error:
	return false;
}

// *************************************************************************************************

__global__ void MultiplyPointwiseReal(REAL_G *arr1_in, REAL_G *arr2_in, REAL_G *arr_out, int size) {
	int tid = (blockIdx.x * blockDim.x) + threadIdx.x;
	if (tid < size) {
		REAL_G a = arr1_in[tid];
		REAL_G b = arr2_in[tid];
		REAL_G c = a * b;
		arr_out[tid] = c;
	}
}

// *************************************************************************************************

__global__ void MultiplyPointwiseComplex(COMPLEX_G *arr1_in, COMPLEX_G *arr2_in, COMPLEX_G *arr_out, int size) {
	int tid = (blockIdx.x * blockDim.x) + threadIdx.x;
	if (tid < size) {
		COMPLEX_G c;
		COMPLEX_G a = arr1_in[tid];
		COMPLEX_G b = arr2_in[tid];
		c.x = (a.x * b.x) - (a.y * b.y);
		c.y = (a.x * b.y) + (a.y * b.x);
		arr_out[tid] = c;
	}
}

//**************************************************************************************************
//* InitializeOptiXOptimizer                                                                       *
//**************************************************************************************************

bool InitializeOptiXOptimizer(
	SRenderParams &params,
	SOptiXRenderParams &params_OptiX,
	bool loadFromFile = false,
	int epoch = 0
) {
	cudaError_t error_CUDA;

	error_CUDA = cudaMalloc(&params_OptiX.bitmap_ref, sizeof(unsigned) * params_OptiX.width * params_OptiX.width * params.NUMBER_OF_POSES);
	if (error_CUDA != cudaSuccess) goto Error;

	error_CUDA = cudaMalloc(&params_OptiX.dL_dparams_1, sizeof(REAL4_G) * params_OptiX.maxNumberOfGaussians2);
	if (error_CUDA != cudaSuccess) goto Error;

	error_CUDA = cudaMalloc(&params_OptiX.dL_dparams_2, sizeof(REAL4_G) * params_OptiX.maxNumberOfGaussians2);
	if (error_CUDA != cudaSuccess) goto Error;

	error_CUDA = cudaMalloc(&params_OptiX.dL_dparams_3, sizeof(REAL4_G) * params_OptiX.maxNumberOfGaussians2);
	if (error_CUDA != cudaSuccess) goto Error;

	error_CUDA = cudaMalloc(&params_OptiX.dL_dparams_4, sizeof(REAL2_G) * params_OptiX.maxNumberOfGaussians2);
	if (error_CUDA != cudaSuccess) goto Error;

	error_CUDA = cudaMalloc(&params_OptiX.loss_device, sizeof(double) * 1);
	if (error_CUDA != cudaSuccess) goto Error;

	error_CUDA = cudaMalloc(&params_OptiX.Gaussians_indices, sizeof(int) * max_Gaussians_per_ray_host * params_OptiX.width * params_OptiX.width);
	if (error_CUDA != cudaSuccess) goto Error;

	error_CUDA = cudaMalloc(&params_OptiX.m11, sizeof(float4) * params_OptiX.maxNumberOfGaussians1);
	if (error_CUDA != cudaSuccess) goto Error;

	error_CUDA = cudaMalloc(&params_OptiX.m12, sizeof(float4) * params_OptiX.maxNumberOfGaussians2);
	if (error_CUDA != cudaSuccess) goto Error;

	error_CUDA = cudaMalloc(&params_OptiX.m21, sizeof(float4) * params_OptiX.maxNumberOfGaussians1);
	if (error_CUDA != cudaSuccess) goto Error;

	error_CUDA = cudaMalloc(&params_OptiX.m22, sizeof(float4) * params_OptiX.maxNumberOfGaussians2);
	if (error_CUDA != cudaSuccess) goto Error;

	error_CUDA = cudaMalloc(&params_OptiX.m31, sizeof(float4) * params_OptiX.maxNumberOfGaussians1);
	if (error_CUDA != cudaSuccess) goto Error;

	error_CUDA = cudaMalloc(&params_OptiX.m32, sizeof(float4) * params_OptiX.maxNumberOfGaussians2);
	if (error_CUDA != cudaSuccess) goto Error;

	error_CUDA = cudaMalloc(&params_OptiX.m41, sizeof(float2) * params_OptiX.maxNumberOfGaussians1);
	if (error_CUDA != cudaSuccess) goto Error;

	error_CUDA = cudaMalloc(&params_OptiX.m42, sizeof(float2) * params_OptiX.maxNumberOfGaussians2);
	if (error_CUDA != cudaSuccess) goto Error;

	error_CUDA = cudaMalloc(&params_OptiX.v11, sizeof(float4) * params_OptiX.maxNumberOfGaussians1);
	if (error_CUDA != cudaSuccess) goto Error;

	error_CUDA = cudaMalloc(&params_OptiX.v12, sizeof(float4) * params_OptiX.maxNumberOfGaussians2);
	if (error_CUDA != cudaSuccess) goto Error;

	error_CUDA = cudaMalloc(&params_OptiX.v21, sizeof(float4) * params_OptiX.maxNumberOfGaussians1);
	if (error_CUDA != cudaSuccess) goto Error;

	error_CUDA = cudaMalloc(&params_OptiX.v22, sizeof(float4) * params_OptiX.maxNumberOfGaussians2);
	if (error_CUDA != cudaSuccess) goto Error;

	error_CUDA = cudaMalloc(&params_OptiX.v31, sizeof(float4) * params_OptiX.maxNumberOfGaussians1);
	if (error_CUDA != cudaSuccess) goto Error;

	error_CUDA = cudaMalloc(&params_OptiX.v32, sizeof(float4) * params_OptiX.maxNumberOfGaussians2);
	if (error_CUDA != cudaSuccess) goto Error;

	error_CUDA = cudaMalloc(&params_OptiX.v41, sizeof(float2) * params_OptiX.maxNumberOfGaussians1);
	if (error_CUDA != cudaSuccess) goto Error;

	error_CUDA = cudaMalloc(&params_OptiX.v42, sizeof(float2) * params_OptiX.maxNumberOfGaussians2);
	if (error_CUDA != cudaSuccess) goto Error;

	error_CUDA = cudaMalloc(&params_OptiX.counter1, sizeof(unsigned) * 1);
	if (error_CUDA != cudaSuccess) goto Error;

	error_CUDA = cudaMalloc(&params_OptiX.counter2, sizeof(unsigned) * 1);
	if (error_CUDA != cudaSuccess) goto Error;

	// ************************************************************************************************

	error_CUDA = cudaMemcpy(params_OptiX.bitmap_ref, params.bitmap_ref, sizeof(unsigned) * params_OptiX.width * params_OptiX.width * params.NUMBER_OF_POSES, cudaMemcpyHostToDevice);
	if (error_CUDA != cudaSuccess) goto Error;

	// ************************************************************************************************

	if (!loadFromFile) {
		error_CUDA = cudaMemset(params_OptiX.m11, 0, sizeof(float4) * params_OptiX.maxNumberOfGaussians1);
		if (error_CUDA != cudaSuccess) goto Error;

		error_CUDA = cudaMemset(params_OptiX.m21, 0, sizeof(float4) * params_OptiX.maxNumberOfGaussians1);
		if (error_CUDA != cudaSuccess) goto Error;

		error_CUDA = cudaMemset(params_OptiX.m31, 0, sizeof(float4) * params_OptiX.maxNumberOfGaussians1);
		if (error_CUDA != cudaSuccess) goto Error;

		error_CUDA = cudaMemset(params_OptiX.m41, 0, sizeof(float2) * params_OptiX.maxNumberOfGaussians1);
		if (error_CUDA != cudaSuccess) goto Error;

		error_CUDA = cudaMemset(params_OptiX.v11, 0, sizeof(float4) * params_OptiX.maxNumberOfGaussians1);
		if (error_CUDA != cudaSuccess) goto Error;

		error_CUDA = cudaMemset(params_OptiX.v21, 0, sizeof(float4) * params_OptiX.maxNumberOfGaussians1);
		if (error_CUDA != cudaSuccess) goto Error;

		error_CUDA = cudaMemset(params_OptiX.v31, 0, sizeof(float4) * params_OptiX.maxNumberOfGaussians1);
		if (error_CUDA != cudaSuccess) goto Error;

		error_CUDA = cudaMemset(params_OptiX.v41, 0, sizeof(float2) * params_OptiX.maxNumberOfGaussians1);
		if (error_CUDA != cudaSuccess) goto Error;
	} else {
		void *buf = malloc(sizeof(float4) * params_OptiX.numberOfGaussians);
		if (buf == NULL) goto Error;

		LoadFromFile("dump/save", epoch, "m1", buf, sizeof(float4) * params_OptiX.numberOfGaussians);
		error_CUDA = cudaMemcpy(params_OptiX.m11, buf, sizeof(float4) * params_OptiX.numberOfGaussians, cudaMemcpyHostToDevice);
		if (error_CUDA != cudaSuccess) goto Error;

		LoadFromFile("dump/save", epoch, "m2", buf, sizeof(float4) * params_OptiX.numberOfGaussians);
		error_CUDA = cudaMemcpy(params_OptiX.m21, buf, sizeof(float4) * params_OptiX.numberOfGaussians, cudaMemcpyHostToDevice);
		if (error_CUDA != cudaSuccess) goto Error;

		LoadFromFile("dump/save", epoch, "m3", buf, sizeof(float4) * params_OptiX.numberOfGaussians);
		error_CUDA = cudaMemcpy(params_OptiX.m31, buf, sizeof(float4) * params_OptiX.numberOfGaussians, cudaMemcpyHostToDevice);
		if (error_CUDA != cudaSuccess) goto Error;

		LoadFromFile("dump/save", epoch, "m4", buf, sizeof(float2) * params_OptiX.numberOfGaussians);
		error_CUDA = cudaMemcpy(params_OptiX.m41, buf, sizeof(float2) * params_OptiX.numberOfGaussians, cudaMemcpyHostToDevice);
		if (error_CUDA != cudaSuccess) goto Error;

		LoadFromFile("dump/save", epoch, "v1", buf, sizeof(float4) * params_OptiX.numberOfGaussians);
		error_CUDA = cudaMemcpy(params_OptiX.v11, buf, sizeof(float4) * params_OptiX.numberOfGaussians, cudaMemcpyHostToDevice);
		if (error_CUDA != cudaSuccess) goto Error;

		LoadFromFile("dump/save", epoch, "v2", buf, sizeof(float4) * params_OptiX.numberOfGaussians);
		error_CUDA = cudaMemcpy(params_OptiX.v21, buf, sizeof(float4) * params_OptiX.numberOfGaussians, cudaMemcpyHostToDevice);
		if (error_CUDA != cudaSuccess) goto Error;

		LoadFromFile("dump/save", epoch, "v3", buf, sizeof(float4) * params_OptiX.numberOfGaussians);
		error_CUDA = cudaMemcpy(params_OptiX.v31, buf, sizeof(float4) * params_OptiX.numberOfGaussians, cudaMemcpyHostToDevice);
		if (error_CUDA != cudaSuccess) goto Error;

		LoadFromFile("dump/save", epoch, "v4", buf, sizeof(float2) * params_OptiX.numberOfGaussians);
		error_CUDA = cudaMemcpy(params_OptiX.v41, buf, sizeof(float2) * params_OptiX.numberOfGaussians, cudaMemcpyHostToDevice);
		if (error_CUDA != cudaSuccess) goto Error;

		free(buf);
	}

	error_CUDA = cudaMemset(params_OptiX.counter2, 0, sizeof(unsigned) * 1);
	if (error_CUDA != cudaSuccess) goto Error;

	//**********************************************************************************************
	//* SSIM                                                                                       *
	// *********************************************************************************************

	const int kernel_size = 11;
	const int kernel_radius = kernel_size >> 1;
	const REAL_G sigma = ((REAL_G)1.5);

	int arraySizeReal = (params_OptiX.width + (kernel_size - 1)) * (params_OptiX.width + (kernel_size - 1)); // !!! !!! !!!
	int arraySizeComplex = (((params_OptiX.width + (kernel_size - 1)) >> 1) + 1) * (params_OptiX.width + (kernel_size - 1)); // !!! !!! !!!

	REAL_G *buf;
	buf = (REAL_G *)malloc(sizeof(REAL_G) * arraySizeReal);
	if (buf == NULL) goto Error;

	cufftResult error_CUFFT;

	// ************************************************************************************************

	error_CUFFT = cufftPlan2d(&params_OptiX.planr2c, params_OptiX.width + (kernel_size - 1), params_OptiX.width + (kernel_size - 1), REAL_TO_COMPLEX_G);
	if (error_CUFFT != CUFFT_SUCCESS) goto Error;

	error_CUFFT = cufftPlan2d(&params_OptiX.planc2r, params_OptiX.width + (kernel_size - 1), params_OptiX.width + (kernel_size - 1), COMPLEX_TO_REAL_G);
	if (error_CUFFT != CUFFT_SUCCESS) goto Error;

	// ************************************************************************************************

	error_CUDA = cudaMalloc(&params_OptiX.F_1, sizeof(COMPLEX_G) * arraySizeComplex);
	if (error_CUDA != cudaSuccess) goto Error;

	error_CUDA = cudaMalloc(&params_OptiX.F_2, sizeof(COMPLEX_G) * arraySizeComplex);
	if (error_CUDA != cudaSuccess) goto Error;

	// ************************************************************************************************

	error_CUDA = cudaMalloc(&params_OptiX.bitmap_ref_R, sizeof(REAL_G) * arraySizeReal * params.NUMBER_OF_POSES);
	if (error_CUDA != cudaSuccess) goto Error;

	error_CUDA = cudaMalloc(&params_OptiX.bitmap_ref_G, sizeof(REAL_G) * arraySizeReal * params.NUMBER_OF_POSES);
	if (error_CUDA != cudaSuccess) goto Error;

	error_CUDA = cudaMalloc(&params_OptiX.bitmap_ref_B, sizeof(REAL_G) * arraySizeReal * params.NUMBER_OF_POSES);
	if (error_CUDA != cudaSuccess) goto Error;

	error_CUDA = cudaMalloc(&params_OptiX.mu_bitmap_ref_R, sizeof(REAL_G) * arraySizeReal * params.NUMBER_OF_POSES);
	if (error_CUDA != cudaSuccess) goto Error;

	error_CUDA = cudaMalloc(&params_OptiX.mu_bitmap_ref_G, sizeof(REAL_G) * arraySizeReal * params.NUMBER_OF_POSES);
	if (error_CUDA != cudaSuccess) goto Error;

	error_CUDA = cudaMalloc(&params_OptiX.mu_bitmap_ref_B, sizeof(REAL_G) * arraySizeReal * params.NUMBER_OF_POSES);
	if (error_CUDA != cudaSuccess) goto Error;

	error_CUDA = cudaMalloc(&params_OptiX.mu_bitmap_out_bitmap_ref_R, sizeof(REAL_G) * arraySizeReal);
	if (error_CUDA != cudaSuccess) goto Error;

	error_CUDA = cudaMalloc(&params_OptiX.mu_bitmap_out_bitmap_ref_G, sizeof(REAL_G) * arraySizeReal);
	if (error_CUDA != cudaSuccess) goto Error;

	error_CUDA = cudaMalloc(&params_OptiX.mu_bitmap_out_bitmap_ref_B, sizeof(REAL_G) * arraySizeReal);
	if (error_CUDA != cudaSuccess) goto Error;

	error_CUDA = cudaMalloc(&params_OptiX.mu_bitmap_ref_R_square, sizeof(REAL_G) * arraySizeReal * params.NUMBER_OF_POSES);
	if (error_CUDA != cudaSuccess) goto Error;

	error_CUDA = cudaMalloc(&params_OptiX.mu_bitmap_ref_G_square, sizeof(REAL_G) * arraySizeReal * params.NUMBER_OF_POSES);
	if (error_CUDA != cudaSuccess) goto Error;

	error_CUDA = cudaMalloc(&params_OptiX.mu_bitmap_ref_B_square, sizeof(REAL_G) * arraySizeReal * params.NUMBER_OF_POSES);
	if (error_CUDA != cudaSuccess) goto Error;

	error_CUDA = cudaMalloc(&params_OptiX.bitmap_out_R, sizeof(REAL_G) * arraySizeReal);
	if (error_CUDA != cudaSuccess) goto Error;

	error_CUDA = cudaMalloc(&params_OptiX.bitmap_out_G, sizeof(REAL_G) * arraySizeReal);
	if (error_CUDA != cudaSuccess) goto Error;

	error_CUDA = cudaMalloc(&params_OptiX.bitmap_out_B, sizeof(REAL_G) * arraySizeReal);
	if (error_CUDA != cudaSuccess) goto Error;

	error_CUDA = cudaMalloc(&params_OptiX.mu_bitmap_out_R, sizeof(REAL_G) * arraySizeReal);
	if (error_CUDA != cudaSuccess) goto Error;
	
	error_CUDA = cudaMalloc(&params_OptiX.mu_bitmap_out_G, sizeof(REAL_G) * arraySizeReal);
	if (error_CUDA != cudaSuccess) goto Error;
	
	error_CUDA = cudaMalloc(&params_OptiX.mu_bitmap_out_B, sizeof(REAL_G) * arraySizeReal);
	if (error_CUDA != cudaSuccess) goto Error;
	
	error_CUDA = cudaMalloc(&params_OptiX.mu_bitmap_out_R_square, sizeof(REAL_G) * arraySizeReal);
	if (error_CUDA != cudaSuccess) goto Error;

	error_CUDA = cudaMalloc(&params_OptiX.mu_bitmap_out_G_square, sizeof(REAL_G) * arraySizeReal);
	if (error_CUDA != cudaSuccess) goto Error;

	error_CUDA = cudaMalloc(&params_OptiX.mu_bitmap_out_B_square, sizeof(REAL_G) * arraySizeReal);
	if (error_CUDA != cudaSuccess) goto Error;

	// ************************************************************************************************

	// Create Gaussian kernel
	memset(buf, 0, sizeof(REAL_G) * arraySizeReal);
	REAL_G sum = ((REAL_G)0.0);
	for (int i = 0; i < kernel_size; ++i) {
		for (int j = 0; j < kernel_size; ++j) {
			REAL_G tmp = exp((-(((j - kernel_radius) * (j - kernel_radius)) + ((i - kernel_radius) * (i - kernel_radius)))) / (((REAL_G)2.0) * sigma * sigma));
			sum += tmp;
			buf[(i * (params_OptiX.width + (kernel_size - 1))) + j] = tmp;
		}
	}
	for (int i = 0; i < kernel_size; ++i) {
		for (int j = 0; j < kernel_size; ++j)
			buf[(i * (params_OptiX.width + (kernel_size - 1))) + j] /= sum;
	}

	// dev_mu_bitmap_out_bitmap_ref_R = dev_kernel
	error_CUDA = cudaMemcpy(params_OptiX.mu_bitmap_out_bitmap_ref_R, buf, sizeof(REAL_G) * arraySizeReal, cudaMemcpyHostToDevice);
	if (error_CUDA != cudaSuccess) goto Error;

	error_CUFFT = DFFT_G(params_OptiX.planr2c, params_OptiX.mu_bitmap_out_bitmap_ref_R, params_OptiX.F_1);
	if (error_CUFFT != CUFFT_SUCCESS) goto Error;

	// ************************************************************************************************

	// Bufor zosta³ uzupe³niony zerami przy okazji tworzenia kernela Gaussowskiego
	for (int pose = 0; pose < params.NUMBER_OF_POSES; ++pose) {
		// R channel
		for (int i = 0; i < params_OptiX.width; ++i) {
			for (int j = 0; j < params_OptiX.width; ++j) {
				unsigned char R = params.bitmap_ref[(pose * (params_OptiX.width * params_OptiX.width)) + ((i * params_OptiX.width) + j)] >> 16;
				buf[(i * (params_OptiX.width + (kernel_size - 1))) + j] = R / ((REAL_G)255.0);
			}
		}

		error_CUDA = cudaMemcpy(params_OptiX.bitmap_ref_R + (pose * arraySizeReal), buf, sizeof(REAL_G) * arraySizeReal, cudaMemcpyHostToDevice);
		if (error_CUDA != cudaSuccess) goto Error;

		// *** *** *** *** ***

		// G channel
		for (int i = 0; i < params_OptiX.width; ++i) {
			for (int j = 0; j < params_OptiX.width; ++j) {
				unsigned char G = (params.bitmap_ref[(pose * (params_OptiX.width * params_OptiX.width)) + ((i * params_OptiX.width) + j)] >> 8) & 255;
				buf[(i * (params_OptiX.width + (kernel_size - 1))) + j] = G / ((REAL_G)255.0);
			}
		}

		error_CUDA = cudaMemcpy(params_OptiX.bitmap_ref_G + (pose * arraySizeReal), buf, sizeof(REAL_G) * arraySizeReal, cudaMemcpyHostToDevice);
		if (error_CUDA != cudaSuccess) goto Error;

		// *** *** *** *** ***

		// B channel
		for (int i = 0; i < params_OptiX.width; ++i) {
			for (int j = 0; j < params_OptiX.width; ++j) {
				unsigned char B = params.bitmap_ref[(pose * (params_OptiX.width * params_OptiX.width)) + ((i * params_OptiX.width) + j)] & 255;
				buf[(i * (params_OptiX.width + (kernel_size - 1))) + j] = B / ((REAL_G)255.0);
			}
		}

		error_CUDA = cudaMemcpy(params_OptiX.bitmap_ref_B + (pose * arraySizeReal), buf, sizeof(REAL_G) * arraySizeReal, cudaMemcpyHostToDevice);
		if (error_CUDA != cudaSuccess) goto Error;
	}

	// ************************************************************************************************

	// Compute mu's for reference images
	for (int pose = 0; pose < params.NUMBER_OF_POSES; ++pose) {
		// R channel
		error_CUFFT = DFFT_G(params_OptiX.planr2c, params_OptiX.bitmap_ref_R + (pose * arraySizeReal), params_OptiX.F_2);
		if (error_CUFFT != CUFFT_SUCCESS) goto Error;

		MultiplyPointwiseComplex<<<(arraySizeComplex + 255) >> 8, 256>>>(params_OptiX.F_2, params_OptiX.F_1, params_OptiX.F_2, arraySizeComplex);
		error_CUDA = cudaGetLastError();
		if (error_CUDA != cudaSuccess) goto Error;

		error_CUFFT = IDFFT_G(params_OptiX.planc2r, params_OptiX.F_2, params_OptiX.mu_bitmap_ref_R + (pose * arraySizeReal));
		if (error_CUFFT != CUFFT_SUCCESS) goto Error;

		// *** *** *** *** ***

		// G channel
		error_CUFFT = DFFT_G(params_OptiX.planr2c, params_OptiX.bitmap_ref_G + (pose * arraySizeReal), params_OptiX.F_2);
		if (error_CUFFT != CUFFT_SUCCESS) goto Error;

		MultiplyPointwiseComplex<<<(arraySizeComplex + 255) >> 8, 256>>>(params_OptiX.F_2, params_OptiX.F_1, params_OptiX.F_2, arraySizeComplex);
		error_CUDA = cudaGetLastError();
		if (error_CUDA != cudaSuccess) goto Error;

		error_CUFFT = IDFFT_G(params_OptiX.planc2r, params_OptiX.F_2, params_OptiX.mu_bitmap_ref_G + (pose * arraySizeReal));
		if (error_CUFFT != CUFFT_SUCCESS) goto Error;

		// *** *** *** *** ***

		// B channel
		error_CUFFT = DFFT_G(params_OptiX.planr2c, params_OptiX.bitmap_ref_B + (pose * arraySizeReal), params_OptiX.F_2);
		if (error_CUFFT != CUFFT_SUCCESS) goto Error;

		MultiplyPointwiseComplex<<<(arraySizeComplex + 255) >> 8, 256>>>(params_OptiX.F_2, params_OptiX.F_1, params_OptiX.F_2, arraySizeComplex);
		error_CUDA = cudaGetLastError();
		if (error_CUDA != cudaSuccess) goto Error;

		error_CUFFT = IDFFT_G(params_OptiX.planc2r, params_OptiX.F_2, params_OptiX.mu_bitmap_ref_B + (pose * arraySizeReal));
		if (error_CUFFT != CUFFT_SUCCESS) goto Error;
	}

	// ************************************************************************************************

	// Compute mu's for reference images square
	for (int pose = 0; pose < params.NUMBER_OF_POSES; ++pose) {
		// R channel
		// mu_bitmap_out_bitmap_ref_R = bitmap_ref_R * bitmap_ref_R
		MultiplyPointwiseReal<<<(arraySizeReal + 255) >> 8, 256>>>(
			params_OptiX.bitmap_ref_R + (pose * arraySizeReal),
			params_OptiX.bitmap_ref_R + (pose * arraySizeReal),
			params_OptiX.mu_bitmap_out_bitmap_ref_R,
			arraySizeReal
			);
		error_CUDA = cudaGetLastError();
		if (error_CUDA != cudaSuccess) goto Error;

		error_CUFFT = DFFT_G(params_OptiX.planr2c, params_OptiX.mu_bitmap_out_bitmap_ref_R, params_OptiX.F_2);
		if (error_CUFFT != CUFFT_SUCCESS) goto Error;

		MultiplyPointwiseComplex<<<(arraySizeComplex + 255) >> 8, 256>>>(params_OptiX.F_2, params_OptiX.F_1, params_OptiX.F_2, arraySizeComplex);
		error_CUDA = cudaGetLastError();
		if (error_CUDA != cudaSuccess) goto Error;

		error_CUFFT = IDFFT_G(params_OptiX.planc2r, params_OptiX.F_2, params_OptiX.mu_bitmap_ref_R_square + (pose * arraySizeReal));
		if (error_CUFFT != CUFFT_SUCCESS) goto Error;

		// *** *** *** *** ***

		// G channel
		// mu_bitmap_out_bitmap_ref_R = bitmap_ref_G * bitmap_ref_G
		MultiplyPointwiseReal<<<(arraySizeReal + 255) >> 8, 256>>>(
			params_OptiX.bitmap_ref_G + (pose * arraySizeReal),
			params_OptiX.bitmap_ref_G + (pose * arraySizeReal),
			params_OptiX.mu_bitmap_out_bitmap_ref_R,
			arraySizeReal
			);
		error_CUDA = cudaGetLastError();
		if (error_CUDA != cudaSuccess) goto Error;

		error_CUFFT = DFFT_G(params_OptiX.planr2c, params_OptiX.mu_bitmap_out_bitmap_ref_R, params_OptiX.F_2);
		if (error_CUFFT != CUFFT_SUCCESS) goto Error;

		MultiplyPointwiseComplex<<<(arraySizeComplex + 255) >> 8, 256>>>(params_OptiX.F_2, params_OptiX.F_1, params_OptiX.F_2, arraySizeComplex);
		error_CUDA = cudaGetLastError();
		if (error_CUDA != cudaSuccess) goto Error;

		error_CUFFT = IDFFT_G(params_OptiX.planc2r, params_OptiX.F_2, params_OptiX.mu_bitmap_ref_G_square + (pose * arraySizeReal));
		if (error_CUFFT != CUFFT_SUCCESS) goto Error;

		// *** *** *** *** ***

		// B channel
		// mu_bitmap_out_bitmap_ref_R = bitmap_ref_B * bitmap_ref_B
		MultiplyPointwiseReal<<<(arraySizeReal + 255) >> 8, 256>>>(
			params_OptiX.bitmap_ref_B + (pose * arraySizeReal),
			params_OptiX.bitmap_ref_B + (pose * arraySizeReal),
			params_OptiX.mu_bitmap_out_bitmap_ref_R,
			arraySizeReal
			);
		error_CUDA = cudaGetLastError();
		if (error_CUDA != cudaSuccess) goto Error;

		error_CUFFT = DFFT_G(params_OptiX.planr2c, params_OptiX.mu_bitmap_out_bitmap_ref_R, params_OptiX.F_2);
		if (error_CUFFT != CUFFT_SUCCESS) goto Error;

		MultiplyPointwiseComplex<<<(arraySizeComplex + 255) >> 8, 256>>>(params_OptiX.F_2, params_OptiX.F_1, params_OptiX.F_2, arraySizeComplex);
		error_CUDA = cudaGetLastError();
		if (error_CUDA != cudaSuccess) goto Error;

		error_CUFFT = IDFFT_G(params_OptiX.planc2r, params_OptiX.F_2, params_OptiX.mu_bitmap_ref_B_square + (pose * arraySizeReal));
		if (error_CUFFT != CUFFT_SUCCESS) goto Error;
	}

	// ************************************************************************************************

	return true;
Error:
	return false;
}

//**************************************************************************************************
//* ZeroOptiXOptimizer                                                                             *
//**************************************************************************************************

extern bool ZeroGradientOptiX(SOptiXRenderParams &params_OptiX) {
	cudaError_t error_CUDA;

	error_CUDA = cudaMemset(params_OptiX.dL_dparams_1, 0, sizeof(REAL4_G) * params_OptiX.numberOfGaussians);
	if (error_CUDA != cudaSuccess) goto Error;

	error_CUDA = cudaMemset(params_OptiX.dL_dparams_2, 0, sizeof(REAL4_G) * params_OptiX.numberOfGaussians);
	if (error_CUDA != cudaSuccess) goto Error;

	error_CUDA = cudaMemset(params_OptiX.dL_dparams_3, 0, sizeof(REAL4_G) * params_OptiX.numberOfGaussians);
	if (error_CUDA != cudaSuccess) goto Error;

	error_CUDA = cudaMemset(params_OptiX.dL_dparams_4, 0, sizeof(REAL2_G) * params_OptiX.numberOfGaussians);
	if (error_CUDA != cudaSuccess) goto Error;

	error_CUDA = cudaMemset(params_OptiX.loss_device, 0, sizeof(double) * 1);
	if (error_CUDA != cudaSuccess) goto Error;

	//**********************************************************************************************

	return true;
Error:
	return false;
}

//**************************************************************************************************
//* RenderOptiX                                                                                    *
//**************************************************************************************************

bool RenderOptiX(SOptiXRenderParams& params_OptiX) {
	cudaError_t error_CUDA;
	OptixResult error_OptiX;
	
	// *********************************************************************************************

	LaunchParams launchParams;
	launchParams.bitmap = params_OptiX.bitmap_out_device;
	launchParams.width = params_OptiX.width;
	launchParams.O = params_OptiX.O;
	launchParams.R = params_OptiX.R;
	launchParams.D = params_OptiX.D;
	launchParams.F = params_OptiX.F;
	launchParams.FOV = params_OptiX.FOV;
	launchParams.traversable = params_OptiX.asHandle;
	launchParams.GC_part_1 = params_OptiX.GC_part_1_1;
	launchParams.GC_part_2 = params_OptiX.GC_part_2_1;
	launchParams.GC_part_3 = params_OptiX.GC_part_3_1;
	launchParams.GC_part_4 = params_OptiX.GC_part_4_1;
	launchParams.Gaussians_indices = params_OptiX.Gaussians_indices;
	launchParams.bitmap_out_R = params_OptiX.bitmap_out_R;
	launchParams.bitmap_out_G = params_OptiX.bitmap_out_G;
	launchParams.bitmap_out_B = params_OptiX.bitmap_out_B;
	launchParams.ray_termination_T_threshold = ray_termination_T_threshold_host; // !!! !!! !!!
	launchParams.last_significant_Gauss_alpha_gradient_precision = last_significant_Gauss_alpha_gradient_precision_host; // !!! !!! !!!
	launchParams.max_Gaussians_per_ray = max_Gaussians_per_ray_host; // !!! !!! !!!

	void *launchParamsBuffer;
	error_CUDA = cudaMalloc(&launchParamsBuffer, sizeof(LaunchParams) * 1);
	if (error_CUDA != cudaSuccess) goto Error;

	error_CUDA = cudaMemcpy(launchParamsBuffer, &launchParams, sizeof(LaunchParams) * 1, cudaMemcpyHostToDevice);
	if (error_CUDA != cudaSuccess) goto Error;

	error_OptiX = optixLaunch(
		params_OptiX.pipeline,
		0,
		(CUdeviceptr)launchParamsBuffer,
		sizeof(LaunchParams) * 1,
		params_OptiX.sbt,
		params_OptiX.width,
		params_OptiX.width,
		1
	);
	if (error_OptiX != OPTIX_SUCCESS) goto Error;

	cudaDeviceSynchronize();
	error_CUDA = cudaGetLastError();
	if (error_CUDA != cudaSuccess) goto Error;

	// *********************************************************************************************

	if (params_OptiX.copyBitmapToHostMemory) {
		error_CUDA = cudaMemcpy(
			params_OptiX.bitmap_out_host,
			params_OptiX.bitmap_out_device,
			sizeof(unsigned) * params_OptiX.width * params_OptiX.width,
			cudaMemcpyDeviceToHost
		);
		if (error_CUDA != cudaSuccess) goto Error;
	}

	// *********************************************************************************************

	return true;
Error:
	return false;
}

//**************************************************************************************************
//* ComputeGradientOptiX                                                                           *
//**************************************************************************************************

__global__ void ComputeGradient(SOptiXRenderParams params_OptiX) {
	int tid = (blockIdx.x * blockDim.x) + threadIdx.x;

	unsigned i = tid / params_OptiX.width;
	unsigned j = tid % params_OptiX.width;

	if ((j < params_OptiX.width) && (i < params_OptiX.width)) {
		REAL3_G d = make_REAL3_G(
			((REAL_G)-0.5) + ((j + ((REAL_G)0.5)) / params_OptiX.width),
			((REAL_G)-0.5) + ((i + ((REAL_G)0.5)) / params_OptiX.width),
			((REAL_G)0.5) / TAN_G(((REAL_G)0.5) * params_OptiX.FOV)
		);
		REAL3_G v = make_REAL3_G(
			MAD_G(params_OptiX.R.x, d.x, MAD_G(params_OptiX.D.x, d.y, params_OptiX.F.x * d.z)),
			MAD_G(params_OptiX.R.y, d.x, MAD_G(params_OptiX.D.y, d.y, params_OptiX.F.y * d.z)),
			MAD_G(params_OptiX.R.z, d.x, MAD_G(params_OptiX.D.z, d.y, params_OptiX.F.z * d.z))
		);

		//******************************************************************************************

		REAL_G R = params_OptiX.bitmap_out_R[(i * (params_OptiX.width + 11 - 1)) + j]; // !!! !!! !!!
		REAL_G G = params_OptiX.bitmap_out_G[(i * (params_OptiX.width + 11 - 1)) + j]; // !!! !!! !!!
		REAL_G B = params_OptiX.bitmap_out_B[(i * (params_OptiX.width + 11 - 1)) + j]; // !!! !!! !!!

		const REAL_G tmp0 = 1 / ((REAL_G)255.0); // !!! !!! !!!
		int color_ref =  params_OptiX.bitmap_ref[(params_OptiX.poseNum * (params_OptiX.width * params_OptiX.width)) + ((i * params_OptiX.width) + j)];
		REAL_G B_ref = (color_ref & 255) * tmp0;
		color_ref = color_ref >> 8;
		REAL_G G_ref = (color_ref & 255) * tmp0;
		color_ref = color_ref >> 8;
		REAL_G R_ref = color_ref * tmp0;

		// Compute loss
		atomicAdd(
			params_OptiX.loss_device,
			MAD_G(R - R_ref, R - R_ref, MAD_G(G - G_ref, G - G_ref, (B - B_ref) * (B - B_ref)))
		);

		REAL_G dI_dalpha = 0;
		
		REAL_G d_dR_dalpha = MAD_G(
			1 - ((REAL_G)lambda),
			(2 * (R - R_ref)) / (3 * params_OptiX.width * params_OptiX.width),
			((REAL_G)lambda) * params_OptiX.mu_bitmap_out_R[((i + 10) * (params_OptiX.width + 11 - 1)) + (j + 10)]
		);
		REAL_G d_dG_dalpha = MAD_G(
			1 - ((REAL_G)lambda),
			(2 * (G - G_ref)) / (3 * params_OptiX.width * params_OptiX.width),
			((REAL_G)lambda) * params_OptiX.mu_bitmap_out_G[((i + 10) * (params_OptiX.width + 11 - 1)) + (j + 10)]
		);
		REAL_G d_dB_dalpha = MAD_G(
			1 - ((REAL_G)lambda),
			(2 * (B - B_ref)) / (3 * params_OptiX.width * params_OptiX.width),
			((REAL_G)lambda) * params_OptiX.mu_bitmap_out_B[((i + 10) * (params_OptiX.width + 11 - 1)) + (j + 10)]
		);

		REAL_G dI_dalpha_max = ABS_G(d_dR_dalpha) + ABS_G(d_dG_dalpha) + ABS_G(d_dB_dalpha); // !!! !!! !!!

		REAL_G alpha_prev = 0; // !!! !!! !!!
		REAL_G alpha_next;

		REAL_G R_Gauss_prev = R; // !!! !!! !!!
		REAL_G G_Gauss_prev = G; // !!! !!! !!!
		REAL_G B_Gauss_prev = B; // !!! !!! !!!

		//******************************************************************************************

		REAL_G T = 1; // !!! !!! !!!
		int k = 0;

		int GaussInd;
		while (k < max_Gaussians_per_ray) {
			GaussInd = params_OptiX.Gaussians_indices[(k * params_OptiX.width * params_OptiX.width) + ((i * params_OptiX.width) + j)]; // !!! !!! !!!
			++k;

			if (GaussInd == -1) return; // !!! !!! !!!

			//**************************************************************************************

			float4 GC_1 = params_OptiX.GC_part_1_1[GaussInd];
			float4 GC_2 = params_OptiX.GC_part_2_1[GaussInd];
			float4 GC_3 = params_OptiX.GC_part_3_1[GaussInd];
			float2 GC_4 = params_OptiX.GC_part_4_1[GaussInd];

			REAL_G aa = ((REAL_G)GC_3.z) * GC_3.z;
			REAL_G bb = ((REAL_G)GC_3.w) * GC_3.w;
			REAL_G cc = ((REAL_G)GC_4.x) * GC_4.x;
			REAL_G dd = ((REAL_G)GC_4.y) * GC_4.y;
			REAL_G s = ((REAL_G)0.5) * (aa + bb + cc + dd);

			REAL_G ab = ((REAL)GC_3.z) * GC_3.w; REAL_G ac = ((REAL_G)GC_3.z) * GC_4.x; REAL_G ad = ((REAL_G)GC_3.z) * GC_4.y;
											     REAL_G bc = ((REAL_G)GC_3.w) * GC_4.x; REAL_G bd = ((REAL_G)GC_3.w) * GC_4.y;
																				        REAL_G cd = ((REAL_G)GC_4.x) * GC_4.y;       

			REAL_G R11 = s - cc - dd;
			REAL_G R12 = bc - ad;
			REAL_G R13 = bd + ac;

			REAL_G R21 = bc + ad;
			REAL_G R22 = s - bb - dd;
			REAL_G R23 = cd - ab;

			REAL_G R31 = bd - ac;
			REAL_G R32 = cd + ab;
			REAL_G R33 = s - bb - cc;

			REAL_G Ox = ((REAL_G)params_OptiX.O.x) - GC_2.x;
			REAL_G Oy = ((REAL_G)params_OptiX.O.y) - GC_2.y;
			REAL_G Oz = ((REAL_G)params_OptiX.O.z) - GC_2.z;

			REAL_G sXInvMinusOne = EXP_G(-GC_2.w); // !!! !!! !!!
			REAL_G sYInvMinusOne = EXP_G(-GC_3.x); // !!! !!! !!!
			REAL_G sZInvMinusOne = EXP_G(-GC_3.y); // !!! !!! !!!

			REAL_G sXInv = 1 + sXInvMinusOne;
			REAL_G Ox_prim = MAD_G(R11, Ox, MAD_G(R21, Oy, R31 * Oz)) * sXInv;
			REAL_G vx_prim = MAD_G(R11, v.x, MAD_G(R21, v.y, R31 * v.z)) * sXInv;

			REAL_G sYInv = 1 + sYInvMinusOne;
			REAL_G Oy_prim = MAD_G(R12, Ox, MAD_G(R22, Oy, R32 * Oz)) * sYInv;
			REAL_G vy_prim = MAD_G(R12, v.x, MAD_G(R22, v.y, R32 * v.z)) * sYInv;

			REAL_G sZInv = 1 + sZInvMinusOne;
			REAL_G Oz_prim = MAD_G(R13, Ox, MAD_G(R23, Oy, R33 * Oz)) * sZInv;
			REAL_G vz_prim = MAD_G(R13, v.x, MAD_G(R23, v.y, R33 * v.z)) * sZInv;

			REAL_G v_dot_v = MAD_G(vx_prim, vx_prim, MAD_G(vy_prim, vy_prim, vz_prim * vz_prim));
			REAL_G O_dot_O = MAD_G(Ox_prim, Ox_prim, MAD_G(Oy_prim, Oy_prim, Oz_prim * Oz_prim));
			REAL_G v_dot_O = MAD_G(vx_prim, Ox_prim, MAD_G(vy_prim, Oy_prim, vz_prim * Oz_prim));
			REAL_G tmp1 = v_dot_O / v_dot_v;
			REAL_G tmp2 = 1 / (1 + EXP_G(-GC_1.w));
			
			#ifndef GRADIENT_OPTIX_USE_DOUBLE_PRECISION
				alpha_next = tmp2 * __saturatef(expf(0.5f * (MAD_G(v_dot_O, tmp1, -O_dot_O) / (s * s)))); // !!! !!! !!!
			#else
				alpha_next = exp(0.5 * (MAD_G(v_dot_O, tmp1, -O_dot_O) / (s * s)));
				alpha_next = (alpha_next < 0) ? 0 : alpha_next;
				alpha_next = (alpha_next > 1) ? 1 : alpha_next;
				alpha_next = tmp2 * alpha_next; // !!! !!! !!!
			#endif
			
			// *************************************************************************************

			REAL_G tmp3 = (1 - alpha_prev); // !!! !!! !!!
			d_dR_dalpha = d_dR_dalpha * tmp3;
			d_dG_dalpha = d_dG_dalpha * tmp3;
			d_dB_dalpha = d_dB_dalpha * tmp3;
			dI_dalpha_max = dI_dalpha_max * tmp3; // !!! !!! !!!
			T = T * tmp3;
			
			// !!! !!! !!!
			dI_dalpha = MAD_G(GC_1.x - R_Gauss_prev, d_dR_dalpha, MAD_G(GC_1.y - G_Gauss_prev, d_dG_dalpha, MAD_G(GC_1.z - B_Gauss_prev, d_dB_dalpha, dI_dalpha)));
			tmp3 = dI_dalpha / (1 - alpha_next);
			tmp3 = (tmp3 < -dI_dalpha_max) ? -dI_dalpha_max : tmp3;
			tmp3 = (tmp3 > dI_dalpha_max) ? dI_dalpha_max : tmp3;
			// !!! !!! !!!

			// *************************************************************************************

			REAL_G dL_dparam;

			// *************************************************************************************

			// dL_d[R, G, B]
			dL_dparam = d_dR_dalpha * alpha_next;
			atomicAdd(((REAL_G *)params_OptiX.dL_dparams_1) + (GaussInd * 4), dL_dparam);
			dL_dparam = d_dG_dalpha * alpha_next;
			atomicAdd(((REAL_G *)params_OptiX.dL_dparams_1) + (GaussInd * 4) + 1, dL_dparam);
			dL_dparam = d_dB_dalpha * alpha_next;
			atomicAdd(((REAL_G *)params_OptiX.dL_dparams_1) + (GaussInd * 4) + 2, dL_dparam);

			// *************************************************************************************

			// !!! !!! !!!
			if ((T * (1 - alpha_next) < ((REAL)ray_termination_T_threshold)) || isnan(tmp3)) {
				dI_dalpha = MAD_G(GC_1.x, d_dR_dalpha, MAD_G(GC_1.y, d_dG_dalpha, GC_1.z * d_dB_dalpha)); // !!! !!! !!!
				break;
			}
			// !!! !!! !!!

			// *************************************************************************************

			tmp3 = (tmp3 * alpha_next);

			// *************************************************************************************

			// dL_dalpha
			dL_dparam = tmp3 * (1.0f - tmp2);
			atomicAdd(((REAL_G *)params_OptiX.dL_dparams_1) + (GaussInd * 4) + 3, dL_dparam);

			// *************************************************************************************

			tmp3 = tmp3 / (s * s);

			// *************************************************************************************

			REAL_G vecx_tmp = tmp3 * MAD_G(-vx_prim, tmp1, Ox_prim); // !!! !!! !!!
			REAL_G vecy_tmp = tmp3 * MAD_G(-vy_prim, tmp1, Oy_prim); // !!! !!! !!!
			REAL_G vecz_tmp = tmp3 * MAD_G(-vz_prim, tmp1, Oz_prim); // !!! !!! !!!

			// *************************************************************************************

			// dL_dsX
			REAL_G dot_product_1 = MAD_G(Ox, R11, MAD_G(Oy, R21, Oz * R31));
			REAL_G dot_product_2 = MAD_G(v.x, R11, MAD_G(v.y, R21, v.z * R31));
			dL_dparam = vecx_tmp * sXInvMinusOne * MAD_G(-tmp1, dot_product_2, dot_product_1);
			atomicAdd(((REAL_G *)params_OptiX.dL_dparams_2) + (GaussInd * 4) + 3, dL_dparam);

			// dL_dsY
			dot_product_1 = MAD_G(Ox, R12, MAD_G(Oy, R22, Oz * R32));
			dot_product_2 = MAD_G(v.x, R12, MAD_G(v.y, R22, v.z * R32));
			dL_dparam = vecy_tmp * sYInvMinusOne * MAD_G(-tmp1, dot_product_2, dot_product_1);
			atomicAdd(((REAL_G *)params_OptiX.dL_dparams_3) + (GaussInd * 4), dL_dparam);

			// dL_dsZ
			dot_product_1 = MAD_G(Ox, R13, MAD_G(Oy, R23, Oz * R33));
			dot_product_2 = MAD_G(v.x, R13, MAD_G(v.y, R23, v.z * R33));
			dL_dparam = vecz_tmp * sZInvMinusOne * MAD_G(-tmp1, dot_product_2, dot_product_1);
			atomicAdd(((REAL_G *)params_OptiX.dL_dparams_3) + (GaussInd * 4) + 1, dL_dparam);

			// *************************************************************************************

			// !!! !!! !!!
			vecx_tmp = vecx_tmp * sXInv;
			vecy_tmp = vecy_tmp * sYInv;
			vecz_tmp = vecz_tmp * sZInv;
			// !!! !!! !!!

			// *************************************************************************************

			// dL_dmX
			dL_dparam = MAD_G(vecx_tmp, R11, MAD_G(vecy_tmp, R12, vecz_tmp * R13));
			atomicAdd(((REAL_G *)params_OptiX.dL_dparams_2) + (GaussInd * 4), dL_dparam);

			// dL_dmY
			dL_dparam = MAD_G(vecx_tmp, R21, MAD_G(vecy_tmp, R22, vecz_tmp * R23));
			atomicAdd(((REAL_G *)params_OptiX.dL_dparams_2) + (GaussInd * 4) + 1, dL_dparam);

			// dL_dmZ
			dL_dparam = MAD_G(vecx_tmp, R31, MAD_G(vecy_tmp, R32, vecz_tmp * R33));
			atomicAdd(((REAL_G *)params_OptiX.dL_dparams_2) + (GaussInd * 4) + 2, dL_dparam);

			// *************************************************************************************

			REAL_G tmp7 = 1 - (aa / s);
			REAL_G tmp8 = 1 - (bb / s);
			REAL_G tmp9 = 1 - (cc / s);
			REAL_G tmp10 = 1 - (dd / s);

			cd = -cd / s; // !!! !!! !!!
			ab = -ab / s; // !!! !!! !!!
			REAL_G a_inv = GC_3.w * cd; // (-bcd) / s
			REAL_G b_inv = GC_3.z * cd; // (-acd) / s
			REAL_G c_inv = ab * GC_4.y; // (-abd) / s
			REAL_G d_inv = ab * GC_4.x; // (-abc) / s

			// *************************************************************************************

			// dL_da
			REAL_G dR11_da = GC_3.z * (tmp7 + tmp8);
			REAL_G dR12_da = MAD_G(-GC_4.y, tmp7, d_inv);
			REAL_G dR13_da = MAD_G(GC_4.x, tmp7, c_inv);

			REAL_G dR21_da = MAD_G(GC_4.y, tmp7, d_inv);
			REAL_G dR22_da = GC_3.z * (tmp7 + tmp9);
			REAL_G dR23_da = MAD_G(-GC_3.w, tmp7, b_inv);

			REAL_G dR31_da = MAD_G(-GC_4.x, tmp7, c_inv);
			REAL_G dR32_da = MAD_G(GC_3.w, tmp7, b_inv);
			REAL_G dR33_da = GC_3.z * (tmp7 + tmp10);

			REAL_G vecx_tmp2 = MAD_G(vecx_tmp, dR11_da, MAD_G(vecy_tmp, dR12_da, vecz_tmp * dR13_da));
			REAL_G vecy_tmp2 = MAD_G(vecx_tmp, dR21_da, MAD_G(vecy_tmp, dR22_da, vecz_tmp * dR23_da));
			REAL_G vecz_tmp2 = MAD_G(vecx_tmp, dR31_da, MAD_G(vecy_tmp, dR32_da, vecz_tmp * dR33_da));

			dot_product_1 = MAD_G(vecx_tmp2, Ox, MAD_G(vecy_tmp2, Oy, vecz_tmp2 * Oz));
			dot_product_2 = MAD_G(vecx_tmp2, v.x, MAD_G(vecy_tmp2, v.y, vecz_tmp2 * v.z));

			dL_dparam = MAD_G(tmp1, dot_product_2, -dot_product_1);
			atomicAdd(((REAL_G *)params_OptiX.dL_dparams_3) + (GaussInd * 4) + 2, dL_dparam);

			// *** *** *** *** ***

			// dL_db
			dR11_da = GC_3.w * (tmp8 + tmp7);
			dR12_da = MAD_G(GC_4.x, tmp8, -c_inv);
			dR13_da = MAD_G(GC_4.y, tmp8, d_inv);

			dR21_da = MAD_G(GC_4.x, tmp8, c_inv);
			dR22_da = -GC_3.w * (tmp8 + tmp10);
			dR23_da = MAD_G(-GC_3.z, tmp8, a_inv);

			dR31_da = MAD_G(GC_4.y, tmp8, -d_inv);
			dR32_da = MAD_G(GC_3.z, tmp8, a_inv);
			dR33_da = -GC_3.w * (tmp8 + tmp9);

			vecx_tmp2 = MAD_G(vecx_tmp, dR11_da, MAD_G(vecy_tmp, dR12_da, vecz_tmp * dR13_da));
			vecy_tmp2 = MAD_G(vecx_tmp, dR21_da, MAD_G(vecy_tmp, dR22_da, vecz_tmp * dR23_da));
			vecz_tmp2 = MAD_G(vecx_tmp, dR31_da, MAD_G(vecy_tmp, dR32_da, vecz_tmp * dR33_da));

			dot_product_1 = MAD_G(vecx_tmp2, Ox, MAD_G(vecy_tmp2, Oy, vecz_tmp2 * Oz));
			dot_product_2 = MAD_G(vecx_tmp2, v.x, MAD_G(vecy_tmp2, v.y, vecz_tmp2 * v.z));

			dL_dparam = MAD_G(tmp1, dot_product_2, -dot_product_1);
			atomicAdd(((REAL_G *)params_OptiX.dL_dparams_3) + (GaussInd * 4) + 3, dL_dparam);

			// *** *** *** *** ***

			// dL_dc
			dR11_da = -GC_4.x * (tmp9 + tmp10);
			dR12_da = MAD_G(GC_3.w, tmp9, -b_inv);
			dR13_da = MAD_G(GC_3.z, tmp9, a_inv);

			dR21_da = MAD_G(GC_3.w, tmp9, b_inv);
			dR22_da = GC_4.x * (tmp9 + tmp7);
			dR23_da = MAD_G(GC_4.y, tmp9, -d_inv);

			dR31_da = MAD_G(-GC_3.z, tmp9, a_inv);
			dR32_da = MAD_G(GC_4.y, tmp9, d_inv);
			dR33_da = -GC_4.x * (tmp9 + tmp8);

			vecx_tmp2 = MAD_G(vecx_tmp, dR11_da, MAD_G(vecy_tmp, dR12_da, vecz_tmp * dR13_da));
			vecy_tmp2 = MAD_G(vecx_tmp, dR21_da, MAD_G(vecy_tmp, dR22_da, vecz_tmp * dR23_da));
			vecz_tmp2 = MAD_G(vecx_tmp, dR31_da, MAD_G(vecy_tmp, dR32_da, vecz_tmp * dR33_da));

			dot_product_1 = MAD_G(vecx_tmp2, Ox, MAD_G(vecy_tmp2, Oy, vecz_tmp2 * Oz));
			dot_product_2 = MAD_G(vecx_tmp2, v.x, MAD_G(vecy_tmp2, v.y, vecz_tmp2 * v.z));

			dL_dparam = MAD_G(tmp1, dot_product_2, -dot_product_1);
			atomicAdd(((REAL_G *)params_OptiX.dL_dparams_4) + (GaussInd * 2), dL_dparam);

			// *** *** *** *** ***

			// dL_dd
			dR11_da = -GC_4.y * (tmp10 + tmp9);
			dR12_da = MAD_G(-GC_3.z, tmp10, a_inv);
			dR13_da = MAD_G(GC_3.w, tmp10, b_inv);

			dR21_da = MAD_G(GC_3.z, tmp10, a_inv);
			dR22_da = -GC_4.y * (tmp10 + tmp8);
			dR23_da = MAD_G(GC_4.x, tmp10, -c_inv);

			dR31_da = MAD_G(GC_3.w, tmp10, -b_inv);
			dR32_da = MAD_G(GC_4.x, tmp10, c_inv);
			dR33_da = GC_4.y * (tmp10 + tmp7);

			vecx_tmp2 = MAD_G(vecx_tmp, dR11_da, MAD_G(vecy_tmp, dR12_da, vecz_tmp * dR13_da));
			vecy_tmp2 = MAD_G(vecx_tmp, dR21_da, MAD_G(vecy_tmp, dR22_da, vecz_tmp * dR23_da));
			vecz_tmp2 = MAD_G(vecx_tmp, dR31_da, MAD_G(vecy_tmp, dR32_da, vecz_tmp * dR33_da));

			dot_product_1 = MAD_G(vecx_tmp2, Ox, MAD_G(vecy_tmp2, Oy, vecz_tmp2 * Oz));
			dot_product_2 = MAD_G(vecx_tmp2, v.x, MAD_G(vecy_tmp2, v.y, vecz_tmp2 * v.z));

			dL_dparam = MAD_G(tmp1, dot_product_2, -dot_product_1);
			atomicAdd(((REAL_G *)params_OptiX.dL_dparams_4) + (GaussInd * 2) + 1, dL_dparam);

			// *************************************************************************************

			alpha_prev = alpha_next;
			R_Gauss_prev = GC_1.x;
			G_Gauss_prev = GC_1.y;
			B_Gauss_prev = GC_1.z;
		}

		// *****************************************************************************************

		if (GaussInd != -1) {
			int LastGaussInd = GaussInd;
			while (k  <max_Gaussians_per_ray) {
				int GaussInd = params_OptiX.Gaussians_indices[(k * params_OptiX.width * params_OptiX.width) + ((i * params_OptiX.width) + j)]; // !!! !!! !!!
				++k;

				if (GaussInd == -1) break; // !!! !!! !!!

				// *********************************************************************************

				float4 GC_1 = params_OptiX.GC_part_1_1[GaussInd];
				float4 GC_2 = params_OptiX.GC_part_2_1[GaussInd];
				float4 GC_3 = params_OptiX.GC_part_3_1[GaussInd];
				float2 GC_4 = params_OptiX.GC_part_4_1[GaussInd];

				REAL_G aa = ((REAL_G)GC_3.z) * GC_3.z;
				REAL_G bb = ((REAL_G)GC_3.w) * GC_3.w;
				REAL_G cc = ((REAL_G)GC_4.x) * GC_4.x;
				REAL_G dd = ((REAL_G)GC_4.y) * GC_4.y;
				REAL_G s = ((REAL_G)0.5) * (aa + bb + cc + dd);

				REAL_G ab = ((REAL)GC_3.z) * GC_3.w; REAL_G ac = ((REAL_G)GC_3.z) * GC_4.x; REAL_G ad = ((REAL_G)GC_3.z) * GC_4.y;
													 REAL_G bc = ((REAL_G)GC_3.w) * GC_4.x; REAL_G bd = ((REAL_G)GC_3.w) * GC_4.y;
																							REAL_G cd = ((REAL_G)GC_4.x) * GC_4.y;       

				REAL_G R11 = s - cc - dd;
				REAL_G R12 = bc - ad;
				REAL_G R13 = bd + ac;

				REAL_G R21 = bc + ad;
				REAL_G R22 = s - bb - dd;
				REAL_G R23 = cd - ab;

				REAL_G R31 = bd - ac;
				REAL_G R32 = cd + ab;
				REAL_G R33 = s - bb - cc;

				REAL_G Ox = ((REAL_G)params_OptiX.O.x) - GC_2.x;
				REAL_G Oy = ((REAL_G)params_OptiX.O.y) - GC_2.y;
				REAL_G Oz = ((REAL_G)params_OptiX.O.z) - GC_2.z;

				REAL_G sXInvMinusOne = EXP_G(-GC_2.w); // !!! !!! !!!
				REAL_G sYInvMinusOne = EXP_G(-GC_3.x); // !!! !!! !!!
				REAL_G sZInvMinusOne = EXP_G(-GC_3.y); // !!! !!! !!!

				REAL_G sXInv = 1 + sXInvMinusOne;
				REAL_G Ox_prim = MAD_G(R11, Ox, MAD_G(R21, Oy, R31 * Oz)) * sXInv;
				REAL_G vx_prim = MAD_G(R11, v.x, MAD_G(R21, v.y, R31 * v.z)) * sXInv;

				REAL_G sYInv = 1 + sYInvMinusOne;
				REAL_G Oy_prim = MAD_G(R12, Ox, MAD_G(R22, Oy, R32 * Oz)) * sYInv;
				REAL_G vy_prim = MAD_G(R12, v.x, MAD_G(R22, v.y, R32 * v.z)) * sYInv;

				REAL_G sZInv = 1 + sZInvMinusOne;
				REAL_G Oz_prim = MAD_G(R13, Ox, MAD_G(R23, Oy, R33 * Oz)) * sZInv;
				REAL_G vz_prim = MAD_G(R13, v.x, MAD_G(R23, v.y, R33 * v.z)) * sZInv;

				REAL_G v_dot_v = MAD_G(vx_prim, vx_prim, MAD_G(vy_prim, vy_prim, vz_prim * vz_prim));
				REAL_G O_dot_O = MAD_G(Ox_prim, Ox_prim, MAD_G(Oy_prim, Oy_prim, Oz_prim * Oz_prim));
				REAL_G v_dot_O = MAD_G(vx_prim, Ox_prim, MAD_G(vy_prim, Oy_prim, vz_prim * Oz_prim));
				REAL_G tmp1 = v_dot_O / v_dot_v;
				REAL_G tmp2 = 1 / (1 + EXP_G(-GC_1.w));

				#ifndef GRADIENT_OPTIX_USE_DOUBLE_PRECISION
					alpha_next = tmp2 * __saturatef(expf(0.5f * (MAD_G(v_dot_O, tmp1, -O_dot_O) / (s * s)))); // !!! !!! !!!
				#else
					alpha_next = exp(0.5 * (MAD_G(v_dot_O, tmp1, -O_dot_O) / (s * s)));
					alpha_next = (alpha_next < 0) ? 0 : alpha_next;
					alpha_next = (alpha_next > 1) ? 1 : alpha_next;
					alpha_next = tmp2 * alpha_next; // !!! !!! !!!
				#endif

				// *********************************************************************************

				dI_dalpha = MAD_G(-alpha_next, MAD_G(GC_1.x, d_dR_dalpha, MAD_G(GC_1.y, d_dG_dalpha, GC_1.z * d_dB_dalpha)), dI_dalpha); // !!! !!! !!!

				tmp2 = (1 - alpha_next);
				d_dR_dalpha = d_dR_dalpha * tmp2;
				d_dG_dalpha = d_dG_dalpha * tmp2;
				d_dB_dalpha = d_dB_dalpha * tmp2;
			}

			// *************************************************************************************

			float4 GC_1 = params_OptiX.GC_part_1_1[LastGaussInd];
			float4 GC_2 = params_OptiX.GC_part_2_1[LastGaussInd];
			float4 GC_3 = params_OptiX.GC_part_3_1[LastGaussInd];
			float2 GC_4 = params_OptiX.GC_part_4_1[LastGaussInd];

			REAL_G aa = ((REAL_G)GC_3.z) * GC_3.z;
			REAL_G bb = ((REAL_G)GC_3.w) * GC_3.w;
			REAL_G cc = ((REAL_G)GC_4.x) * GC_4.x;
			REAL_G dd = ((REAL_G)GC_4.y) * GC_4.y;
			REAL_G s = ((REAL_G)0.5) * (aa + bb + cc + dd);

			REAL_G ab = ((REAL)GC_3.z) * GC_3.w;   REAL_G ac = ((REAL_G)GC_3.z) * GC_4.x; REAL_G ad = ((REAL_G)GC_3.z) * GC_4.y;
												   REAL_G bc = ((REAL_G)GC_3.w) * GC_4.x; REAL_G bd = ((REAL_G)GC_3.w) * GC_4.y;
																						  REAL_G cd = ((REAL_G)GC_4.x) * GC_4.y;       

			REAL_G R11 = s - cc - dd;
			REAL_G R12 = bc - ad;
			REAL_G R13 = bd + ac;

			REAL_G R21 = bc + ad;
			REAL_G R22 = s - bb - dd;
			REAL_G R23 = cd - ab;

			REAL_G R31 = bd - ac;
			REAL_G R32 = cd + ab;
			REAL_G R33 = s - bb - cc;

			REAL_G Ox = ((REAL_G)params_OptiX.O.x) - GC_2.x;
			REAL_G Oy = ((REAL_G)params_OptiX.O.y) - GC_2.y;
			REAL_G Oz = ((REAL_G)params_OptiX.O.z) - GC_2.z;

			REAL_G sXInvMinusOne = EXP_G(-GC_2.w); // !!! !!! !!!
			REAL_G sYInvMinusOne = EXP_G(-GC_3.x); // !!! !!! !!!
			REAL_G sZInvMinusOne = EXP_G(-GC_3.y); // !!! !!! !!!

			REAL_G sXInv = 1 + sXInvMinusOne;
			REAL_G Ox_prim = MAD_G(R11, Ox, MAD_G(R21, Oy, R31 * Oz)) * sXInv;
			REAL_G vx_prim = MAD_G(R11, v.x, MAD_G(R21, v.y, R31 * v.z)) * sXInv;

			REAL_G sYInv = 1 + sYInvMinusOne;
			REAL_G Oy_prim = MAD_G(R12, Ox, MAD_G(R22, Oy, R32 * Oz)) * sYInv;
			REAL_G vy_prim = MAD_G(R12, v.x, MAD_G(R22, v.y, R32 * v.z)) * sYInv;

			REAL_G sZInv = 1 + sZInvMinusOne;
			REAL_G Oz_prim = MAD_G(R13, Ox, MAD_G(R23, Oy, R33 * Oz)) * sZInv;
			REAL_G vz_prim = MAD_G(R13, v.x, MAD_G(R23, v.y, R33 * v.z)) * sZInv;

			REAL_G v_dot_v = MAD_G(vx_prim, vx_prim, MAD_G(vy_prim, vy_prim, vz_prim * vz_prim));
			REAL_G O_dot_O = MAD_G(Ox_prim, Ox_prim, MAD_G(Oy_prim, Oy_prim, Oz_prim * Oz_prim));
			REAL_G v_dot_O = MAD_G(vx_prim, Ox_prim, MAD_G(vy_prim, Oy_prim, vz_prim * Oz_prim));
			REAL_G tmp1 = v_dot_O / v_dot_v;
			REAL_G tmp2 = 1 / (1 + EXP_G(-GC_1.w));

			#ifndef GRADIENT_OPTIX_USE_DOUBLE_PRECISION
				alpha_next = tmp2 * __saturatef(expf(0.5f * (MAD_G(v_dot_O, tmp1, -O_dot_O) / (s * s)))); // !!! !!! !!!
			#else
				alpha_next = exp(0.5 * (MAD_G(v_dot_O, tmp1, -O_dot_O) / (s * s)));
				alpha_next = (alpha_next < 0) ? 0 : alpha_next;
				alpha_next = (alpha_next > 1) ? 1 : alpha_next;
				alpha_next = tmp2 * alpha_next; // !!! !!! !!!
			#endif

			// *************************************************************************************

			REAL_G tmp3 = dI_dalpha; // !!! !!! !!!
			
			// *************************************************************************************

			REAL_G dL_dparam;

			// *************************************************************************************

			tmp3 = (tmp3 * alpha_next);

			// *************************************************************************************

			// dL_dalpha
			dL_dparam = tmp3 * (1.0f - tmp2);
			atomicAdd(((REAL_G *)params_OptiX.dL_dparams_1) + (LastGaussInd * 4) + 3, dL_dparam);

			// *************************************************************************************

			tmp3 = tmp3 / (s * s);

			// *************************************************************************************

			REAL_G vecx_tmp = tmp3 * MAD_G(-vx_prim, tmp1, Ox_prim); // !!! !!! !!!
			REAL_G vecy_tmp = tmp3 * MAD_G(-vy_prim, tmp1, Oy_prim); // !!! !!! !!!
			REAL_G vecz_tmp = tmp3 * MAD_G(-vz_prim, tmp1, Oz_prim); // !!! !!! !!!

			// *************************************************************************************

			// dL_dsX
			REAL_G dot_product_1 = MAD_G(Ox, R11, MAD_G(Oy, R21, Oz * R31));
			REAL_G dot_product_2 = MAD_G(v.x, R11, MAD_G(v.y, R21, v.z * R31));
			dL_dparam = vecx_tmp * sXInvMinusOne * MAD_G(-tmp1, dot_product_2, dot_product_1);
			atomicAdd(((REAL_G *)params_OptiX.dL_dparams_2) + (LastGaussInd * 4) + 3, dL_dparam);

			// dL_dsY
			dot_product_1 = MAD_G(Ox, R12, MAD_G(Oy, R22, Oz * R32));
			dot_product_2 = MAD_G(v.x, R12, MAD_G(v.y, R22, v.z * R32));
			dL_dparam = vecy_tmp * sYInvMinusOne * MAD_G(-tmp1, dot_product_2, dot_product_1);
			atomicAdd(((REAL_G *)params_OptiX.dL_dparams_3) + (LastGaussInd * 4), dL_dparam);

			// dL_dsZ
			dot_product_1 = MAD_G(Ox, R13, MAD_G(Oy, R23, Oz * R33));
			dot_product_2 = MAD_G(v.x, R13, MAD_G(v.y, R23, v.z * R33));
			dL_dparam = vecz_tmp * sZInvMinusOne * MAD_G(-tmp1, dot_product_2, dot_product_1);
			atomicAdd(((REAL_G *)params_OptiX.dL_dparams_3) + (LastGaussInd * 4) + 1, dL_dparam);

			// *************************************************************************************

			// !!! !!! !!!
			vecx_tmp = vecx_tmp * sXInv;
			vecy_tmp = vecy_tmp * sYInv;
			vecz_tmp = vecz_tmp * sZInv;
			// !!! !!! !!!

			// *************************************************************************************

			// dL_dmX
			dL_dparam = MAD_G(vecx_tmp, R11, MAD_G(vecy_tmp, R12, vecz_tmp * R13));
			atomicAdd(((REAL_G *)params_OptiX.dL_dparams_2) + (LastGaussInd * 4), dL_dparam);

			// dL_dmY
			dL_dparam = MAD_G(vecx_tmp, R21, MAD_G(vecy_tmp, R22, vecz_tmp * R23));
			atomicAdd(((REAL_G *)params_OptiX.dL_dparams_2) + (LastGaussInd * 4) + 1, dL_dparam);

			// dL_dmZ
			dL_dparam = MAD_G(vecx_tmp, R31, MAD_G(vecy_tmp, R32, vecz_tmp * R33));
			atomicAdd(((REAL_G *)params_OptiX.dL_dparams_2) + (LastGaussInd * 4) + 2, dL_dparam);

			// *************************************************************************************

			REAL_G tmp7 = 1 - (aa / s);
			REAL_G tmp8 = 1 - (bb / s);
			REAL_G tmp9 = 1 - (cc / s);
			REAL_G tmp10 = 1 - (dd / s);

			cd = -cd / s; // !!! !!! !!!
			ab = -ab / s; // !!! !!! !!!
			REAL_G a_inv = GC_3.w * cd; // (-bcd) / s
			REAL_G b_inv = GC_3.z * cd; // (-acd) / s
			REAL_G c_inv = ab * GC_4.y; // (-abd) / s
			REAL_G d_inv = ab * GC_4.x; // (-abc) / s

			// *************************************************************************************

			// dL_da
			REAL_G dR11_da = GC_3.z * (tmp7 + tmp8);
			REAL_G dR12_da = MAD_G(-GC_4.y, tmp7, d_inv);
			REAL_G dR13_da = MAD_G(GC_4.x, tmp7, c_inv);

			REAL_G dR21_da = MAD_G(GC_4.y, tmp7, d_inv);
			REAL_G dR22_da = GC_3.z * (tmp7 + tmp9);
			REAL_G dR23_da = MAD_G(-GC_3.w, tmp7, b_inv);

			REAL_G dR31_da = MAD_G(-GC_4.x, tmp7, c_inv);
			REAL_G dR32_da = MAD_G(GC_3.w, tmp7, b_inv);
			REAL_G dR33_da = GC_3.z * (tmp7 + tmp10);

			REAL_G vecx_tmp2 = MAD_G(vecx_tmp, dR11_da, MAD_G(vecy_tmp, dR12_da, vecz_tmp * dR13_da));
			REAL_G vecy_tmp2 = MAD_G(vecx_tmp, dR21_da, MAD_G(vecy_tmp, dR22_da, vecz_tmp * dR23_da));
			REAL_G vecz_tmp2 = MAD_G(vecx_tmp, dR31_da, MAD_G(vecy_tmp, dR32_da, vecz_tmp * dR33_da));

			dot_product_1 = MAD_G(vecx_tmp2, Ox, MAD_G(vecy_tmp2, Oy, vecz_tmp2 * Oz));
			dot_product_2 = MAD_G(vecx_tmp2, v.x, MAD_G(vecy_tmp2, v.y, vecz_tmp2 * v.z));

			dL_dparam = MAD_G(tmp1, dot_product_2, -dot_product_1);
			atomicAdd(((REAL_G *)params_OptiX.dL_dparams_3) + (LastGaussInd * 4) + 2, dL_dparam);

			// *** *** *** *** ***

			// dL_db
			dR11_da = GC_3.w * (tmp8 + tmp7);
			dR12_da = MAD_G(GC_4.x, tmp8, -c_inv);
			dR13_da = MAD_G(GC_4.y, tmp8, d_inv);

			dR21_da = MAD_G(GC_4.x, tmp8, c_inv);
			dR22_da = -GC_3.w * (tmp8 + tmp10);
			dR23_da = MAD_G(-GC_3.z, tmp8, a_inv);

			dR31_da = MAD_G(GC_4.y, tmp8, -d_inv);
			dR32_da = MAD_G(GC_3.z, tmp8, a_inv);
			dR33_da = -GC_3.w * (tmp8 + tmp9);

			vecx_tmp2 = MAD_G(vecx_tmp, dR11_da, MAD_G(vecy_tmp, dR12_da, vecz_tmp * dR13_da));
			vecy_tmp2 = MAD_G(vecx_tmp, dR21_da, MAD_G(vecy_tmp, dR22_da, vecz_tmp * dR23_da));
			vecz_tmp2 = MAD_G(vecx_tmp, dR31_da, MAD_G(vecy_tmp, dR32_da, vecz_tmp * dR33_da));

			dot_product_1 = MAD_G(vecx_tmp2, Ox, MAD_G(vecy_tmp2, Oy, vecz_tmp2 * Oz));
			dot_product_2 = MAD_G(vecx_tmp2, v.x, MAD_G(vecy_tmp2, v.y, vecz_tmp2 * v.z));

			dL_dparam = MAD_G(tmp1, dot_product_2, -dot_product_1);
			atomicAdd(((REAL_G *)params_OptiX.dL_dparams_3) + (LastGaussInd * 4) + 3, dL_dparam);

			// *** *** *** *** ***

			// dL_dc
			dR11_da = -GC_4.x * (tmp9 + tmp10);
			dR12_da = MAD_G(GC_3.w, tmp9, -b_inv);
			dR13_da = MAD_G(GC_3.z, tmp9, a_inv);

			dR21_da = MAD_G(GC_3.w, tmp9, b_inv);
			dR22_da = GC_4.x * (tmp9 + tmp7);
			dR23_da = MAD_G(GC_4.y, tmp9, -d_inv);

			dR31_da = MAD_G(-GC_3.z, tmp9, a_inv);
			dR32_da = MAD_G(GC_4.y, tmp9, d_inv);
			dR33_da = -GC_4.x * (tmp9 + tmp8);

			vecx_tmp2 = MAD_G(vecx_tmp, dR11_da, MAD_G(vecy_tmp, dR12_da, vecz_tmp * dR13_da));
			vecy_tmp2 = MAD_G(vecx_tmp, dR21_da, MAD_G(vecy_tmp, dR22_da, vecz_tmp * dR23_da));
			vecz_tmp2 = MAD_G(vecx_tmp, dR31_da, MAD_G(vecy_tmp, dR32_da, vecz_tmp * dR33_da));

			dot_product_1 = MAD_G(vecx_tmp2, Ox, MAD_G(vecy_tmp2, Oy, vecz_tmp2 * Oz));
			dot_product_2 = MAD_G(vecx_tmp2, v.x, MAD_G(vecy_tmp2, v.y, vecz_tmp2 * v.z));

			dL_dparam = MAD_G(tmp1, dot_product_2, -dot_product_1);
			atomicAdd(((REAL_G *)params_OptiX.dL_dparams_4) + (LastGaussInd * 2), dL_dparam);

			// *** *** *** *** ***

			// dL_dd
			dR11_da = -GC_4.y * (tmp10 + tmp9);
			dR12_da = MAD_G(-GC_3.z, tmp10, a_inv);
			dR13_da = MAD_G(GC_3.w, tmp10, b_inv);

			dR21_da = MAD_G(GC_3.z, tmp10, a_inv);
			dR22_da = -GC_4.y * (tmp10 + tmp8);
			dR23_da = MAD_G(GC_4.x, tmp10, -c_inv);

			dR31_da = MAD_G(GC_3.w, tmp10, -b_inv);
			dR32_da = MAD_G(GC_4.x, tmp10, c_inv);
			dR33_da = GC_4.y * (tmp10 + tmp7);

			vecx_tmp2 = MAD_G(vecx_tmp, dR11_da, MAD_G(vecy_tmp, dR12_da, vecz_tmp * dR13_da));
			vecy_tmp2 = MAD_G(vecx_tmp, dR21_da, MAD_G(vecy_tmp, dR22_da, vecz_tmp * dR23_da));
			vecz_tmp2 = MAD_G(vecx_tmp, dR31_da, MAD_G(vecy_tmp, dR32_da, vecz_tmp * dR33_da));

			dot_product_1 = MAD_G(vecx_tmp2, Ox, MAD_G(vecy_tmp2, Oy, vecz_tmp2 * Oz));
			dot_product_2 = MAD_G(vecx_tmp2, v.x, MAD_G(vecy_tmp2, v.y, vecz_tmp2 * v.z));

			dL_dparam = MAD_G(tmp1, dot_product_2, -dot_product_1);
			atomicAdd(((REAL_G *)params_OptiX.dL_dparams_4) + (LastGaussInd * 2) + 1, dL_dparam);
		}
	}
}

//**************************************************************************************************
//* UpdateGradientOptiX                                                                            *
//**************************************************************************************************

static __device__ float RandomFloat(unsigned n) {
	const unsigned a = 1664525;
	const unsigned c = 1013904223;

	unsigned tmp1 = 1;
	unsigned tmp2 = a;
	unsigned tmp3 = 0;
	while (n != 0) {
		if ((n & 1) != 0) tmp3 = (tmp2 * tmp3) + tmp1;
		tmp1 = (tmp2 * tmp1) + tmp1;
		tmp2 = tmp2 * tmp2;
		n >>= 1;
	}
	float result = __uint_as_float(1065353216 | ((tmp3 * c) & 8388607)) - 0.99999994f;
	return result;
}

// *************************************************************************************************

static __device__ void RandomNormalFloat(unsigned n, float& Z1, float& Z2) {
	float U1 = RandomFloat(n);
	float U2 = RandomFloat(n + 1); // !!! !!! !!!
	float tmp1 = sqrtf(-2.0f * __logf(U1));
	float tmp2 = 2.0f * M_PI * U2;
	float sine;
	float cosine;
	sincosf(tmp2, &sine, &cosine);
	Z1 = tmp1 * cosine;
	Z2 = tmp1 * sine;
}

// *************************************************************************************************

static __device__ void RandomMultinormalFloat(
	float3 m,
	float3 scale,
	float4 q,
	float Z1, float Z2, float Z3,
	float3 &P
) {
	float3 P_prim = make_float3(Z1 * scale.x, Z2 * scale.y, Z3 * scale.z);

	float aa = q.x * q.x;
	float bb = q.y * q.y;
	float cc = q.z * q.z;
	float dd = q.w * q.w;

	float s = 2.0f / (aa + bb + cc + dd);

	float bs = q.y * s;  float cs = q.z * s;  float ds = q.w * s;
	float ab = q.x * bs; float ac = q.x * cs; float ad = q.x * ds;
	bb = bb * s;         float bc = q.y * cs; float bd = q.y * ds;
	cc = cc * s;         float cd = q.z * ds;       dd = dd * s;

	float R11 = 1.0f - cc - dd;
	float R12 = bc - ad;
	float R13 = bd + ac;

	float R21 = bc + ad;
	float R22 = 1.0f - bb - dd;
	float R23 = cd - ab;

	float R31 = bd - ac;
	float R32 = cd + ab;
	float R33 = 1.0f - bb - cc;

	P = make_float3(
		m.x + ((R11 * P_prim.x) + (R12 * P_prim.y) + (R13 * P_prim.z)),
		m.y + ((R21 * P_prim.x) + (R22 * P_prim.y) + (R23 * P_prim.z)),
		m.z + ((R31 * P_prim.x) + (R32 * P_prim.y) + (R33 * P_prim.z))
	);
}

// *************************************************************************************************

__global__ void dev_UpdateGradientOptiX(SOptiXRenderParams params_OptiX) {
	const float beta1 = 0.9f;
	const float beta2 = 0.999f;
	const float epsilon = 0.00000001f;
	float t = params_OptiX.epoch;

	float tmp1 = 1.0f / (1.0f - powf(beta1, t));
	float tmp2 = 1.0f / (1.0f - powf(beta2, t));

	// *****************************************************************************************

	__shared__ unsigned counter1;
	__shared__ unsigned counter2;

	int tid = (blockIdx.x * blockDim.x) + threadIdx.x;

	// *****************************************************************************************

	REAL4_G dL_dparam_1;
	float4 m1;
	float4 v1;
	float4 GC_1;

	REAL4_G dL_dparam_2;
	float4 m2;
	float4 v2;
	float4 GC_2;

	float3 dm;

	REAL4_G dL_dparam_3;
	float4 m3;
	float4 v3;
	float4 GC_3;

	float3 scale;

	REAL2_G dL_dparam_4;
	float2 m4;
	float2 v4;
	float2 GC_4;

	bool isOpaqueEnough;
	bool isMovedEnough;
	bool isBigEnough;
	bool isNotTooBig;
	bool isBigEnoughToSplit;

	if (tid < params_OptiX.numberOfGaussians) {
		dL_dparam_1 = ((REAL4_G *)params_OptiX.dL_dparams_1)[tid];
		m1 = ((float4 *)params_OptiX.m11)[tid];
		v1 = ((float4 *)params_OptiX.v11)[tid];
		GC_1 = ((float4 *)params_OptiX.GC_part_1_1)[tid];

		m1.x = (beta1 * m1.x) + ((1.0f - beta1) * dL_dparam_1.x);
		m1.y = (beta1 * m1.y) + ((1.0f - beta1) * dL_dparam_1.y);
		m1.z = (beta1 * m1.z) + ((1.0f - beta1) * dL_dparam_1.z);
		m1.w = (beta1 * m1.w) + ((1.0f - beta1) * dL_dparam_1.w);

		v1.x = (beta2 * v1.x) + ((1.0f - beta2) * dL_dparam_1.x * dL_dparam_1.x);
		v1.y = (beta2 * v1.y) + ((1.0f - beta2) * dL_dparam_1.y * dL_dparam_1.y);
		v1.z = (beta2 * v1.z) + ((1.0f - beta2) * dL_dparam_1.z * dL_dparam_1.z);
		v1.w = (beta2 * v1.w) + ((1.0f - beta2) * dL_dparam_1.w * dL_dparam_1.w);

		float lr = lr_RGB * expf(params_OptiX.epoch * lr_RGB_exponential_decay_coefficient);
		if (lr < 0.001f) lr = 0.001f;

		// R
		GC_1.x -= ((lr * (m1.x * tmp1)) / (sqrtf(v1.x * tmp2) + epsilon));
		GC_1.x = __saturatef(GC_1.x);
		// G
		GC_1.y -= ((lr * (m1.y * tmp1)) / (sqrtf(v1.y * tmp2) + epsilon));
		GC_1.y = __saturatef(GC_1.y);
		// B
		GC_1.z -= ((lr * (m1.z * tmp1)) / (sqrtf(v1.z * tmp2) + epsilon));
		GC_1.z = __saturatef(GC_1.z);

		lr = lr_alpha * expf(params_OptiX.epoch * lr_alpha_exponential_decay_coefficient);
		if (lr < 0.01f) lr = 0.01f;

		// alpha
		GC_1.w -= ((lr * (m1.w * tmp1)) / (sqrtf(v1.w * tmp2) + epsilon));

		isOpaqueEnough = (GC_1.w >= alpha_threshold_for_Gauss_removal);

		// *****************************************************************************************

		dL_dparam_2 = ((REAL4_G *)params_OptiX.dL_dparams_2)[tid];
		m2 = ((float4 *)params_OptiX.m21)[tid];
		v2 = ((float4 *)params_OptiX.v21)[tid];
		GC_2 = ((float4 *)params_OptiX.GC_part_2_1)[tid];

		m2.x = (beta1 * m2.x) + ((1.0f - beta1) * dL_dparam_2.x);
		m2.y = (beta1 * m2.y) + ((1.0f - beta1) * dL_dparam_2.y);
		m2.z = (beta1 * m2.z) + ((1.0f - beta1) * dL_dparam_2.z);
		m2.w = (beta1 * m2.w) + ((1.0f - beta1) * dL_dparam_2.w);

		v2.x = (beta2 * v2.x) + ((1.0f - beta2) * dL_dparam_2.x * dL_dparam_2.x);
		v2.y = (beta2 * v2.y) + ((1.0f - beta2) * dL_dparam_2.y * dL_dparam_2.y);
		v2.z = (beta2 * v2.z) + ((1.0f - beta2) * dL_dparam_2.z * dL_dparam_2.z);
		v2.w = (beta2 * v2.w) + ((1.0f - beta2) * dL_dparam_2.w * dL_dparam_2.w);

		lr = scene_extent * lr_m * expf(params_OptiX.epoch * lr_m_exponential_decay_coefficient);
		if (lr < 0.000003f) lr = 0.000003f;

		// mX, mY, mZ
		dm = make_float3(
			((lr * (m2.x * tmp1)) / (sqrtf(v2.x * tmp2) + epsilon)),
			((lr * (m2.y * tmp1)) / (sqrtf(v2.y * tmp2) + epsilon)),
			((lr * (m2.z * tmp1)) / (sqrtf(v2.z * tmp2) + epsilon))
		);

		lr = scene_extent * lr_s * expf(params_OptiX.epoch * lr_s_exponential_decay_coefficient);
		if (lr < 0.001f) lr = 0.001f;

		// sX
		GC_2.w -= ((lr * (m2.w * tmp1)) / (sqrtf(v2.w * tmp2) + epsilon));

		isMovedEnough = (sqrtf((dm.x * dm.x) + (dm.y * dm.y) + (dm.z * dm.z)) >= scene_extent * mu_grad_norm_threshold_for_densification);

		// *****************************************************************************************

		dL_dparam_3 = ((REAL4_G *)params_OptiX.dL_dparams_3)[tid];
		m3 = ((float4 *)params_OptiX.m31)[tid];
		v3 = ((float4 *)params_OptiX.v31)[tid];
		GC_3 = ((float4 *)params_OptiX.GC_part_3_1)[tid];

		m3.x = (beta1 * m3.x) + ((1.0f - beta1) * dL_dparam_3.x);
		m3.y = (beta1 * m3.y) + ((1.0f - beta1) * dL_dparam_3.y);
		m3.z = (beta1 * m3.z) + ((1.0f - beta1) * dL_dparam_3.z);
		m3.w = (beta1 * m3.w) + ((1.0f - beta1) * dL_dparam_3.w);

		v3.x = (beta2 * v3.x) + ((1.0f - beta2) * dL_dparam_3.x * dL_dparam_3.x);
		v3.y = (beta2 * v3.y) + ((1.0f - beta2) * dL_dparam_3.y * dL_dparam_3.y);
		v3.z = (beta2 * v3.z) + ((1.0f - beta2) * dL_dparam_3.z * dL_dparam_3.z);
		v3.w = (beta2 * v3.w) + ((1.0f - beta2) * dL_dparam_3.w * dL_dparam_3.w);

		// sY
		GC_3.x -= ((lr * (m3.x * tmp1)) / (sqrtf(v3.x * tmp2) + epsilon));
		// sZ
		GC_3.y -= ((lr * (m3.y * tmp1)) / (sqrtf(v3.y * tmp2) + epsilon));

		lr = lr_q * expf(params_OptiX.epoch * lr_q_exponential_decay_coefficient);
		if (lr < 0.001f) lr = 0.001f;

		// qr
		GC_3.z -= ((lr * (m3.z * tmp1)) / (sqrtf(v3.z * tmp2) + epsilon));
		// qi
		GC_3.w -= ((lr * (m3.w * tmp1)) / (sqrtf(v3.w * tmp2) + epsilon));

		scale = make_float3(
			1.0f / (1.0f + expf(-GC_2.w)),
			1.0f / (1.0f + expf(-GC_3.x)),
			1.0f / (1.0f + expf(-GC_3.y))
		);
	
		float length = sqrtf((scale.x * scale.x) + (scale.y * scale.y) + (scale.z * scale.z));

		isBigEnough = (length >= scene_extent * min_s_norm_threshold_for_Gauss_removal);
		isNotTooBig = (length <= scene_extent * max_s_norm_threshold_for_Gauss_removal);
		isBigEnoughToSplit = (length > scene_extent * s_norm_threshold_for_split_strategy);

		// *****************************************************************************************

		dL_dparam_4 = ((REAL2_G *)params_OptiX.dL_dparams_4)[tid];
		m4 = ((float2 *)params_OptiX.m41)[tid];
		v4 = ((float2 *)params_OptiX.v41)[tid];
		GC_4 = ((float2 *)params_OptiX.GC_part_4_1)[tid];

		m4.x = (beta1 * m4.x) + ((1.0f - beta1) * dL_dparam_4.x);
		m4.y = (beta1 * m4.y) + ((1.0f - beta1) * dL_dparam_4.y);

		v4.x = (beta2 * v4.x) + ((1.0f - beta2) * dL_dparam_4.x * dL_dparam_4.x);
		v4.y = (beta2 * v4.y) + ((1.0f - beta2) * dL_dparam_4.y * dL_dparam_4.y);

		// qj
		GC_4.x -= ((lr * (m4.x * tmp1)) / (sqrtf(v4.x * tmp2) + epsilon));
		// qk
		GC_4.y -= ((lr * (m4.y * tmp1)) / (sqrtf(v4.y * tmp2) + epsilon));
	}

	// *********************************************************************************************

	bool densification_epoch = (
		(params_OptiX.epoch >= densification_start_epoch) &&
		(params_OptiX.epoch <= densification_end_epoch) &&
		((params_OptiX.epoch % densification_frequency) == 0)
	);
	
	unsigned GaussInd;
	unsigned sampleNum;

	if (densification_epoch) {
		if (threadIdx.x == 0) {
			counter1 = 0;
			counter2 = 0;
		}
		__syncthreads();

		if (tid < params_OptiX.numberOfGaussians) {
			if ((isOpaqueEnough) && (isBigEnough) && (isNotTooBig)) {
				if (isMovedEnough) {
					GaussInd = atomicAdd(&counter1, 2);
					if (isBigEnoughToSplit) sampleNum = atomicAdd(&counter2, 6);
				} else
					GaussInd = atomicAdd(&counter1, 1);
			}
		}
		__syncthreads();

		if (threadIdx.x == 0) {
			counter1 = atomicAdd((unsigned *)params_OptiX.counter1, counter1);
			counter2 = atomicAdd((unsigned *)params_OptiX.counter2, counter2);
		}
		__syncthreads();

		GaussInd += counter1;
		sampleNum += counter2;
	}

	// *********************************************************************************************

	if (tid < params_OptiX.numberOfGaussians) {	
		float aa = ((float)GC_3.z) * GC_3.z;
		float bb = ((float)GC_3.w) * GC_3.w;
		float cc = ((float)GC_4.x) * GC_4.x;
		float dd = ((float)GC_4.y) * GC_4.y;
		float s = 2.0f / (aa + bb + cc + dd);

		float bs = GC_3.w * s;  float cs = GC_4.x * s;  float ds = GC_4.y * s;
		float ab = GC_3.z * bs; float ac = GC_3.z * cs; float ad = GC_3.z * ds;
		bb = bb * s;            float bc = GC_3.w * cs; float bd = GC_3.w * ds;
		cc = cc * s;            float cd = GC_4.x * ds;       dd = dd * s;

		float R11 = 1.0f - cc - dd;
		float R12 = bc - ad;
		float R13 = bd + ac;

		float R21 = bc + ad;
		float R22 = 1.0f - bb - dd;
		float R23 = cd - ab;

		float R31 = bd - ac;
		float R32 = cd + ab;
		float R33 = 1.0f - bb - cc;

		// *****************************************************************************************

		//if (((params_OptiX.epoch - 1) % 3000) == 0) GC_1.w = -logf(254.0f - 1.0f); // !!! !!! !!!

		float3 lower_bound;
		float3 upper_bound;

		if (!densification_epoch) {
			scale.x = ((scale.x < scene_extent * min_s_coefficients_clipping_threshold) ? scene_extent * min_s_coefficients_clipping_threshold : scale.x);
			scale.y = ((scale.y < scene_extent * min_s_coefficients_clipping_threshold) ? scene_extent * min_s_coefficients_clipping_threshold : scale.y);
			scale.z = ((scale.z < scene_extent * min_s_coefficients_clipping_threshold) ? scene_extent * min_s_coefficients_clipping_threshold : scale.z);

			scale.x = ((scale.x > scene_extent * max_s_coefficients_clipping_threshold) ? scene_extent * max_s_coefficients_clipping_threshold : scale.x);
			scale.y = ((scale.y > scene_extent * max_s_coefficients_clipping_threshold) ? scene_extent * max_s_coefficients_clipping_threshold : scale.y);
			scale.z = ((scale.z > scene_extent * max_s_coefficients_clipping_threshold) ? scene_extent * max_s_coefficients_clipping_threshold : scale.z);

			GC_2.w = -logf((1.0f / scale.x) - 1.0f);
			GC_3.x = -logf((1.0f / scale.y) - 1.0f);
			GC_3.y = -logf((1.0f / scale.z) - 1.0f);

			float tmpX = sqrtf(11.3449f * ((scale.x * scale.x * R11 * R11) + (scale.y * scale.y * R12 * R12) + (scale.z * scale.z * R13 * R13)));
			float tmpY = sqrtf(11.3449f * ((scale.x * scale.x * R21 * R21) + (scale.y * scale.y * R22 * R22) + (scale.z * scale.z * R23 * R23)));
			float tmpZ = sqrtf(11.3449f * ((scale.x * scale.x * R31 * R31) + (scale.y * scale.y * R32 * R32) + (scale.z * scale.z * R33 * R33)));

			// *** *** *** *** ***

			((float4 *)params_OptiX.GC_part_1_1)[tid] = GC_1;
			((float4 *)params_OptiX.m11)[tid] = m1;
			((float4 *)params_OptiX.v11)[tid] = v1;

			// mX
			GC_2.x -= dm.x;
			// mY
			GC_2.y -= dm.y;
			// mZ
			GC_2.z -= dm.z;

			((float4 *)params_OptiX.GC_part_2_1)[tid] = GC_2;
			((float4 *)params_OptiX.m21)[tid] = m2;
			((float4 *)params_OptiX.v21)[tid] = v2;

			((float4 *)params_OptiX.GC_part_3_1)[tid] = GC_3;
			((float4 *)params_OptiX.m31)[tid] = m3;
			((float4 *)params_OptiX.v31)[tid] = v3;

			((float2 *)params_OptiX.GC_part_4_1)[tid] = GC_4;
			((float2 *)params_OptiX.m41)[tid] = m4;
			((float2 *)params_OptiX.v41)[tid] = v4;

			lower_bound.x = GC_2.x - tmpX; // !!! !!! !!!
			upper_bound.x = GC_2.x + tmpX; // !!! !!! !!!

			lower_bound.y = GC_2.y - tmpY; // !!! !!! !!!
			upper_bound.y = GC_2.y + tmpY; // !!! !!! !!!

			lower_bound.z = GC_2.z - tmpZ; // !!! !!! !!!
			upper_bound.z = GC_2.z + tmpZ; // !!! !!! !!!

			((float *)params_OptiX.aabbBuffer)[(tid * 6) + 0] = GC_2.x - tmpX; // !!! !!! !!!
			((float *)params_OptiX.aabbBuffer)[(tid * 6) + 3] = GC_2.x + tmpX; // !!! !!! !!!

			((float *)params_OptiX.aabbBuffer)[(tid * 6) + 1] = GC_2.y - tmpY; // !!! !!! !!!
			((float *)params_OptiX.aabbBuffer)[(tid * 6) + 4] = GC_2.y + tmpY; // !!! !!! !!!

			((float *)params_OptiX.aabbBuffer)[(tid * 6) + 2] = GC_2.z - tmpZ; // !!! !!! !!!
			((float *)params_OptiX.aabbBuffer)[(tid * 6) + 5] = GC_2.z + tmpZ; // !!! !!! !!!
		} else {
			if ((isOpaqueEnough) && (isBigEnough) && (isNotTooBig)) {
				if (isMovedEnough) {
					if (isBigEnoughToSplit) {
						float Z1, Z2;
						RandomNormalFloat(sampleNum, Z1, Z2); // !!! !!! !!!
						float Z3, Z4;
						RandomNormalFloat(sampleNum + 2, Z3, Z4); // !!! !!! !!!
						float Z5, Z6;
						RandomNormalFloat(sampleNum + 4, Z5, Z6); // !!! !!! !!!

						float3 P1;
						RandomMultinormalFloat(
							make_float3(GC_2.x, GC_2.y, GC_2.z),
							scale,
							make_float4(GC_3.z, GC_3.w, GC_4.x, GC_4.y),
							Z1, Z2, Z3,
							P1
						);

						float3 P2;
						RandomMultinormalFloat(
							make_float3(GC_2.x, GC_2.y, GC_2.z),
							scale,
							make_float4(GC_3.z, GC_3.w, GC_4.x, GC_4.y),
							Z4, Z5, Z6,
							P2
						);

						// *** *** *** *** ***

						scale.x /= split_ratio;
						scale.y /= split_ratio;
						scale.z /= split_ratio;

						// *** *** *** *** ***

						scale.x = ((scale.x < scene_extent * min_s_coefficients_clipping_threshold) ? scene_extent * min_s_coefficients_clipping_threshold : scale.x);
						scale.y = ((scale.y < scene_extent * min_s_coefficients_clipping_threshold) ? scene_extent * min_s_coefficients_clipping_threshold : scale.y);
						scale.z = ((scale.z < scene_extent * min_s_coefficients_clipping_threshold) ? scene_extent * min_s_coefficients_clipping_threshold : scale.z);

						scale.x = ((scale.x > scene_extent * max_s_coefficients_clipping_threshold) ? scene_extent * max_s_coefficients_clipping_threshold : scale.x);
						scale.y = ((scale.y > scene_extent * max_s_coefficients_clipping_threshold) ? scene_extent * max_s_coefficients_clipping_threshold : scale.y);
						scale.z = ((scale.z > scene_extent * max_s_coefficients_clipping_threshold) ? scene_extent * max_s_coefficients_clipping_threshold : scale.z);

						GC_2.w = -logf((1.0f / scale.x) - 1.0f);
						GC_3.x = -logf((1.0f / scale.y) - 1.0f);
						GC_3.y = -logf((1.0f / scale.z) - 1.0f);

						float tmpX = sqrtf(11.3449f * ((scale.x * scale.x * R11 * R11) + (scale.y * scale.y * R12 * R12) + (scale.z * scale.z * R13 * R13)));
						float tmpY = sqrtf(11.3449f * ((scale.x * scale.x * R21 * R21) + (scale.y * scale.y * R22 * R22) + (scale.z * scale.z * R23 * R23)));
						float tmpZ = sqrtf(11.3449f * ((scale.x * scale.x * R31 * R31) + (scale.y * scale.y * R32 * R32) + (scale.z * scale.z * R33 * R33)));

						// *** *** *** *** ***

						((float4 *)params_OptiX.GC_part_1_2)[GaussInd] = GC_1;
						((float4 *)params_OptiX.m12)[GaussInd] = m1;
						((float4 *)params_OptiX.v12)[GaussInd] = v1;

						GC_2.x = P1.x;
						GC_2.y = P1.y;
						GC_2.z = P1.z;
						
						((float4 *)params_OptiX.GC_part_2_2)[GaussInd] = GC_2;
						((float4 *)params_OptiX.m22)[GaussInd] = m2;
						((float4 *)params_OptiX.v22)[GaussInd] = v2;

						((float4 *)params_OptiX.GC_part_3_2)[GaussInd] = GC_3;
						((float4 *)params_OptiX.m32)[GaussInd] = m3;
						((float4 *)params_OptiX.v32)[GaussInd] = v3;

						((float2 *)params_OptiX.GC_part_4_2)[GaussInd] = GC_4;
						((float2 *)params_OptiX.m42)[GaussInd] = m4;
						((float2 *)params_OptiX.v42)[GaussInd] = v4;

						lower_bound.x = GC_2.x - tmpX; // !!! !!! !!!
						upper_bound.x = GC_2.x + tmpX; // !!! !!! !!!

						lower_bound.y = GC_2.y - tmpY; // !!! !!! !!!
						upper_bound.y = GC_2.y + tmpY; // !!! !!! !!!

						lower_bound.z = GC_2.z - tmpZ; // !!! !!! !!!
						upper_bound.z = GC_2.z + tmpZ; // !!! !!! !!!

						((float *)params_OptiX.aabbBuffer)[(GaussInd * 6) + 0] = GC_2.x - tmpX; // !!! !!! !!!
						((float *)params_OptiX.aabbBuffer)[(GaussInd * 6) + 3] = GC_2.x + tmpX; // !!! !!! !!!

						((float *)params_OptiX.aabbBuffer)[(GaussInd * 6) + 1] = GC_2.y - tmpY; // !!! !!! !!!
						((float *)params_OptiX.aabbBuffer)[(GaussInd * 6) + 4] = GC_2.y + tmpY; // !!! !!! !!!

						((float *)params_OptiX.aabbBuffer)[(GaussInd * 6) + 2] = GC_2.z - tmpZ; // !!! !!! !!!
						((float *)params_OptiX.aabbBuffer)[(GaussInd * 6) + 5] = GC_2.z + tmpZ; // !!! !!! !!!

						// *** *** *** *** ***

						((float4 *)params_OptiX.GC_part_1_2)[GaussInd + 1] = GC_1;
						((float4 *)params_OptiX.m12)[GaussInd + 1] = m1;
						((float4 *)params_OptiX.v12)[GaussInd + 1] = v1;

						GC_2.x = P2.x;
						GC_2.y = P2.y;
						GC_2.z = P2.z;

						((float4 *)params_OptiX.GC_part_2_2)[GaussInd + 1] = GC_2;
						((float4 *)params_OptiX.m22)[GaussInd + 1] = m2;
						((float4 *)params_OptiX.v22)[GaussInd + 1] = v2;

						((float4 *)params_OptiX.GC_part_3_2)[GaussInd + 1] = GC_3;
						((float4 *)params_OptiX.m32)[GaussInd + 1] = m3;
						((float4 *)params_OptiX.v32)[GaussInd + 1] = v3;

						((float2 *)params_OptiX.GC_part_4_2)[GaussInd + 1] = GC_4;
						((float2 *)params_OptiX.m42)[GaussInd + 1] = m4;
						((float2 *)params_OptiX.v42)[GaussInd + 1] = v4;

						lower_bound.x = ((GC_2.x - tmpX < lower_bound.x) ? GC_2.x - tmpX : lower_bound.x); // !!! !!! !!!
						upper_bound.x = ((GC_2.x + tmpX > upper_bound.x) ? GC_2.x + tmpX : upper_bound.x); // !!! !!! !!!

						lower_bound.y = ((GC_2.y - tmpY < lower_bound.y) ? GC_2.y - tmpY : lower_bound.y); // !!! !!! !!!
						upper_bound.y = ((GC_2.y + tmpY > upper_bound.y) ? GC_2.y + tmpY : upper_bound.y); // !!! !!! !!!

						lower_bound.z = ((GC_2.z - tmpZ < lower_bound.z) ? GC_2.z - tmpZ : lower_bound.z); // !!! !!! !!!
						upper_bound.z = ((GC_2.z + tmpZ > upper_bound.z) ? GC_2.z + tmpZ : upper_bound.z); // !!! !!! !!!

						((float *)params_OptiX.aabbBuffer)[((GaussInd + 1) * 6) + 0] = GC_2.x - tmpX; // !!! !!! !!!
						((float *)params_OptiX.aabbBuffer)[((GaussInd + 1) * 6) + 3] = GC_2.x + tmpX; // !!! !!! !!!

						((float *)params_OptiX.aabbBuffer)[((GaussInd + 1) * 6) + 1] = GC_2.y - tmpY; // !!! !!! !!!
						((float *)params_OptiX.aabbBuffer)[((GaussInd + 1) * 6) + 4] = GC_2.y + tmpY; // !!! !!! !!!

						((float *)params_OptiX.aabbBuffer)[((GaussInd + 1) * 6) + 2] = GC_2.z - tmpZ; // !!! !!! !!!
						((float *)params_OptiX.aabbBuffer)[((GaussInd + 1) * 6) + 5] = GC_2.z + tmpZ; // !!! !!! !!!
					} else {
						scale.x = ((scale.x < scene_extent * min_s_coefficients_clipping_threshold) ? scene_extent * min_s_coefficients_clipping_threshold : scale.x);
						scale.y = ((scale.y < scene_extent * min_s_coefficients_clipping_threshold) ? scene_extent * min_s_coefficients_clipping_threshold : scale.y);
						scale.z = ((scale.z < scene_extent * min_s_coefficients_clipping_threshold) ? scene_extent * min_s_coefficients_clipping_threshold : scale.z);

						scale.x = ((scale.x > scene_extent * max_s_coefficients_clipping_threshold) ? scene_extent * max_s_coefficients_clipping_threshold : scale.x);
						scale.y = ((scale.y > scene_extent * max_s_coefficients_clipping_threshold) ? scene_extent * max_s_coefficients_clipping_threshold : scale.y);
						scale.z = ((scale.z > scene_extent * max_s_coefficients_clipping_threshold) ? scene_extent * max_s_coefficients_clipping_threshold : scale.z);

						GC_2.w = -logf((1.0f / scale.x) - 1.0f);
						GC_3.x = -logf((1.0f / scale.y) - 1.0f);
						GC_3.y = -logf((1.0f / scale.z) - 1.0f);

						float tmpX = sqrtf(11.3449f * ((scale.x * scale.x * R11 * R11) + (scale.y * scale.y * R12 * R12) + (scale.z * scale.z * R13 * R13)));
						float tmpY = sqrtf(11.3449f * ((scale.x * scale.x * R21 * R21) + (scale.y * scale.y * R22 * R22) + (scale.z * scale.z * R23 * R23)));
						float tmpZ = sqrtf(11.3449f * ((scale.x * scale.x * R31 * R31) + (scale.y * scale.y * R32 * R32) + (scale.z * scale.z * R33 * R33)));

						// *** *** *** *** ***

						((float4 *)params_OptiX.GC_part_1_2)[GaussInd] = GC_1;
						((float4 *)params_OptiX.m12)[GaussInd] = m1;
						((float4 *)params_OptiX.v12)[GaussInd] = v1;

						((float4 *)params_OptiX.GC_part_2_2)[GaussInd] = GC_2;
						((float4 *)params_OptiX.m22)[GaussInd] = m2;
						((float4 *)params_OptiX.v22)[GaussInd] = v2;

						((float4 *)params_OptiX.GC_part_3_2)[GaussInd] = GC_3;
						((float4 *)params_OptiX.m32)[GaussInd] = m3;
						((float4 *)params_OptiX.v32)[GaussInd] = v3;

						((float2 *)params_OptiX.GC_part_4_2)[GaussInd] = GC_4;
						((float2 *)params_OptiX.m42)[GaussInd] = m4;
						((float2 *)params_OptiX.v42)[GaussInd] = v4;

						lower_bound.x = GC_2.x - tmpX; // !!! !!! !!!
						upper_bound.x = GC_2.x + tmpX; // !!! !!! !!!

						lower_bound.y = GC_2.y - tmpY; // !!! !!! !!!
						upper_bound.y = GC_2.y + tmpY; // !!! !!! !!!

						lower_bound.z = GC_2.z - tmpZ; // !!! !!! !!!
						upper_bound.z = GC_2.z + tmpZ; // !!! !!! !!!

						((float *)params_OptiX.aabbBuffer)[(GaussInd * 6) + 0] = GC_2.x - tmpX; // !!! !!! !!!
						((float *)params_OptiX.aabbBuffer)[(GaussInd * 6) + 3] = GC_2.x + tmpX; // !!! !!! !!!

						((float *)params_OptiX.aabbBuffer)[(GaussInd * 6) + 1] = GC_2.y - tmpY; // !!! !!! !!!
						((float *)params_OptiX.aabbBuffer)[(GaussInd * 6) + 4] = GC_2.y + tmpY; // !!! !!! !!!

						((float *)params_OptiX.aabbBuffer)[(GaussInd * 6) + 2] = GC_2.z - tmpZ; // !!! !!! !!!
						((float *)params_OptiX.aabbBuffer)[(GaussInd * 6) + 5] = GC_2.z + tmpZ; // !!! !!! !!!

						// *** *** *** *** ***

						((float4 *)params_OptiX.GC_part_1_2)[GaussInd + 1] = GC_1;
						((float4 *)params_OptiX.m12)[GaussInd + 1] = m1;
						((float4 *)params_OptiX.v12)[GaussInd + 1] = v1;

						// mX
						GC_2.x -= dm.x;
						// mY
						GC_2.y -= dm.y;
						// mZ
						GC_2.z -= dm.z;

						((float4 *)params_OptiX.GC_part_2_2)[GaussInd + 1] = GC_2;
						((float4 *)params_OptiX.m22)[GaussInd + 1] = m2;
						((float4 *)params_OptiX.v22)[GaussInd + 1] = v2;

						((float4 *)params_OptiX.GC_part_3_2)[GaussInd + 1] = GC_3;
						((float4 *)params_OptiX.m32)[GaussInd + 1] = m3;
						((float4 *)params_OptiX.v32)[GaussInd + 1] = v3;

						((float2 *)params_OptiX.GC_part_4_2)[GaussInd + 1] = GC_4;
						((float2 *)params_OptiX.m42)[GaussInd + 1] = m4;
						((float2 *)params_OptiX.v42)[GaussInd + 1] = v4;

						lower_bound.x = ((GC_2.x - tmpX < lower_bound.x) ? GC_2.x - tmpX : lower_bound.x); // !!! !!! !!!
						upper_bound.x = ((GC_2.x + tmpX > upper_bound.x) ? GC_2.x + tmpX : upper_bound.x); // !!! !!! !!!

						lower_bound.y = ((GC_2.y - tmpY < lower_bound.y) ? GC_2.y - tmpY : lower_bound.y); // !!! !!! !!!
						upper_bound.y = ((GC_2.y + tmpY > upper_bound.y) ? GC_2.y + tmpY : upper_bound.y); // !!! !!! !!!

						lower_bound.z = ((GC_2.z - tmpZ < lower_bound.z) ? GC_2.z - tmpZ : lower_bound.z); // !!! !!! !!!
						upper_bound.z = ((GC_2.z + tmpZ > upper_bound.z) ? GC_2.z + tmpZ : upper_bound.z); // !!! !!! !!!

						((float *)params_OptiX.aabbBuffer)[((GaussInd + 1) * 6) + 0] = GC_2.x - tmpX; // !!! !!! !!!
						((float *)params_OptiX.aabbBuffer)[((GaussInd + 1) * 6) + 3] = GC_2.x + tmpX; // !!! !!! !!!

						((float *)params_OptiX.aabbBuffer)[((GaussInd + 1) * 6) + 1] = GC_2.y - tmpY; // !!! !!! !!!
						((float *)params_OptiX.aabbBuffer)[((GaussInd + 1) * 6) + 4] = GC_2.y + tmpY; // !!! !!! !!!

						((float *)params_OptiX.aabbBuffer)[((GaussInd + 1) * 6) + 2] = GC_2.z - tmpZ; // !!! !!! !!!
						((float *)params_OptiX.aabbBuffer)[((GaussInd + 1) * 6) + 5] = GC_2.z + tmpZ; // !!! !!! !!!
					}
				} else {
					scale.x = ((scale.x < scene_extent * min_s_coefficients_clipping_threshold) ? scene_extent * min_s_coefficients_clipping_threshold : scale.x);
					scale.y = ((scale.y < scene_extent * min_s_coefficients_clipping_threshold) ? scene_extent * min_s_coefficients_clipping_threshold : scale.y);
					scale.z = ((scale.z < scene_extent * min_s_coefficients_clipping_threshold) ? scene_extent * min_s_coefficients_clipping_threshold : scale.z);

					scale.x = ((scale.x > scene_extent * max_s_coefficients_clipping_threshold) ? scene_extent * max_s_coefficients_clipping_threshold : scale.x);
					scale.y = ((scale.y > scene_extent * max_s_coefficients_clipping_threshold) ? scene_extent * max_s_coefficients_clipping_threshold : scale.y);
					scale.z = ((scale.z > scene_extent * max_s_coefficients_clipping_threshold) ? scene_extent * max_s_coefficients_clipping_threshold : scale.z);

					GC_2.w = -logf((1.0f / scale.x) - 1.0f);
					GC_3.x = -logf((1.0f / scale.y) - 1.0f);
					GC_3.y = -logf((1.0f / scale.z) - 1.0f);
					
					float tmpX = sqrtf(11.3449f * ((scale.x * scale.x * R11 * R11) + (scale.y * scale.y * R12 * R12) + (scale.z * scale.z * R13 * R13)));
					float tmpY = sqrtf(11.3449f * ((scale.x * scale.x * R21 * R21) + (scale.y * scale.y * R22 * R22) + (scale.z * scale.z * R23 * R23)));
					float tmpZ = sqrtf(11.3449f * ((scale.x * scale.x * R31 * R31) + (scale.y * scale.y * R32 * R32) + (scale.z * scale.z * R33 * R33)));

					// *** *** *** *** ***

					((float4 *)params_OptiX.GC_part_1_2)[GaussInd] = GC_1;
					((float4 *)params_OptiX.m12)[GaussInd] = m1;
					((float4 *)params_OptiX.v12)[GaussInd] = v1;

					// mX
					GC_2.x -= dm.x;
					// mY
					GC_2.y -= dm.y;
					// mZ
					GC_2.z -= dm.z;

					((float4 *)params_OptiX.GC_part_2_2)[GaussInd] = GC_2;
					((float4 *)params_OptiX.m22)[GaussInd] = m2;
					((float4 *)params_OptiX.v22)[GaussInd] = v2;

					((float4 *)params_OptiX.GC_part_3_2)[GaussInd] = GC_3;
					((float4 *)params_OptiX.m32)[GaussInd] = m3;
					((float4 *)params_OptiX.v32)[GaussInd] = v3;

					((float2 *)params_OptiX.GC_part_4_2)[GaussInd] = GC_4;
					((float2 *)params_OptiX.m42)[GaussInd] = m4;
					((float2 *)params_OptiX.v42)[GaussInd] = v4;

					lower_bound.x = GC_2.x - tmpX; // !!! !!! !!!
					upper_bound.x = GC_2.x + tmpX; // !!! !!! !!!

					lower_bound.y = GC_2.y - tmpY; // !!! !!! !!!
					upper_bound.y = GC_2.y + tmpY; // !!! !!! !!!

					lower_bound.z = GC_2.z - tmpZ; // !!! !!! !!!
					upper_bound.z = GC_2.z + tmpZ; // !!! !!! !!!

					((float *)params_OptiX.aabbBuffer)[(GaussInd * 6) + 0] = GC_2.x - tmpX; // !!! !!! !!!
					((float *)params_OptiX.aabbBuffer)[(GaussInd * 6) + 3] = GC_2.x + tmpX; // !!! !!! !!!

					((float *)params_OptiX.aabbBuffer)[(GaussInd * 6) + 1] = GC_2.y - tmpY; // !!! !!! !!!
					((float *)params_OptiX.aabbBuffer)[(GaussInd * 6) + 4] = GC_2.y + tmpY; // !!! !!! !!!

					((float *)params_OptiX.aabbBuffer)[(GaussInd * 6) + 2] = GC_2.z - tmpZ; // !!! !!! !!!
					((float *)params_OptiX.aabbBuffer)[(GaussInd * 6) + 5] = GC_2.z + tmpZ; // !!! !!! !!!
				}
			}
		}

		if ((!densification_epoch) || ((isOpaqueEnough) && (isBigEnough) && (isNotTooBig))) {
			atomicMin(&auxiliary_values.scene_lower_bound.x, dev_Float2SortableUint(lower_bound.x));
			atomicMax(&auxiliary_values.scene_upper_bound.x, dev_Float2SortableUint(upper_bound.x));

			atomicMin(&auxiliary_values.scene_lower_bound.y, dev_Float2SortableUint(lower_bound.y));
			atomicMax(&auxiliary_values.scene_upper_bound.y, dev_Float2SortableUint(upper_bound.y));

			atomicMin(&auxiliary_values.scene_lower_bound.z, dev_Float2SortableUint(lower_bound.z));
			atomicMax(&auxiliary_values.scene_upper_bound.z, dev_Float2SortableUint(upper_bound.z));
		}
	}
}

// *************************************************************************************************

__global__ void ComputeArraysForGradientComputation(
	REAL_G *dev_mu1R, REAL_G *dev_mu2R,
	REAL_G *dev_sigma12R,
	REAL_G *dev_sigma1R_square, REAL_G *dev_sigma2R_square,

	REAL_G *dev_mu1G, REAL_G *dev_mu2G,
	REAL_G *dev_sigma12G,
	REAL_G *dev_sigma1G_square, REAL_G *dev_sigma2G_square,

	REAL_G *dev_mu1B, REAL_G *dev_mu2B,
	REAL_G *dev_sigma12B,
	REAL_G *dev_sigma1B_square, REAL_G *dev_sigma2B_square,

	REAL_G *dev_tmp1R, REAL_G *dev_tmp2R, REAL_G *dev_tmp3R,
	REAL_G *dev_tmp1G, REAL_G *dev_tmp2G, REAL_G *dev_tmp3G,
	REAL_G *dev_tmp1B, REAL_G *dev_tmp2B, REAL_G *dev_tmp3B,

	int width, int height, int kernel_radius
) {
	int tid = (blockIdx.x * blockDim.x) + threadIdx.x;
	int x = tid % width;
	int y = tid / width;
	int size = width * height;
	if (tid < size) {
		if ((x >= kernel_radius) && (x < width - kernel_radius) && (y >= kernel_radius) && (y < height - kernel_radius)) {
			REAL_G mu1R = dev_mu1R[tid] / size;
			REAL_G mu2R = dev_mu2R[tid] / size;
			REAL_G sigma12R =  dev_sigma12R[tid] / size;
			REAL_G sigma1R_square = dev_sigma1R_square[tid] / size;
			REAL_G sigma2R_square = dev_sigma2R_square[tid] / size;

			REAL_G mu1G = dev_mu1G[tid] / size;
			REAL_G mu2G = dev_mu2G[tid] / size;
			REAL_G sigma12G =  dev_sigma12G[tid] / size;
			REAL_G sigma1G_square = dev_sigma1G_square[tid] / size;
			REAL_G sigma2G_square = dev_sigma2G_square[tid] / size;

			REAL_G mu1B = dev_mu1B[tid] / size;
			REAL_G mu2B = dev_mu2B[tid] / size;
			REAL_G sigma12B =  dev_sigma12B[tid] / size;
			REAL_G sigma1B_square = dev_sigma1B_square[tid] / size;
			REAL_G sigma2B_square = dev_sigma2B_square[tid] / size;

			REAL_G c1 = ((REAL_G)(0.01 * 0.01));
			REAL_G c2 = ((REAL_G)(0.03 * 0.03));

			REAL_G AR = (2 * mu1R * mu2R) + c1;
			REAL_G BR = (2 * (sigma12R - (mu1R * mu2R))) + c2;
			REAL_G CR = ((mu1R * mu1R) + (mu2R * mu2R) + c1);
			REAL_G DR = ((sigma1R_square - (mu1R * mu1R)) + (sigma2R_square - (mu2R * mu2R)) + c2);

			REAL_G AG = (2 * mu1G * mu2G) + c1;
			REAL_G BG = (2 * (sigma12G - (mu1G * mu2G))) + c2;
			REAL_G CG = ((mu1G * mu1G) + (mu2G * mu2G) + c1);
			REAL_G DG = ((sigma1G_square - (mu1G * mu1G)) + (sigma2G_square - (mu2G * mu2G)) + c2);

			REAL_G AB = (2 * mu1B * mu2B) + c1;
			REAL_G BB = (2 * (sigma12B - (mu1B * mu2B))) + c2;
			REAL_G CB = ((mu1B * mu1B) + (mu2B * mu2B) + c1);
			REAL_G DB = ((sigma1B_square - (mu1B * mu1B)) + (sigma2B_square - (mu2B * mu2B)) + c2);

			REAL_G tmp1R = (2 * ((CR * DR * mu2R * (BR - AR)) - (AR * BR * mu1R * (DR - CR)))) / (CR * CR * DR * DR);
			REAL_G tmp2R = (2 * AR * CR * DR) / (CR * CR * DR * DR);
			REAL_G tmp3R = (2 * AR * BR * CR) / (CR * CR * DR * DR);

			REAL_G tmp1G = (2 * ((CG * DG * mu2G * (BG - AG)) - (AG * BG * mu1G * (DG - CG)))) / (CG * CG * DG * DG);
			REAL_G tmp2G = (2 * AG * CG * DG) / (CG * CG * DG * DG);
			REAL_G tmp3G = (2 * AG * BG * CG) / (CG * CG * DG * DG);

			REAL_G tmp1B = (2 * ((CB * DB * mu2B * (BB - AB)) - (AB * BB * mu1B * (DB - CB)))) / (CB * CB * DB * DB);
			REAL_G tmp2B = (2 * AB * CB * DB) / (CB * CB * DB * DB);
			REAL_G tmp3B = (2 * AB * BB * CB) / (CB * CB * DB * DB);

			dev_tmp1R[tid] = tmp1R;
			dev_tmp2R[tid] = tmp2R;
			dev_tmp3R[tid] = tmp3R;

			dev_tmp1G[tid] = tmp1G;
			dev_tmp2G[tid] = tmp2G;
			dev_tmp3G[tid] = tmp3G;

			dev_tmp1B[tid] = tmp1B;
			dev_tmp2B[tid] = tmp2B;
			dev_tmp3B[tid] = tmp3B;
		} else {
			dev_tmp1R[tid] = 0;
			dev_tmp2R[tid] = 0;
			dev_tmp3R[tid] = 0;

			dev_tmp1G[tid] = 0;
			dev_tmp2G[tid] = 0;
			dev_tmp3G[tid] = 0;

			dev_tmp1B[tid] = 0;
			dev_tmp2B[tid] = 0;
			dev_tmp3B[tid] = 0;
		}
	}
}

// *************************************************************************************************

__global__ void ComputeGradientSSIM(
	REAL_G *dev_conv1R,
	REAL_G *dev_conv2R, REAL_G *dev_img2R,
	REAL_G *dev_conv3R, REAL_G *dev_img1R,

	REAL_G *dev_conv1G,
	REAL_G *dev_conv2G, REAL_G *dev_img2G,
	REAL_G *dev_conv3G, REAL_G *dev_img1G,

	REAL_G *dev_conv1B,
	REAL_G *dev_conv2B, REAL_G *dev_img2B,
	REAL_G *dev_conv3B, REAL_G *dev_img1B,

	REAL_G *dev_gradR, REAL_G *dev_gradG, REAL_G *dev_gradB,

	int width, int height, int kernel_size
) {
	int tid = (blockIdx.x * blockDim.x) + threadIdx.x;
	int x = tid % width;
	int y = tid / width;
	int size = width * height;
	if ((tid < size) && (x >= kernel_size - 1) && (y >= kernel_size - 1)) {
		int ind = ((y - (kernel_size - 1)) * width) + (x - (kernel_size - 1));

		REAL_G conv1R = dev_conv1R[tid] / size;
		REAL_G conv2R = dev_conv2R[tid] / size;
		REAL_G img2R = dev_img2R[ind];
		REAL_G conv3R = dev_conv3R[tid] / size;
		REAL_G img1R = dev_img1R[ind];

		REAL_G conv1G = dev_conv1G[tid] / size;
		REAL_G conv2G = dev_conv2G[tid] / size;
		REAL_G img2G = dev_img2G[ind];
		REAL_G conv3G = dev_conv3G[tid] / size;
		REAL_G img1G = dev_img1G[ind];

		REAL_G conv1B = dev_conv1B[tid] / size;
		REAL_G conv2B = dev_conv2B[tid] / size;
		REAL_G img2B = dev_img2B[ind];
		REAL_G conv3B = dev_conv3B[tid] / size;
		REAL_G img1B = dev_img1B[ind];

		REAL_G gradR = ((conv3R * img1R) - conv1R - (conv2R * img2R)) / (2 * 3 * (width - (kernel_size - 1)) * (height - (kernel_size - 1)));
		REAL_G gradG = ((conv3G * img1G) - conv1G - (conv2G * img2G)) / (2 * 3 * (width - (kernel_size - 1)) * (height - (kernel_size - 1)));
		REAL_G gradB = ((conv3B * img1B) - conv1B - (conv2B * img2B)) / (2 * 3 * (width - (kernel_size - 1)) * (height - (kernel_size - 1)));

		dev_gradR[tid] = gradR;
		dev_gradG[tid] = gradG;
		dev_gradB[tid] = gradB;
	}
}

// *************************************************************************************************

// DEBUG GRADIENT
extern bool DumpParameters(SOptiXRenderParams& params_OptiX);

bool UpdateGradientOptiX(SOptiXRenderParams& params_OptiX, int &state) {
	cudaError_t error_CUDA;
	OptixResult error_OptiX;

	//**********************************************************************************************
	// SSIM                                                                                        *
	//**********************************************************************************************

	const int kernel_size = 11;
	const int kernel_radius = kernel_size >> 1;
	const REAL_G sigma = ((REAL_G)1.5);

	int arraySizeReal = (params_OptiX.width + (kernel_size - 1)) * (params_OptiX.width + (kernel_size - 1)); // !!! !!! !!!
	int arraySizeComplex = (((params_OptiX.width + (kernel_size - 1)) >> 1) + 1) * (params_OptiX.width + (kernel_size - 1)); // !!! !!! !!!

	cufftResult error_CUFFT;

	// *********************************************************************************************

	//********************************
	// Compute mu's for output image *
	//********************************

	// R channel
	error_CUFFT = DFFT_G(params_OptiX.planr2c, params_OptiX.bitmap_out_R, params_OptiX.F_2);
	if (error_CUFFT != CUFFT_SUCCESS) goto Error;

	MultiplyPointwiseComplex<<<(arraySizeComplex + 255) >> 8, 256>>>(params_OptiX.F_2, params_OptiX.F_1, params_OptiX.F_2, arraySizeComplex);
	error_CUDA = cudaGetLastError();
	if (error_CUDA != cudaSuccess) goto Error;

	error_CUFFT = IDFFT_G(params_OptiX.planc2r, params_OptiX.F_2, params_OptiX.mu_bitmap_out_R);
	if (error_CUFFT != CUFFT_SUCCESS) goto Error;

	// *** *** *** *** ***

	// G channel
	error_CUFFT = DFFT_G(params_OptiX.planr2c, params_OptiX.bitmap_out_G, params_OptiX.F_2);
	if (error_CUFFT != CUFFT_SUCCESS) goto Error;

	MultiplyPointwiseComplex<<<(arraySizeComplex + 255) >> 8, 256>>>(params_OptiX.F_2, params_OptiX.F_1, params_OptiX.F_2, arraySizeComplex);
	error_CUDA = cudaGetLastError();
	if (error_CUDA != cudaSuccess) goto Error;

	error_CUFFT = IDFFT_G(params_OptiX.planc2r, params_OptiX.F_2, params_OptiX.mu_bitmap_out_G);
	if (error_CUFFT != CUFFT_SUCCESS) goto Error;

	// *** *** *** *** ***

	// B channel
	error_CUFFT = DFFT_G(params_OptiX.planr2c, params_OptiX.bitmap_out_B, params_OptiX.F_2);
	if (error_CUFFT != CUFFT_SUCCESS) goto Error;

	MultiplyPointwiseComplex<<<(arraySizeComplex + 255) >> 8, 256>>>(params_OptiX.F_2, params_OptiX.F_1, params_OptiX.F_2, arraySizeComplex);
	error_CUDA = cudaGetLastError();
	if (error_CUDA != cudaSuccess) goto Error;

	error_CUFFT = IDFFT_G(params_OptiX.planc2r, params_OptiX.F_2, params_OptiX.mu_bitmap_out_B);
	if (error_CUFFT != CUFFT_SUCCESS) goto Error;

	// *********************************************************************************************

	//***************************************
	// Compute mu's for output image square *
	//***************************************

	// R channel
	// mu_bitmap_out_bitmap_ref_R = bitmap_out_R * bitmap_out_R
	MultiplyPointwiseReal<<<(arraySizeReal + 255) >> 8, 256>>>(
		params_OptiX.bitmap_out_R,
		params_OptiX.bitmap_out_R,
		params_OptiX.mu_bitmap_out_bitmap_ref_R,
		arraySizeReal
		);
	error_CUDA = cudaGetLastError();
	if (error_CUDA != cudaSuccess) goto Error;

	error_CUFFT = DFFT_G(params_OptiX.planr2c, params_OptiX.mu_bitmap_out_bitmap_ref_R, params_OptiX.F_2);
	if (error_CUFFT != CUFFT_SUCCESS) goto Error;

	MultiplyPointwiseComplex<<<(arraySizeComplex + 255) >> 8, 256>>>(params_OptiX.F_2, params_OptiX.F_1, params_OptiX.F_2, arraySizeComplex);
	error_CUDA = cudaGetLastError();
	if (error_CUDA != cudaSuccess) goto Error;

	error_CUFFT = IDFFT_G(params_OptiX.planc2r, params_OptiX.F_2, params_OptiX.mu_bitmap_out_R_square);
	if (error_CUFFT != CUFFT_SUCCESS) goto Error;

	// *** *** *** *** ***

	// G channel
	// mu_bitmap_out_bitmap_ref_R = bitmap_out_G * bitmap_out_G
	MultiplyPointwiseReal<<<(arraySizeReal + 255) >> 8, 256>>>(
		params_OptiX.bitmap_out_G,
		params_OptiX.bitmap_out_G,
		params_OptiX.mu_bitmap_out_bitmap_ref_R,
		arraySizeReal
		);
	error_CUDA = cudaGetLastError();
	if (error_CUDA != cudaSuccess) goto Error;

	error_CUFFT = DFFT_G(params_OptiX.planr2c, params_OptiX.mu_bitmap_out_bitmap_ref_R, params_OptiX.F_2);
	if (error_CUFFT != CUFFT_SUCCESS) goto Error;

	MultiplyPointwiseComplex<<<(arraySizeComplex + 255) >> 8, 256>>>(params_OptiX.F_2, params_OptiX.F_1, params_OptiX.F_2, arraySizeComplex);
	error_CUDA = cudaGetLastError();
	if (error_CUDA != cudaSuccess) goto Error;

	error_CUFFT = IDFFT_G(params_OptiX.planc2r, params_OptiX.F_2, params_OptiX.mu_bitmap_out_G_square);
	if (error_CUFFT != CUFFT_SUCCESS) goto Error;

	// *** *** *** *** ***

	// B channel
	// mu_bitmap_out_bitmap_ref_R = bitmap_out_B * bitmap_out_B
	MultiplyPointwiseReal<<<(arraySizeReal + 255) >> 8, 256>>>(
		params_OptiX.bitmap_out_B,
		params_OptiX.bitmap_out_B,
		params_OptiX.mu_bitmap_out_bitmap_ref_R,
		arraySizeReal
		);
	error_CUDA = cudaGetLastError();
	if (error_CUDA != cudaSuccess) goto Error;

	error_CUFFT = DFFT_G(params_OptiX.planr2c, params_OptiX.mu_bitmap_out_bitmap_ref_R, params_OptiX.F_2);
	if (error_CUFFT != CUFFT_SUCCESS) goto Error;

	MultiplyPointwiseComplex<<<(arraySizeComplex + 255) >> 8, 256>>>(params_OptiX.F_2, params_OptiX.F_1, params_OptiX.F_2, arraySizeComplex);
	error_CUDA = cudaGetLastError();
	if (error_CUDA != cudaSuccess) goto Error;

	error_CUFFT = IDFFT_G(params_OptiX.planc2r, params_OptiX.F_2, params_OptiX.mu_bitmap_out_B_square);
	if (error_CUFFT != CUFFT_SUCCESS) goto Error;

	// *********************************************************************************************

	//***************************************************************
	// Compute mu's for product of output image and reference image *
	//***************************************************************

	// R channel
	// mu_bitmap_out_bitmap_ref_R = bitmap_out_R * bitmap_ref_R
	MultiplyPointwiseReal<<<(arraySizeReal + 255) >> 8, 256>>>(
		params_OptiX.bitmap_out_R,
		params_OptiX.bitmap_ref_R + (params_OptiX.poseNum * arraySizeReal), // !!! !!! !!!
		params_OptiX.mu_bitmap_out_bitmap_ref_R,
		arraySizeReal
		);
	error_CUDA = cudaGetLastError();
	if (error_CUDA != cudaSuccess) goto Error;

	error_CUFFT = DFFT_G(params_OptiX.planr2c, params_OptiX.mu_bitmap_out_bitmap_ref_R, params_OptiX.F_2);
	if (error_CUFFT != CUFFT_SUCCESS) goto Error;

	MultiplyPointwiseComplex<<<(arraySizeComplex + 255) >> 8, 256>>>(params_OptiX.F_2, params_OptiX.F_1, params_OptiX.F_2, arraySizeComplex);
	error_CUDA = cudaGetLastError();
	if (error_CUDA != cudaSuccess) goto Error;

	error_CUFFT = IDFFT_G(params_OptiX.planc2r, params_OptiX.F_2, params_OptiX.mu_bitmap_out_bitmap_ref_R);
	if (error_CUFFT != CUFFT_SUCCESS) goto Error;

	// *** *** *** *** ***

	// G channel
	// mu_bitmap_out_bitmap_ref_G = bitmap_out_G * bitmap_ref_G
	MultiplyPointwiseReal<<<(arraySizeReal + 255) >> 8, 256>>>(
		params_OptiX.bitmap_out_G,
		params_OptiX.bitmap_ref_G + (params_OptiX.poseNum * arraySizeReal), // !!! !!! !!!
		params_OptiX.mu_bitmap_out_bitmap_ref_G,
		arraySizeReal
		);
	error_CUDA = cudaGetLastError();
	if (error_CUDA != cudaSuccess) goto Error;

	error_CUFFT = DFFT_G(params_OptiX.planr2c, params_OptiX.mu_bitmap_out_bitmap_ref_G, params_OptiX.F_2);
	if (error_CUFFT != CUFFT_SUCCESS) goto Error;

	MultiplyPointwiseComplex<<<(arraySizeComplex + 255) >> 8, 256>>>(params_OptiX.F_2, params_OptiX.F_1, params_OptiX.F_2, arraySizeComplex);
	error_CUDA = cudaGetLastError();
	if (error_CUDA != cudaSuccess) goto Error;

	error_CUFFT = IDFFT_G(params_OptiX.planc2r, params_OptiX.F_2, params_OptiX.mu_bitmap_out_bitmap_ref_G);
	if (error_CUFFT != CUFFT_SUCCESS) goto Error;

	// *** *** *** *** ***

	// B channel
	// mu_bitmap_out_bitmap_ref_B = bitmap_out_B * bitmap_ref_B
	MultiplyPointwiseReal<<<(arraySizeReal + 255) >> 8, 256>>>(
		params_OptiX.bitmap_out_B,
		params_OptiX.bitmap_ref_B + (params_OptiX.poseNum * arraySizeReal), // !!! !!! !!!
		params_OptiX.mu_bitmap_out_bitmap_ref_B,
		arraySizeReal
		);
	error_CUDA = cudaGetLastError();
	if (error_CUDA != cudaSuccess) goto Error;

	error_CUFFT = DFFT_G(params_OptiX.planr2c, params_OptiX.mu_bitmap_out_bitmap_ref_B, params_OptiX.F_2);
	if (error_CUFFT != CUFFT_SUCCESS) goto Error;

	MultiplyPointwiseComplex<<<(arraySizeComplex + 255) >> 8, 256>>>(params_OptiX.F_2, params_OptiX.F_1, params_OptiX.F_2, arraySizeComplex);
	error_CUDA = cudaGetLastError();
	if (error_CUDA != cudaSuccess) goto Error;

	error_CUFFT = IDFFT_G(params_OptiX.planc2r, params_OptiX.F_2, params_OptiX.mu_bitmap_out_bitmap_ref_B);
	if (error_CUFFT != CUFFT_SUCCESS) goto Error;

	// *********************************************************************************************

	ComputeArraysForGradientComputation<<<(arraySizeReal + 255) >> 8, 256>>>(
		params_OptiX.mu_bitmap_out_R, params_OptiX.mu_bitmap_ref_R + (params_OptiX.poseNum * arraySizeReal),
		params_OptiX.mu_bitmap_out_bitmap_ref_R,
		params_OptiX.mu_bitmap_out_R_square, params_OptiX.mu_bitmap_ref_R_square + (params_OptiX.poseNum * arraySizeReal),

		params_OptiX.mu_bitmap_out_G, params_OptiX.mu_bitmap_ref_G + (params_OptiX.poseNum * arraySizeReal),
		params_OptiX.mu_bitmap_out_bitmap_ref_G,
		params_OptiX.mu_bitmap_out_G_square, params_OptiX.mu_bitmap_ref_G_square + (params_OptiX.poseNum * arraySizeReal),

		params_OptiX.mu_bitmap_out_B, params_OptiX.mu_bitmap_ref_B + (params_OptiX.poseNum * arraySizeReal),
		params_OptiX.mu_bitmap_out_bitmap_ref_B,
		params_OptiX.mu_bitmap_out_B_square, params_OptiX.mu_bitmap_ref_B_square + (params_OptiX.poseNum * arraySizeReal),

		// !!! !!! !!!
		params_OptiX.mu_bitmap_out_R, params_OptiX.mu_bitmap_out_bitmap_ref_R, params_OptiX.mu_bitmap_out_R_square,
		params_OptiX.mu_bitmap_out_G, params_OptiX.mu_bitmap_out_bitmap_ref_G, params_OptiX.mu_bitmap_out_G_square,
		params_OptiX.mu_bitmap_out_B, params_OptiX.mu_bitmap_out_bitmap_ref_B, params_OptiX.mu_bitmap_out_B_square,
		// !!! !!! !!!

		params_OptiX.width + (kernel_size - 1), params_OptiX.width + (kernel_size - 1), kernel_radius
	);
	error_CUDA = cudaGetLastError();
	if (error_CUDA != cudaSuccess) goto Error;

	// *********************************************************************************************

	//**********************************************************
	// Compute auxiliary convolutions for gradient computation *
	//**********************************************************

	// convolution_1_R
	error_CUFFT = DFFT_G(params_OptiX.planr2c, params_OptiX.mu_bitmap_out_R, params_OptiX.F_2);
	if (error_CUFFT != CUFFT_SUCCESS) goto Error;

	MultiplyPointwiseComplex<<<(arraySizeComplex + 255) >> 8, 256>>>(params_OptiX.F_2, params_OptiX.F_1, params_OptiX.F_2, arraySizeComplex);
	error_CUDA = cudaGetLastError();
	if (error_CUDA != cudaSuccess) goto Error;

	error_CUFFT = IDFFT_G(params_OptiX.planc2r, params_OptiX.F_2, params_OptiX.mu_bitmap_out_R);
	if (error_CUFFT != CUFFT_SUCCESS) goto Error;

	// *** *** *** *** ***

	// convolution_1_G
	error_CUFFT = DFFT_G(params_OptiX.planr2c, params_OptiX.mu_bitmap_out_G, params_OptiX.F_2);
	if (error_CUFFT != CUFFT_SUCCESS) goto Error;

	MultiplyPointwiseComplex<<<(arraySizeComplex + 255) >> 8, 256>>>(params_OptiX.F_2, params_OptiX.F_1, params_OptiX.F_2, arraySizeComplex);
	error_CUDA = cudaGetLastError();
	if (error_CUDA != cudaSuccess) goto Error;

	error_CUFFT = IDFFT_G(params_OptiX.planc2r, params_OptiX.F_2, params_OptiX.mu_bitmap_out_G);
	if (error_CUFFT != CUFFT_SUCCESS) goto Error;

	// *** *** *** *** ***

	// convolution_1_B
	error_CUFFT = DFFT_G(params_OptiX.planr2c, params_OptiX.mu_bitmap_out_B, params_OptiX.F_2);
	if (error_CUFFT != CUFFT_SUCCESS) goto Error;

	MultiplyPointwiseComplex<<<(arraySizeComplex + 255) >> 8, 256>>>(params_OptiX.F_2, params_OptiX.F_1, params_OptiX.F_2, arraySizeComplex);
	error_CUDA = cudaGetLastError();
	if (error_CUDA != cudaSuccess) goto Error;

	error_CUFFT = IDFFT_G(params_OptiX.planc2r, params_OptiX.F_2, params_OptiX.mu_bitmap_out_B);
	if (error_CUFFT != CUFFT_SUCCESS) goto Error;

	// *** *** *** *** ***

	// convolution_2_R
	error_CUFFT = DFFT_G(params_OptiX.planr2c, params_OptiX.mu_bitmap_out_bitmap_ref_R, params_OptiX.F_2);
	if (error_CUFFT != CUFFT_SUCCESS) goto Error;

	MultiplyPointwiseComplex<<<(arraySizeComplex + 255) >> 8, 256>>>(params_OptiX.F_2, params_OptiX.F_1, params_OptiX.F_2, arraySizeComplex);
	error_CUDA = cudaGetLastError();
	if (error_CUDA != cudaSuccess) goto Error;

	error_CUFFT = IDFFT_G(params_OptiX.planc2r, params_OptiX.F_2, params_OptiX.mu_bitmap_out_bitmap_ref_R);
	if (error_CUFFT != CUFFT_SUCCESS) goto Error;

	// *** *** *** *** ***

	// convolution_2_G
	error_CUFFT = DFFT_G(params_OptiX.planr2c, params_OptiX.mu_bitmap_out_bitmap_ref_G, params_OptiX.F_2);
	if (error_CUFFT != CUFFT_SUCCESS) goto Error;

	MultiplyPointwiseComplex<<<(arraySizeComplex + 255) >> 8, 256>>>(params_OptiX.F_2, params_OptiX.F_1, params_OptiX.F_2, arraySizeComplex);
	error_CUDA = cudaGetLastError();
	if (error_CUDA != cudaSuccess) goto Error;

	error_CUFFT = IDFFT_G(params_OptiX.planc2r, params_OptiX.F_2, params_OptiX.mu_bitmap_out_bitmap_ref_G);
	if (error_CUFFT != CUFFT_SUCCESS) goto Error;

	// *** *** *** *** ***

	// convolution_2_B
	error_CUFFT = DFFT_G(params_OptiX.planr2c, params_OptiX.mu_bitmap_out_bitmap_ref_B, params_OptiX.F_2);
	if (error_CUFFT != CUFFT_SUCCESS) goto Error;

	MultiplyPointwiseComplex<<<(arraySizeComplex + 255) >> 8, 256>>>(params_OptiX.F_2, params_OptiX.F_1, params_OptiX.F_2, arraySizeComplex);
	error_CUDA = cudaGetLastError();
	if (error_CUDA != cudaSuccess) goto Error;

	error_CUFFT = IDFFT_G(params_OptiX.planc2r, params_OptiX.F_2, params_OptiX.mu_bitmap_out_bitmap_ref_B);
	if (error_CUFFT != CUFFT_SUCCESS) goto Error;

	// *** *** *** *** ***

	// convolution_3_R
	error_CUFFT = DFFT_G(params_OptiX.planr2c, params_OptiX.mu_bitmap_out_R_square, params_OptiX.F_2);
	if (error_CUFFT != CUFFT_SUCCESS) goto Error;

	MultiplyPointwiseComplex<<<(arraySizeComplex + 255) >> 8, 256>>>(params_OptiX.F_2, params_OptiX.F_1, params_OptiX.F_2, arraySizeComplex);
	error_CUDA = cudaGetLastError();
	if (error_CUDA != cudaSuccess) goto Error;

	error_CUFFT = IDFFT_G(params_OptiX.planc2r, params_OptiX.F_2, params_OptiX.mu_bitmap_out_R_square);
	if (error_CUFFT != CUFFT_SUCCESS) goto Error;

	// *** *** *** *** ***

	// convolution_3_G
	error_CUFFT = DFFT_G(params_OptiX.planr2c, params_OptiX.mu_bitmap_out_G_square, params_OptiX.F_2);
	if (error_CUFFT != CUFFT_SUCCESS) goto Error;

	MultiplyPointwiseComplex<<<(arraySizeComplex + 255) >> 8, 256>>>(params_OptiX.F_2, params_OptiX.F_1, params_OptiX.F_2, arraySizeComplex);
	error_CUDA = cudaGetLastError();
	if (error_CUDA != cudaSuccess) goto Error;

	error_CUFFT = IDFFT_G(params_OptiX.planc2r, params_OptiX.F_2, params_OptiX.mu_bitmap_out_G_square);
	if (error_CUFFT != CUFFT_SUCCESS) goto Error;

	// *** *** *** *** ***

	// convolution_3_B
	error_CUFFT = DFFT_G(params_OptiX.planr2c, params_OptiX.mu_bitmap_out_B_square, params_OptiX.F_2);
	if (error_CUFFT != CUFFT_SUCCESS) goto Error;

	MultiplyPointwiseComplex<<<(arraySizeComplex + 255) >> 8, 256>>>(params_OptiX.F_2, params_OptiX.F_1, params_OptiX.F_2, arraySizeComplex);
	error_CUDA = cudaGetLastError();
	if (error_CUDA != cudaSuccess) goto Error;

	error_CUFFT = IDFFT_G(params_OptiX.planc2r, params_OptiX.F_2, params_OptiX.mu_bitmap_out_B_square);
	if (error_CUFFT != CUFFT_SUCCESS) goto Error;

	// *********************************************************************************************

	ComputeGradientSSIM<<<(arraySizeReal + 255) >> 8, 256>>>(
		params_OptiX.mu_bitmap_out_R,
		params_OptiX.mu_bitmap_out_bitmap_ref_R, params_OptiX.bitmap_ref_R + (params_OptiX.poseNum * arraySizeReal),
		params_OptiX.mu_bitmap_out_R_square, params_OptiX.bitmap_out_R,

		params_OptiX.mu_bitmap_out_G,
		params_OptiX.mu_bitmap_out_bitmap_ref_G, params_OptiX.bitmap_ref_G + (params_OptiX.poseNum * arraySizeReal),
		params_OptiX.mu_bitmap_out_G_square, params_OptiX.bitmap_out_G,

		params_OptiX.mu_bitmap_out_B,
		params_OptiX.mu_bitmap_out_bitmap_ref_B, params_OptiX.bitmap_ref_B + (params_OptiX.poseNum * arraySizeReal),
		params_OptiX.mu_bitmap_out_B_square, params_OptiX.bitmap_out_B,

		// !!! !!! !!!
		params_OptiX.mu_bitmap_out_R, params_OptiX.mu_bitmap_out_G, params_OptiX.mu_bitmap_out_B,
		// !!! !!! !!!

		params_OptiX.width + (kernel_size - 1), params_OptiX.width + (kernel_size - 1), kernel_size
	);
	error_CUDA = cudaGetLastError();
	if (error_CUDA != cudaSuccess) goto Error;

	// *********************************************************************************************

	// TEST
	/*if (params_OptiX.epoch == 1000) {
		REAL_G *buf = NULL;
		buf = (REAL_G *)malloc(sizeof(REAL_G) * arraySizeReal);
		if (buf == NULL) goto Error;
		unsigned *bitmap_ref = (unsigned *)malloc(sizeof(unsigned) * params_OptiX.width * params_OptiX.width);

		// R channel
		error_CUDA = cudaMemcpy(buf, params_OptiX.mu_bitmap_out_R, sizeof(REAL_G) * arraySizeReal, cudaMemcpyDeviceToHost);
		if (error_CUDA != cudaSuccess) goto Error;

		for (int i = 0; i < params_OptiX.width; ++i) {
			for (int j = 0; j < params_OptiX.width; ++j) {
				REAL_G Rf = buf[((kernel_radius + i) * (params_OptiX.width + (kernel_size - 1))) + (kernel_radius + j)] / arraySizeReal;
				if (Rf < ((REAL_G)0.0)) Rf = ((REAL_G)0.0);
				if (Rf > ((REAL_G)1.0)) Rf = ((REAL_G)1.0);
				unsigned char Ri = Rf * ((REAL_G)255.0);
				bitmap_ref[(i * params_OptiX.width) + j] = (((unsigned) Ri) << 16);
			}
		}

		// G channel
		error_CUDA = cudaMemcpy(buf, params_OptiX.mu_bitmap_out_G, sizeof(REAL_G) *  arraySizeReal, cudaMemcpyDeviceToHost);
		if (error_CUDA != cudaSuccess) goto Error;

		for (int i = 0; i < params_OptiX.width; ++i) {
			for (int j = 0; j < params_OptiX.width; ++j) {
				REAL_G Gf = buf[((kernel_radius + i) * (params_OptiX.width + (kernel_size - 1))) + (kernel_radius + j)] / arraySizeReal;
				if (Gf < ((REAL_G)0.0)) Gf = ((REAL_G)0.0);
				if (Gf > ((REAL_G)1.0)) Gf = ((REAL_G)1.0);
				unsigned char Gi = Gf * ((REAL_G)255.0);
				bitmap_ref[(i * params_OptiX.width) + j] |= (((unsigned) Gi) << 8);
			}
		}

		// B channel
		error_CUDA = cudaMemcpy(buf, params_OptiX.mu_bitmap_out_B, sizeof(REAL_G) *  arraySizeReal, cudaMemcpyDeviceToHost);
		if (error_CUDA != cudaSuccess) goto Error;

		for (int i = 0; i < params_OptiX.width; ++i) {
			for (int j = 0; j < params_OptiX.width; ++j) {
				REAL_G Bf = buf[((kernel_radius + i) * (params_OptiX.width + (kernel_size - 1))) + (kernel_radius + j)] / arraySizeReal;
				if (Bf < ((REAL_G)0.0)) Bf = ((REAL_G)0.0);
				if (Bf > ((REAL_G)1.0)) Bf = ((REAL_G)1.0);
				unsigned char Bi = Bf * ((REAL_G)255.0);
				bitmap_ref[(i * params_OptiX.width) + j] |= ((unsigned)Bi);
			}
		}

		// Copy to bitmap on hdd
		unsigned char *foo = (unsigned char *)malloc(3 * params_OptiX.width * params_OptiX.width);
		for (int i = 0; i < params_OptiX.width; ++i) {
			for (int j = 0; j < params_OptiX.width; ++j) {
				unsigned char R = bitmap_ref[(i * params_OptiX.width) + j] >> 16;
				unsigned char G = (bitmap_ref[(i * params_OptiX.width) + j] >> 8) & 255;
				unsigned char B = bitmap_ref[(i * params_OptiX.width) + j] & 255;
				foo[((((params_OptiX.width - 1 - i) * params_OptiX.width) + j) * 3) + 2] = R;
				foo[((((params_OptiX.width - 1 - i) * params_OptiX.width) + j) * 3) + 1] = G;
				foo[(((params_OptiX.width - 1 - i) * params_OptiX.width) + j) * 3] = B;		
			}
		}

		FILE *f = fopen("test.bmp", "rb+");
		fseek(f, 54, SEEK_SET);
		fwrite(foo, sizeof(int) * params_OptiX.width * params_OptiX.width, 1, f);
		fclose(f);

		free(buf);
		free(bitmap_ref);
	}*/

	//***********************************************************************************************

	ComputeGradient<<<((params_OptiX.width * params_OptiX.width) + 63) >> 6, 64>>>(params_OptiX);
	error_CUDA = cudaGetLastError();
	if (error_CUDA != cudaSuccess) goto Error;

	if (
		(params_OptiX.epoch >= densification_start_epoch_host) &&
		(params_OptiX.epoch <= densification_end_epoch_host) &&
		((params_OptiX.epoch % densification_frequency_host) == 0)
	) {
		error_CUDA = cudaMemset(params_OptiX.counter1, 0, sizeof(unsigned) * 1);
		if (error_CUDA != cudaSuccess) goto Error;
	}

	// DEBUG GRADIENT
	//if (params_OptiX.epoch == 1) {
	//	DumpParameters(params_OptiX);
	//}

	error_CUDA = cudaMemcpyToSymbol(auxiliary_values, &initial_values, sizeof(SAuxiliaryValues) * 1);
	if (error_CUDA != cudaSuccess) goto Error;

	dev_UpdateGradientOptiX<<<(params_OptiX.numberOfGaussians + 63) >> 6, 64>>>(params_OptiX);
	error_CUDA = cudaGetLastError();
	if (error_CUDA != cudaSuccess) goto Error;

	SAuxiliaryValues auxiliary_values_local;

	error_CUDA = cudaMemcpyFromSymbol(&auxiliary_values_local, auxiliary_values, sizeof(SAuxiliaryValues) * 1);
	if (error_CUDA != cudaSuccess) goto Error;

	float dX = SortableUint2Float(auxiliary_values_local.scene_upper_bound.x) - SortableUint2Float(auxiliary_values_local.scene_lower_bound.x);
	float dY = SortableUint2Float(auxiliary_values_local.scene_upper_bound.y) - SortableUint2Float(auxiliary_values_local.scene_lower_bound.y);
	float dZ = SortableUint2Float(auxiliary_values_local.scene_upper_bound.z) - SortableUint2Float(auxiliary_values_local.scene_lower_bound.z);

	float scene_extent_local = sqrtf((dX * dX) + (dY * dY) + (dZ * dZ));

	error_CUDA = cudaMemcpyToSymbol(scene_extent, &scene_extent_local, sizeof(float) * 1);
	if (error_CUDA != cudaSuccess) goto Error;

	if (
		(params_OptiX.epoch >= densification_start_epoch_host) &&
		(params_OptiX.epoch <= densification_end_epoch_host) &&
		((params_OptiX.epoch % densification_frequency_host) == 0)
	) {
		error_CUDA = cudaMemcpy(&params_OptiX.numberOfGaussians, params_OptiX.counter1, sizeof(unsigned) * 1, cudaMemcpyDeviceToHost);
		if (error_CUDA != cudaSuccess) goto Error;
	}

	error_CUDA = cudaMemcpy(&params_OptiX.loss_host, params_OptiX.loss_device, sizeof(double) * 1, cudaMemcpyDeviceToHost);
	if (error_CUDA != cudaSuccess) goto Error;

	if (
		(params_OptiX.epoch >= densification_start_epoch_host) &&
		(params_OptiX.epoch <= densification_end_epoch_host) &&
		((params_OptiX.epoch % densification_frequency_host) == 0)
	) {
		int tmp1;

		tmp1 = params_OptiX.maxNumberOfGaussians1;
		params_OptiX.maxNumberOfGaussians1 = params_OptiX.maxNumberOfGaussians2;
		params_OptiX.maxNumberOfGaussians2 = tmp1;

		void *tmp2;

		tmp2 = params_OptiX.GC_part_1_1; params_OptiX.GC_part_1_1 = params_OptiX.GC_part_1_2; params_OptiX.GC_part_1_2 = (float4 *)tmp2;
		tmp2 = params_OptiX.GC_part_2_1; params_OptiX.GC_part_2_1 = params_OptiX.GC_part_2_2; params_OptiX.GC_part_2_2 = (float4 *)tmp2;
		tmp2 = params_OptiX.GC_part_3_1; params_OptiX.GC_part_3_1 = params_OptiX.GC_part_3_2; params_OptiX.GC_part_3_2 = (float4 *)tmp2;
		tmp2 = params_OptiX.GC_part_4_1; params_OptiX.GC_part_4_1 = params_OptiX.GC_part_4_2; params_OptiX.GC_part_4_2 = (float2 *)tmp2;

		tmp2 = params_OptiX.m11; params_OptiX.m11 = params_OptiX.m12; params_OptiX.m12 = (float4 *)tmp2;
		tmp2 = params_OptiX.m21; params_OptiX.m21 = params_OptiX.m22; params_OptiX.m22 = (float4 *)tmp2;
		tmp2 = params_OptiX.m31; params_OptiX.m31 = params_OptiX.m32; params_OptiX.m32 = (float4 *)tmp2;
		tmp2 = params_OptiX.m41; params_OptiX.m41 = params_OptiX.m42; params_OptiX.m42 = (float2 *)tmp2;

		tmp2 = params_OptiX.v11; params_OptiX.v11 = params_OptiX.v12; params_OptiX.v12 = (float4 *)tmp2;
		tmp2 = params_OptiX.v21; params_OptiX.v21 = params_OptiX.v22; params_OptiX.v22 = (float4 *)tmp2;
		tmp2 = params_OptiX.v31; params_OptiX.v31 = params_OptiX.v32; params_OptiX.v32 = (float4 *)tmp2;
		tmp2 = params_OptiX.v41; params_OptiX.v41 = params_OptiX.v42; params_OptiX.v42 = (float2 *)tmp2;

		if ((params_OptiX.numberOfGaussians * 2) > params_OptiX.maxNumberOfGaussians2) {
			params_OptiX.maxNumberOfGaussians2 = params_OptiX.numberOfGaussians * 3;

			// *************************************************************************************

			if (params_OptiX.maxNumberOfGaussians2 > params_OptiX.maxNumberOfGaussians1) {
				// !!! !!! !!!
				// Nale¿y przenieœæ zawrtoœæ aktualnej tablicy bounding box'ów do nowej tablicy
				void *tmp;
			
				error_CUDA = cudaMalloc(&tmp, sizeof(float) * 6 * params_OptiX.maxNumberOfGaussians2);
				if (error_CUDA != cudaSuccess) goto Error;

				error_CUDA = cudaMemcpy(tmp, params_OptiX.aabbBuffer, sizeof(float) * 6 * params_OptiX.numberOfGaussians, cudaMemcpyDeviceToDevice);
				if (error_CUDA != cudaSuccess) goto Error;

				error_CUDA = cudaFree(params_OptiX.aabbBuffer);
				if (error_CUDA != cudaSuccess) goto Error;

				params_OptiX.aabbBuffer = tmp;
				// !!! !!! !!!
			}

			// *************************************************************************************

			error_CUDA = cudaFree(params_OptiX.GC_part_1_2);
			if (error_CUDA != cudaSuccess) goto Error;

			error_CUDA = cudaFree(params_OptiX.GC_part_2_2);
			if (error_CUDA != cudaSuccess) goto Error;

			error_CUDA = cudaFree(params_OptiX.GC_part_3_2);
			if (error_CUDA != cudaSuccess) goto Error;

			error_CUDA = cudaFree(params_OptiX.GC_part_4_2);
			if (error_CUDA != cudaSuccess) goto Error;

			error_CUDA = cudaMalloc(&params_OptiX.GC_part_1_2, sizeof(float4) * params_OptiX.maxNumberOfGaussians2);
			if (error_CUDA != cudaSuccess) goto Error;

			error_CUDA = cudaMalloc(&params_OptiX.GC_part_2_2, sizeof(float4) * params_OptiX.maxNumberOfGaussians2);
			if (error_CUDA != cudaSuccess) goto Error;

			error_CUDA = cudaMalloc(&params_OptiX.GC_part_3_2, sizeof(float4) * params_OptiX.maxNumberOfGaussians2);
			if (error_CUDA != cudaSuccess) goto Error;

			error_CUDA = cudaMalloc(&params_OptiX.GC_part_4_2, sizeof(float2) * params_OptiX.maxNumberOfGaussians2);
			if (error_CUDA != cudaSuccess) goto Error;

			// ************************************************************************************************

			if (params_OptiX.maxNumberOfGaussians2 > params_OptiX.maxNumberOfGaussians1) {
				error_CUDA = cudaFree(params_OptiX.dL_dparams_1);
				if (error_CUDA != cudaSuccess) goto Error;

				error_CUDA = cudaFree(params_OptiX.dL_dparams_2);
				if (error_CUDA != cudaSuccess) goto Error;

				error_CUDA = cudaFree(params_OptiX.dL_dparams_3);
				if (error_CUDA != cudaSuccess) goto Error;

				error_CUDA = cudaFree(params_OptiX.dL_dparams_4);
				if (error_CUDA != cudaSuccess) goto Error;

				error_CUDA = cudaMalloc(&params_OptiX.dL_dparams_1, sizeof(REAL4_G) * params_OptiX.maxNumberOfGaussians2);
				if (error_CUDA != cudaSuccess) goto Error;

				error_CUDA = cudaMalloc(&params_OptiX.dL_dparams_2, sizeof(REAL4_G) * params_OptiX.maxNumberOfGaussians2);
				if (error_CUDA != cudaSuccess) goto Error;

				error_CUDA = cudaMalloc(&params_OptiX.dL_dparams_3, sizeof(REAL4_G) * params_OptiX.maxNumberOfGaussians2);
				if (error_CUDA != cudaSuccess) goto Error;

				error_CUDA = cudaMalloc(&params_OptiX.dL_dparams_4, sizeof(REAL2_G) * params_OptiX.maxNumberOfGaussians2);
				if (error_CUDA != cudaSuccess) goto Error;
			}

			// ************************************************************************************************

			error_CUDA = cudaFree(params_OptiX.m12);
			if (error_CUDA != cudaSuccess) goto Error;

			error_CUDA = cudaFree(params_OptiX.m22);
			if (error_CUDA != cudaSuccess) goto Error;

			error_CUDA = cudaFree(params_OptiX.m32);
			if (error_CUDA != cudaSuccess) goto Error;

			error_CUDA = cudaFree(params_OptiX.m42);
			if (error_CUDA != cudaSuccess) goto Error;

			error_CUDA = cudaMalloc(&params_OptiX.m12, sizeof(float4) * params_OptiX.maxNumberOfGaussians2);
			if (error_CUDA != cudaSuccess) goto Error;

			error_CUDA = cudaMalloc(&params_OptiX.m22, sizeof(float4) * params_OptiX.maxNumberOfGaussians2);
			if (error_CUDA != cudaSuccess) goto Error;

			error_CUDA = cudaMalloc(&params_OptiX.m32, sizeof(float4) * params_OptiX.maxNumberOfGaussians2);
			if (error_CUDA != cudaSuccess) goto Error;

			error_CUDA = cudaMalloc(&params_OptiX.m42, sizeof(float2) * params_OptiX.maxNumberOfGaussians2);
			if (error_CUDA != cudaSuccess) goto Error;

			// ************************************************************************************************

			error_CUDA = cudaFree(params_OptiX.v12);
			if (error_CUDA != cudaSuccess) goto Error;

			error_CUDA = cudaFree(params_OptiX.v22);
			if (error_CUDA != cudaSuccess) goto Error;

			error_CUDA = cudaFree(params_OptiX.v32);
			if (error_CUDA != cudaSuccess) goto Error;

			error_CUDA = cudaFree(params_OptiX.v42);
			if (error_CUDA != cudaSuccess) goto Error;

			error_CUDA = cudaMalloc(&params_OptiX.v12, sizeof(float4) * params_OptiX.maxNumberOfGaussians2);
			if (error_CUDA != cudaSuccess) goto Error;

			error_CUDA = cudaMalloc(&params_OptiX.v22, sizeof(float4) * params_OptiX.maxNumberOfGaussians2);
			if (error_CUDA != cudaSuccess) goto Error;

			error_CUDA = cudaMalloc(&params_OptiX.v32, sizeof(float4) * params_OptiX.maxNumberOfGaussians2);
			if (error_CUDA != cudaSuccess) goto Error;

			error_CUDA = cudaMalloc(&params_OptiX.v42, sizeof(float2) * params_OptiX.maxNumberOfGaussians2);
			if (error_CUDA != cudaSuccess) goto Error;
		}
	}

	// ************************************************************************************************

	OptixAccelBuildOptions accel_options = {};
	accel_options.buildFlags = OPTIX_BUILD_FLAG_ALLOW_COMPACTION;
	accel_options.operation  = OPTIX_BUILD_OPERATION_BUILD;

	OptixBuildInput aabb_input = {};
	aabb_input.type                               = OPTIX_BUILD_INPUT_TYPE_CUSTOM_PRIMITIVES;
	aabb_input.customPrimitiveArray.aabbBuffers   = (CUdeviceptr *)&params_OptiX.aabbBuffer;
	aabb_input.customPrimitiveArray.numPrimitives = params_OptiX.numberOfGaussians;

	unsigned aabb_input_flags[1]                  = {OPTIX_GEOMETRY_FLAG_NONE};
	aabb_input.customPrimitiveArray.flags         = (const unsigned int *)aabb_input_flags;
	aabb_input.customPrimitiveArray.numSbtRecords = 1;

	// *********************************************************************************************

	OptixAccelBufferSizes blasBufferSizes;
	error_OptiX = optixAccelComputeMemoryUsage(
		params_OptiX.optixContext,
		&accel_options,
		&aabb_input,
		1,
		&blasBufferSizes
	);
	if (error_OptiX != OPTIX_SUCCESS) goto Error;

	// *********************************************************************************************

	OptixAccelEmitDesc emitDesc;
	emitDesc.type   = OPTIX_PROPERTY_TYPE_COMPACTED_SIZE;
	emitDesc.result = (CUdeviceptr)params_OptiX.compactedSizeBuffer;

	if (blasBufferSizes.tempSizeInBytes > params_OptiX.tempBufferSize) {
		error_CUDA = cudaFree(params_OptiX.tempBuffer);
		if (error_CUDA != cudaSuccess) goto Error;

		params_OptiX.tempBufferSize = blasBufferSizes.tempSizeInBytes * 2; // !!! !!! !!!
		error_CUDA = cudaMalloc(&params_OptiX.tempBuffer, params_OptiX.tempBufferSize);
		if (error_CUDA != cudaSuccess) goto Error;
	}

	if (blasBufferSizes.outputSizeInBytes > params_OptiX.outputBufferSize) {
		error_CUDA = cudaFree(params_OptiX.outputBuffer);
		if (error_CUDA != cudaSuccess) goto Error;

		params_OptiX.outputBufferSize = blasBufferSizes.outputSizeInBytes * 2; // !!! !!! !!!
		error_CUDA = cudaMalloc(&params_OptiX.outputBuffer, params_OptiX.outputBufferSize);
		if (error_CUDA != cudaSuccess) goto Error;
	}

	// *********************************************************************************************

	error_OptiX = optixAccelBuild(
		params_OptiX.optixContext,
		0,
		&accel_options,
		&aabb_input,
		1,  
		(CUdeviceptr)params_OptiX.tempBuffer,
		blasBufferSizes.tempSizeInBytes,
		(CUdeviceptr)params_OptiX.outputBuffer,
		blasBufferSizes.outputSizeInBytes,
		&params_OptiX.asHandle,
		&emitDesc,
		1
	);
	if (error_OptiX != OPTIX_SUCCESS) goto Error;

	cudaDeviceSynchronize();
	error_CUDA = cudaGetLastError();
	if (error_CUDA != cudaSuccess) goto Error;

	unsigned long long compactedSize;
	error_CUDA = cudaMemcpy(&compactedSize, params_OptiX.compactedSizeBuffer, 8, cudaMemcpyDeviceToHost);
	if (error_CUDA != cudaSuccess) goto Error;

	if (compactedSize > params_OptiX.asBufferSize) {
		error_CUDA = cudaFree(params_OptiX.asBuffer);
		if (error_CUDA != cudaSuccess) goto Error;

		params_OptiX.asBufferSize = compactedSize * 2; // !!! !!! !!! 
		error_CUDA = cudaMalloc(&params_OptiX.asBuffer, params_OptiX.asBufferSize);
		if (error_CUDA != cudaSuccess) goto Error;
	}

	error_OptiX = optixAccelCompact(
		params_OptiX.optixContext,
		0,
		params_OptiX.asHandle,
		(CUdeviceptr)params_OptiX.asBuffer,
		compactedSize,
		&params_OptiX.asHandle
	);
	if (error_OptiX != OPTIX_SUCCESS) goto Error;

	cudaDeviceSynchronize();
	error_CUDA = cudaGetLastError();
	if (error_CUDA != cudaSuccess) goto Error;

	// *********************************************************************************************

	return true;
Error:
	return false;
}

// *************************************************************************************************

bool SetConfigurationOptiX(SOptiXRenderConfig& config_OptiX) {
	cudaMemcpyToSymbol(lr_RGB, &config_OptiX.lr_RGB, sizeof(float));
	cudaMemcpyToSymbol(lr_RGB_exponential_decay_coefficient, &config_OptiX.lr_RGB_exponential_decay_coefficient, sizeof(float));
	cudaMemcpyToSymbol(lr_alpha, &config_OptiX.lr_alpha, sizeof(float));
	cudaMemcpyToSymbol(lr_alpha_exponential_decay_coefficient, &config_OptiX.lr_alpha_exponential_decay_coefficient, sizeof(float));
	cudaMemcpyToSymbol(lr_m, &config_OptiX.lr_m, sizeof(float));
	cudaMemcpyToSymbol(lr_m_exponential_decay_coefficient, &config_OptiX.lr_m_exponential_decay_coefficient, sizeof(float));
	cudaMemcpyToSymbol(lr_s, &config_OptiX.lr_s, sizeof(float));
	cudaMemcpyToSymbol(lr_s_exponential_decay_coefficient, &config_OptiX.lr_s_exponential_decay_coefficient, sizeof(float));
	cudaMemcpyToSymbol(lr_q, &config_OptiX.lr_q, sizeof(float));
	cudaMemcpyToSymbol(lr_q_exponential_decay_coefficient, &config_OptiX.lr_q_exponential_decay_coefficient, sizeof(float));

	cudaMemcpyToSymbol(densification_frequency, &config_OptiX.densification_frequency, sizeof(int));
	densification_frequency_host = config_OptiX.densification_frequency;

	cudaMemcpyToSymbol(densification_start_epoch, &config_OptiX.densification_start_epoch, sizeof(int));
	densification_start_epoch_host = config_OptiX.densification_start_epoch;

	cudaMemcpyToSymbol(densification_end_epoch, &config_OptiX.densification_end_epoch, sizeof(int));
	densification_end_epoch_host = config_OptiX.densification_end_epoch;

	cudaMemcpyToSymbol(alpha_threshold_for_Gauss_removal, &config_OptiX.alpha_threshold_for_Gauss_removal, sizeof(float));

	cudaMemcpyToSymbol(min_s_coefficients_clipping_threshold, &config_OptiX.min_s_coefficients_clipping_threshold, sizeof(float));
	min_s_coefficients_clipping_threshold_host = config_OptiX.min_s_coefficients_clipping_threshold;

	cudaMemcpyToSymbol(max_s_coefficients_clipping_threshold, &config_OptiX.max_s_coefficients_clipping_threshold, sizeof(float));
	max_s_coefficients_clipping_threshold_host = config_OptiX.max_s_coefficients_clipping_threshold;

	cudaMemcpyToSymbol(min_s_norm_threshold_for_Gauss_removal, &config_OptiX.min_s_norm_threshold_for_Gauss_removal, sizeof(float));
	cudaMemcpyToSymbol(max_s_norm_threshold_for_Gauss_removal, &config_OptiX.max_s_norm_threshold_for_Gauss_removal, sizeof(float));
	cudaMemcpyToSymbol(mu_grad_norm_threshold_for_densification, &config_OptiX.mu_grad_norm_threshold_for_densification, sizeof(float));
	cudaMemcpyToSymbol(s_norm_threshold_for_split_strategy, &config_OptiX.s_norm_threshold_for_split_strategy, sizeof(float));
	cudaMemcpyToSymbol(split_ratio, &config_OptiX.split_ratio, sizeof(float));
	cudaMemcpyToSymbol(lambda, &config_OptiX.lambda, sizeof(float));
	
	cudaMemcpyToSymbol(ray_termination_T_threshold, &config_OptiX.ray_termination_T_threshold, sizeof(float));
	ray_termination_T_threshold_host = config_OptiX.ray_termination_T_threshold;

	last_significant_Gauss_alpha_gradient_precision_host = config_OptiX.last_significant_Gauss_alpha_gradient_precision;

	cudaMemcpyToSymbol(max_Gaussians_per_ray, &config_OptiX.max_Gaussians_per_ray, sizeof(int));
	max_Gaussians_per_ray_host = config_OptiX.max_Gaussians_per_ray;

	return true;
}

// *************************************************************************************************

void GetSceneBoundsOptiX(float& lB, float& rB, float& uB, float& dB, float& bB, float& fB, float &scene_extent_param) {
	SAuxiliaryValues values;
	cudaMemcpyFromSymbol(&values, auxiliary_values, sizeof(SAuxiliaryValues));
	cudaMemcpyFromSymbol(&scene_extent_param, scene_extent, sizeof(float));
	lB = SortableUint2Float(values.scene_lower_bound.x);
	rB = SortableUint2Float(values.scene_upper_bound.x);
	uB = SortableUint2Float(values.scene_lower_bound.y);
	dB = SortableUint2Float(values.scene_upper_bound.y);
	bB = SortableUint2Float(values.scene_lower_bound.z);
	fB = SortableUint2Float(values.scene_upper_bound.z);
}