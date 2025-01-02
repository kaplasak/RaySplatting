#include "Header.cuh"

// *************************************************************************************************

static void DumpToFile(const char *fPath, int epochNum, const char *fExtension, void *buf, int size) {
	FILE *f;

	char fName[256];
	sprintf_s(fName, "%s/%d.%s", fPath, epochNum, fExtension);

	fopen_s(&f, fName, "wb");
	fwrite(buf, size, 1, f);
	fclose(f);
}

// *************************************************************************************************

bool DumpParameters(SOptiXRenderParams& params_OptiX) {
	cudaError_t error_CUDA;

	void *buf = malloc(sizeof(REAL4_G) * params_OptiX.numberOfGaussians);
	if (buf == NULL) goto Error;

	error_CUDA = cudaMemcpy(buf, params_OptiX.GC_part_1_1, sizeof(float4) * params_OptiX.numberOfGaussians, cudaMemcpyDeviceToHost);
	if (error_CUDA != cudaSuccess) goto Error;
	DumpToFile("dump/save", params_OptiX.epoch, "GC1", buf, sizeof(float4) * params_OptiX.numberOfGaussians);

	error_CUDA = cudaMemcpy(buf, params_OptiX.GC_part_2_1, sizeof(float4) * params_OptiX.numberOfGaussians, cudaMemcpyDeviceToHost);
	if (error_CUDA != cudaSuccess) goto Error;
	DumpToFile("dump/save", params_OptiX.epoch, "GC2", buf, sizeof(float4) * params_OptiX.numberOfGaussians);

	error_CUDA = cudaMemcpy(buf, params_OptiX.GC_part_3_1, sizeof(float4) * params_OptiX.numberOfGaussians, cudaMemcpyDeviceToHost);
	if (error_CUDA != cudaSuccess) goto Error;
	DumpToFile("dump/save", params_OptiX.epoch, "GC3", buf, sizeof(float4) * params_OptiX.numberOfGaussians);

	error_CUDA = cudaMemcpy(buf, params_OptiX.GC_part_4_1, sizeof(float2) * params_OptiX.numberOfGaussians, cudaMemcpyDeviceToHost);
	if (error_CUDA != cudaSuccess) goto Error;
	DumpToFile("dump/save", params_OptiX.epoch, "GC4", buf, sizeof(float2) * params_OptiX.numberOfGaussians);

	error_CUDA = cudaMemcpy(buf, params_OptiX.m11, sizeof(float4) * params_OptiX.numberOfGaussians, cudaMemcpyDeviceToHost);
	if (error_CUDA != cudaSuccess) goto Error;
	DumpToFile("dump/save", params_OptiX.epoch, "m1", buf, sizeof(float4) * params_OptiX.numberOfGaussians);

	error_CUDA = cudaMemcpy(buf, params_OptiX.m21, sizeof(float4) * params_OptiX.numberOfGaussians, cudaMemcpyDeviceToHost);
	if (error_CUDA != cudaSuccess) goto Error;
	DumpToFile("dump/save", params_OptiX.epoch, "m2", buf, sizeof(float4) * params_OptiX.numberOfGaussians);

	error_CUDA = cudaMemcpy(buf, params_OptiX.m31, sizeof(float4) * params_OptiX.numberOfGaussians, cudaMemcpyDeviceToHost);
	if (error_CUDA != cudaSuccess) goto Error;
	DumpToFile("dump/save", params_OptiX.epoch, "m3", buf, sizeof(float4) * params_OptiX.numberOfGaussians);

	error_CUDA = cudaMemcpy(buf, params_OptiX.m41, sizeof(float2) * params_OptiX.numberOfGaussians, cudaMemcpyDeviceToHost);
	if (error_CUDA != cudaSuccess) goto Error;
	DumpToFile("dump/save", params_OptiX.epoch, "m4", buf, sizeof(float2) * params_OptiX.numberOfGaussians);

	error_CUDA = cudaMemcpy(buf, params_OptiX.v11, sizeof(float4) * params_OptiX.numberOfGaussians, cudaMemcpyDeviceToHost);
	if (error_CUDA != cudaSuccess) goto Error;
	DumpToFile("dump/save", params_OptiX.epoch, "v1", buf, sizeof(float4) * params_OptiX.numberOfGaussians);

	error_CUDA = cudaMemcpy(buf, params_OptiX.v21, sizeof(float4) * params_OptiX.numberOfGaussians, cudaMemcpyDeviceToHost);
	if (error_CUDA != cudaSuccess) goto Error;
	DumpToFile("dump/save", params_OptiX.epoch, "v2", buf, sizeof(float4) * params_OptiX.numberOfGaussians);

	error_CUDA = cudaMemcpy(buf, params_OptiX.v31, sizeof(float4) * params_OptiX.numberOfGaussians, cudaMemcpyDeviceToHost);
	if (error_CUDA != cudaSuccess) goto Error;
	DumpToFile("dump/save", params_OptiX.epoch, "v3", buf, sizeof(float4) * params_OptiX.numberOfGaussians);

	error_CUDA = cudaMemcpy(buf, params_OptiX.v41, sizeof(float2) * params_OptiX.numberOfGaussians, cudaMemcpyDeviceToHost);
	if (error_CUDA != cudaSuccess) goto Error;
	DumpToFile("dump/save", params_OptiX.epoch, "v4", buf, sizeof(float2) * params_OptiX.numberOfGaussians);

	// *** *** *** *** ***

	// DEBUG GRADIENT
	/*error_CUDA = cudaMemcpy(buf, params_OptiX.dL_dparams_1, sizeof(REAL4_G) * params_OptiX.numberOfGaussians, cudaMemcpyDeviceToHost);
	if (error_CUDA != cudaSuccess) goto Error;
	DumpToFile("dump/save", params_OptiX.epoch, "params1", buf, sizeof(REAL4_G) * params_OptiX.numberOfGaussians);

	error_CUDA = cudaMemcpy(buf, params_OptiX.dL_dparams_2, sizeof(REAL4_G) * params_OptiX.numberOfGaussians, cudaMemcpyDeviceToHost);
	if (error_CUDA != cudaSuccess) goto Error;
	DumpToFile("dump/save", params_OptiX.epoch, "params2", buf, sizeof(REAL4_G) * params_OptiX.numberOfGaussians);

	error_CUDA = cudaMemcpy(buf, params_OptiX.dL_dparams_3, sizeof(REAL4_G) * params_OptiX.numberOfGaussians, cudaMemcpyDeviceToHost);
	if (error_CUDA != cudaSuccess) goto Error;
	DumpToFile("dump/save", params_OptiX.epoch, "params3", buf, sizeof(REAL4_G) * params_OptiX.numberOfGaussians);

	error_CUDA = cudaMemcpy(buf, params_OptiX.dL_dparams_4, sizeof(REAL2_G) * params_OptiX.numberOfGaussians, cudaMemcpyDeviceToHost);
	if (error_CUDA != cudaSuccess) goto Error;
	DumpToFile("dump/save", params_OptiX.epoch, "params4", buf, sizeof(REAL2_G) * params_OptiX.numberOfGaussians);

	int *foo = (int *)malloc(sizeof(int) * params_OptiX.width * params_OptiX.width * dlugosc_promienia);
	error_CUDA = cudaMemcpy(foo, params_OptiX.Gaussians_indices, sizeof(int) * params_OptiX.width * params_OptiX.width * dlugosc_promienia, cudaMemcpyDeviceToHost);
	if (error_CUDA != cudaSuccess) goto Error;
	for (int i = 0; i < params_OptiX.width; ++i) {
		for (int j = 0; j < params_OptiX.width; ++j) {
			int k = 0;
			int ind;
			do {
				ind = foo[(k * (params_OptiX.width * params_OptiX.width)) + (i * params_OptiX.width) + j];
				++k;
			} while ((ind != -1) && (k < MAX_RAY_LENGTH));
			while (k < MAX_RAY_LENGTH) {
				foo[(k * (params_OptiX.width * params_OptiX.width)) + (i * params_OptiX.width) + j] = -1;
				++k;
			}
		}
	}
	DumpToFile("dump/save", params_OptiX.epoch, "indices", foo, sizeof(int) * params_OptiX.width * params_OptiX.width * dlugosc_promienia);
	free(foo);*/

	// *** *** *** *** ***

	free(buf);

	return true;
Error:
	return false;
}

// *************************************************************************************************