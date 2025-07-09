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

template<int SH_degree>
bool DumpParametersOptiX(SOptiXRenderParams<SH_degree> &params_OptiX) {
	cudaError_t error_CUDA;

	void *buf = malloc(sizeof(REAL4_G) * params_OptiX.numberOfGaussians);
	if (buf == NULL) goto Error;

	// *** *** *** *** ***

	error_CUDA = cudaMemcpy(buf, params_OptiX.GC_part_1_1, sizeof(float4) * params_OptiX.numberOfGaussians, cudaMemcpyDeviceToHost);
	if (error_CUDA != cudaSuccess) goto Error;
	DumpToFile("dump/save", params_OptiX.epoch, "GC1", buf, sizeof(float4) * params_OptiX.numberOfGaussians);

	error_CUDA = cudaMemcpy(buf, params_OptiX.GC_part_2_1, sizeof(float4) * params_OptiX.numberOfGaussians, cudaMemcpyDeviceToHost);
	if (error_CUDA != cudaSuccess) goto Error;
	DumpToFile("dump/save", params_OptiX.epoch, "GC2", buf, sizeof(float4) * params_OptiX.numberOfGaussians);

	error_CUDA = cudaMemcpy(buf, params_OptiX.GC_part_3_1, sizeof(float4) * params_OptiX.numberOfGaussians, cudaMemcpyDeviceToHost);
	if (error_CUDA != cudaSuccess) goto Error;
	DumpToFile("dump/save", params_OptiX.epoch, "GC3", buf, sizeof(float4) * params_OptiX.numberOfGaussians);

	// !!! !!! !!!
	error_CUDA = cudaMemcpy(buf, params_OptiX.GC_part_4_1, sizeof(float2) * params_OptiX.numberOfGaussians, cudaMemcpyDeviceToHost);
	if (error_CUDA != cudaSuccess) goto Error;
	DumpToFile("dump/save", params_OptiX.epoch, "GC4", buf, sizeof(float2) * params_OptiX.numberOfGaussians);

	// *** *** *** *** ***

	// Spherical harmonics
	if constexpr (SH_degree >= 1) {
		error_CUDA = cudaMemcpy(buf, params_OptiX.GC_SH_1, sizeof(float4) * params_OptiX.numberOfGaussians, cudaMemcpyDeviceToHost);
		if (error_CUDA != cudaSuccess) goto Error;
		DumpToFile("dump/save", params_OptiX.epoch, "GC_SH_1", buf, sizeof(float4) * params_OptiX.numberOfGaussians);

		error_CUDA = cudaMemcpy(buf, params_OptiX.GC_SH_2, sizeof(float4) * params_OptiX.numberOfGaussians, cudaMemcpyDeviceToHost);
		if (error_CUDA != cudaSuccess) goto Error;
		DumpToFile("dump/save", params_OptiX.epoch, "GC_SH_2", buf, sizeof(float4) * params_OptiX.numberOfGaussians);

		if constexpr (SH_degree >= 2) {
			error_CUDA = cudaMemcpy(buf, params_OptiX.GC_SH_3, sizeof(float4) * params_OptiX.numberOfGaussians, cudaMemcpyDeviceToHost);
			if (error_CUDA != cudaSuccess) goto Error;
			DumpToFile("dump/save", params_OptiX.epoch, "GC_SH_3", buf, sizeof(float4) * params_OptiX.numberOfGaussians);

			error_CUDA = cudaMemcpy(buf, params_OptiX.GC_SH_4, sizeof(float4) * params_OptiX.numberOfGaussians, cudaMemcpyDeviceToHost);
			if (error_CUDA != cudaSuccess) goto Error;
			DumpToFile("dump/save", params_OptiX.epoch, "GC_SH_4", buf, sizeof(float4) * params_OptiX.numberOfGaussians);

			error_CUDA = cudaMemcpy(buf, params_OptiX.GC_SH_5, sizeof(float4) * params_OptiX.numberOfGaussians, cudaMemcpyDeviceToHost);
			if (error_CUDA != cudaSuccess) goto Error;
			DumpToFile("dump/save", params_OptiX.epoch, "GC_SH_5", buf, sizeof(float4) * params_OptiX.numberOfGaussians);

			error_CUDA = cudaMemcpy(buf, params_OptiX.GC_SH_6, sizeof(float4) * params_OptiX.numberOfGaussians, cudaMemcpyDeviceToHost);
			if (error_CUDA != cudaSuccess) goto Error;
			DumpToFile("dump/save", params_OptiX.epoch, "GC_SH_6", buf, sizeof(float4) * params_OptiX.numberOfGaussians);

			if constexpr (SH_degree >= 3) {
				error_CUDA = cudaMemcpy(buf, params_OptiX.GC_SH_7, sizeof(float4) * params_OptiX.numberOfGaussians, cudaMemcpyDeviceToHost);
				if (error_CUDA != cudaSuccess) goto Error;
				DumpToFile("dump/save", params_OptiX.epoch, "GC_SH_7", buf, sizeof(float4) * params_OptiX.numberOfGaussians);

				error_CUDA = cudaMemcpy(buf, params_OptiX.GC_SH_8, sizeof(float4) * params_OptiX.numberOfGaussians, cudaMemcpyDeviceToHost);
				if (error_CUDA != cudaSuccess) goto Error;
				DumpToFile("dump/save", params_OptiX.epoch, "GC_SH_8", buf, sizeof(float4) * params_OptiX.numberOfGaussians);

				error_CUDA = cudaMemcpy(buf, params_OptiX.GC_SH_9, sizeof(float4) * params_OptiX.numberOfGaussians, cudaMemcpyDeviceToHost);
				if (error_CUDA != cudaSuccess) goto Error;
				DumpToFile("dump/save", params_OptiX.epoch, "GC_SH_9", buf, sizeof(float4) * params_OptiX.numberOfGaussians);

				error_CUDA = cudaMemcpy(buf, params_OptiX.GC_SH_10, sizeof(float4) * params_OptiX.numberOfGaussians, cudaMemcpyDeviceToHost);
				if (error_CUDA != cudaSuccess) goto Error;
				DumpToFile("dump/save", params_OptiX.epoch, "GC_SH_10", buf, sizeof(float4) * params_OptiX.numberOfGaussians);

				error_CUDA = cudaMemcpy(buf, params_OptiX.GC_SH_11, sizeof(float4) * params_OptiX.numberOfGaussians, cudaMemcpyDeviceToHost);
				if (error_CUDA != cudaSuccess) goto Error;
				DumpToFile("dump/save", params_OptiX.epoch, "GC_SH_11", buf, sizeof(float4) * params_OptiX.numberOfGaussians);

				if constexpr (SH_degree >= 4) {
					error_CUDA = cudaMemcpy(buf, params_OptiX.GC_SH_12, sizeof(float4) * params_OptiX.numberOfGaussians, cudaMemcpyDeviceToHost);
					if (error_CUDA != cudaSuccess) goto Error;
					DumpToFile("dump/save", params_OptiX.epoch, "GC_SH_12", buf, sizeof(float4) * params_OptiX.numberOfGaussians);

					error_CUDA = cudaMemcpy(buf, params_OptiX.GC_SH_13, sizeof(float4) * params_OptiX.numberOfGaussians, cudaMemcpyDeviceToHost);
					if (error_CUDA != cudaSuccess) goto Error;
					DumpToFile("dump/save", params_OptiX.epoch, "GC_SH_13", buf, sizeof(float4) * params_OptiX.numberOfGaussians);

					error_CUDA = cudaMemcpy(buf, params_OptiX.GC_SH_14, sizeof(float4) * params_OptiX.numberOfGaussians, cudaMemcpyDeviceToHost);
					if (error_CUDA != cudaSuccess) goto Error;
					DumpToFile("dump/save", params_OptiX.epoch, "GC_SH_14", buf, sizeof(float4) * params_OptiX.numberOfGaussians);

					error_CUDA = cudaMemcpy(buf, params_OptiX.GC_SH_15, sizeof(float4) * params_OptiX.numberOfGaussians, cudaMemcpyDeviceToHost);
					if (error_CUDA != cudaSuccess) goto Error;
					DumpToFile("dump/save", params_OptiX.epoch, "GC_SH_15", buf, sizeof(float4) * params_OptiX.numberOfGaussians);

					error_CUDA = cudaMemcpy(buf, params_OptiX.GC_SH_16, sizeof(float4) * params_OptiX.numberOfGaussians, cudaMemcpyDeviceToHost);
					if (error_CUDA != cudaSuccess) goto Error;
					DumpToFile("dump/save", params_OptiX.epoch, "GC_SH_16", buf, sizeof(float4) * params_OptiX.numberOfGaussians);

					error_CUDA = cudaMemcpy(buf, params_OptiX.GC_SH_17, sizeof(float4) * params_OptiX.numberOfGaussians, cudaMemcpyDeviceToHost);
					if (error_CUDA != cudaSuccess) goto Error;
					DumpToFile("dump/save", params_OptiX.epoch, "GC_SH_17", buf, sizeof(float4) * params_OptiX.numberOfGaussians);

					error_CUDA = cudaMemcpy(buf, params_OptiX.GC_SH_18, sizeof(float4) * params_OptiX.numberOfGaussians, cudaMemcpyDeviceToHost);
					if (error_CUDA != cudaSuccess) goto Error;
					DumpToFile("dump/save", params_OptiX.epoch, "GC_SH_18", buf, sizeof(float4) * params_OptiX.numberOfGaussians);
				} else {
					// !!! !!! !!!
					error_CUDA = cudaMemcpy(buf, params_OptiX.GC_SH_12, sizeof(float) * params_OptiX.numberOfGaussians, cudaMemcpyDeviceToHost);
					if (error_CUDA != cudaSuccess) goto Error;
					DumpToFile("dump/save", params_OptiX.epoch, "GC_SH_12", buf, sizeof(float) * params_OptiX.numberOfGaussians);
				}
			}
		} else {
			// !!! !!! !!!
			error_CUDA = cudaMemcpy(buf, params_OptiX.GC_SH_3, sizeof(float) * params_OptiX.numberOfGaussians, cudaMemcpyDeviceToHost);
			if (error_CUDA != cudaSuccess) goto Error;
			DumpToFile("dump/save", params_OptiX.epoch, "GC_SH_3", buf, sizeof(float) * params_OptiX.numberOfGaussians);
		}
	}

	// *** *** *** *** ***

	error_CUDA = cudaMemcpy(buf, params_OptiX.m11, sizeof(float4) * params_OptiX.numberOfGaussians, cudaMemcpyDeviceToHost);
	if (error_CUDA != cudaSuccess) goto Error;
	DumpToFile("dump/save", params_OptiX.epoch, "m1", buf, sizeof(float4) * params_OptiX.numberOfGaussians);

	error_CUDA = cudaMemcpy(buf, params_OptiX.m21, sizeof(float4) * params_OptiX.numberOfGaussians, cudaMemcpyDeviceToHost);
	if (error_CUDA != cudaSuccess) goto Error;
	DumpToFile("dump/save", params_OptiX.epoch, "m2", buf, sizeof(float4) * params_OptiX.numberOfGaussians);

	error_CUDA = cudaMemcpy(buf, params_OptiX.m31, sizeof(float4) * params_OptiX.numberOfGaussians, cudaMemcpyDeviceToHost);
	if (error_CUDA != cudaSuccess) goto Error;
	DumpToFile("dump/save", params_OptiX.epoch, "m3", buf, sizeof(float4) * params_OptiX.numberOfGaussians);

	// !!! !!! !!!
	error_CUDA = cudaMemcpy(buf, params_OptiX.m41, sizeof(float2) * params_OptiX.numberOfGaussians, cudaMemcpyDeviceToHost);
	if (error_CUDA != cudaSuccess) goto Error;
	DumpToFile("dump/save", params_OptiX.epoch, "m4", buf, sizeof(float2) * params_OptiX.numberOfGaussians);

	// *** *** *** *** ***

	// Spherical harmonics
	if constexpr (SH_degree >= 1) {
		error_CUDA = cudaMemcpy(buf, params_OptiX.m_SH_1, sizeof(float4) * params_OptiX.numberOfGaussians, cudaMemcpyDeviceToHost);
		if (error_CUDA != cudaSuccess) goto Error;
		DumpToFile("dump/save", params_OptiX.epoch, "m_SH_1", buf, sizeof(float4) * params_OptiX.numberOfGaussians);

		error_CUDA = cudaMemcpy(buf, params_OptiX.m_SH_2, sizeof(float4) * params_OptiX.numberOfGaussians, cudaMemcpyDeviceToHost);
		if (error_CUDA != cudaSuccess) goto Error;
		DumpToFile("dump/save", params_OptiX.epoch, "m_SH_2", buf, sizeof(float4) * params_OptiX.numberOfGaussians);

		if constexpr (SH_degree >= 2) {
			error_CUDA = cudaMemcpy(buf, params_OptiX.m_SH_3, sizeof(float4) * params_OptiX.numberOfGaussians, cudaMemcpyDeviceToHost);
			if (error_CUDA != cudaSuccess) goto Error;
			DumpToFile("dump/save", params_OptiX.epoch, "m_SH_3", buf, sizeof(float4) * params_OptiX.numberOfGaussians);

			error_CUDA = cudaMemcpy(buf, params_OptiX.m_SH_4, sizeof(float4) * params_OptiX.numberOfGaussians, cudaMemcpyDeviceToHost);
			if (error_CUDA != cudaSuccess) goto Error;
			DumpToFile("dump/save", params_OptiX.epoch, "m_SH_4", buf, sizeof(float4) * params_OptiX.numberOfGaussians);

			error_CUDA = cudaMemcpy(buf, params_OptiX.m_SH_5, sizeof(float4) * params_OptiX.numberOfGaussians, cudaMemcpyDeviceToHost);
			if (error_CUDA != cudaSuccess) goto Error;
			DumpToFile("dump/save", params_OptiX.epoch, "m_SH_5", buf, sizeof(float4) * params_OptiX.numberOfGaussians);

			error_CUDA = cudaMemcpy(buf, params_OptiX.m_SH_6, sizeof(float4) * params_OptiX.numberOfGaussians, cudaMemcpyDeviceToHost);
			if (error_CUDA != cudaSuccess) goto Error;
			DumpToFile("dump/save", params_OptiX.epoch, "m_SH_6", buf, sizeof(float4) * params_OptiX.numberOfGaussians);

			if constexpr (SH_degree >= 3) {
				error_CUDA = cudaMemcpy(buf, params_OptiX.m_SH_7, sizeof(float4) * params_OptiX.numberOfGaussians, cudaMemcpyDeviceToHost);
				if (error_CUDA != cudaSuccess) goto Error;
				DumpToFile("dump/save", params_OptiX.epoch, "m_SH_7", buf, sizeof(float4) * params_OptiX.numberOfGaussians);

				error_CUDA = cudaMemcpy(buf, params_OptiX.m_SH_8, sizeof(float4) * params_OptiX.numberOfGaussians, cudaMemcpyDeviceToHost);
				if (error_CUDA != cudaSuccess) goto Error;
				DumpToFile("dump/save", params_OptiX.epoch, "m_SH_8", buf, sizeof(float4) * params_OptiX.numberOfGaussians);

				error_CUDA = cudaMemcpy(buf, params_OptiX.m_SH_9, sizeof(float4) * params_OptiX.numberOfGaussians, cudaMemcpyDeviceToHost);
				if (error_CUDA != cudaSuccess) goto Error;
				DumpToFile("dump/save", params_OptiX.epoch, "m_SH_9", buf, sizeof(float4) * params_OptiX.numberOfGaussians);

				error_CUDA = cudaMemcpy(buf, params_OptiX.m_SH_10, sizeof(float4) * params_OptiX.numberOfGaussians, cudaMemcpyDeviceToHost);
				if (error_CUDA != cudaSuccess) goto Error;
				DumpToFile("dump/save", params_OptiX.epoch, "m_SH_10", buf, sizeof(float4) * params_OptiX.numberOfGaussians);

				error_CUDA = cudaMemcpy(buf, params_OptiX.m_SH_11, sizeof(float4) * params_OptiX.numberOfGaussians, cudaMemcpyDeviceToHost);
				if (error_CUDA != cudaSuccess) goto Error;
				DumpToFile("dump/save", params_OptiX.epoch, "m_SH_11", buf, sizeof(float4) * params_OptiX.numberOfGaussians);

				if constexpr (SH_degree >= 4) {
					error_CUDA = cudaMemcpy(buf, params_OptiX.m_SH_12, sizeof(float4) * params_OptiX.numberOfGaussians, cudaMemcpyDeviceToHost);
					if (error_CUDA != cudaSuccess) goto Error;
					DumpToFile("dump/save", params_OptiX.epoch, "m_SH_12", buf, sizeof(float4) * params_OptiX.numberOfGaussians);

					error_CUDA = cudaMemcpy(buf, params_OptiX.m_SH_13, sizeof(float4) * params_OptiX.numberOfGaussians, cudaMemcpyDeviceToHost);
					if (error_CUDA != cudaSuccess) goto Error;
					DumpToFile("dump/save", params_OptiX.epoch, "m_SH_13", buf, sizeof(float4) * params_OptiX.numberOfGaussians);

					error_CUDA = cudaMemcpy(buf, params_OptiX.m_SH_14, sizeof(float4) * params_OptiX.numberOfGaussians, cudaMemcpyDeviceToHost);
					if (error_CUDA != cudaSuccess) goto Error;
					DumpToFile("dump/save", params_OptiX.epoch, "m_SH_14", buf, sizeof(float4) * params_OptiX.numberOfGaussians);

					error_CUDA = cudaMemcpy(buf, params_OptiX.m_SH_15, sizeof(float4) * params_OptiX.numberOfGaussians, cudaMemcpyDeviceToHost);
					if (error_CUDA != cudaSuccess) goto Error;
					DumpToFile("dump/save", params_OptiX.epoch, "m_SH_15", buf, sizeof(float4) * params_OptiX.numberOfGaussians);

					error_CUDA = cudaMemcpy(buf, params_OptiX.m_SH_16, sizeof(float4) * params_OptiX.numberOfGaussians, cudaMemcpyDeviceToHost);
					if (error_CUDA != cudaSuccess) goto Error;
					DumpToFile("dump/save", params_OptiX.epoch, "m_SH_16", buf, sizeof(float4) * params_OptiX.numberOfGaussians);

					error_CUDA = cudaMemcpy(buf, params_OptiX.m_SH_17, sizeof(float4) * params_OptiX.numberOfGaussians, cudaMemcpyDeviceToHost);
					if (error_CUDA != cudaSuccess) goto Error;
					DumpToFile("dump/save", params_OptiX.epoch, "m_SH_17", buf, sizeof(float4) * params_OptiX.numberOfGaussians);

					error_CUDA = cudaMemcpy(buf, params_OptiX.m_SH_18, sizeof(float4) * params_OptiX.numberOfGaussians, cudaMemcpyDeviceToHost);
					if (error_CUDA != cudaSuccess) goto Error;
					DumpToFile("dump/save", params_OptiX.epoch, "m_SH_18", buf, sizeof(float4) * params_OptiX.numberOfGaussians);
				} else {
					// !!! !!! !!!
					error_CUDA = cudaMemcpy(buf, params_OptiX.m_SH_12, sizeof(float) * params_OptiX.numberOfGaussians, cudaMemcpyDeviceToHost);
					if (error_CUDA != cudaSuccess) goto Error;
					DumpToFile("dump/save", params_OptiX.epoch, "m_SH_12", buf, sizeof(float) * params_OptiX.numberOfGaussians);
				}
			}
		} else {
			// !!! !!! !!!
			error_CUDA = cudaMemcpy(buf, params_OptiX.m_SH_3, sizeof(float) * params_OptiX.numberOfGaussians, cudaMemcpyDeviceToHost);
			if (error_CUDA != cudaSuccess) goto Error;
			DumpToFile("dump/save", params_OptiX.epoch, "m_SH_3", buf, sizeof(float) * params_OptiX.numberOfGaussians);
		}
	}

	// *** *** *** *** ***

	error_CUDA = cudaMemcpy(buf, params_OptiX.v11, sizeof(float4) * params_OptiX.numberOfGaussians, cudaMemcpyDeviceToHost);
	if (error_CUDA != cudaSuccess) goto Error;
	DumpToFile("dump/save", params_OptiX.epoch, "v1", buf, sizeof(float4) * params_OptiX.numberOfGaussians);

	error_CUDA = cudaMemcpy(buf, params_OptiX.v21, sizeof(float4) * params_OptiX.numberOfGaussians, cudaMemcpyDeviceToHost);
	if (error_CUDA != cudaSuccess) goto Error;
	DumpToFile("dump/save", params_OptiX.epoch, "v2", buf, sizeof(float4) * params_OptiX.numberOfGaussians);

	error_CUDA = cudaMemcpy(buf, params_OptiX.v31, sizeof(float4) * params_OptiX.numberOfGaussians, cudaMemcpyDeviceToHost);
	if (error_CUDA != cudaSuccess) goto Error;
	DumpToFile("dump/save", params_OptiX.epoch, "v3", buf, sizeof(float4) * params_OptiX.numberOfGaussians);

	// !!! !!! !!!
	error_CUDA = cudaMemcpy(buf, params_OptiX.v41, sizeof(float2) * params_OptiX.numberOfGaussians, cudaMemcpyDeviceToHost);
	if (error_CUDA != cudaSuccess) goto Error;
	DumpToFile("dump/save", params_OptiX.epoch, "v4", buf, sizeof(float2) * params_OptiX.numberOfGaussians);

	// *** *** *** *** ***

	// Spherical harmonics
	if constexpr (SH_degree >= 1) {
		error_CUDA = cudaMemcpy(buf, params_OptiX.v_SH_1, sizeof(float4) * params_OptiX.numberOfGaussians, cudaMemcpyDeviceToHost);
		if (error_CUDA != cudaSuccess) goto Error;
		DumpToFile("dump/save", params_OptiX.epoch, "v_SH_1", buf, sizeof(float4) * params_OptiX.numberOfGaussians);

		error_CUDA = cudaMemcpy(buf, params_OptiX.v_SH_2, sizeof(float4) * params_OptiX.numberOfGaussians, cudaMemcpyDeviceToHost);
		if (error_CUDA != cudaSuccess) goto Error;
		DumpToFile("dump/save", params_OptiX.epoch, "v_SH_2", buf, sizeof(float4) * params_OptiX.numberOfGaussians);

		if constexpr (SH_degree >= 2) {
			error_CUDA = cudaMemcpy(buf, params_OptiX.v_SH_3, sizeof(float4) * params_OptiX.numberOfGaussians, cudaMemcpyDeviceToHost);
			if (error_CUDA != cudaSuccess) goto Error;
			DumpToFile("dump/save", params_OptiX.epoch, "v_SH_3", buf, sizeof(float4) * params_OptiX.numberOfGaussians);

			error_CUDA = cudaMemcpy(buf, params_OptiX.v_SH_4, sizeof(float4) * params_OptiX.numberOfGaussians, cudaMemcpyDeviceToHost);
			if (error_CUDA != cudaSuccess) goto Error;
			DumpToFile("dump/save", params_OptiX.epoch, "v_SH_4", buf, sizeof(float4) * params_OptiX.numberOfGaussians);

			error_CUDA = cudaMemcpy(buf, params_OptiX.v_SH_5, sizeof(float4) * params_OptiX.numberOfGaussians, cudaMemcpyDeviceToHost);
			if (error_CUDA != cudaSuccess) goto Error;
			DumpToFile("dump/save", params_OptiX.epoch, "v_SH_5", buf, sizeof(float4) * params_OptiX.numberOfGaussians);

			error_CUDA = cudaMemcpy(buf, params_OptiX.v_SH_6, sizeof(float4) * params_OptiX.numberOfGaussians, cudaMemcpyDeviceToHost);
			if (error_CUDA != cudaSuccess) goto Error;
			DumpToFile("dump/save", params_OptiX.epoch, "v_SH_6", buf, sizeof(float4) * params_OptiX.numberOfGaussians);

			if constexpr (SH_degree >= 3) {
				error_CUDA = cudaMemcpy(buf, params_OptiX.v_SH_7, sizeof(float4) * params_OptiX.numberOfGaussians, cudaMemcpyDeviceToHost);
				if (error_CUDA != cudaSuccess) goto Error;
				DumpToFile("dump/save", params_OptiX.epoch, "v_SH_7", buf, sizeof(float4) * params_OptiX.numberOfGaussians);

				error_CUDA = cudaMemcpy(buf, params_OptiX.v_SH_8, sizeof(float4) * params_OptiX.numberOfGaussians, cudaMemcpyDeviceToHost);
				if (error_CUDA != cudaSuccess) goto Error;
				DumpToFile("dump/save", params_OptiX.epoch, "v_SH_8", buf, sizeof(float4) * params_OptiX.numberOfGaussians);

				error_CUDA = cudaMemcpy(buf, params_OptiX.v_SH_9, sizeof(float4) * params_OptiX.numberOfGaussians, cudaMemcpyDeviceToHost);
				if (error_CUDA != cudaSuccess) goto Error;
				DumpToFile("dump/save", params_OptiX.epoch, "v_SH_9", buf, sizeof(float4) * params_OptiX.numberOfGaussians);

				error_CUDA = cudaMemcpy(buf, params_OptiX.v_SH_10, sizeof(float4) * params_OptiX.numberOfGaussians, cudaMemcpyDeviceToHost);
				if (error_CUDA != cudaSuccess) goto Error;
				DumpToFile("dump/save", params_OptiX.epoch, "v_SH_10", buf, sizeof(float4) * params_OptiX.numberOfGaussians);

				error_CUDA = cudaMemcpy(buf, params_OptiX.v_SH_11, sizeof(float4) * params_OptiX.numberOfGaussians, cudaMemcpyDeviceToHost);
				if (error_CUDA != cudaSuccess) goto Error;
				DumpToFile("dump/save", params_OptiX.epoch, "v_SH_11", buf, sizeof(float4) * params_OptiX.numberOfGaussians);

				if constexpr (SH_degree >= 4) {
					error_CUDA = cudaMemcpy(buf, params_OptiX.v_SH_12, sizeof(float4) * params_OptiX.numberOfGaussians, cudaMemcpyDeviceToHost);
					if (error_CUDA != cudaSuccess) goto Error;
					DumpToFile("dump/save", params_OptiX.epoch, "v_SH_12", buf, sizeof(float4) * params_OptiX.numberOfGaussians);

					error_CUDA = cudaMemcpy(buf, params_OptiX.v_SH_13, sizeof(float4) * params_OptiX.numberOfGaussians, cudaMemcpyDeviceToHost);
					if (error_CUDA != cudaSuccess) goto Error;
					DumpToFile("dump/save", params_OptiX.epoch, "v_SH_13", buf, sizeof(float4) * params_OptiX.numberOfGaussians);

					error_CUDA = cudaMemcpy(buf, params_OptiX.v_SH_14, sizeof(float4) * params_OptiX.numberOfGaussians, cudaMemcpyDeviceToHost);
					if (error_CUDA != cudaSuccess) goto Error;
					DumpToFile("dump/save", params_OptiX.epoch, "v_SH_14", buf, sizeof(float4) * params_OptiX.numberOfGaussians);

					error_CUDA = cudaMemcpy(buf, params_OptiX.v_SH_15, sizeof(float4) * params_OptiX.numberOfGaussians, cudaMemcpyDeviceToHost);
					if (error_CUDA != cudaSuccess) goto Error;
					DumpToFile("dump/save", params_OptiX.epoch, "v_SH_15", buf, sizeof(float4) * params_OptiX.numberOfGaussians);

					error_CUDA = cudaMemcpy(buf, params_OptiX.v_SH_16, sizeof(float4) * params_OptiX.numberOfGaussians, cudaMemcpyDeviceToHost);
					if (error_CUDA != cudaSuccess) goto Error;
					DumpToFile("dump/save", params_OptiX.epoch, "v_SH_16", buf, sizeof(float4) * params_OptiX.numberOfGaussians);

					error_CUDA = cudaMemcpy(buf, params_OptiX.v_SH_17, sizeof(float4) * params_OptiX.numberOfGaussians, cudaMemcpyDeviceToHost);
					if (error_CUDA != cudaSuccess) goto Error;
					DumpToFile("dump/save", params_OptiX.epoch, "v_SH_17", buf, sizeof(float4) * params_OptiX.numberOfGaussians);

					error_CUDA = cudaMemcpy(buf, params_OptiX.v_SH_18, sizeof(float4) * params_OptiX.numberOfGaussians, cudaMemcpyDeviceToHost);
					if (error_CUDA != cudaSuccess) goto Error;
					DumpToFile("dump/save", params_OptiX.epoch, "v_SH_18", buf, sizeof(float4) * params_OptiX.numberOfGaussians);
				} else {
					// !!! !!! !!!
					error_CUDA = cudaMemcpy(buf, params_OptiX.v_SH_12, sizeof(float) * params_OptiX.numberOfGaussians, cudaMemcpyDeviceToHost);
					if (error_CUDA != cudaSuccess) goto Error;
					DumpToFile("dump/save", params_OptiX.epoch, "v_SH_12", buf, sizeof(float) * params_OptiX.numberOfGaussians);
				}
			}
		} else {
			// !!! !!! !!!
			error_CUDA = cudaMemcpy(buf, params_OptiX.v_SH_3, sizeof(float) * params_OptiX.numberOfGaussians, cudaMemcpyDeviceToHost);
			if (error_CUDA != cudaSuccess) goto Error;
			DumpToFile("dump/save", params_OptiX.epoch, "v_SH_3", buf, sizeof(float) * params_OptiX.numberOfGaussians);
		}
	}

	// *** *** *** *** ***

	// DEBUG GRADIENT
	#ifdef DEBUG_GRADIENT
	error_CUDA = cudaMemcpy(buf, params_OptiX.dL_dparams_1, sizeof(REAL4_G) * params_OptiX.numberOfGaussians, cudaMemcpyDeviceToHost);
	if (error_CUDA != cudaSuccess) goto Error;
	DumpToFile("dump/save", params_OptiX.epoch, "dparams1", buf, sizeof(REAL4_G) * params_OptiX.numberOfGaussians);

	error_CUDA = cudaMemcpy(buf, params_OptiX.dL_dparams_2, sizeof(REAL4_G) * params_OptiX.numberOfGaussians, cudaMemcpyDeviceToHost);
	if (error_CUDA != cudaSuccess) goto Error;
	DumpToFile("dump/save", params_OptiX.epoch, "dparams2", buf, sizeof(REAL4_G) * params_OptiX.numberOfGaussians);

	error_CUDA = cudaMemcpy(buf, params_OptiX.dL_dparams_3, sizeof(REAL4_G) * params_OptiX.numberOfGaussians, cudaMemcpyDeviceToHost);
	if (error_CUDA != cudaSuccess) goto Error;
	DumpToFile("dump/save", params_OptiX.epoch, "dparams3", buf, sizeof(REAL4_G) * params_OptiX.numberOfGaussians);

	// !!! !!! !!!
	error_CUDA = cudaMemcpy(buf, params_OptiX.dL_dparams_4, sizeof(REAL2_G) * params_OptiX.numberOfGaussians, cudaMemcpyDeviceToHost);
	if (error_CUDA != cudaSuccess) goto Error;
	DumpToFile("dump/save", params_OptiX.epoch, "dparams4", buf, sizeof(REAL2_G) * params_OptiX.numberOfGaussians);

	// *** *** *** *** ***

	// Spherical harmonics
	if constexpr (SH_degree >= 1) {
		error_CUDA = cudaMemcpy(buf, params_OptiX.dL_dparams_SH_1, sizeof(REAL4_G) * params_OptiX.numberOfGaussians, cudaMemcpyDeviceToHost);
		if (error_CUDA != cudaSuccess) goto Error;
		DumpToFile("dump/save", params_OptiX.epoch, "dparams_SH_1", buf, sizeof(REAL4_G) * params_OptiX.numberOfGaussians);

		error_CUDA = cudaMemcpy(buf, params_OptiX.dL_dparams_SH_2, sizeof(REAL4_G) * params_OptiX.numberOfGaussians, cudaMemcpyDeviceToHost);
		if (error_CUDA != cudaSuccess) goto Error;
		DumpToFile("dump/save", params_OptiX.epoch, "dparams_SH_2", buf, sizeof(REAL4_G) * params_OptiX.numberOfGaussians);

		if constexpr (SH_degree >= 2) {
			error_CUDA = cudaMemcpy(buf, params_OptiX.dL_dparams_SH_3, sizeof(REAL4_G) * params_OptiX.numberOfGaussians, cudaMemcpyDeviceToHost);
			if (error_CUDA != cudaSuccess) goto Error;
			DumpToFile("dump/save", params_OptiX.epoch, "dparams_SH_3", buf, sizeof(REAL4_G) * params_OptiX.numberOfGaussians);

			error_CUDA = cudaMemcpy(buf, params_OptiX.dL_dparams_SH_4, sizeof(REAL4_G) * params_OptiX.numberOfGaussians, cudaMemcpyDeviceToHost);
			if (error_CUDA != cudaSuccess) goto Error;
			DumpToFile("dump/save", params_OptiX.epoch, "dparams_SH_4", buf, sizeof(REAL4_G) * params_OptiX.numberOfGaussians);

			error_CUDA = cudaMemcpy(buf, params_OptiX.dL_dparams_SH_5, sizeof(REAL4_G) * params_OptiX.numberOfGaussians, cudaMemcpyDeviceToHost);
			if (error_CUDA != cudaSuccess) goto Error;
			DumpToFile("dump/save", params_OptiX.epoch, "dparams_SH_5", buf, sizeof(REAL4_G) * params_OptiX.numberOfGaussians);

			error_CUDA = cudaMemcpy(buf, params_OptiX.dL_dparams_SH_6, sizeof(REAL4_G) * params_OptiX.numberOfGaussians, cudaMemcpyDeviceToHost);
			if (error_CUDA != cudaSuccess) goto Error;
			DumpToFile("dump/save", params_OptiX.epoch, "dparams_SH_6", buf, sizeof(REAL4_G) * params_OptiX.numberOfGaussians);

			if constexpr (SH_degree >= 3) {
				error_CUDA = cudaMemcpy(buf, params_OptiX.dL_dparams_SH_7, sizeof(REAL4_G) * params_OptiX.numberOfGaussians, cudaMemcpyDeviceToHost);
				if (error_CUDA != cudaSuccess) goto Error;
				DumpToFile("dump/save", params_OptiX.epoch, "dparams_SH_7", buf, sizeof(REAL4_G) * params_OptiX.numberOfGaussians);

				error_CUDA = cudaMemcpy(buf, params_OptiX.dL_dparams_SH_8, sizeof(REAL4_G) * params_OptiX.numberOfGaussians, cudaMemcpyDeviceToHost);
				if (error_CUDA != cudaSuccess) goto Error;
				DumpToFile("dump/save", params_OptiX.epoch, "dparams_SH_8", buf, sizeof(REAL4_G) * params_OptiX.numberOfGaussians);

				error_CUDA = cudaMemcpy(buf, params_OptiX.dL_dparams_SH_9, sizeof(REAL4_G) * params_OptiX.numberOfGaussians, cudaMemcpyDeviceToHost);
				if (error_CUDA != cudaSuccess) goto Error;
				DumpToFile("dump/save", params_OptiX.epoch, "dparams_SH_9", buf, sizeof(REAL4_G) * params_OptiX.numberOfGaussians);

				error_CUDA = cudaMemcpy(buf, params_OptiX.dL_dparams_SH_10, sizeof(REAL4_G) * params_OptiX.numberOfGaussians, cudaMemcpyDeviceToHost);
				if (error_CUDA != cudaSuccess) goto Error;
				DumpToFile("dump/save", params_OptiX.epoch, "dparams_SH_10", buf, sizeof(REAL4_G) * params_OptiX.numberOfGaussians);

				error_CUDA = cudaMemcpy(buf, params_OptiX.dL_dparams_SH_11, sizeof(REAL4_G) * params_OptiX.numberOfGaussians, cudaMemcpyDeviceToHost);
				if (error_CUDA != cudaSuccess) goto Error;
				DumpToFile("dump/save", params_OptiX.epoch, "dparams_SH_11", buf, sizeof(REAL4_G) * params_OptiX.numberOfGaussians);

				if constexpr (SH_degree >= 4) {
					error_CUDA = cudaMemcpy(buf, params_OptiX.dL_dparams_SH_12, sizeof(REAL4_G) * params_OptiX.numberOfGaussians, cudaMemcpyDeviceToHost);
					if (error_CUDA != cudaSuccess) goto Error;
					DumpToFile("dump/save", params_OptiX.epoch, "dparams_SH_12", buf, sizeof(REAL4_G) * params_OptiX.numberOfGaussians);

					error_CUDA = cudaMemcpy(buf, params_OptiX.dL_dparams_SH_13, sizeof(REAL4_G) * params_OptiX.numberOfGaussians, cudaMemcpyDeviceToHost);
					if (error_CUDA != cudaSuccess) goto Error;
					DumpToFile("dump/save", params_OptiX.epoch, "dparams_SH_13", buf, sizeof(REAL4_G) * params_OptiX.numberOfGaussians);

					error_CUDA = cudaMemcpy(buf, params_OptiX.dL_dparams_SH_14, sizeof(REAL4_G) * params_OptiX.numberOfGaussians, cudaMemcpyDeviceToHost);
					if (error_CUDA != cudaSuccess) goto Error;
					DumpToFile("dump/save", params_OptiX.epoch, "dparams_SH_14", buf, sizeof(REAL4_G) * params_OptiX.numberOfGaussians);

					error_CUDA = cudaMemcpy(buf, params_OptiX.dL_dparams_SH_15, sizeof(REAL4_G) * params_OptiX.numberOfGaussians, cudaMemcpyDeviceToHost);
					if (error_CUDA != cudaSuccess) goto Error;
					DumpToFile("dump/save", params_OptiX.epoch, "dparams_SH_15", buf, sizeof(REAL4_G) * params_OptiX.numberOfGaussians);

					error_CUDA = cudaMemcpy(buf, params_OptiX.dL_dparams_SH_16, sizeof(REAL4_G) * params_OptiX.numberOfGaussians, cudaMemcpyDeviceToHost);
					if (error_CUDA != cudaSuccess) goto Error;
					DumpToFile("dump/save", params_OptiX.epoch, "dparams_SH_16", buf, sizeof(REAL4_G) * params_OptiX.numberOfGaussians);

					error_CUDA = cudaMemcpy(buf, params_OptiX.dL_dparams_SH_17, sizeof(REAL4_G) * params_OptiX.numberOfGaussians, cudaMemcpyDeviceToHost);
					if (error_CUDA != cudaSuccess) goto Error;
					DumpToFile("dump/save", params_OptiX.epoch, "dparams_SH_17", buf, sizeof(REAL4_G) * params_OptiX.numberOfGaussians);

					error_CUDA = cudaMemcpy(buf, params_OptiX.dL_dparams_SH_18, sizeof(REAL4_G) * params_OptiX.numberOfGaussians, cudaMemcpyDeviceToHost);
					if (error_CUDA != cudaSuccess) goto Error;
					DumpToFile("dump/save", params_OptiX.epoch, "dparams_SH_18", buf, sizeof(REAL4_G) * params_OptiX.numberOfGaussians);
				} else {
					// !!! !!! !!!
					error_CUDA = cudaMemcpy(buf, params_OptiX.dL_dparams_SH_12, sizeof(REAL_G) * params_OptiX.numberOfGaussians, cudaMemcpyDeviceToHost);
					if (error_CUDA != cudaSuccess) goto Error;
					DumpToFile("dump/save", params_OptiX.epoch, "dparams_SH_12", buf, sizeof(REAL_G) * params_OptiX.numberOfGaussians);
				}
			}
		} else {
			// !!! !!! !!!
			error_CUDA = cudaMemcpy(buf, params_OptiX.dL_dparams_SH_3, sizeof(REAL_G) * params_OptiX.numberOfGaussians, cudaMemcpyDeviceToHost);
			if (error_CUDA != cudaSuccess) goto Error;
			DumpToFile("dump/save", params_OptiX.epoch, "dparams_SH_3", buf, sizeof(REAL_G) * params_OptiX.numberOfGaussians);
		}
	}

	// *** *** *** *** ***

	int *foo = (int *)malloc(sizeof(int) * params_OptiX.width * params_OptiX.height * max_Gaussians_per_ray_host);
	error_CUDA = cudaMemcpy(foo, params_OptiX.Gaussians_indices, sizeof(int) * params_OptiX.width * params_OptiX.height * max_Gaussians_per_ray_host, cudaMemcpyDeviceToHost);
	if (error_CUDA != cudaSuccess) goto Error;
	for (int i = 0; i < params_OptiX.height; ++i) {
		for (int j = 0; j < params_OptiX.width; ++j) {
			int k = 0;
			int ind;
			do {
				ind = foo[(k * (params_OptiX.width * params_OptiX.height)) + (i * params_OptiX.width) + j];
				++k;
			} while ((ind != -1) && (k < max_Gaussians_per_ray_host));
			while (k < max_Gaussians_per_ray_host) {
				foo[(k * (params_OptiX.width * params_OptiX.height)) + (i * params_OptiX.width) + j] = -1;
				++k;
			}
		}
	}
	DumpToFile("dump/save", params_OptiX.epoch, "indices", foo, sizeof(int) * params_OptiX.width * params_OptiX.height * max_Gaussians_per_ray_host);
	free(foo);
	#endif

	// *** *** *** *** ***

	free(buf);

	return true;
Error:
	return false;
}

// *************************************************************************************************

bool DumpParametersOptiXSH0(SOptiXRenderParams<0> &params_OptiX) {
	return DumpParametersOptiX<0>(params_OptiX);
}

// *************************************************************************************************

bool DumpParametersOptiXSH1(SOptiXRenderParams<1> &params_OptiX) {
	return DumpParametersOptiX<1>(params_OptiX);
}

// *************************************************************************************************

bool DumpParametersOptiXSH2(SOptiXRenderParams<2> &params_OptiX) {
	return DumpParametersOptiX<2>(params_OptiX);
}

// *************************************************************************************************

bool DumpParametersOptiXSH3(SOptiXRenderParams<3> &params_OptiX) {
	return DumpParametersOptiX<3>(params_OptiX);
}

// *************************************************************************************************

bool DumpParametersOptiXSH4(SOptiXRenderParams<4> &params_OptiX) {
	return DumpParametersOptiX<4>(params_OptiX);
}