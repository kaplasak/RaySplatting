#include "Header.cuh"

// *** *** *** *** ***

extern bool InitializeCUDARenderer(SRenderParams& params, SCUDARenderParams& dev_params) {
	cudaError_t cudaStatus;

	dev_params.bitmap = NULL;
	cudaStatus = cudaMalloc((void**)&dev_params.bitmap, 4 * params.w * params.h);
	if (cudaStatus != cudaSuccess)
		goto Error;

	const int kernel_size = 11;
	int arraySizeReal = (params.w + (kernel_size - 1)) * (params.h + (kernel_size - 1)); // !!! !!! !!!

	dev_params.dev_bitmap_out_R = NULL;
	cudaStatus = cudaMalloc((void**)&dev_params.dev_bitmap_out_R, sizeof(REAL) * arraySizeReal);
	if (cudaStatus != cudaSuccess)
		goto Error;

	dev_params.dev_bitmap_out_G = NULL;
	cudaStatus = cudaMalloc((void**)&dev_params.dev_bitmap_out_G, sizeof(REAL) * arraySizeReal);
	if (cudaStatus != cudaSuccess)
		goto Error;

	dev_params.dev_bitmap_out_B = NULL;
	cudaStatus = cudaMalloc((void**)&dev_params.dev_bitmap_out_B, sizeof(REAL) * arraySizeReal);
	if (cudaStatus != cudaSuccess)
		goto Error;

	// *** *** *** *** ***

	// camera
	dev_params.Ox = params.Ox; dev_params.Oy = params.Oy; dev_params.Oz = params.Oz;
	dev_params.Rx = params.Rx; dev_params.Ry = params.Ry; dev_params.Rz = params.Rz;
	dev_params.Dx = params.Dx; dev_params.Dy = params.Dy; dev_params.Dz = params.Dz;
	dev_params.Fx = params.Fx; dev_params.Fy = params.Fy; dev_params.Fz = params.Fz;
	
	// *** *** *** *** ***

	// bitmap
	dev_params.w = params.w;
	dev_params.h = params.h;

	return true;
Error:
	if (dev_params.bitmap != NULL)
		cudaFree(dev_params.bitmap);

	if (dev_params.bitmap != NULL)
		cudaFree(dev_params.dev_bitmap_out_R);

	if (dev_params.bitmap != NULL)
		cudaFree(dev_params.dev_bitmap_out_G);

	if (dev_params.bitmap != NULL)
		cudaFree(dev_params.dev_bitmap_out_B);

	return false;
}

// *** *** *** *** ***

extern bool InitializeCUDARendererAS(SRenderParams& params, SCUDARenderParams& dev_params) {
	cudaError_t cudaStatus;

	dev_params.tree_part_1 = NULL;
	cudaStatus = cudaMalloc((void**)&dev_params.tree_part_1, sizeof(float4) * params.H);
	if (cudaStatus != cudaSuccess)
		goto Error;

	dev_params.tree_part_2 = NULL;
	cudaStatus = cudaMalloc((void**)&dev_params.tree_part_2, sizeof(float4) * params.H);
	if (cudaStatus != cudaSuccess)
		goto Error;

	dev_params.GC_part_1 = NULL;
	cudaStatus = cudaMalloc((void**)&dev_params.GC_part_1, sizeof(float4) * params.numberOfGaussians);
	if (cudaStatus != cudaSuccess)
		goto Error;

	dev_params.GC_part_2 = NULL;
	cudaStatus = cudaMalloc((void**)&dev_params.GC_part_2, sizeof(float4) * params.numberOfGaussians);
	if (cudaStatus != cudaSuccess)
		goto Error;

	dev_params.GC_part_3 = NULL;
	cudaStatus = cudaMalloc((void**)&dev_params.GC_part_3, sizeof(float4) * params.numberOfGaussians);
	if (cudaStatus != cudaSuccess)
		goto Error;

	dev_params.GC_part_4 = NULL;
	cudaStatus = cudaMalloc((void**)&dev_params.GC_part_4, sizeof(float2) * params.numberOfGaussians);
	if (cudaStatus != cudaSuccess)
		goto Error;

	dev_params.d = NULL;
	cudaStatus = cudaMalloc((void**)&dev_params.d, sizeof(int) * params.D);
	if (cudaStatus != cudaSuccess)
		goto Error;

	// *** *** *** *** ***

	// tree_part_1
	float4* tree_part_1 = NULL;
	tree_part_1 = (float4*)malloc(sizeof(float4) * params.H);
	if (tree_part_1 == NULL)
		goto Error;

	for (int i = 0; i < params.H; ++i) {
		SLBVHTreeNode node = params.tree[i];

		*((int*)&tree_part_1[i].x) = (((int)node.info) << 30) | node.lNode;
		*((int*)&tree_part_1[i].y) = node.rNode;
		tree_part_1[i].z = node.lB;
		tree_part_1[i].w = node.rB;
	}

	cudaStatus = cudaMemcpy(dev_params.tree_part_1, tree_part_1, sizeof(float4) * params.H, cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess)
		goto Error;

	free(tree_part_1);

	// *** *** *** *** ***

	// tree_part_2
	float4* tree_part_2 = NULL;
	tree_part_2 = (float4*)malloc(sizeof(float4) * params.H);
	if (tree_part_2 == NULL)
		goto Error;

	for (int i = 0; i < params.H; ++i) {
		SLBVHTreeNode node = params.tree[i];

		tree_part_2[i].x = node.uB;
		tree_part_2[i].y = node.dB;
		tree_part_2[i].z = node.bB;
		tree_part_2[i].w = node.fB;
	}

	cudaStatus = cudaMemcpy(dev_params.tree_part_2, tree_part_2, sizeof(float4) * params.H, cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess)
		goto Error;

	free(tree_part_2);

	// *** *** *** *** ***

	// GC_part_1
	float4* GC_part_1 = NULL;
	GC_part_1 = (float4*)malloc(sizeof(float4) * params.numberOfGaussians);
	if (GC_part_1 == NULL)
		goto Error;

	for (int i = 0; i < params.numberOfGaussians; ++i) {
		SGaussianComponent gc = params.GC[i];

		GC_part_1[i].x = gc.R;
		GC_part_1[i].y = gc.G;
		GC_part_1[i].z = gc.B;
		GC_part_1[i].w = gc.alpha;
	}

	cudaStatus = cudaMemcpy(dev_params.GC_part_1, GC_part_1, sizeof(float4) * params.numberOfGaussians, cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess)
		goto Error;

	free(GC_part_1);

	// *** *** *** *** ***

	// GC_part_2
	float4* GC_part_2 = NULL;
	GC_part_2 = (float4*)malloc(sizeof(float4) * params.numberOfGaussians);
	if (GC_part_2 == NULL)
		goto Error;

	for (int i = 0; i < params.numberOfGaussians; ++i) {
		SGaussianComponent gc = params.GC[i];

		GC_part_2[i].x = gc.mX;
		GC_part_2[i].y = gc.mY;
		GC_part_2[i].z = gc.mZ;
		GC_part_2[i].w = gc.sX;
	}

	cudaStatus = cudaMemcpy(dev_params.GC_part_2, GC_part_2, sizeof(float4) * params.numberOfGaussians, cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess)
		goto Error;

	free(GC_part_2);

	// *** *** *** *** ***

	// GC_part_3
	float4* GC_part_3 = NULL;
	GC_part_3 = (float4*)malloc(sizeof(float4) * params.numberOfGaussians);
	if (GC_part_3 == NULL)
		goto Error;

	for (int i = 0; i < params.numberOfGaussians; ++i) {
		SGaussianComponent gc = params.GC[i];

		GC_part_3[i].x = gc.sY;
		GC_part_3[i].y = gc.sZ;
		GC_part_3[i].z = gc.qr;
		GC_part_3[i].w = gc.qi;
	}

	cudaStatus = cudaMemcpy(dev_params.GC_part_3, GC_part_3, sizeof(float4) * params.numberOfGaussians, cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess)
		goto Error;

	free(GC_part_3);

	// *** *** *** *** ***

	// GC_part_4
	float2* GC_part_4 = NULL;
	GC_part_4 = (float2*)malloc(sizeof(float2) * params.numberOfGaussians);
	if (GC_part_4 == NULL)
		goto Error;

	for (int i = 0; i < params.numberOfGaussians; ++i) {
		SGaussianComponent gc = params.GC[i];

		GC_part_4[i].x = gc.qj;
		GC_part_4[i].y = gc.qk;
	}

	cudaStatus = cudaMemcpy(dev_params.GC_part_4, GC_part_4, sizeof(float2) * params.numberOfGaussians, cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess)
		goto Error;

	free(GC_part_4);

	// *** *** *** *** ***

	// d
	cudaStatus = cudaMemcpy(dev_params.d, params.d, sizeof(int) * params.D, cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess)
		goto Error;

	dev_params.D = params.D;
	dev_params.H = params.H;

	return true;
Error:
	if (dev_params.tree_part_1 != NULL)
		cudaFree(dev_params.tree_part_1);

	if (dev_params.tree_part_2 != NULL)
		cudaFree(dev_params.tree_part_2);

	if (dev_params.GC_part_1 != NULL)
		cudaFree(dev_params.GC_part_1);

	if (dev_params.GC_part_2 != NULL)
		cudaFree(dev_params.GC_part_2);

	if (dev_params.GC_part_3 != NULL)
		cudaFree(dev_params.GC_part_3);

	if (dev_params.GC_part_4 != NULL)
		cudaFree(dev_params.GC_part_4);

	if (dev_params.d != NULL)
		cudaFree(dev_params.d);

	if (tree_part_1 != NULL)
		free(tree_part_1);

	if (tree_part_2 != NULL)
		free(tree_part_2);

	if (GC_part_1 != NULL)
		free(GC_part_1);

	if (GC_part_2 != NULL)
		free(GC_part_2);

	if (GC_part_3 != NULL)
		free(GC_part_3);

	if (GC_part_4 != NULL)
		free(GC_part_4);

	return false;
}

// *** *** *** *** ***

__global__ void MultiplyPointwiseReal(REAL *arr1_in, REAL *arr2_in, REAL *arr_out, int size) {
	int tid = (blockIdx.x * blockDim.x) + threadIdx.x;
	if (tid < size) {
		REAL a = arr1_in[tid];
		REAL b = arr2_in[tid];
		REAL c = a * b;
		arr_out[tid] = c;
	}
}

// *** *** *** *** ***

__global__ void MultiplyPointwiseComplex(COMPLEX *arr1_in, COMPLEX *arr2_in, COMPLEX *arr_out, int size) {
	int tid = (blockIdx.x * blockDim.x) + threadIdx.x;
	if (tid < size) {
		COMPLEX c;
		COMPLEX a = arr1_in[tid];
		COMPLEX b = arr2_in[tid];
		c.x = (a.x * b.x) - (a.y * b.y);
		c.y = (a.x * b.y) + (a.y * b.x);
		arr_out[tid] = c;
	}
}

//   -*-   -*-   -*-

extern bool InitializeCUDAGradient(SRenderParams& params, SCUDARenderParams& dev_params) {
	cudaError_t cudaStatus;

	dev_params.bitmap_ref = NULL;
	cudaStatus = cudaMalloc((void **)&dev_params.bitmap_ref, 4 * params.w * params.h * params.NUMBER_OF_POSES);
	if (cudaStatus != cudaSuccess)
		goto Error;

	dev_params.dL_dparams_1 = NULL;
	cudaStatus = cudaMalloc((void **)&dev_params.dL_dparams_1, 4 * sizeof(REAL) * params.numberOfGaussians * 10); // (x2) !!! !!! !!!
	if (cudaStatus != cudaSuccess)
		goto Error;

	dev_params.dL_dparams_2 = NULL;
	cudaStatus = cudaMalloc((void **)&dev_params.dL_dparams_2, 4 * sizeof(REAL) * params.numberOfGaussians * 10); // (x2) !!! !!! !!!
	if (cudaStatus != cudaSuccess)
		goto Error;

	dev_params.dL_dparams_3 = NULL;
	cudaStatus = cudaMalloc((void **)&dev_params.dL_dparams_3, 4 * sizeof(REAL) * params.numberOfGaussians * 10); // (x2) !!! !!! !!!
	if (cudaStatus != cudaSuccess)
		goto Error;

	dev_params.dL_dparams_4 = NULL;
	cudaStatus = cudaMalloc((void **)&dev_params.dL_dparams_4, 2 * sizeof(REAL) * params.numberOfGaussians * 10); // (x2) !!! !!! !!!
	if (cudaStatus != cudaSuccess)
		goto Error;

	dev_params.output_params = NULL;
	cudaStatus = cudaMalloc((void **)&dev_params.output_params, 8 * 1);
	if (cudaStatus != cudaSuccess)
		goto Error;

	// !!! !!! !!!
	dev_params.dump = NULL;
	cudaStatus = cudaMalloc((void **)&dev_params.dump, sizeof(int) * MAX_RAY_LENGTH * params.w * params.h);
	if (cudaStatus != cudaSuccess)
		goto Error;
	// !!! !!! !!!

	cudaStatus = cudaMemcpy(dev_params.bitmap_ref, params.bitmap_ref, 4 * params.w * params.h * params.NUMBER_OF_POSES, cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess)
		goto Error;

	// !!! !!! !!!
	params.dL_dparams_1 = malloc(4 * sizeof(REAL) * params.numberOfGaussians * 10); // (x2) !!! !!! !!!
	params.dL_dparams_2 = malloc(4 * sizeof(REAL) * params.numberOfGaussians * 10); // (x2) !!! !!! !!!
	params.dL_dparams_3 = malloc(4 * sizeof(REAL) * params.numberOfGaussians * 10); // (x2) !!! !!! !!!
	params.dL_dparams_4 = malloc(2 * sizeof(REAL) * params.numberOfGaussians * 10); // (x2) !!! !!! !!!

	params.m1 = malloc(4 * sizeof(REAL) * params.numberOfGaussians * 2); // (x2) !!! !!! !!!
	params.m2 = malloc(4 * sizeof(REAL) * params.numberOfGaussians * 2); // (x2) !!! !!! !!!
	params.m3 = malloc(4 * sizeof(REAL) * params.numberOfGaussians * 2); // (x2) !!! !!! !!!
	params.m4 = malloc(2 * sizeof(REAL) * params.numberOfGaussians * 2); // (x2) !!! !!! !!!

	params.v1 = malloc(4 * sizeof(REAL) * params.numberOfGaussians * 2); // (x2) !!! !!! !!!
	params.v2 = malloc(4 * sizeof(REAL) * params.numberOfGaussians * 2); // (x2) !!! !!! !!!
	params.v3 = malloc(4 * sizeof(REAL) * params.numberOfGaussians * 2); // (x2) !!! !!! !!!
	params.v4 = malloc(2 * sizeof(REAL) * params.numberOfGaussians * 2); // (x2) !!! !!! !!!

	memset(params.m1, 0, 4 * sizeof(REAL) * params.numberOfGaussians * 2); // (x2) !!! !!! !!!
	memset(params.m2, 0, 4 * sizeof(REAL) * params.numberOfGaussians * 2); // (x2) !!! !!! !!!
	memset(params.m3, 0, 4 * sizeof(REAL) * params.numberOfGaussians * 2); // (x2) !!! !!! !!!
	memset(params.m4, 0, 2 * sizeof(REAL) * params.numberOfGaussians * 2); // (x2) !!! !!! !!!

	memset(params.v1, 0, 4 * sizeof(REAL) * params.numberOfGaussians * 2); // (x2) !!! !!! !!!
	memset(params.v2, 0, 4 * sizeof(REAL) * params.numberOfGaussians * 2); // (x2) !!! !!! !!!
	memset(params.v3, 0, 4 * sizeof(REAL) * params.numberOfGaussians * 2); // (x2) !!! !!! !!!
	memset(params.v4, 0, 2 * sizeof(REAL) * params.numberOfGaussians * 2); // (x2) !!! !!! !!!

	params.dump = (int *)malloc(sizeof(int) * MAX_RAY_LENGTH * params.w * params.h);
	// !!! !!! !!!

	//*************************************************************************************************
	// SSIM                                                                                           *
	//*************************************************************************************************

	const int kernel_size = 11;
	const int kernel_radius = kernel_size >> 1;
	const REAL sigma = ((REAL)1.5);

	int arraySizeReal = (params.w + (kernel_size - 1)) * (params.h + (kernel_size - 1)); // !!! !!! !!!
	int arraySizeComplex = (((params.w + (kernel_size - 1)) >> 1) + 1) * (params.h + (kernel_size - 1)); // !!! !!! !!!

	REAL *buf = NULL;
	buf = (REAL *)malloc(sizeof(REAL) * arraySizeReal);
	if (buf == NULL) goto Error;

	cufftResult error_CUFFT;

	// *** *** *** *** ***

	error_CUFFT = cufftPlan2d(&params.planr2c, params.h + (kernel_size - 1), params.w + (kernel_size - 1), REAL_TO_COMPLEX);
	if (error_CUFFT != CUFFT_SUCCESS) goto Error;

	error_CUFFT = cufftPlan2d(&params.planc2r, params.h + (kernel_size - 1), params.w + (kernel_size - 1), COMPLEX_TO_REAL);
	if (error_CUFFT != CUFFT_SUCCESS) goto Error;

	// *** *** *** *** ***

	cudaStatus = cudaMalloc((void **)&params.dev_F_1, sizeof(COMPLEX) * arraySizeComplex);
	if (cudaStatus != cudaSuccess)
		goto Error;

	cudaStatus = cudaMalloc((void **)&params.dev_F_2, sizeof(COMPLEX) * arraySizeComplex);
	if (cudaStatus != cudaSuccess)
		goto Error;

	// *** *** *** *** ***

	cudaStatus = cudaMalloc((void **)&params.dev_bitmap_ref_R, sizeof(REAL) * arraySizeReal * params.NUMBER_OF_POSES);
	if (cudaStatus != cudaSuccess)
		goto Error;

	cudaStatus = cudaMalloc((void **)&params.dev_bitmap_ref_G, sizeof(REAL) * arraySizeReal * params.NUMBER_OF_POSES);
	if (cudaStatus != cudaSuccess)
		goto Error;

	cudaStatus = cudaMalloc((void **)&params.dev_bitmap_ref_B, sizeof(REAL) * arraySizeReal * params.NUMBER_OF_POSES);
	if (cudaStatus != cudaSuccess)
		goto Error;

	cudaStatus = cudaMalloc((void **)&params.dev_mu_bitmap_ref_R, sizeof(REAL) * arraySizeReal * params.NUMBER_OF_POSES);
	if (cudaStatus != cudaSuccess)
		goto Error;

	cudaStatus = cudaMalloc((void **)&params.dev_mu_bitmap_ref_G, sizeof(REAL) * arraySizeReal * params.NUMBER_OF_POSES);
	if (cudaStatus != cudaSuccess)
		goto Error;

	cudaStatus = cudaMalloc((void **)&params.dev_mu_bitmap_ref_B, sizeof(REAL) * arraySizeReal * params.NUMBER_OF_POSES);
	if (cudaStatus != cudaSuccess)
		goto Error;

	cudaStatus = cudaMalloc((void **)&params.dev_mu_bitmap_out_bitmap_ref_R, sizeof(REAL) * arraySizeReal); // !!! !!! !!!
	if (cudaStatus != cudaSuccess)
		goto Error;

	cudaStatus = cudaMalloc((void **)&params.dev_mu_bitmap_out_bitmap_ref_G, sizeof(REAL) * arraySizeReal); // !!! !!! !!!
	if (cudaStatus != cudaSuccess)
		goto Error;

	cudaStatus = cudaMalloc((void **)&params.dev_mu_bitmap_out_bitmap_ref_B, sizeof(REAL) * arraySizeReal); // !!! !!! !!!
	if (cudaStatus != cudaSuccess)
		goto Error;

	cudaStatus = cudaMalloc((void **)&params.dev_mu_bitmap_ref_R_square, sizeof(REAL) * arraySizeReal * params.NUMBER_OF_POSES);
	if (cudaStatus != cudaSuccess)
		goto Error;

	cudaStatus = cudaMalloc((void **)&params.dev_mu_bitmap_ref_G_square, sizeof(REAL) * arraySizeReal * params.NUMBER_OF_POSES);
	if (cudaStatus != cudaSuccess)
		goto Error;

	cudaStatus = cudaMalloc((void **)&params.dev_mu_bitmap_ref_B_square, sizeof(REAL) * arraySizeReal * params.NUMBER_OF_POSES);
	if (cudaStatus != cudaSuccess)
		goto Error;

	// !!! !!! !!! WNIMANIE !!! !!! !!!
	cudaStatus = cudaMalloc((void **)&params.dev_mu_bitmap_out_R, sizeof(REAL) * arraySizeReal); // !!! !!! !!!
	if (cudaStatus != cudaSuccess)
		goto Error;
	dev_params.dev_mu_bitmap_out_R = params.dev_mu_bitmap_out_R; // !!! !!! !!!

	// !!! !!! !!! WNIMANIE !!! !!! !!!
	cudaStatus = cudaMalloc((void **)&params.dev_mu_bitmap_out_G, sizeof(REAL) * arraySizeReal); // !!! !!! !!!
	if (cudaStatus != cudaSuccess)
		goto Error;
	dev_params.dev_mu_bitmap_out_G = params.dev_mu_bitmap_out_G; // !!! !!! !!!

	// !!! !!! !!! WNIMANIE !!! !!! !!!
	cudaStatus = cudaMalloc((void **)&params.dev_mu_bitmap_out_B, sizeof(REAL) * arraySizeReal); // !!! !!! !!!
	if (cudaStatus != cudaSuccess)
		goto Error;
	dev_params.dev_mu_bitmap_out_B = params.dev_mu_bitmap_out_B; // !!! !!! !!!

	cudaStatus = cudaMalloc((void **)&params.dev_mu_bitmap_out_R_square, sizeof(REAL) * arraySizeReal); // !!! !!! !!!
	if (cudaStatus != cudaSuccess)
		goto Error;

	cudaStatus = cudaMalloc((void **)&params.dev_mu_bitmap_out_G_square, sizeof(REAL) * arraySizeReal); // !!! !!! !!!
	if (cudaStatus != cudaSuccess)
		goto Error;

	cudaStatus = cudaMalloc((void **)&params.dev_mu_bitmap_out_B_square, sizeof(REAL) * arraySizeReal); // !!! !!! !!!
	if (cudaStatus != cudaSuccess)
		goto Error;

	// *** *** *** *** ***

	// Create Gaussian kernel
	memset(buf, 0, sizeof(REAL) * arraySizeReal);
	REAL sum = ((REAL)0.0);
	for (int i = 0; i < kernel_size; ++i) {
		for (int j = 0; j < kernel_size; ++j) {
			float tmp = exp((-(((j - kernel_radius) * (j - kernel_radius)) + ((i - kernel_radius) * (i - kernel_radius)))) / (((REAL)2.0) * sigma * sigma));
			sum += tmp;
			buf[(i * (params.w + (kernel_size - 1))) + j] = tmp;
		}
	}
	for (int i = 0; i < kernel_size; ++i) {
		for (int j = 0; j < kernel_size; ++j)
			buf[(i * (params.w + (kernel_size - 1))) + j] /= sum;
	}

	// dev_mu_bitmap_out_bitmap_ref_R = dev_kernel
	cudaStatus = cudaMemcpy(params.dev_mu_bitmap_out_bitmap_ref_R, buf, sizeof(REAL) * arraySizeReal, cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess)
		goto Error;

	error_CUFFT = DFFT(params.planr2c, params.dev_mu_bitmap_out_bitmap_ref_R, params.dev_F_1);
	if (error_CUFFT != CUFFT_SUCCESS) goto Error;

	// *** *** *** *** ***

	// Bufor został uzupełniony zerami przy okazji tworzenia kernela Gaussowskiego
	for (int pose = 0; pose < params.NUMBER_OF_POSES; ++pose) {
		// R channel
		for (int i = 0; i < params.h; ++i) {
			for (int j = 0; j < params.w; ++j) {
				unsigned char R = params.bitmap_ref[(pose * (params.w * params.h)) + ((i * params.w) + j)] >> 16;
				buf[(i * (params.w + (kernel_size - 1))) + j] = R / ((REAL)255.0);
			}
		}

		// *** *** *** *** ***

		cudaStatus = cudaMemcpy(params.dev_bitmap_ref_R + (pose * arraySizeReal), buf, sizeof(REAL) * arraySizeReal, cudaMemcpyHostToDevice);
		if (cudaStatus != cudaSuccess)
			goto Error;

		// G channel
		for (int i = 0; i < params.h; ++i) {
			for (int j = 0; j < params.w; ++j) {
				unsigned char G = (params.bitmap_ref[(pose * (params.w * params.h)) + ((i * params.w) + j)] >> 8) & 255;
				buf[(i * (params.w + (kernel_size - 1))) + j] = G / ((REAL)255.0);
			}
		}

		// *** *** *** *** ***

		cudaStatus = cudaMemcpy(params.dev_bitmap_ref_G + (pose * arraySizeReal), buf, sizeof(REAL) * arraySizeReal, cudaMemcpyHostToDevice);
		if (cudaStatus != cudaSuccess)
			goto Error;

		// B channel
		for (int i = 0; i < params.h; ++i) {
			for (int j = 0; j < params.w; ++j) {
				unsigned char B = params.bitmap_ref[(pose * (params.w * params.h)) + ((i * params.w) + j)] & 255;
				buf[(i * (params.w + (kernel_size - 1))) + j] = B / ((REAL)255.0);
			}
		}

		cudaStatus = cudaMemcpy(params.dev_bitmap_ref_B + (pose * arraySizeReal), buf, sizeof(REAL) * arraySizeReal, cudaMemcpyHostToDevice);
		if (cudaStatus != cudaSuccess)
			goto Error;
	}

	// *** *** *** *** ***

	// Compute mu's for reference images
	for (int pose = 0; pose < params.NUMBER_OF_POSES; ++pose) {
		// R channel
		error_CUFFT = DFFT(params.planr2c, params.dev_bitmap_ref_R + (pose * arraySizeReal), params.dev_F_2);
		if (error_CUFFT != CUFFT_SUCCESS) goto Error;

		MultiplyPointwiseComplex<<<(arraySizeComplex + 255) >> 8, 256>>>(params.dev_F_2, params.dev_F_1, params.dev_F_2, arraySizeComplex);
		cudaStatus = cudaGetLastError();
		if (cudaStatus != cudaSuccess) goto Error;

		error_CUFFT = IDFFT(params.planc2r, params.dev_F_2, params.dev_mu_bitmap_ref_R + (pose * arraySizeReal));
		if (error_CUFFT != CUFFT_SUCCESS) goto Error;

		// *** *** *** *** ***

		// G channel
		error_CUFFT = DFFT(params.planr2c, params.dev_bitmap_ref_G + (pose * arraySizeReal), params.dev_F_2);
		if (error_CUFFT != CUFFT_SUCCESS) goto Error;

		MultiplyPointwiseComplex<<<(arraySizeComplex + 255) >> 8, 256>>>(params.dev_F_2, params.dev_F_1, params.dev_F_2, arraySizeComplex);
		cudaStatus = cudaGetLastError();
		if (cudaStatus != cudaSuccess) goto Error;

		error_CUFFT = IDFFT(params.planc2r, params.dev_F_2, params.dev_mu_bitmap_ref_G + (pose * arraySizeReal));
		if (error_CUFFT != CUFFT_SUCCESS) goto Error;

		// *** *** *** *** ***

		// B channel
		error_CUFFT = DFFT(params.planr2c, params.dev_bitmap_ref_B + (pose * arraySizeReal), params.dev_F_2);
		if (error_CUFFT != CUFFT_SUCCESS) goto Error;

		MultiplyPointwiseComplex<<<(arraySizeComplex + 255) >> 8, 256>>>(params.dev_F_2, params.dev_F_1, params.dev_F_2, arraySizeComplex);
		cudaStatus = cudaGetLastError();
		if (cudaStatus != cudaSuccess) goto Error;

		error_CUFFT = IDFFT(params.planc2r, params.dev_F_2, params.dev_mu_bitmap_ref_B + (pose * arraySizeReal));
		if (error_CUFFT != CUFFT_SUCCESS) goto Error;
	}

	// *** *** *** *** ***

	// Compute mu's for reference images square
	for (int pose = 0; pose < params.NUMBER_OF_POSES; ++pose) {
		// R channel
		// dev_mu_bitmap_out_bitmap_ref_R = dev_bitmap_ref_R * dev_bitmap_ref_R
		MultiplyPointwiseReal<<<(arraySizeReal + 255) >> 8, 256>>>(
			params.dev_bitmap_ref_R + (pose * arraySizeReal),
			params.dev_bitmap_ref_R + (pose * arraySizeReal),
			params.dev_mu_bitmap_out_bitmap_ref_R,
			arraySizeReal
		);
		cudaStatus = cudaGetLastError();
		if (cudaStatus != cudaSuccess) goto Error;

		error_CUFFT = DFFT(params.planr2c, params.dev_mu_bitmap_out_bitmap_ref_R, params.dev_F_2);
		if (error_CUFFT != CUFFT_SUCCESS) goto Error;

		MultiplyPointwiseComplex<<<(arraySizeComplex + 255) >> 8, 256>>>(params.dev_F_2, params.dev_F_1, params.dev_F_2, arraySizeComplex);
		cudaStatus = cudaGetLastError();
		if (cudaStatus != cudaSuccess) goto Error;

		error_CUFFT = IDFFT(params.planc2r, params.dev_F_2, params.dev_mu_bitmap_ref_R_square + (pose * arraySizeReal));
		if (error_CUFFT != CUFFT_SUCCESS) goto Error;

		// *** *** *** *** ***

		// G channel
		// dev_mu_bitmap_out_bitmap_ref_R = dev_bitmap_ref_G * dev_bitmap_ref_G
		MultiplyPointwiseReal<<<(arraySizeReal + 255) >> 8, 256>>>(
			params.dev_bitmap_ref_G + (pose * arraySizeReal),
			params.dev_bitmap_ref_G + (pose * arraySizeReal),
			params.dev_mu_bitmap_out_bitmap_ref_R,
			arraySizeReal
			);
		cudaStatus = cudaGetLastError();
		if (cudaStatus != cudaSuccess) goto Error;

		error_CUFFT = DFFT(params.planr2c, params.dev_mu_bitmap_out_bitmap_ref_R, params.dev_F_2);
		if (error_CUFFT != CUFFT_SUCCESS) goto Error;

		MultiplyPointwiseComplex<<<(arraySizeComplex + 255) >> 8, 256>>>(params.dev_F_2, params.dev_F_1, params.dev_F_2, arraySizeComplex);
		cudaStatus = cudaGetLastError();
		if (cudaStatus != cudaSuccess) goto Error;

		error_CUFFT = IDFFT(params.planc2r, params.dev_F_2, params.dev_mu_bitmap_ref_G_square + (pose * arraySizeReal));
		if (error_CUFFT != CUFFT_SUCCESS) goto Error;

		// *** *** *** *** ***

		// B channel
		// dev_mu_bitmap_out_bitmap_ref_R = dev_bitmap_ref_B * dev_bitmap_ref_B
		MultiplyPointwiseReal<<<(arraySizeReal + 255) >> 8, 256>>>(
			params.dev_bitmap_ref_B + (pose * arraySizeReal),
			params.dev_bitmap_ref_B + (pose * arraySizeReal),
			params.dev_mu_bitmap_out_bitmap_ref_R,
			arraySizeReal
			);
		cudaStatus = cudaGetLastError();
		if (cudaStatus != cudaSuccess) goto Error;

		error_CUFFT = DFFT(params.planr2c, params.dev_mu_bitmap_out_bitmap_ref_R, params.dev_F_2);
		if (error_CUFFT != CUFFT_SUCCESS) goto Error;

		MultiplyPointwiseComplex<<<(arraySizeComplex + 255) >> 8, 256>>>(params.dev_F_2, params.dev_F_1, params.dev_F_2, arraySizeComplex);
		cudaStatus = cudaGetLastError();
		if (cudaStatus != cudaSuccess) goto Error;

		error_CUFFT = IDFFT(params.planc2r, params.dev_F_2, params.dev_mu_bitmap_ref_B_square + (pose * arraySizeReal));
		if (error_CUFFT != CUFFT_SUCCESS) goto Error;
	}

	// *** *** *** *** ***

	// TEST 1
	/*int pose = 0;

	// R channel
	cudaStatus = cudaMemcpy(buf, params.dev_mu_bitmap_ref_R + (pose * arraySizeReal), sizeof(REAL) * arraySizeReal, cudaMemcpyDeviceToHost);
	if (cudaStatus != cudaSuccess) goto Error;

	for (int i = 0; i < params.h; ++i) {
		for (int j = 0; j < params.w; ++j) {
			REAL Rf = buf[((kernel_radius + i) * (params.w + (kernel_size - 1))) + (kernel_radius + j)] / arraySizeReal;
			if (Rf < ((REAL)0.0)) Rf = ((REAL)0.0);
			if (Rf > ((REAL)1.0)) Rf = ((REAL)1.0);
			unsigned char Ri = Rf * ((REAL)255.0);
			params.bitmap_ref[(pose * (params.w * params.h)) + ((i * params.w) + j)] = (((unsigned) Ri) << 16);
		}
	}

	// G channel
	cudaStatus = cudaMemcpy(buf, params.dev_mu_bitmap_ref_G + (pose * arraySizeReal), sizeof(REAL) *  arraySizeReal, cudaMemcpyDeviceToHost);
	if (cudaStatus != cudaSuccess) goto Error;

	for (int i = 0; i < params.h; ++i) {
		for (int j = 0; j < params.w; ++j) {
			REAL Gf = buf[((kernel_radius + i) * (params.w + (kernel_size - 1))) + (kernel_radius + j)] / arraySizeReal;
			if (Gf < ((REAL)0.0)) Gf = ((REAL)0.0);
			if (Gf > ((REAL)1.0)) Gf = ((REAL)1.0);
			unsigned char Gi = Gf * ((REAL)255.0);
			params.bitmap_ref[(pose * (params.w * params.h)) + ((i * params.w) + j)] |= (((unsigned) Gi) << 8);
		}
	}

	// B channel
	cudaStatus = cudaMemcpy(buf, params.dev_mu_bitmap_ref_B + (pose * arraySizeReal), sizeof(REAL) *  arraySizeReal, cudaMemcpyDeviceToHost);
	if (cudaStatus != cudaSuccess) goto Error;

	for (int i = 0; i < params.h; ++i) {
		for (int j = 0; j < params.w; ++j) {
			REAL Bf = buf[((kernel_radius + i) * (params.w + (kernel_size - 1))) + (kernel_radius + j)] / arraySizeReal;
			if (Bf < ((REAL)0.0)) Bf = ((REAL)0.0);
			if (Bf > ((REAL)1.0)) Bf = ((REAL)1.0);
			unsigned char Bi = Bf * ((REAL)255.0);
			params.bitmap_ref[(pose * (params.w * params.h)) + ((i * params.w) + j)] |= ((unsigned)Bi);
		}
	}

	// Copy to bitmap on hdd
	unsigned char *foo = (unsigned char *)malloc(3 * params.w * params.h);
	for (int i = 0; i < params.h; ++i) {
		for (int j = 0; j < params.w; ++j) {
			unsigned char R = params.bitmap_ref[(pose * (params.w * params.h)) + ((i * params.w) + j)] >> 16;
			unsigned char G = (params.bitmap_ref[(pose * (params.w * params.h)) + ((i * params.w) + j)] >> 8) & 255;
			unsigned char B = params.bitmap_ref[(pose * (params.w * params.h)) + ((i * params.w) + j)] & 255;
			foo[((((params.h - 1 - i) * params.w) + j) * 3) + 2] = R;
			foo[((((params.h - 1 - i) * params.w) + j) * 3) + 1] = G;
			foo[(((params.h - 1 - i) * params.w) + j) * 3] = B;		
		}
	}

	FILE *f = fopen("test.bmp", "rb+");
	fseek(f, 54, SEEK_SET);
	fwrite(foo, sizeof(int) * params.w * params.h, 1, f);
	fclose(f);*/

	// TEST 2
	/*int pose = 0;
	
	cudaStatus = cudaMemcpy(buf, params.dev_mu_bitmap_ref_B_square + (pose * arraySizeReal), sizeof(REAL) * arraySizeReal, cudaMemcpyDeviceToHost);
	if (cudaStatus != cudaSuccess) goto Error;

	FILE *f = fopen("output.txt", "wt");
	for (int i = 0; i < 32; ++i) {
		REAL value = buf[((kernel_radius + 399) * (params.w + (kernel_size - 1))) + (kernel_radius + 400 - 16 + i)] / arraySizeReal;
		char buffer[256];
#ifndef USE_DOUBLE_PRECISION
		sprintf(buffer, "%.20f\n", value);
#else
		sprintf(buffer, "%.20lf\n", value);
#endif
		fprintf(f, "%s", buffer);
	}
	fclose(f);*/

	//*************************************************************************************************

	return true;
Error:
	if (dev_params.bitmap_ref != NULL)
		cudaFree(dev_params.bitmap_ref);

	if (dev_params.dL_dparams_1 != NULL)
		cudaFree(dev_params.dL_dparams_1);

	if (dev_params.dL_dparams_2 != NULL)
		cudaFree(dev_params.dL_dparams_2);

	if (dev_params.dL_dparams_3 != NULL)
		cudaFree(dev_params.dL_dparams_3);

	if (dev_params.dL_dparams_4 != NULL)
		cudaFree(dev_params.dL_dparams_4);

	if (dev_params.output_params != NULL)
		cudaFree(dev_params.output_params);

	return false;
}

// *** *** *** *** ***

extern bool ZeroCUDAGradient(SRenderParams& params, SCUDARenderParams& dev_params) {
	cudaError_t cudaStatus;

	cudaStatus = cudaMemset(dev_params.dL_dparams_1, 0, 4 * sizeof(REAL) * params.numberOfGaussians);
	if (cudaStatus != cudaSuccess)
		goto Error;

	cudaStatus = cudaMemset(dev_params.dL_dparams_2, 0, 4 * sizeof(REAL) * params.numberOfGaussians);
	if (cudaStatus != cudaSuccess)
		goto Error;

	cudaStatus = cudaMemset(dev_params.dL_dparams_3, 0, 4 * sizeof(REAL) * params.numberOfGaussians);
	if (cudaStatus != cudaSuccess)
		goto Error;

	cudaStatus = cudaMemset(dev_params.dL_dparams_4, 0, 2 * sizeof(REAL) * params.numberOfGaussians);
	if (cudaStatus != cudaSuccess)
		goto Error;

	cudaStatus = cudaMemset(dev_params.output_params, 0, 1 * 8);
	if (cudaStatus != cudaSuccess)
		goto Error;

	// *** *** ***

	// !!! !!! !!!
	// -1 = [255, 255, 255, 255]
	// !!! !!! !!!
	cudaStatus = cudaMemset(dev_params.dump, 255, sizeof(int) * MAX_RAY_LENGTH * params.w * params.h);
	if (cudaStatus != cudaSuccess)
		goto Error;

	// *** *** ***

	return true;
Error:
	return false;
}

// *** *** *** *** ***

static unsigned seed_float = 0;

static float RandomFloat() {
	float result;

	*((unsigned*)&result) = (127 << 23) | (seed_float & ((1 << 23) - 1));
	seed_float = (1664525 * seed_float) + 1013904223;
	return result - 1.0f;
}

// *** *** *** *** ***

static unsigned RandomInteger() {
	unsigned result;

	result = seed_float;
	seed_float = (1664525 * seed_float) + 1013904223;
	return result;
}

// *** *** *** *** ***

static void RandomNormalFloat(float& Z1, float& Z2) {
	float U1;
	do {
		U1 = RandomFloat();
	} while (U1 == 0.0f);
	float U2 = RandomFloat();
	float tmp1 = sqrt(-2.0f * log(U1));
	float tmp2 = 2.0f * M_PI * U2;
	Z1 = tmp1 * cos(tmp2);
	Z2 = tmp1 * sin(tmp2);
}

// *** *** *** *** ***

static void RandomMultinormalFloat(
	SGaussianComponent &gc,
	float Z1, float Z2, float Z3,
	float &X, float &Y, float &Z
) {
	float sX = 1.0f / (1.0f + expf(-gc.sX));
	float sY = 1.0f / (1.0f + expf(-gc.sY));
	float sZ = 1.0f / (1.0f + expf(-gc.sZ));

	float X_prim = Z1 * sX;
	float Y_prim = Z2 * sY;
	float Z_prim = Z3 * sZ;

	// *** *** *** *** ***

	float aa = gc.qr * gc.qr;
	float bb = gc.qi * gc.qi;
	float cc = gc.qj * gc.qj;
	float dd = gc.qk * gc.qk;

	float s = 2.0f / (aa + bb + cc + dd);

	float bs = gc.qi * s;  float cs = gc.qj * s;  float ds = gc.qk * s;
	float ab = gc.qr * bs; float ac = gc.qr * cs; float ad = gc.qr * ds;
	bb = bb * s;           float bc = gc.qi * cs; float bd = gc.qi * ds;
	cc = cc * s;           float cd = gc.qj * ds;       dd = dd * s;

	float R11 = 1.0f - cc - dd;
	float R12 = bc - ad;
	float R13 = bd + ac;

	float R21 = bc + ad;
	float R22 = 1.0f - bb - dd;
	float R23 = cd - ab;

	float R31 = bd - ac;
	float R32 = cd + ab;
	float R33 = 1.0f - bb - cc;

	X = gc.mX + ((R11 * X_prim) + (R12 * Y_prim) + (R13 * Z_prim));
	Y = gc.mY + ((R21 * X_prim) + (R22 * Y_prim) + (R23 * Z_prim));
	Z = gc.mZ + ((R31 * X_prim) + (R32 * Y_prim) + (R33 * Z_prim));
}

// *** *** *** *** ***

extern bool UpdateCUDAGradient(SRenderParams& params, SCUDARenderParams& dev_params) {
	cudaError_t cudaStatus;

	// *** *** *** *** ***

	cudaStatus = cudaMemcpy(params.dL_dparams_1, dev_params.dL_dparams_1, 4 * sizeof(REAL) * params.numberOfGaussians, cudaMemcpyDeviceToHost);
	if (cudaStatus != cudaSuccess)
		goto Error;

	cudaStatus = cudaMemcpy(params.dL_dparams_2, dev_params.dL_dparams_2, 4 * sizeof(REAL) * params.numberOfGaussians, cudaMemcpyDeviceToHost);
	if (cudaStatus != cudaSuccess)
		goto Error;

	cudaStatus = cudaMemcpy(params.dL_dparams_3, dev_params.dL_dparams_3, 4 * sizeof(REAL) * params.numberOfGaussians, cudaMemcpyDeviceToHost);
	if (cudaStatus != cudaSuccess)
		goto Error;

	cudaStatus = cudaMemcpy(params.dL_dparams_4, dev_params.dL_dparams_4, 2 * sizeof(REAL) * params.numberOfGaussians, cudaMemcpyDeviceToHost);
	if (cudaStatus != cudaSuccess)
		goto Error;

	cudaStatus = cudaMemcpy(&params.loss, dev_params.output_params, 1 * 8, cudaMemcpyDeviceToHost);
	if (cudaStatus != cudaSuccess)
		goto Error;

	// *** *** *** *** ***

	float lr_RGB;
	float lr_A;
	float lr_m;
	float lr_s;
	float lr_q;
	
	lr_RGB = 0.01f; //
	lr_A = 0.1; //
	lr_m = 0.001f; // 0.01
	lr_s = 0.0001f; //  0.001f
	lr_q = 0.1f;

	/*if (params.epoch > 1) {
		if ((params.loss / (3.0f * params.w * params.h)) < 0.001f) {
			lr_RGB *= ((params.loss * 1000.0f) / (3.0f * params.w * params.h));
			lr_m *= ((params.loss * 1000.0f) / (3.0f * params.w * params.h));
			lr_s *= ((params.loss * 1000.0f) / (3.0f * params.w * params.h));
			lr_q *= ((params.loss * 1000.0f) / (3.0f * params.w * params.h));
		}
	}*/

	if (params.epoch > 2048) {
		lr_RGB *= (2048.0f / params.epoch);
		lr_m *= (2048.0f / params.epoch);
		lr_s *= (2048.0f / params.epoch);
		lr_q *= (2048.0f / params.epoch);
	}

	int numberOfGaussians = 0;
	SGaussianComponent *GC_new = (SGaussianComponent *)malloc(sizeof(SGaussianComponent) * params.numberOfGaussians * 2); // (x2) !!! !!! !!!

	void *m1_new = malloc(4 * sizeof(REAL) * params.numberOfGaussians * 2); // (x2) !!! !!! !!!
	void *m2_new = malloc(4 * sizeof(REAL) * params.numberOfGaussians * 2); // (x2) !!! !!! !!!
	void *m3_new = malloc(4 * sizeof(REAL) * params.numberOfGaussians * 2); // (x2) !!! !!! !!!
	void *m4_new = malloc(2 * sizeof(REAL) * params.numberOfGaussians * 2); // (x2) !!! !!! !!!

	void *v1_new = malloc(4 * sizeof(REAL) * params.numberOfGaussians * 2); // (x2) !!! !!! !!!
	void *v2_new = malloc(4 * sizeof(REAL) * params.numberOfGaussians * 2); // (x2) !!! !!! !!!
	void *v3_new = malloc(4 * sizeof(REAL) * params.numberOfGaussians * 2); // (x2) !!! !!! !!!
	void *v4_new = malloc(2 * sizeof(REAL) * params.numberOfGaussians * 2); // (x2) !!! !!! !!!

	for (int i = 0; i < params.numberOfGaussians; ++i) {
		float beta1 = 0.9f;
		float beta2 = 0.999f;
		float epsilon = 0.00000001f;
		float t = params.epoch;

		float grad;
		float m;
		float v;
		float m_hat;
		float v_hat;

		//grad = (((REAL *)params.dL_dparams_1)[i * 4] / (3.0f * /*params.NUMBER_OF_POSES **/ params.w * params.h));
		grad = ((REAL *)params.dL_dparams_1)[i * 4];
		m = (beta1 * ((float *)params.m1)[i * 4]) + ((1.0f - beta1) * grad);
		v = (beta2 * ((float *)params.v1)[i * 4]) + ((1.0f - beta2) * grad * grad);
		((float *)params.m1)[i * 4] = m;
		((float *)params.v1)[i * 4] = v;
		m_hat = m / (1.0f - powf(beta1, t));
		v_hat = v / (1.0f - powf(beta2, t));
		params.GC[i].R -= (lr_RGB * m_hat) / (sqrtf(v_hat) + epsilon);
		
		params.GC[i].R = (params.GC[i].R < 0.0f) ? 0.0f : params.GC[i].R;
		params.GC[i].R = (params.GC[i].R > 1.0f) ? 1.0f : params.GC[i].R;

		//grad = (((REAL *)params.dL_dparams_1)[(i * 4) + 1] / (3.0f * /*params.NUMBER_OF_POSES **/ params.w * params.h));
		grad = ((REAL *)params.dL_dparams_1)[(i * 4) + 1];
		m = (beta1 * ((float *)params.m1)[(i * 4) + 1]) + ((1.0f - beta1) * grad);
		v = (beta2 * ((float *)params.v1)[(i * 4) + 1]) + ((1.0f - beta2) * grad * grad);
		((float *)params.m1)[(i * 4) + 1] = m;
		((float *)params.v1)[(i * 4) + 1] = v;
		m_hat = m / (1.0f - powf(beta1, t));
		v_hat = v / (1.0f - powf(beta2, t));
		params.GC[i].G -= (lr_RGB * m_hat) / (sqrtf(v_hat) + epsilon);

		params.GC[i].G = (params.GC[i].G < 0.0f) ? 0.0f : params.GC[i].G;
		params.GC[i].G = (params.GC[i].G > 1.0f) ? 1.0f : params.GC[i].G;

		//grad = (((REAL *)params.dL_dparams_1)[(i * 4) + 2] / (3.0f * /*params.NUMBER_OF_POSES **/ params.w * params.h));
		grad = ((REAL *)params.dL_dparams_1)[(i * 4) + 2];
		m = (beta1 * ((float *)params.m1)[(i * 4) + 2]) + ((1.0f - beta1) * grad);
		v = (beta2 * ((float *)params.v1)[(i * 4) + 2]) + ((1.0f - beta2) * grad * grad);
		((float *)params.m1)[(i * 4) + 2] = m;
		((float *)params.v1)[(i * 4) + 2] = v;
		m_hat = m / (1.0f - powf(beta1, t));
		v_hat = v / (1.0f - powf(beta2, t));
		params.GC[i].B -= (lr_RGB * m_hat) / (sqrtf(v_hat) + epsilon);

		params.GC[i].B = (params.GC[i].B < 0.0f) ? 0.0f : params.GC[i].B;
		params.GC[i].B = (params.GC[i].B > 1.0f) ? 1.0f : params.GC[i].B;

		//grad = (((REAL *)params.dL_dparams_1)[(i * 4) + 3] / (3.0f * /*params.NUMBER_OF_POSES **/ params.w * params.h));
		grad = ((REAL *)params.dL_dparams_1)[(i * 4) + 3];
		m = (beta1 * ((float *)params.m1)[(i * 4) + 3]) + ((1.0f - beta1) * grad);
		v = (beta2 * ((float *)params.v1)[(i * 4) + 3]) + ((1.0f - beta2) * grad * grad);
		((float *)params.m1)[(i * 4) + 3] = m;
		((float *)params.v1)[(i * 4) + 3] = v;
		m_hat = m / (1.0f - powf(beta1, t));
		v_hat = v / (1.0f - powf(beta2, t));
		params.GC[i].alpha -= (lr_A * m_hat) / (sqrtf(v_hat) + epsilon);

		// *** *** *** *** ***

		//grad = (((REAL *)params.dL_dparams_2)[i * 4] / (3.0f * /*params.NUMBER_OF_POSES **/ params.w * params.h));
		grad = ((REAL *)params.dL_dparams_2)[i * 4];
		m = (beta1 * ((float *)params.m2)[i * 4]) + ((1.0f - beta1) * grad);
		v = (beta2 * ((float *)params.v2)[i * 4]) + ((1.0f - beta2) * grad * grad);
		((float *)params.m2)[i * 4] = m;
		((float *)params.v2)[i * 4] = v;
		m_hat = m / (1.0f - powf(beta1, t));
		v_hat = v / (1.0f - powf(beta2, t));
		float dX = (lr_m * m_hat) / (sqrtf(v_hat) + epsilon);
		//if (dX < -1.0f) dX = -1.0f;
		//if (dX > 1.0f) dX = 1.0f;

		//grad = (((REAL *)params.dL_dparams_2)[(i * 4) + 1] / (3.0f * /*params.NUMBER_OF_POSES **/ params.w * params.h));
		grad = ((REAL *)params.dL_dparams_2)[(i * 4) + 1];
		m = (beta1 * ((float *)params.m2)[(i * 4) + 1]) + ((1.0f - beta1) * grad);
		v = (beta2 * ((float *)params.v2)[(i * 4) + 1]) + ((1.0f - beta2) * grad * grad);
		((float *)params.m2)[(i * 4) + 1] = m;
		((float *)params.v2)[(i * 4) + 1] = v;
		m_hat = m / (1.0f - powf(beta1, t));
		v_hat = v / (1.0f - powf(beta2, t));
		float dY = (lr_m * m_hat) / (sqrtf(v_hat) + epsilon);
		//if (dY < -1.0f) dY = -1.0f;
		//if (dY > 1.0f) dY = 1.0f;

		//grad = (((REAL *)params.dL_dparams_2)[(i * 4) + 2] / (3.0f * /*params.NUMBER_OF_POSES **/ params.w * params.h));
		grad = ((REAL *)params.dL_dparams_2)[(i * 4) + 2];
		m = (beta1 * ((float *)params.m2)[(i * 4) + 2]) + ((1.0f - beta1) * grad);
		v = (beta2 * ((float *)params.v2)[(i * 4) + 2]) + ((1.0f - beta2) * grad * grad);
		((float *)params.m2)[(i * 4) + 2] = m;
		((float *)params.v2)[(i * 4) + 2] = v;
		m_hat = m / (1.0f - powf(beta1, t));
		v_hat = v / (1.0f - powf(beta2, t));
		float dZ = (lr_m * m_hat) / (sqrtf(v_hat) + epsilon);
		//if (dZ < -1.0f) dZ = -1.0f;
		//if (dZ > 1.0f) dZ = 1.0f;

		/*params.GC[i].mX -= dX;
		params.GC[i].mY -= dY;
		params.GC[i].mZ -= dZ;

		if (params.GC[i].mX < -2.0f) params.GC[i].mX = -2.0f;
		if (params.GC[i].mX > 2.0f) params.GC[i].mX = 2.0f;

		if (params.GC[i].mY < -2.0f) params.GC[i].mY = -2.0f;
		if (params.GC[i].mY > 2.0f) params.GC[i].mY = 2.0f;

		if (params.GC[i].mZ < -2.0f) params.GC[i].mZ = -2.0f;
		if (params.GC[i].mZ > 2.0f) params.GC[i].mZ = 2.0f;*/

		// *** *** *** *** ***

		//float sX = lr_s * (((REAL *)params.dL_dparams_2)[(i * 4) + 3] / (3.0f * /*params.NUMBER_OF_POSES **/ params.w * params.h));
		//float sY = lr_s * (((REAL *)params.dL_dparams_3)[i * 4] / (3.0f * /*params.NUMBER_OF_POSES **/ params.w * params.h));
		//float sZ = lr_s * (((REAL *)params.dL_dparams_3)[(i * 4) + 1] / (3.0f * /*params.NUMBER_OF_POSES **/ params.w * params.h));

		float s1 = 1.0f / (1.0f + expf(-params.GC[i].sX));

		//grad = (((REAL *)params.dL_dparams_2)[(i * 4) + 3] / (3.0f * /*params.NUMBER_OF_POSES **/ params.w * params.h));
		grad = ((REAL *)params.dL_dparams_2)[(i * 4) + 3];
		m = (beta1 * ((float *)params.m2)[(i * 4) + 3]) + ((1.0f - beta1) * grad);
		v = (beta2 * ((float *)params.v2)[(i * 4) + 3]) + ((1.0f - beta2) * grad * grad);
		((float *)params.m2)[(i * 4) + 3] = m;
		((float *)params.v2)[(i * 4) + 3] = v;
		m_hat = m / (1.0f - powf(beta1, t));
		v_hat = v / (1.0f - powf(beta2, t));
		float s2 = params.GC[i].sX - (lr_s * m_hat) / (sqrtf(v_hat) + epsilon);
		
		s2 = 1.0f / (1.0f + expf(-s2));
		if (s2 < 0.001f) s2 = 0.001f;
		//if ((s2 / s1) > 1.01f) s2 = s1 * 1.01;
		//if ((s2 / s1) < 0.99f) s2 = s1 * 0.99f;
		//if (s2 > 0.025f) s2 = s1;
		params.GC[i].sX = -log((1.0f / s2) - 1.0f);

		s1 = 1.0f / (1.0f + expf(-params.GC[i].sY));

		//grad = (((REAL *)params.dL_dparams_3)[i * 4] / (3.0f * /*params.NUMBER_OF_POSES **/ params.w * params.h));
		grad = ((REAL *)params.dL_dparams_3)[i * 4];
		m = (beta1 * ((float *)params.m3)[i * 4]) + ((1.0f - beta1) * grad);
		v = (beta2 * ((float *)params.v3)[i * 4]) + ((1.0f - beta2) * grad * grad);
		((float *)params.m3)[i * 4] = m;
		((float *)params.v3)[i * 4] = v;
		m_hat = m / (1.0f - powf(beta1, t));
		v_hat = v / (1.0f - powf(beta2, t));
		s2 = params.GC[i].sY - (lr_s * m_hat) / (sqrtf(v_hat) + epsilon);
				
		s2 = 1.0f / (1.0f + expf(-s2));
		if (s2 < 0.001f) s2 = 0.001f;
		//if ((s2 / s1) > 1.01f) s2 = s1 * 1.01;
		//if ((s2 / s1) < 0.99f) s2 = s1 * 0.99f;
		//if (s2 > 0.025f) s2 = s1;
		params.GC[i].sY = -log((1.0f / s2) - 1.0f);

		s1 = 1.0f / (1.0f + expf(-params.GC[i].sZ));

		//grad = (((REAL *)params.dL_dparams_3)[(i * 4) + 1] / (3.0f * /*params.NUMBER_OF_POSES **/ params.w * params.h));
		grad = ((REAL *)params.dL_dparams_3)[(i * 4) + 1];
		m = (beta1 * ((float *)params.m3)[(i * 4) + 1]) + ((1.0f - beta1) * grad);
		v = (beta2 * ((float *)params.v3)[(i * 4) + 1]) + ((1.0f - beta2) * grad * grad);
		((float *)params.m3)[(i * 4) + 1] = m;
		((float *)params.v3)[(i * 4) + 1] = v;
		m_hat = m / (1.0f - powf(beta1, t));
		v_hat = v / (1.0f - powf(beta2, t));
		s2 = params.GC[i].sZ - (lr_s * m_hat) / (sqrtf(v_hat) + epsilon);
		
		s2 = 1.0f / (1.0f + expf(-s2));
		if (s2 < 0.001f) s2 = 0.001f;
		//if ((s2 / s1) > 1.01f) s2 = s1 * 1.01;
		//if ((s2 / s1) < 0.99f) s2 = s1 * 0.99f;
		//if (s2 > 0.025f) s2 = s1;
		params.GC[i].sZ = -log((1.0f / s2) - 1.0f);

		//params.GC[i].sX -= sX;
		//params.GC[i].sY -= sY;
		//params.GC[i].sZ -= sZ;

		// *** *** *** *** ***

		//params.GC[i].qr -= lr_q * (((REAL *)params.dL_dparams_3)[(i * 4) + 2] / (3.0f * /*params.NUMBER_OF_POSES **/ params.w * params.h));
		//params.GC[i].qi -= lr_q * (((REAL *)params.dL_dparams_3)[(i * 4) + 3] / (3.0f * /*params.NUMBER_OF_POSES **/ params.w * params.h));
		//params.GC[i].qj -= lr_q * (((REAL *)params.dL_dparams_4)[i * 2] / (3.0f * /*params.NUMBER_OF_POSES **/ params.w * params.h));
		//params.GC[i].qk -= lr_q * (((REAL *)params.dL_dparams_4)[(i * 2) + 1] / (3.0f * /*params.NUMBER_OF_POSES **/ params.w * params.h));
		
		//grad = (((REAL *)params.dL_dparams_3)[(i * 4) + 2] / (3.0f * /*params.NUMBER_OF_POSES **/ params.w * params.h));
		grad = ((REAL *)params.dL_dparams_3)[(i * 4) + 2];
		m = (beta1 * ((float *)params.m3)[(i * 4) + 2]) + ((1.0f - beta1) * grad);
		v = (beta2 * ((float *)params.v3)[(i * 4) + 2]) + ((1.0f - beta2) * grad * grad);
		((float *)params.m3)[(i * 4) + 2] = m;
		((float *)params.v3)[(i * 4) + 2] = v;
		m_hat = m / (1.0f - powf(beta1, t));
		v_hat = v / (1.0f - powf(beta2, t));
		params.GC[i].qr -= (lr_q * m_hat) / (sqrtf(v_hat) + epsilon);

		//grad = (((REAL *)params.dL_dparams_3)[(i * 4) + 3] / (3.0f * /*params.NUMBER_OF_POSES **/ params.w * params.h));
		grad = ((REAL *)params.dL_dparams_3)[(i * 4) + 3];
		m = (beta1 * ((float *)params.m3)[(i * 4) + 3]) + ((1.0f - beta1) * grad);
		v = (beta2 * ((float *)params.v3)[(i * 4) + 3]) + ((1.0f - beta2) * grad * grad);
		((float *)params.m3)[(i * 4) + 3] = m;
		((float *)params.v3)[(i * 4) + 3] = v;
		m_hat = m / (1.0f - powf(beta1, t));
		v_hat = v / (1.0f - powf(beta2, t));
		params.GC[i].qi -= (lr_q * m_hat) / (sqrtf(v_hat) + epsilon);

		//grad = (((REAL *)params.dL_dparams_4)[i * 2] / (3.0f * /*params.NUMBER_OF_POSES **/ params.w * params.h));
		grad = ((REAL *)params.dL_dparams_4)[i * 2];
		m = (beta1 * ((float *)params.m4)[i * 2]) + ((1.0f - beta1) * grad);
		v = (beta2 * ((float *)params.v4)[i * 2]) + ((1.0f - beta2) * grad * grad);
		((float *)params.m4)[i * 2] = m;
		((float *)params.v4)[i * 2] = v;
		m_hat = m / (1.0f - powf(beta1, t));
		v_hat = v / (1.0f - powf(beta2, t));
		params.GC[i].qj -= (lr_q * m_hat) / (sqrtf(v_hat) + epsilon);

		//grad = (((REAL *)params.dL_dparams_4)[(i * 2) + 1] / (3.0f * /*params.NUMBER_OF_POSES **/ params.w * params.h));
		grad = ((REAL *)params.dL_dparams_4)[(i * 2) + 1];
		m = (beta1 * ((float *)params.m4)[(i * 2) + 1]) + ((1.0f - beta1) * grad);
		v = (beta2 * ((float *)params.v4)[(i * 2) + 1]) + ((1.0f - beta2) * grad * grad);
		((float *)params.m4)[(i * 2) + 1] = m;
		((float *)params.v4)[(i * 2) + 1] = v;
		m_hat = m / (1.0f - powf(beta1, t));
		v_hat = v / (1.0f - powf(beta2, t));
		params.GC[i].qk -= (lr_q * m_hat) / (sqrtf(v_hat) + epsilon);

		// *** *** *** *** ***

		/*float dL_dm_proj_norm_avg = 0.0;
		for (int j = 0; j < params.NUMBER_OF_POSES; ++j) {
			float dL_dm_proj_X = (params.poses[j].Rx * dX) + (params.poses[j].Ry * dY) + (params.poses[j].Rz * dZ);
			float dL_dm_proj_Y = (params.poses[j].Dx * dX) + (params.poses[j].Dy * dY) + (params.poses[j].Dz * dZ);
			float dL_dm_proj_norm = sqrtf((dL_dm_proj_X * dL_dm_proj_X) + (dL_dm_proj_Y * dL_dm_proj_Y));
			dL_dm_proj_norm_avg += dL_dm_proj_norm;
		}
		dL_dm_proj_norm_avg /= params.NUMBER_OF_POSES;*/

		float sX = 1.0f / (1.0f + expf(-params.GC[i].sX));
		float sY = 1.0f / (1.0f + expf(-params.GC[i].sY));
		float sZ = 1.0f / (1.0f + expf(-params.GC[i].sZ));

		// !!! !!! !!!
		/*if (
			(params.GC[i].alpha >= -log((1.0f / (1.0f / 255.0f)) - 1.0f)) &&
			(
				((params.epoch % 100) != 0) ||
				(
					(sqrtf((sX * sX) + (sY * sY) + (sZ * sZ)) <= 0.025f) &&
					(sqrtf((sX * sX) + (sY * sY) + (sZ * sZ)) >= 0.001f)
				)
			)
		) {*/

		if (
			((params.epoch % 100) != 0) ||
			(
				(params.GC[i].alpha >= -log((1.0f / (1.0f / 255.0f)) - 1.0f)) &&
				(sqrtf((sX * sX) + (sY * sY) + (sZ * sZ)) <= 0.025f) &&
				(sqrtf((sX * sX) + (sY * sY) + (sZ * sZ)) >= 0.001f)
			)
		) {
			if (((params.epoch - 1) % 10000) == 0)
				params.GC[i].alpha = -log((1.0f / (1.01f / 255.0f)));

			//dX = ((float *)params.dL_dparams_2)[i * 4];
			//dY = ((float *)params.dL_dparams_2)[(i * 4) + 1];
			//dZ = ((float *)params.dL_dparams_2)[(i * 4) + 2];

			/*float mX = params.GC[i].mX;
			float mY = params.GC[i].mY;
			float mZ = params.GC[i].mZ;

			float d = (0.5f * dev_params.w) / tanf(0.5f * dev_params.FOV);

			// !!! !!! !!!
			float pixel_delta;
			if ((params.epoch > 1) && ((params.loss / (3.0f * params.w * params.h)) < 0.01f))
				pixel_delta = (params.loss / (3.0f * params.w * params.h)) * 1.0f; // !!! !!! !!!
			else
				pixel_delta = 0.01f; // !!! !!! !!!
			// !!! !!! !!!

			float lambdaAbsMin = INFINITY;
			bool clip = false;
			for (int j = 0; j < params.NUMBER_OF_POSES; ++j) {
				float dX1 = (params.poses[j].Rx * dX) + (params.poses[j].Ry * dY) + (params.poses[j].Rz * dZ);
				float dY1 = (params.poses[j].Dx * dX) + (params.poses[j].Dy * dY) + (params.poses[j].Dz * dZ);
				float dZ1 = (params.poses[j].Fx * dX) + (params.poses[j].Fy * dY) + (params.poses[j].Fz * dZ);
			
				float mX1 = (params.poses[j].Rx * (mX - params.poses[j].Ox)) + (params.poses[j].Ry * (mY - params.poses[j].Oy)) + (params.poses[j].Rz * (mZ - params.poses[j].Oz));
				float mY1 = (params.poses[j].Dx * (mX - params.poses[j].Ox)) + (params.poses[j].Dy * (mY - params.poses[j].Oy)) + (params.poses[j].Dz * (mZ - params.poses[j].Oz));
				float mZ1 = (params.poses[j].Fx * (mX - params.poses[j].Ox)) + (params.poses[j].Fy * (mY - params.poses[j].Oy)) + (params.poses[j].Fz * (mZ - params.poses[j].Oz));
			
				float X1 = (mX1 * d) / mZ1;
				float X2 = ((mX1 - dX1) * d) / (mZ1 - dZ1);
				float Y1 = (mY1 * d) / mZ1;
				float Y2 = ((mY1 - dY1) * d) / (mZ1 - dZ1);
				if ((fabs(X2 - X1) > 1.0) || (fabs(Y2 - Y1) > 1.0)) clip = true;

				float lambda;
				if (fabs(dX1) >= fabs(dY1))
					lambda = (pixel_delta * mZ1 * mZ1) / ((d * ((mZ1 * dX1) - (mX1 * dZ1))) + (pixel_delta * mZ1 * dZ1));
				else
					lambda = (pixel_delta * mZ1 * mZ1) / ((d * ((mZ1 * dY1) - (mY1 * dZ1))) + (pixel_delta * mZ1 * dZ1));

				if ((mZ1 > 0.0f) && (fabs(lambda) < fabs(lambdaAbsMin)))
					lambdaAbsMin = fabs(lambda);
			}*/

			// *** *** ***
			
			//if (false) {

			//if ((params.epoch % 10000) == 0)
			//	params.GC[i].alpha = -log((1.0f / (10.0f / 255.0f)) - 1.0f);

			if (((params.epoch % 100) == 0) && (sqrtf((dX * dX) + (dY * dY) + (dZ * dZ)) >= 0.0005f)) {
				if (sqrtf((sX * sX) + (sY * sY) + (sZ * sZ)) > 0.01f) {
					sX = sX / 1.6f;
					sY = sY / 1.6f;
					sZ = sZ / 1.6f;

					// *** *** *** *** ***

					float Z1, Z2;
					RandomNormalFloat(Z1, Z2);
					float Z3, Z4;
					RandomNormalFloat(Z3, Z4);
					float Z5, Z6;
					RandomNormalFloat(Z5, Z6);

					// *** *** *** *** ***

					float m1X, m1Y, m1Z;
					RandomMultinormalFloat(params.GC[i], Z1, Z2, Z3, m1X, m1Y, m1Z);
					GC_new[numberOfGaussians] = params.GC[i];

					GC_new[numberOfGaussians].mX = m1X;
					GC_new[numberOfGaussians].mY = m1Y;
					GC_new[numberOfGaussians].mZ = m1Z;

					GC_new[numberOfGaussians].sX = -log((1.0f / sX) - 1.0f);
					GC_new[numberOfGaussians].sY = -log((1.0f / sY) - 1.0f);
					GC_new[numberOfGaussians].sZ = -log((1.0f / sZ) - 1.0f);

					((float *)m1_new)[4 * numberOfGaussians] = ((float *)params.m1)[4 * i];
					((float *)m1_new)[(4 * numberOfGaussians) + 1] = ((float *)params.m1)[(4 * i) + 1];
					((float *)m1_new)[(4 * numberOfGaussians) + 2] = ((float *)params.m1)[(4 * i) + 2];
					((float *)m1_new)[(4 * numberOfGaussians) + 3] = ((float *)params.m1)[(4 * i) + 3];

					((float *)m2_new)[4 * numberOfGaussians] = ((float *)params.m2)[4 * i];
					((float *)m2_new)[(4 * numberOfGaussians) + 1] = ((float *)params.m2)[(4 * i) + 1];
					((float *)m2_new)[(4 * numberOfGaussians) + 2] = ((float *)params.m2)[(4 * i) + 2];
					((float *)m2_new)[(4 * numberOfGaussians) + 3] = ((float *)params.m2)[(4 * i) + 3];

					((float *)m3_new)[4 * numberOfGaussians] = ((float *)params.m3)[4 * i];
					((float *)m3_new)[(4 * numberOfGaussians) + 1] = ((float *)params.m3)[(4 * i) + 1];
					((float *)m3_new)[(4 * numberOfGaussians) + 2] = ((float *)params.m3)[(4 * i) + 2];
					((float *)m3_new)[(4 * numberOfGaussians) + 3] = ((float *)params.m3)[(4 * i) + 3];

					((float *)m4_new)[2 * numberOfGaussians] = ((float *)params.m4)[2 * i];
					((float *)m4_new)[(2 * numberOfGaussians) + 1] = ((float *)params.m4)[(2 * i) + 1];

					((float *)v1_new)[4 * numberOfGaussians] = ((float *)params.v1)[4 * i];
					((float *)v1_new)[(4 * numberOfGaussians) + 1] = ((float *)params.v1)[(4 * i) + 1];
					((float *)v1_new)[(4 * numberOfGaussians) + 2] = ((float *)params.v1)[(4 * i) + 2];
					((float *)v1_new)[(4 * numberOfGaussians) + 3] = ((float *)params.v1)[(4 * i) + 3];

					((float *)v2_new)[4 * numberOfGaussians] = ((float *)params.v2)[4 * i];
					((float *)v2_new)[(4 * numberOfGaussians) + 1] = ((float *)params.v2)[(4 * i) + 1];
					((float *)v2_new)[(4 * numberOfGaussians) + 2] = ((float *)params.v2)[(4 * i) + 2];
					((float *)v2_new)[(4 * numberOfGaussians) + 3] = ((float *)params.v2)[(4 * i) + 3];

					((float *)v3_new)[4 * numberOfGaussians] = ((float *)params.v3)[4 * i];
					((float *)v3_new)[(4 * numberOfGaussians) + 1] = ((float *)params.v3)[(4 * i) + 1];
					((float *)v3_new)[(4 * numberOfGaussians) + 2] = ((float *)params.v3)[(4 * i) + 2];
					((float *)v3_new)[(4 * numberOfGaussians) + 3] = ((float *)params.v3)[(4 * i) + 3];

					((float *)v4_new)[2 * numberOfGaussians] = ((float *)params.v4)[2 * i];
					((float *)v4_new)[(2 * numberOfGaussians) + 1] = ((float *)params.v4)[(2 * i) + 1];
				
					++numberOfGaussians;

					// *** *** *** *** ***

					float m2X, m2Y, m2Z;
					RandomMultinormalFloat(params.GC[i], Z4, Z5, Z6, m2X, m2Y, m2Z);

					GC_new[numberOfGaussians] = params.GC[i];

					GC_new[numberOfGaussians].mX = m2X;
					GC_new[numberOfGaussians].mY = m2Y;
					GC_new[numberOfGaussians].mZ = m2Z;

					GC_new[numberOfGaussians].sX = -log((1.0f / sX) - 1.0f);
					GC_new[numberOfGaussians].sY = -log((1.0f / sY) - 1.0f);
					GC_new[numberOfGaussians].sZ = -log((1.0f / sZ) - 1.0f);

					((float *)m1_new)[4 * numberOfGaussians] = ((float *)params.m1)[4 * i];
					((float *)m1_new)[(4 * numberOfGaussians) + 1] = ((float *)params.m1)[(4 * i) + 1];
					((float *)m1_new)[(4 * numberOfGaussians) + 2] = ((float *)params.m1)[(4 * i) + 2];
					((float *)m1_new)[(4 * numberOfGaussians) + 3] = ((float *)params.m1)[(4 * i) + 3];

					((float *)m2_new)[4 * numberOfGaussians] = ((float *)params.m2)[4 * i];
					((float *)m2_new)[(4 * numberOfGaussians) + 1] = ((float *)params.m2)[(4 * i) + 1];
					((float *)m2_new)[(4 * numberOfGaussians) + 2] = ((float *)params.m2)[(4 * i) + 2];
					((float *)m2_new)[(4 * numberOfGaussians) + 3] = ((float *)params.m2)[(4 * i) + 3];

					((float *)m3_new)[4 * numberOfGaussians] = ((float *)params.m3)[4 * i];
					((float *)m3_new)[(4 * numberOfGaussians) + 1] = ((float *)params.m3)[(4 * i) + 1];
					((float *)m3_new)[(4 * numberOfGaussians) + 2] = ((float *)params.m3)[(4 * i) + 2];
					((float *)m3_new)[(4 * numberOfGaussians) + 3] = ((float *)params.m3)[(4 * i) + 3];

					((float *)m4_new)[2 * numberOfGaussians] = ((float *)params.m4)[2 * i];
					((float *)m4_new)[(2 * numberOfGaussians) + 1] = ((float *)params.m4)[(2 * i) + 1];

					((float *)v1_new)[4 * numberOfGaussians] = ((float *)params.v1)[4 * i];
					((float *)v1_new)[(4 * numberOfGaussians) + 1] = ((float *)params.v1)[(4 * i) + 1];
					((float *)v1_new)[(4 * numberOfGaussians) + 2] = ((float *)params.v1)[(4 * i) + 2];
					((float *)v1_new)[(4 * numberOfGaussians) + 3] = ((float *)params.v1)[(4 * i) + 3];

					((float *)v2_new)[4 * numberOfGaussians] = ((float *)params.v2)[4 * i];
					((float *)v2_new)[(4 * numberOfGaussians) + 1] = ((float *)params.v2)[(4 * i) + 1];
					((float *)v2_new)[(4 * numberOfGaussians) + 2] = ((float *)params.v2)[(4 * i) + 2];
					((float *)v2_new)[(4 * numberOfGaussians) + 3] = ((float *)params.v2)[(4 * i) + 3];

					((float *)v3_new)[4 * numberOfGaussians] = ((float *)params.v3)[4 * i];
					((float *)v3_new)[(4 * numberOfGaussians) + 1] = ((float *)params.v3)[(4 * i) + 1];
					((float *)v3_new)[(4 * numberOfGaussians) + 2] = ((float *)params.v3)[(4 * i) + 2];
					((float *)v3_new)[(4 * numberOfGaussians) + 3] = ((float *)params.v3)[(4 * i) + 3];

					((float *)v4_new)[2 * numberOfGaussians] = ((float *)params.v4)[2 * i];
					((float *)v4_new)[(2 * numberOfGaussians) + 1] = ((float *)params.v4)[(2 * i) + 1];

					++numberOfGaussians;	
				} else {
					GC_new[numberOfGaussians] = params.GC[i];

					GC_new[numberOfGaussians].sX = -log((1.0f / sX) - 1.0f);
					GC_new[numberOfGaussians].sY = -log((1.0f / sY) - 1.0f);
					GC_new[numberOfGaussians].sZ = -log((1.0f / sZ) - 1.0f);

					((float *)m1_new)[4 * numberOfGaussians] = ((float *)params.m1)[4 * i];
					((float *)m1_new)[(4 * numberOfGaussians) + 1] = ((float *)params.m1)[(4 * i) + 1];
					((float *)m1_new)[(4 * numberOfGaussians) + 2] = ((float *)params.m1)[(4 * i) + 2];
					((float *)m1_new)[(4 * numberOfGaussians) + 3] = ((float *)params.m1)[(4 * i) + 3];

					((float *)m2_new)[4 * numberOfGaussians] = ((float *)params.m2)[4 * i];
					((float *)m2_new)[(4 * numberOfGaussians) + 1] = ((float *)params.m2)[(4 * i) + 1];
					((float *)m2_new)[(4 * numberOfGaussians) + 2] = ((float *)params.m2)[(4 * i) + 2];
					((float *)m2_new)[(4 * numberOfGaussians) + 3] = ((float *)params.m2)[(4 * i) + 3];

					((float *)m3_new)[4 * numberOfGaussians] = ((float *)params.m3)[4 * i];
					((float *)m3_new)[(4 * numberOfGaussians) + 1] = ((float *)params.m3)[(4 * i) + 1];
					((float *)m3_new)[(4 * numberOfGaussians) + 2] = ((float *)params.m3)[(4 * i) + 2];
					((float *)m3_new)[(4 * numberOfGaussians) + 3] = ((float *)params.m3)[(4 * i) + 3];

					((float *)m4_new)[2 * numberOfGaussians] = ((float *)params.m4)[2 * i];
					((float *)m4_new)[(2 * numberOfGaussians) + 1] = ((float *)params.m4)[(2 * i) + 1];

					((float *)v1_new)[4 * numberOfGaussians] = ((float *)params.v1)[4 * i];
					((float *)v1_new)[(4 * numberOfGaussians) + 1] = ((float *)params.v1)[(4 * i) + 1];
					((float *)v1_new)[(4 * numberOfGaussians) + 2] = ((float *)params.v1)[(4 * i) + 2];
					((float *)v1_new)[(4 * numberOfGaussians) + 3] = ((float *)params.v1)[(4 * i) + 3];

					((float *)v2_new)[4 * numberOfGaussians] = ((float *)params.v2)[4 * i];
					((float *)v2_new)[(4 * numberOfGaussians) + 1] = ((float *)params.v2)[(4 * i) + 1];
					((float *)v2_new)[(4 * numberOfGaussians) + 2] = ((float *)params.v2)[(4 * i) + 2];
					((float *)v2_new)[(4 * numberOfGaussians) + 3] = ((float *)params.v2)[(4 * i) + 3];

					((float *)v3_new)[4 * numberOfGaussians] = ((float *)params.v3)[4 * i];
					((float *)v3_new)[(4 * numberOfGaussians) + 1] = ((float *)params.v3)[(4 * i) + 1];
					((float *)v3_new)[(4 * numberOfGaussians) + 2] = ((float *)params.v3)[(4 * i) + 2];
					((float *)v3_new)[(4 * numberOfGaussians) + 3] = ((float *)params.v3)[(4 * i) + 3];

					((float *)v4_new)[2 * numberOfGaussians] = ((float *)params.v4)[2 * i];
					((float *)v4_new)[(2 * numberOfGaussians) + 1] = ((float *)params.v4)[(2 * i) + 1];

					++numberOfGaussians;

					// *** *** *** *** ***

					GC_new[numberOfGaussians] = params.GC[i];

					GC_new[numberOfGaussians].mX -= dX;
					GC_new[numberOfGaussians].mY -= dY;
					GC_new[numberOfGaussians].mZ -= dZ;

					GC_new[numberOfGaussians].sX = -log((1.0f / sX) - 1.0f);
					GC_new[numberOfGaussians].sY = -log((1.0f / sY) - 1.0f);
					GC_new[numberOfGaussians].sZ = -log((1.0f / sZ) - 1.0f);

					((float *)m1_new)[4 * numberOfGaussians] = ((float *)params.m1)[4 * i];
					((float *)m1_new)[(4 * numberOfGaussians) + 1] = ((float *)params.m1)[(4 * i) + 1];
					((float *)m1_new)[(4 * numberOfGaussians) + 2] = ((float *)params.m1)[(4 * i) + 2];
					((float *)m1_new)[(4 * numberOfGaussians) + 3] = ((float *)params.m1)[(4 * i) + 3];

					((float *)m2_new)[4 * numberOfGaussians] = ((float *)params.m2)[4 * i];
					((float *)m2_new)[(4 * numberOfGaussians) + 1] = ((float *)params.m2)[(4 * i) + 1];
					((float *)m2_new)[(4 * numberOfGaussians) + 2] = ((float *)params.m2)[(4 * i) + 2];
					((float *)m2_new)[(4 * numberOfGaussians) + 3] = ((float *)params.m2)[(4 * i) + 3];

					((float *)m3_new)[4 * numberOfGaussians] = ((float *)params.m3)[4 * i];
					((float *)m3_new)[(4 * numberOfGaussians) + 1] = ((float *)params.m3)[(4 * i) + 1];
					((float *)m3_new)[(4 * numberOfGaussians) + 2] = ((float *)params.m3)[(4 * i) + 2];
					((float *)m3_new)[(4 * numberOfGaussians) + 3] = ((float *)params.m3)[(4 * i) + 3];

					((float *)m4_new)[2 * numberOfGaussians] = ((float *)params.m4)[2 * i];
					((float *)m4_new)[(2 * numberOfGaussians) + 1] = ((float *)params.m4)[(2 * i) + 1];

					((float *)v1_new)[4 * numberOfGaussians] = ((float *)params.v1)[4 * i];
					((float *)v1_new)[(4 * numberOfGaussians) + 1] = ((float *)params.v1)[(4 * i) + 1];
					((float *)v1_new)[(4 * numberOfGaussians) + 2] = ((float *)params.v1)[(4 * i) + 2];
					((float *)v1_new)[(4 * numberOfGaussians) + 3] = ((float *)params.v1)[(4 * i) + 3];

					((float *)v2_new)[4 * numberOfGaussians] = ((float *)params.v2)[4 * i];
					((float *)v2_new)[(4 * numberOfGaussians) + 1] = ((float *)params.v2)[(4 * i) + 1];
					((float *)v2_new)[(4 * numberOfGaussians) + 2] = ((float *)params.v2)[(4 * i) + 2];
					((float *)v2_new)[(4 * numberOfGaussians) + 3] = ((float *)params.v2)[(4 * i) + 3];

					((float *)v3_new)[4 * numberOfGaussians] = ((float *)params.v3)[4 * i];
					((float *)v3_new)[(4 * numberOfGaussians) + 1] = ((float *)params.v3)[(4 * i) + 1];
					((float *)v3_new)[(4 * numberOfGaussians) + 2] = ((float *)params.v3)[(4 * i) + 2];
					((float *)v3_new)[(4 * numberOfGaussians) + 3] = ((float *)params.v3)[(4 * i) + 3];

					((float *)v4_new)[2 * numberOfGaussians] = ((float *)params.v4)[2 * i];
					((float *)v4_new)[(2 * numberOfGaussians) + 1] = ((float *)params.v4)[(2 * i) + 1];

					++numberOfGaussians;	
				}
			} else {
				GC_new[numberOfGaussians] = params.GC[i];

				GC_new[numberOfGaussians].mX -= dX;
				GC_new[numberOfGaussians].mY -= dY;
				GC_new[numberOfGaussians].mZ -= dZ;

				GC_new[numberOfGaussians].sX = -log((1.0f / sX) - 1.0f);
				GC_new[numberOfGaussians].sY = -log((1.0f / sY) - 1.0f);
				GC_new[numberOfGaussians].sZ = -log((1.0f / sZ) - 1.0f);

				((float *)m1_new)[4 * numberOfGaussians] = ((float *)params.m1)[4 * i];
				((float *)m1_new)[(4 * numberOfGaussians) + 1] = ((float *)params.m1)[(4 * i) + 1];
				((float *)m1_new)[(4 * numberOfGaussians) + 2] = ((float *)params.m1)[(4 * i) + 2];
				((float *)m1_new)[(4 * numberOfGaussians) + 3] = ((float *)params.m1)[(4 * i) + 3];

				((float *)m2_new)[4 * numberOfGaussians] = ((float *)params.m2)[4 * i];
				((float *)m2_new)[(4 * numberOfGaussians) + 1] = ((float *)params.m2)[(4 * i) + 1];
				((float *)m2_new)[(4 * numberOfGaussians) + 2] = ((float *)params.m2)[(4 * i) + 2];
				((float *)m2_new)[(4 * numberOfGaussians) + 3] = ((float *)params.m2)[(4 * i) + 3];

				((float *)m3_new)[4 * numberOfGaussians] = ((float *)params.m3)[4 * i];
				((float *)m3_new)[(4 * numberOfGaussians) + 1] = ((float *)params.m3)[(4 * i) + 1];
				((float *)m3_new)[(4 * numberOfGaussians) + 2] = ((float *)params.m3)[(4 * i) + 2];
				((float *)m3_new)[(4 * numberOfGaussians) + 3] = ((float *)params.m3)[(4 * i) + 3];

				((float *)m4_new)[2 * numberOfGaussians] = ((float *)params.m4)[2 * i];
				((float *)m4_new)[(2 * numberOfGaussians) + 1] = ((float *)params.m4)[(2 * i) + 1];

				((float *)v1_new)[4 * numberOfGaussians] = ((float *)params.v1)[4 * i];
				((float *)v1_new)[(4 * numberOfGaussians) + 1] = ((float *)params.v1)[(4 * i) + 1];
				((float *)v1_new)[(4 * numberOfGaussians) + 2] = ((float *)params.v1)[(4 * i) + 2];
				((float *)v1_new)[(4 * numberOfGaussians) + 3] = ((float *)params.v1)[(4 * i) + 3];

				((float *)v2_new)[4 * numberOfGaussians] = ((float *)params.v2)[4 * i];
				((float *)v2_new)[(4 * numberOfGaussians) + 1] = ((float *)params.v2)[(4 * i) + 1];
				((float *)v2_new)[(4 * numberOfGaussians) + 2] = ((float *)params.v2)[(4 * i) + 2];
				((float *)v2_new)[(4 * numberOfGaussians) + 3] = ((float *)params.v2)[(4 * i) + 3];

				((float *)v3_new)[4 * numberOfGaussians] = ((float *)params.v3)[4 * i];
				((float *)v3_new)[(4 * numberOfGaussians) + 1] = ((float *)params.v3)[(4 * i) + 1];
				((float *)v3_new)[(4 * numberOfGaussians) + 2] = ((float *)params.v3)[(4 * i) + 2];
				((float *)v3_new)[(4 * numberOfGaussians) + 3] = ((float *)params.v3)[(4 * i) + 3];

				((float *)v4_new)[2 * numberOfGaussians] = ((float *)params.v4)[2 * i];
				((float *)v4_new)[(2 * numberOfGaussians) + 1] = ((float *)params.v4)[(2 * i) + 1];

				++numberOfGaussians;
			}

			/*if (lambdaAbsMin < INFINITY) {
				GC_new[numberOfGaussians].mX -= (dX * lambdaAbsMin);
				GC_new[numberOfGaussians].mY -= (dY * lambdaAbsMin);
				GC_new[numberOfGaussians].mZ -= (dZ * lambdaAbsMin);
			} else {
				GC_new[numberOfGaussians].mX -= dX;
				GC_new[numberOfGaussians].mY -= dY;
				GC_new[numberOfGaussians].mZ -= dZ;
			}*/

			/*if (dX < -1.0f) dX = -1.0f;
			if (dX > 1.0f) dX = 1.0f;
			if (dY < -1.0f) dY = -1.0f;
			if (dY > 1.0f) dY = 1.0f;
			if (dZ < -1.0f) dZ = -1.0f;
			if (dZ > 1.0f) dZ = 1.0f;*/

			/*float dL_dm_proj_norm_avg = 0.0;
			float dL_dm_proj_X = (params.poses[dev_params.poseNum].Rx * dX) + (params.poses[dev_params.poseNum].Ry * dY) + (params.poses[dev_params.poseNum].Rz * dZ);
			float dL_dm_proj_Y = (params.poses[dev_params.poseNum].Dx * dX) + (params.poses[dev_params.poseNum].Dy * dY) + (params.poses[dev_params.poseNum].Dz * dZ);
			float dL_dm_proj_norm = sqrtf((dL_dm_proj_X * dL_dm_proj_X) + (dL_dm_proj_Y * dL_dm_proj_Y));
			dL_dm_proj_norm_avg += dL_dm_proj_norm;*/
			//if (dL_dm_proj_norm_avg > 1.0f / fminf(params.w, params.h))
			//	dL_dm_proj_norm_avg = 1.0f / fminf(params.w, params.h);

			/*float dL_dm_proj_norm_avg = 0.0;
			for (int j = 0; j < params.NUMBER_OF_POSES; ++j) {
				float dL_dm_proj_X = (params.poses[j].Rx * dX) + (params.poses[j].Ry * dY) + (params.poses[j].Rz * dZ);
				float dL_dm_proj_Y = (params.poses[j].Dx * dX) + (params.poses[j].Dy * dY) + (params.poses[j].Dz * dZ);
				float dL_dm_proj_norm = sqrtf((dL_dm_proj_X * dL_dm_proj_X) + (dL_dm_proj_Y * dL_dm_proj_Y));
				dL_dm_proj_norm_avg += dL_dm_proj_norm;
			}
			dL_dm_proj_norm_avg /= params.NUMBER_OF_POSES;*/

			/*GC_new[numberOfGaussians] = params.GC[i];
			if (fabs(lambdaAbsMin) < INFINITY) {
				GC_new[numberOfGaussians].mX -= (dX * lambdaAbsMin);
				GC_new[numberOfGaussians].mY -= (dY * lambdaAbsMin);
				GC_new[numberOfGaussians].mZ -= (dZ * lambdaAbsMin);
			}*/

			/*if (dL_dm_proj_norm_avg >= 0.0001f) {
				GC_new[numberOfGaussians].mX -= (dX / (fminf(params.w, params.h) * dL_dm_proj_norm_avg));
				GC_new[numberOfGaussians].mY -= (dY / (fminf(params.w, params.h) * dL_dm_proj_norm_avg));
				GC_new[numberOfGaussians].mZ -= (dZ / (fminf(params.w, params.h) * dL_dm_proj_norm_avg));
			}*/
			
			/*if (dX <= 0.0f) GC_new[numberOfGaussians].mX += 0.001f;
			else
				GC_new[numberOfGaussians].mX -= 0.001f;
			if (dY <= 0.0f) GC_new[numberOfGaussians].mY += 0.001f;
			else
				GC_new[numberOfGaussians].mY -= 0.001f;
			if (dZ <= 0.0f) GC_new[numberOfGaussians].mZ += 0.001f;
			else
				GC_new[numberOfGaussians].mZ -= 0.001f;*/

			/*if (((params.epoch % 25) != 0) || (dL_dm_proj_norm_avg <= 1.0f / fminf(params.w, params.h))) {
				if (dL_dm_proj_norm_avg > 1.0f / fminf(params.w, params.h)) {
					dX = dX * (1.0f / (fminf(params.w, params.h) * dL_dm_proj_norm_avg));
					dY = dY * (1.0f / (fminf(params.w, params.h) * dL_dm_proj_norm_avg));
					dZ = dZ * (1.0f / (fminf(params.w, params.h) * dL_dm_proj_norm_avg));
				}

				GC_new[numberOfGaussians].mX -= dX;
				GC_new[numberOfGaussians].mY -= dY;
				GC_new[numberOfGaussians].mZ -= dZ;
				++numberOfGaussians;
			} else {
				float sX = 1.0f / (1.0f + expf(-params.GC[i].sX));
				float sY = 1.0f / (1.0f + expf(-params.GC[i].sY));
				float sZ = 1.0f / (1.0f + expf(-params.GC[i].sZ));
				float s_norm = sqrtf((sX * sX) + (sY * sY) + (sZ * sZ));

				GC_new[numberOfGaussians + 1] = params.GC[i];
				GC_new[numberOfGaussians + 1].mX -= dX;
				GC_new[numberOfGaussians + 1].mY -= dY;
				GC_new[numberOfGaussians + 1].mZ -= dZ;

				if (s_norm > 10.0f / fminf(params.w, params.h)) {
					sX = sX / 1.6;
					sY = sY / 1.6;
					sZ = sZ / 1.6;

					GC_new[numberOfGaussians].sX = -log((1.0f / sX) - 1.0f);
					GC_new[numberOfGaussians].sY = -log((1.0f / sY) - 1.0f);
					GC_new[numberOfGaussians].sZ = -log((1.0f / sZ) - 1.0f);

					GC_new[numberOfGaussians + 1].sX = -log((1.0f / sX) - 1.0f);
					GC_new[numberOfGaussians + 1].sY = -log((1.0f / sY) - 1.0f);
					GC_new[numberOfGaussians + 1].sZ = -log((1.0f / sZ) - 1.0f);
				}

				numberOfGaussians += 2;
			}*/
		}
		// !!! !!! !!!
	}

	free(params.m1);
	free(params.m2);
	free(params.m3);
	free(params.m4);

	free(params.v1);
	free(params.v2);
	free(params.v3);
	free(params.v4);

	params.m1 = m1_new;
	params.m2 = m2_new;
	params.m3 = m3_new;
	params.m4 = m4_new;

	params.v1 = v1_new;
	params.v2 = v2_new;
	params.v3 = v3_new;
	params.v4 = v4_new;

	// *** *** *** *** ***

	// !!! !!! !!!
	params.numberOfGaussians = numberOfGaussians;
	free(params.GC);
	params.GC = GC_new;
	// !!! !!! !!!

	return true;
Error:
	return false;
}

// *** *** *** *** ***

__device__ void dev_DeinterleaveBits(unsigned code, unsigned& x, unsigned& y) {
	x = code & 1431655765U;
	x = (x | (x >> 1)) & 858993459U;
	x = (x | (x >> 2)) & 252645135U;
	x = (x | (x >> 4)) & 16711935U;
	x = (x | (x >> 8)) & 65535U;

	y = (code >> 1) & 1431655765U;
	y = (y | (y >> 1)) & 858993459U;
	y = (y | (y >> 2)) & 252645135U;
	y = (y | (y >> 4)) & 16711935U;
	y = (y | (y >> 8)) & 65535U;
}

// *** *** *** *** ***

__device__ static bool IntersectRayAABB(
	float Ox, float Oy, float Oz,
	float vx, float vy, float vz,
	float lB, float rB, float uB, float dB, float bB, float fB,
	float& tHit1, float& tHit2,
	float tStart
) {
	float vxInv = __frcp_rn(vx);
	float vyInv = __frcp_rn(vy);
	float vzInv = __frcp_rn(vz);
	float t1 = (lB - Ox) * vxInv;
	float t2 = (rB - Ox) * vxInv;
	float t3 = (uB - Oy) * vyInv;
	float t4 = (dB - Oy) * vyInv;
	float t5 = (bB - Oz) * vzInv;
	float t6 = (fB - Oz) * vzInv;
	tHit1 = fmaxf(fmaxf(fminf(t1, t2), fminf(t3, t4)), fminf(t5, t6));
	tHit2 = fminf(fminf(fmaxf(t1, t2), fmaxf(t3, t4)), fmaxf(t5, t6));
	return (tHit2 > tStart) && (tHit2 >= tHit1);
}

__device__ static bool IntersectRayAABBDouble(
	double Ox, double Oy, double Oz,
	double vx, double vy, double vz,
	float lB, float rB, float uB, float dB, float bB, float fB,
	double& tHit1, double& tHit2,
	double tStart
) {
	double vxInv = 1.0 / vx;
	double vyInv = 1.0 / vy;
	double vzInv = 1.0 / vz;
	double t1 = (lB - Ox) * vxInv;
	double t2 = (rB - Ox) * vxInv;
	double t3 = (uB - Oy) * vyInv;
	double t4 = (dB - Oy) * vyInv;
	double t5 = (bB - Oz) * vzInv;
	double t6 = (fB - Oz) * vzInv;
	tHit1 = fmax(fmax(fmin(t1, t2), fmin(t3, t4)), fmin(t5, t6));
	tHit2 = fmin(fmin(fmax(t1, t2), fmax(t3, t4)), fmax(t5, t6));
	return (tHit2 > tStart) && (tHit2 >= tHit1);
}

// *** *** *** *** ***

__device__ static bool IntersectRayGaussianComponent(
	float Ox, float Oy, float Oz,
	float vx, float vy, float vz,
	float mX, float mY, float mZ,
	float sX, float sY, float sZ,
	float qr, float qi, float qj, float qk,
	float& tHit1, float& tHit2,
	float tStart
) {
	Ox = Ox - mX;
	Oy = Oy - mY;
	Oz = Oz - mZ;

	float aa = qr * qr;
	float bb = qi * qi;
	float cc = qj * qj;
	float dd = qk * qk;
	float s = 2.0f * __frcp_rn(aa + bb + cc + dd);

	float bs = qi * s;  float cs = qj * s;  float ds = qk * s;
	float ab = qr * bs; float ac = qr * cs; float ad = qr * ds;
	bb = bb * s;        float bc = qi * cs; float bd = qi * ds;
	cc = cc * s;        float cd = qj * ds;       dd = dd * s;

	float R11 = 1.0f - cc - dd;
	float R12 = bc - ad;
	float R13 = bd + ac;

	float R21 = bc + ad;
	float R22 = 1.0f - bb - dd;
	float R23 = cd - ab;

	float R31 = bd - ac;
	float R32 = cd + ab;
	float R33 = 1.0f - bb - cc;

	// *** *** ***

	// 1
	/*float sXInv = 1.0f + __expf(-sX);
	float sYInv = 1.0f + __expf(-sY);
	float sZInv = 1.0f + __expf(-sZ);
	float Ox_prim = ((R11 * Ox * sXInv) + (R21 * Oy * sYInv) + (R31 * Oz * sZInv));
	float vx_prim = ((R11 * vx * sXInv) + (R21 * vy * sYInv) + (R31 * vz * sZInv));
	float Oy_prim = ((R12 * Ox * sXInv) + (R22 * Oy * sYInv) + (R32 * Oz * sZInv));
	float vy_prim = ((R12 * vx * sXInv) + (R22 * vy * sYInv) + (R32 * vz * sZInv));
	float Oz_prim = ((R13 * Ox * sXInv) + (R23 * Oy * sYInv) + (R33 * Oz * sZInv));
	float vz_prim = ((R13 * vx * sXInv) + (R23 * vy * sYInv) + (R33 * vz * sZInv));*/

	// *** *** ***

	// 2
	float sXInv = 1.0f + __expf(-sX);
	float Ox_prim = ((R11 * Ox) + (R21 * Oy) + (R31 * Oz)) * sXInv;
	float vx_prim = ((R11 * vx) + (R21 * vy) + (R31 * vz)) * sXInv;

	float sYInv = 1.0f + __expf(-sY);
	float Oy_prim = ((R12 * Ox) + (R22 * Oy) + (R32 * Oz)) * sYInv;
	float vy_prim = ((R12 * vx) + (R22 * vy) + (R32 * vz)) * sYInv;

	float sZInv = 1.0f + __expf(-sZ);
	float Oz_prim = ((R13 * Ox) + (R23 * Oy) + (R33 * Oz)) * sZInv;
	float vz_prim = ((R13 * vx) + (R23 * vy) + (R33 * vz)) * sZInv;

	// *** *** ***

	float a = (vx_prim * vx_prim) + (vy_prim * vy_prim) + (vz_prim * vz_prim);
	float b = 2.0f * ((Ox_prim * vx_prim) + (Oy_prim * vy_prim) + (Oz_prim * vz_prim));
	float c = ((Ox_prim * Ox_prim) + (Oy_prim * Oy_prim) + (Oz_prim * Oz_prim)) - 11.3449f;
	float delta = (b * b) - (4.0f * a * c);
	if (delta >= 0.0f) {
		float tmp1 = __fsqrt_rn(delta);
		float tmp2 = __frcp_rn(2.0f * a);

		tHit1 = (-b - tmp1) * tmp2;
		tHit2 = (-b + tmp1) * tmp2;
		if (tHit1 > tStart)
			return true;
		else
			return false;
	}
	else
		return false;
}

__device__ static bool IntersectRayGaussianComponentDouble(
	double Ox, double Oy, double Oz,
	double vx, double vy, double vz,
	float mX, float mY, float mZ,
	float sX, float sY, float sZ,
	float qr, float qi, float qj, float qk,
	double& tHit1, double& tHit2,
	double tStart
) {
	Ox = Ox - mX;
	Oy = Oy - mY;
	Oz = Oz - mZ;

	double aa = ((double)qr) * qr;
	double bb = ((double)qi) * qi;
	double cc = ((double)qj) * qj;
	double dd = ((double)qk) * qk;
	double s = 2.0 / (aa + bb + cc + dd);

	double bs = qi * s;  double cs = qj * s;  double ds = qk * s;
	double ab = qr * bs; double ac = qr * cs; double ad = qr * ds;
	bb = bb * s;         double bc = qi * cs; double bd = qi * ds;
	cc = cc * s;         double cd = qj * ds;        dd = dd * s;

	double R11 = 1.0 - cc - dd;
	double R12 = bc - ad;
	double R13 = bd + ac;

	double R21 = bc + ad;
	double R22 = 1.0 - bb - dd;
	double R23 = cd - ab;

	double R31 = bd - ac;
	double R32 = cd + ab;
	double R33 = 1.0 - bb - cc;

	double sXInv = 1.0 + exp((double)-sX);
	double Ox_prim = ((R11 * Ox) + (R21 * Oy) + (R31 * Oz)) * sXInv;
	double vx_prim = ((R11 * vx) + (R21 * vy) + (R31 * vz)) * sXInv;

	double sYInv = 1.0 + exp((double)-sY);
	double Oy_prim = ((R12 * Ox) + (R22 * Oy) + (R32 * Oz)) * sYInv;
	double vy_prim = ((R12 * vx) + (R22 * vy) + (R32 * vz)) * sYInv;

	double sZInv = 1.0 + exp((double)-sZ);
	double Oz_prim = ((R13 * Ox) + (R23 * Oy) + (R33 * Oz)) * sZInv;
	double vz_prim = ((R13 * vx) + (R23 * vy) + (R33 * vz)) * sZInv;

	double a = (vx_prim * vx_prim) + (vy_prim * vy_prim) + (vz_prim * vz_prim);
	double b = 2.0 * ((Ox_prim * vx_prim) + (Oy_prim * vy_prim) + (Oz_prim * vz_prim));
	double c = ((Ox_prim * Ox_prim) + (Oy_prim * Oy_prim) + (Oz_prim * Oz_prim)) - 11.3449;
	double delta = (b * b) - (4.0 * a * c);
	if (delta >= 0.0) {
		double tmp1 = sqrt(delta);
		double tmp2 = 0.5 / a;

		tHit1 = (-b - tmp1) * tmp2;
		tHit2 = (-b + tmp1) * tmp2;
		if (tHit1 > tStart)
			return true;
		else
			return false;
	}
	else
		return false;
}

// *** *** *** *** ***

// modulacja przezroczystości równa 1 - odległość Mahalanobisa do kwadratu
__global__ void RenderCUDAKernel(SCUDARenderParams dev_params) {
	int GaussIndices[MAX_RAY_LENGTH];

	int tid = (blockIdx.x * blockDim.x) + threadIdx.x;

	// 2
	unsigned i, j;
	dev_DeinterleaveBits(tid, j, i);
	if ((j < dev_params.w) && (i < dev_params.h)) {
		// NIE TRZEBA. WYPEŁNIAMY TABLICĘ INDEKSÓW BAJTAMI 255, CO DAJE -1

		// !!! !!! !!!
		//for (int ii = 0; ii < MAX_RAY_LENGTH; ++ii)
			//dev_params.dump[(ii * dev_params.w * dev_params.h) + (i * dev_params.w) + j] = -1;
		// !!! !!! !!!

#ifndef USE_DOUBLE_PRECISION
		float wInv = __frcp_rn(dev_params.w);
		float hInv = __frcp_rn(dev_params.h);
#else
		double wInv = 1.0 / dev_params.w;
		double hInv = 1.0 / dev_params.h;
#endif

		REAL dX, dY, dZ;
		if (dev_params.h <= dev_params.w) {
			dX = dev_params.w * hInv * (-0.5f + ((j + 0.5f) * wInv));
			dY = -0.5f + ((i + 0.5f) * hInv);
		}
		else {
			dX = -0.5f + ((j + 0.5f) * wInv);
			dY = dev_params.h * wInv * (-0.5f + ((i + 0.5f) * hInv));
		}
#ifndef USE_DOUBLE_PRECISION
		dZ = 0.5f * __frcp_rn(__tanf(0.5f * dev_params.FOV));
#else
		dZ = 0.5 / (tan(0.5 * dev_params.FOV));
#endif

		REAL vx = (dev_params.Rx * dX) + (dev_params.Dx * dY) + (dev_params.Fx * dZ);
		REAL vy = (dev_params.Ry * dX) + (dev_params.Dy * dY) + (dev_params.Fy * dZ);
		REAL vz = (dev_params.Rz * dX) + (dev_params.Dz * dY) + (dev_params.Fz * dZ);

		unsigned long long nodeTrail = 1ULL;
		unsigned long long notCompletedMask = 1ULL;

		REAL tStart = 0.0f;

		REAL T = 1.0f;

		REAL R = 0.0f;
		REAL G = 0.0f;
		REAL B = 0.0f;

		int GaussInd;

		int numberOfGaussians = 0;
		do {
			REAL tHit = INFINITY;
			GaussInd = -1;

			unsigned long long nodeTrailTmp = nodeTrail;
			unsigned long long notCompletedMaskTmp = notCompletedMask;
			unsigned long long nodeInd = (nodeTrailTmp + dev_params.d[nodeTrailTmp & (dev_params.D - 1)]) % dev_params.H;

			bool firstIntersection = true;

			do {
				float4 tree_1 = dev_params.tree_part_1[nodeInd];
				float4 tree_2 = dev_params.tree_part_2[nodeInd];

#ifndef USE_DOUBLE_PRECISION
				float t1, t2;
				bool intersection = IntersectRayAABB(
					dev_params.Ox, dev_params.Oy, dev_params.Oz,
					vx, vy, vz,
					tree_1.z, tree_1.w, tree_2.x, tree_2.y, tree_2.z, tree_2.w,
					t1, t2,
					tStart
				);
#else
				double t1, t2;
				bool intersection = IntersectRayAABBDouble(
					dev_params.Ox, dev_params.Oy, dev_params.Oz,
					vx, vy, vz,
					tree_1.z, tree_1.w, tree_2.x, tree_2.y, tree_2.z, tree_2.w,
					t1, t2,
					tStart
				);
#endif

				if ((intersection) && (t1 < tHit)) {
					unsigned tmp = __float_as_uint(tree_1.x);
					int info = tmp >> 30;
					int lNode = tmp & 1073741823;
					int rNode = __float_as_uint(tree_1.y);

					switch (info) {
						case 0: {
							nodeTrailTmp = (nodeTrailTmp << 1) + (vx <= 0.0f);
							break;
						}
						case 1: {
							nodeTrailTmp = (nodeTrailTmp << 1) + (vy <= 0.0f);
							break;
						}
						case 2: {
							nodeTrailTmp = (nodeTrailTmp << 1) + (vz <= 0.0f);
							break;
						}
					}

					if (info != 3) {
						notCompletedMaskTmp = (notCompletedMaskTmp << 1) + 1;
						if ((nodeTrailTmp & 1) == 0)
							nodeInd = lNode;
						else
							nodeInd = rNode;
					}
					else {
						for (int i = lNode; i <= rNode; ++i) {
#ifndef USE_DOUBLE_PRECISION
							float tHit1, tHit2;
#else
							double tHit1, tHit2;
#endif

							float4 GC_2 = dev_params.GC_part_2[i];
							float4 GC_3 = dev_params.GC_part_3[i];
							float2 GC_4 = dev_params.GC_part_4[i];

#ifndef USE_DOUBLE_PRECISION
							bool intersection = IntersectRayGaussianComponent(
								dev_params.Ox, dev_params.Oy, dev_params.Oz,
								vx, vy, vz,
								GC_2.x, GC_2.y, GC_2.z,
								GC_2.w, GC_3.x, GC_3.y,
								GC_3.z, GC_3.w, GC_4.x, GC_4.y,
								tHit1, tHit2,
								tStart
							);
#else
							bool intersection = IntersectRayGaussianComponentDouble(
								dev_params.Ox, dev_params.Oy, dev_params.Oz,
								vx, vy, vz,
								GC_2.x, GC_2.y, GC_2.z,
								GC_2.w, GC_3.x, GC_3.y,
								GC_3.z, GC_3.w, GC_4.x, GC_4.y,
								tHit1, tHit2,
								tStart
							);
#endif
							if ((intersection) && (tHit1 <= tHit)) {
								tHit = tHit1;
								GaussInd = i;
								if (firstIntersection) {
									nodeTrail = nodeTrailTmp;
									notCompletedMask = notCompletedMaskTmp;
									firstIntersection = false;
								}
							}
						}

						unsigned long index = __ffsll(notCompletedMaskTmp) - 1;
						nodeTrailTmp = (nodeTrailTmp >> index) ^ 1;
						notCompletedMaskTmp = (notCompletedMaskTmp >> index) ^ 1;

						nodeInd = (nodeTrailTmp + dev_params.d[nodeTrailTmp & (dev_params.D - 1)]) % dev_params.H;
					}
				}
				else {
					unsigned long index = __ffsll(notCompletedMaskTmp) - 1;
					nodeTrailTmp = (nodeTrailTmp >> index) ^ 1;
					notCompletedMaskTmp = (notCompletedMaskTmp >> index) ^ 1;

					nodeInd = (nodeTrailTmp + dev_params.d[nodeTrailTmp & (dev_params.D - 1)]) % dev_params.H;
				}
			} while (notCompletedMaskTmp != 0);

			if (GaussInd != -1) {
				float4 GC_1 = dev_params.GC_part_1[GaussInd];

				// *** *** ***

				float4 GC_2 = dev_params.GC_part_2[GaussInd];
				float4 GC_3 = dev_params.GC_part_3[GaussInd];
				float2 GC_4 = dev_params.GC_part_4[GaussInd];

				REAL Ox = ((REAL)dev_params.Ox) - GC_2.x;
				REAL Oy = ((REAL)dev_params.Oy) - GC_2.y;
				REAL Oz = ((REAL)dev_params.Oz) - GC_2.z;

				REAL aa = ((REAL)GC_3.z) * GC_3.z;
				REAL bb = ((REAL)GC_3.w) * GC_3.w;
				REAL cc = ((REAL)GC_4.x) * GC_4.x;
				REAL dd = ((REAL)GC_4.y) * GC_4.y;
#ifndef USE_DOUBLE_PRECISION
				float s = 2.0f * __frcp_rn(aa + bb + cc + dd);
#else
				double s = 2.0 / (aa + bb + cc + dd);
#endif

				REAL bs = GC_3.w * s;  REAL cs = GC_4.x * s;  REAL ds = GC_4.y * s;
				REAL ab = GC_3.z * bs; REAL ac = GC_3.z * cs; REAL ad = GC_3.z * ds;
				bb = bb * s;           REAL bc = GC_3.w * cs; REAL bd = GC_3.w * ds;
				cc = cc * s;           REAL cd = GC_4.x * ds;      dd = dd * s;

				REAL R11 = 1.0f - cc - dd;
				REAL R12 = bc - ad;
				REAL R13 = bd + ac;

				REAL R21 = bc + ad;
				REAL R22 = 1.0f - bb - dd;
				REAL R23 = cd - ab;

				REAL R31 = bd - ac;
				REAL R32 = cd + ab;
				REAL R33 = 1.0f - bb - cc;

#ifndef USE_DOUBLE_PRECISION
				float sXInv = 1.0f + __expf(-GC_2.w);
#else
				double sXInv = 1.0 + exp((double)-GC_2.w);
#endif
				REAL Ox_prim = ((R11 * Ox) + (R21 * Oy) + (R31 * Oz)) * sXInv;
				REAL vx_prim = ((R11 * vx) + (R21 * vy) + (R31 * vz)) * sXInv;

#ifndef USE_DOUBLE_PRECISION
				float sYInv = 1.0f + __expf(-GC_3.x);
#else
				double sYInv = 1.0 + exp((double)-GC_3.x);
#endif
				REAL Oy_prim = ((R12 * Ox) + (R22 * Oy) + (R32 * Oz)) * sYInv;
				REAL vy_prim = ((R12 * vx) + (R22 * vy) + (R32 * vz)) * sYInv;

#ifndef USE_DOUBLE_PRECISION
				float sZInv = 1.0f + __expf(-GC_3.y);
#else
				double sZInv = 1.0 + exp((double)-GC_3.y);
#endif
				REAL Oz_prim = ((R13 * Ox) + (R23 * Oy) + (R33 * Oz)) * sZInv;
				REAL vz_prim = ((R13 * vx) + (R23 * vy) + (R33 * vz)) * sZInv;

#ifndef USE_DOUBLE_PRECISION
				float v_dot_v_inv = __frcp_rn((vx_prim * vx_prim) + (vy_prim * vy_prim) + (vz_prim * vz_prim));
#else
				double v_dot_v_inv = 1.0 / ((vx_prim * vx_prim) + (vy_prim * vy_prim) + (vz_prim * vz_prim));
#endif
				REAL O_dot_O = (Ox_prim * Ox_prim) + (Oy_prim * Oy_prim) + (Oz_prim * Oz_prim);
				REAL v_dot_O = (vx_prim * Ox_prim) + (vy_prim * Oy_prim) + (vz_prim * Oz_prim);

#ifndef USE_DOUBLE_PRECISION
				float alpha = fmaxf(1.0f + (((v_dot_O * v_dot_O * v_dot_v_inv) - O_dot_O) / 11.3449f), 0.0f);
				alpha = alpha * __frcp_rn(1.0f + __expf(-GC_1.w));
#else
				double alpha = fmax(1.0 + (((v_dot_O * v_dot_O * v_dot_v_inv) - O_dot_O) / 11.3449), 0.0);
				alpha = alpha / (1.0 + exp((double)-GC_1.w));
#endif

				// *** *** ***

				REAL tmp1 = alpha * T;
				REAL tmp2 = 1.0f - alpha;

				R = R + (GC_1.x * tmp1);
				G = G + (GC_1.y * tmp1);
				B = B + (GC_1.z * tmp1);

				T = T * tmp2;

				tStart = tHit;

				// !!! !!! !!!
				dev_params.dump[(numberOfGaussians * dev_params.w * dev_params.h) + (i * dev_params.w) + j] = GaussInd;
				GaussIndices[numberOfGaussians] = GaussInd;
				// !!! !!! !!!

				++numberOfGaussians;
			}
		} while ((GaussInd != -1) && (T >= 1.0 / 255.0) && (numberOfGaussians < MAX_RAY_LENGTH));
	
		dev_params.bitmap[(i * dev_params.w) + j] = (((int)(R * 255.0f)) << 16) + (((int)(G * 255.0f)) << 8) + ((int)(B * 255.0f));
		dev_params.dev_bitmap_out_R[(i * (dev_params.w + 11 - 1)) + j] = R; // !!! !!! !!!
		dev_params.dev_bitmap_out_G[(i * (dev_params.w + 11 - 1)) + j] = G; // !!! !!! !!!
		dev_params.dev_bitmap_out_B[(i * (dev_params.w + 11 - 1)) + j] = B; // !!! !!! !!!
	
		if (dev_params.render) return; // !!! !!! !!!

		/**********************/
		/* COMPUTING GRADIENT */
		/**********************/
		/*const REAL tmp0 = 1.0 / 255.0; // !!! !!! !!!
		int color_ref = dev_params.bitmap_ref[(dev_params.poseNum * (dev_params.w * dev_params.h)) + (i * dev_params.w) + j];
		REAL B_ref = (color_ref & 255) * tmp0;
		color_ref = color_ref >> 8;
		REAL G_ref = (color_ref & 255) * tmp0;
		color_ref = color_ref >> 8;
		REAL R_ref = color_ref * tmp0;

		atomicAdd(
			(double *)dev_params.output_params,
			((R - R_ref) * (R - R_ref)) + ((G - G_ref) * (G - G_ref)) + ((B - B_ref) * (B - B_ref))
		);

		// *** *** *** *** ***

		const REAL tmp1 = 1.0 / (2.0 * 11.3449); // !!! !!! !!!
		REAL tmp2 = 1.0 / 11.3449; // !!! !!! !!!

		REAL dR_dalpha;
		REAL dG_dalpha;
		REAL dB_dalpha;

		REAL d_dR_dalpha = (2.0f * (R - R_ref));
		REAL d_dG_dalpha = (2.0f * (G - G_ref));
		REAL d_dB_dalpha = (2.0f * (B - B_ref));

		REAL alpha_prev, alpha_next;
		REAL R_Gauss_prev, G_Gauss_prev, B_Gauss_prev;

		for (int k = 0; k < numberOfGaussians; ++k) {
			GaussInd = GaussIndices[k];

			float4 GC_1 = dev_params.GC_part_1[GaussInd];
			float4 GC_2 = dev_params.GC_part_2[GaussInd];
			float4 GC_3 = dev_params.GC_part_3[GaussInd];
			float2 GC_4 = dev_params.GC_part_4[GaussInd];

			REAL aa = ((REAL)GC_3.z) * GC_3.z;
			REAL bb = ((REAL)GC_3.w) * GC_3.w;
			REAL cc = ((REAL)GC_4.x) * GC_4.x;
			REAL dd = ((REAL)GC_4.y) * GC_4.y;
#ifndef USE_DOUBLE_PRECISION
			float s = 2.0f * __frcp_rn(aa + bb + cc + dd);
#else
			double s = 2.0 / (aa + bb + cc + dd);
#endif

			REAL bs = GC_3.w * s;  REAL cs = GC_4.x * s;  REAL ds = GC_4.y * s;
			REAL ab = GC_3.z * bs; REAL ac = GC_3.z * cs; REAL ad = GC_3.z * ds;
			bb = bb * s;           REAL bc = GC_3.w * cs; REAL bd = GC_3.w * ds;
			cc = cc * s;           REAL cd = GC_4.x * ds;      dd = dd * s;

			REAL R11 = 1.0f - cc - dd;
			REAL R12 = bc - ad;
			REAL R13 = bd + ac;

			REAL R21 = bc + ad;
			REAL R22 = 1.0f - bb - dd;
			REAL R23 = cd - ab;

			REAL R31 = bd - ac;
			REAL R32 = cd + ab;
			REAL R33 = 1.0f - bb - cc;

			REAL Ox = ((REAL)dev_params.Ox) - GC_2.x;
			REAL Oy = ((REAL)dev_params.Oy) - GC_2.y;
			REAL Oz = ((REAL)dev_params.Oz) - GC_2.z;

#ifndef USE_DOUBLE_PRECISION
			float sXInvMinusOne = __expf(-GC_2.w); // !!! !!! !!!
			float sYInvMinusOne = __expf(-GC_3.x); // !!! !!! !!!
			float sZInvMinusOne = __expf(-GC_3.y); // !!! !!! !!!
#else
			double sXInvMinusOne = exp(-GC_2.w); // !!! !!! !!!
			double sYInvMinusOne = exp(-GC_3.x); // !!! !!! !!!
			double sZInvMinusOne = exp(-GC_3.y); // !!! !!! !!!
#endif

			REAL sXInv = 1.0f + sXInvMinusOne;
			REAL Ox_prim = ((R11 * Ox) + (R21 * Oy) + (R31 * Oz)) * sXInv;
			REAL vx_prim = ((R11 * vx) + (R21 * vy) + (R31 * vz)) * sXInv;

			REAL sYInv = 1.0f + sYInvMinusOne;
			REAL Oy_prim = ((R12 * Ox) + (R22 * Oy) + (R32 * Oz)) * sYInv;
			REAL vy_prim = ((R12 * vx) + (R22 * vy) + (R32 * vz)) * sYInv;

			REAL sZInv = 1.0f + sZInvMinusOne;
			REAL Oz_prim = ((R13 * Ox) + (R23 * Oy) + (R33 * Oz)) * sZInv;
			REAL vz_prim = ((R13 * vx) + (R23 * vy) + (R33 * vz)) * sZInv;

			REAL v_dot_v = (vx_prim * vx_prim) + (vy_prim * vy_prim) + (vz_prim * vz_prim);
			REAL O_dot_O = (Ox_prim * Ox_prim) + (Oy_prim * Oy_prim) + (Oz_prim * Oz_prim);
			REAL v_dot_O = (vx_prim * Ox_prim) + (vy_prim * Oy_prim) + (vz_prim * Oz_prim);
#ifndef USE_DOUBLE_PRECISION
			float tmp3 = __frcp_rn(v_dot_v);
#else
			double tmp3 = 1.0 / v_dot_v;
#endif
			REAL tmp4 = v_dot_O * tmp3;
#ifndef USE_DOUBLE_PRECISION
			float tmp5 = __frcp_rn(1.0f + __expf(-GC_1.w));
#else
			double tmp5 = 1.0 / (1.0 + exp((double)-GC_1.w));
#endif
			REAL tmp6;
			
#ifndef USE_DOUBLE_PRECISION
			alpha_next = tmp5 * fmaxf(1.0f + (((v_dot_O * tmp4) - O_dot_O) / 11.3449f), 0.0f); // !!! !!! !!!
#else
			alpha_next = tmp5 * fmax(1.0 + (((v_dot_O * tmp4) - O_dot_O) / 11.3449), 0.0); // !!! !!! !!!
#endif

			if (k > 0) {
				d_dR_dalpha = d_dR_dalpha * (1.0f - alpha_prev);
				d_dG_dalpha = d_dG_dalpha * (1.0f - alpha_prev);
				d_dB_dalpha = d_dB_dalpha * (1.0f - alpha_prev);

				dR_dalpha = dR_dalpha + ((GC_1.x - R_Gauss_prev) * d_dR_dalpha);
				dG_dalpha = dG_dalpha + ((GC_1.y - G_Gauss_prev) * d_dG_dalpha);
				dB_dalpha = dB_dalpha + ((GC_1.z - B_Gauss_prev) * d_dB_dalpha);

#ifndef USE_DOUBLE_PRECISION
				tmp6 = (dR_dalpha + dG_dalpha + dB_dalpha) * __frcp_rn(1.0f - alpha_next);
#else
				tmp6 = (dR_dalpha + dG_dalpha + dB_dalpha) / (1.0 - alpha_next);
#endif
			} else {
#ifndef USE_DOUBLE_PRECISION
				dR_dalpha = (GC_1.x - R) * __frcp_rn(1.0f - alpha_next);
				dG_dalpha = (GC_1.y - G) * __frcp_rn(1.0f - alpha_next);
				dB_dalpha = (GC_1.z - B) * __frcp_rn(1.0f - alpha_next);
#else
				dR_dalpha = (GC_1.x - R) / (1.0 - alpha_next);
				dG_dalpha = (GC_1.y - G) / (1.0 - alpha_next);
				dB_dalpha = (GC_1.z - B) / (1.0 - alpha_next);
#endif

				dR_dalpha = d_dR_dalpha * dR_dalpha;
				dG_dalpha = d_dG_dalpha * dG_dalpha;
				dB_dalpha = d_dB_dalpha * dB_dalpha;

				tmp6 = (dR_dalpha + dG_dalpha + dB_dalpha);
			}

			REAL vecx_tmp = Ox_prim - (vx_prim * tmp4);
			REAL vecy_tmp = Oy_prim - (vy_prim * tmp4);
			REAL vecz_tmp = Oz_prim - (vz_prim * tmp4);

			// *** *** *** *** ***

			REAL dL_dparam;

			// *** *** *** *** ***

			// dL_d[R, G, B, alpha]
			dL_dparam = d_dR_dalpha * alpha_next;
			atomicAdd(((REAL *)dev_params.dL_dparams_1) + (GaussInd * 4), dL_dparam);
			dL_dparam = d_dG_dalpha * alpha_next;
			atomicAdd(((REAL *)dev_params.dL_dparams_1) + (GaussInd * 4) + 1, dL_dparam);
			dL_dparam = d_dB_dalpha * alpha_next;
			atomicAdd(((REAL *)dev_params.dL_dparams_1) + (GaussInd * 4) + 2, dL_dparam);
			dL_dparam = tmp6 * alpha_next * (1.0f - tmp5);
			atomicAdd(((REAL *)dev_params.dL_dparams_1) + (GaussInd * 4) + 3, dL_dparam);
		
			// *** *** *** *** ***

#ifndef USE_DOUBLE_PRECISION
			tmp6 = tmp6 * tmp5 * (2.0f / 11.3449f);
#else
			tmp6 = tmp6 * tmp5 * (2.0 / 11.3449);
#endif

			// *** *** *** *** ***

			// dL_dmX
			REAL dot_product = ((vecx_tmp * R11 * sXInv) + (vecy_tmp * R12 * sYInv) + (vecz_tmp * R13 * sZInv));
			dL_dparam = tmp6 * dot_product;
			atomicAdd(((REAL *)dev_params.dL_dparams_2) + (GaussInd * 4), dL_dparam);

			// dL_dmY
			dot_product = ((vecx_tmp * R21 * sXInv) + (vecy_tmp * R22 * sYInv) + (vecz_tmp * R23 * sZInv));
			dL_dparam = tmp6 * dot_product;
			atomicAdd(((REAL *)dev_params.dL_dparams_2) + (GaussInd * 4) + 1, dL_dparam);

			// dL_dmZ
			dot_product = ((vecx_tmp * R31 * sXInv) + (vecy_tmp * R32 * sYInv) + (vecz_tmp * R33 * sZInv));
			dL_dparam = tmp6 * dot_product;
			atomicAdd(((REAL *)dev_params.dL_dparams_2) + (GaussInd * 4) + 2, dL_dparam);

			// *** *** *** *** ***

			// dL_dsX
			// tmp4 * vec_tmp.x * ((-vx)*R11*sXInvMinusOne + (-vy)*R21*sXInvMinusOne + (-vz)*R31*sXInvMinusOne) -
			// vec_tmp.x * (-(Ox-mx)*R11*sXInvMinusOne + -(Oy-my)*R21*sXInvMinusOne + -(Oz-mz)*R31*sXInvMinusOne) =

			// vec_tmp.x * sXInvMinusOne * (((Ox-mx)*R11) + ((Oy-my)*R21) + ((Oz-mz)*R31)) -
			// tmp4 * vec_tmp.x * sXInvMinusOne * ((vx*R11) + (vy*R21) + (vz*R31))

			// dot_product_1 = ((Ox-mx)*R11) + ((Oy-my)*R21) + ((Oz-mz)*R31)
			// dot_product_2 = (vx*R11) + (vy*R21) + (vz*R31)
			// vec_tmp.x * sXInvMinusOne * (dot_product_1 - (tmp4 * dot_product_2))

			REAL dot_product_1 = (Ox * R11) + (Oy * R21) + (Oz * R31);
			REAL dot_product_2 = (vx * R11) + (vy * R21) + (vz * R31);
			dL_dparam = vecx_tmp * sXInvMinusOne * (dot_product_1 - (tmp4 * dot_product_2));
			dL_dparam = tmp6 * dL_dparam;
			atomicAdd(((REAL *)dev_params.dL_dparams_2) + (GaussInd * 4) + 3, dL_dparam);

			// dL_dsY
			dot_product_1 = (Ox * R12) + (Oy * R22) + (Oz * R32);
			dot_product_2 = (vx * R12) + (vy * R22) + (vz * R32);
			dL_dparam = vecy_tmp * sYInvMinusOne * (dot_product_1 - (tmp4 * dot_product_2));
			dL_dparam = tmp6 * dL_dparam;
			atomicAdd(((REAL *)dev_params.dL_dparams_3) + (GaussInd * 4), dL_dparam);

			// dL_dsZ
			dot_product_1 = (Ox * R13) + (Oy * R23) + (Oz * R33);
			dot_product_2 = (vx * R13) + (vy * R23) + (vz * R33);
			dL_dparam = vecz_tmp * sZInvMinusOne * (dot_product_1 - (tmp4 * dot_product_2));
			dL_dparam = tmp6 * dL_dparam;
			atomicAdd(((REAL *)dev_params.dL_dparams_3) + (GaussInd * 4) + 1, dL_dparam);

			// *** *** *** *** ***

			REAL tmp7 = s * (1.0f - (aa * s));
			REAL tmp8 = s * (1.0f - bb);
			REAL tmp9 = s * (1.0f - cc);
			REAL tmp10 = s * (1.0f - dd);

			REAL tmp11 = -(ab * cd); // -abcd(s^2)

#ifndef USE_DOUBLE_PRECISION
			REAL a_inv = __frcp_rn(GC_3.z);
			REAL b_inv = __frcp_rn(GC_3.w);
			REAL c_inv = __frcp_rn(GC_4.x);
			REAL d_inv = __frcp_rn(GC_4.y);
#else
			REAL a_inv = 1.0 / GC_3.z;
			REAL b_inv = 1.0 / GC_3.w;
			REAL c_inv = 1.0 / GC_4.x;
			REAL d_inv = 1.0 / GC_4.y;
#endif

			// !!! !!! !!!
			vecx_tmp = vecx_tmp * sXInv;
			vecy_tmp = vecy_tmp * sYInv;
			vecz_tmp = vecz_tmp * sZInv;
			// !!! !!! !!!

			// *** *** *** *** ***

			// dL_da
			REAL dR11_da = GC_3.z * (tmp7 + tmp8);
			REAL dR12_da = (d_inv * tmp11) - (GC_4.y * tmp7);
			REAL dR13_da = (GC_4.x * tmp7) + (c_inv * tmp11);

			REAL dR21_da = (GC_4.y * tmp7) + (d_inv * tmp11);
			REAL dR22_da = GC_3.z * (tmp7 + tmp9);
			REAL dR23_da = (b_inv * tmp11) - (GC_3.w * tmp7);

			REAL dR31_da = (c_inv * tmp11) - (GC_4.x * tmp7);
			REAL dR32_da = (GC_3.w * tmp7) + (b_inv * tmp11);
			REAL dR33_da = GC_3.z * (tmp7 + tmp10);

			REAL vecx_tmp2 = ((vecx_tmp * dR11_da) + (vecy_tmp * dR12_da) + (vecz_tmp * dR13_da));
			REAL vecy_tmp2 = ((vecx_tmp * dR21_da) + (vecy_tmp * dR22_da) + (vecz_tmp * dR23_da));
			REAL vecz_tmp2 = ((vecx_tmp * dR31_da) + (vecy_tmp * dR32_da) + (vecz_tmp * dR33_da));

			dot_product_1 = (vecx_tmp2 * Ox) + (vecy_tmp2 * Oy) + (vecz_tmp2 * Oz);
			dot_product_2 = (vecx_tmp2 * vx) + (vecy_tmp2 * vy) + (vecz_tmp2 * vz);

			dL_dparam = (tmp4 * dot_product_2) - dot_product_1;
			dL_dparam = tmp6 * dL_dparam; 
			atomicAdd(((REAL *)dev_params.dL_dparams_3) + (GaussInd * 4) + 2, dL_dparam);

			// *** *** *** *** ***

			// dL_db
			dR11_da = GC_3.w * (tmp8 + tmp7);
			dR12_da = (GC_4.x * tmp8) - (c_inv * tmp11);
			dR13_da = (GC_4.y * tmp8) + (d_inv * tmp11);

			dR21_da = (GC_4.x * tmp8) + (c_inv * tmp11);
			dR22_da = -GC_3.w * (tmp8 + tmp10);
			dR23_da = (a_inv * tmp11) - (GC_3.z * tmp8);

			dR31_da = (GC_4.y * tmp8) - (d_inv * tmp11);
			dR32_da = (GC_3.z * tmp8) + (a_inv * tmp11);
			dR33_da = -GC_3.w * (tmp8 + tmp9);

			vecx_tmp2 = ((vecx_tmp * dR11_da) + (vecy_tmp * dR12_da) + (vecz_tmp * dR13_da));
			vecy_tmp2 = ((vecx_tmp * dR21_da) + (vecy_tmp * dR22_da) + (vecz_tmp * dR23_da));
			vecz_tmp2 = ((vecx_tmp * dR31_da) + (vecy_tmp * dR32_da) + (vecz_tmp * dR33_da));

			dot_product_1 = (vecx_tmp2 * Ox) + (vecy_tmp2 * Oy) + (vecz_tmp2 * Oz);
			dot_product_2 = (vecx_tmp2 * vx) + (vecy_tmp2 * vy) + (vecz_tmp2 * vz);

			dL_dparam = (tmp4 * dot_product_2) - dot_product_1;
			dL_dparam = tmp6 * dL_dparam; 
			atomicAdd(((REAL *)dev_params.dL_dparams_3) + (GaussInd * 4) + 3, dL_dparam);

			// *** *** *** *** ***

			// dL_dc
			dR11_da = -GC_4.x * (tmp9 + tmp10);
			dR12_da = (GC_3.w * tmp9) - (b_inv * tmp11);
			dR13_da = (GC_3.z * tmp9) + (a_inv * tmp11);

			dR21_da = (GC_3.w * tmp9) + (b_inv * tmp11);
			dR22_da = GC_4.x * (tmp9 + tmp7);
			dR23_da = (GC_4.y * tmp9) - (d_inv * tmp11);

			dR31_da = (a_inv * tmp11) - (GC_3.z * tmp9);
			dR32_da = (GC_4.y * tmp9) + (d_inv * tmp11);
			dR33_da = -GC_4.x * (tmp9 + tmp8);

			vecx_tmp2 = ((vecx_tmp * dR11_da) + (vecy_tmp * dR12_da) + (vecz_tmp * dR13_da));
			vecy_tmp2 = ((vecx_tmp * dR21_da) + (vecy_tmp * dR22_da) + (vecz_tmp * dR23_da));
			vecz_tmp2 = ((vecx_tmp * dR31_da) + (vecy_tmp * dR32_da) + (vecz_tmp * dR33_da));

			dot_product_1 = (vecx_tmp2 * Ox) + (vecy_tmp2 * Oy) + (vecz_tmp2 * Oz);
			dot_product_2 = (vecx_tmp2 * vx) + (vecy_tmp2 * vy) + (vecz_tmp2 * vz);

			dL_dparam = (tmp4 * dot_product_2) - dot_product_1;
			dL_dparam = tmp6 * dL_dparam;
			atomicAdd(((REAL *)dev_params.dL_dparams_4) + (GaussInd * 2), dL_dparam);

			// *** *** *** *** ***

			// dL_dd
			dR11_da = -GC_4.y * (tmp10 + tmp9);
			dR12_da = (a_inv * tmp11) - (GC_3.z * tmp10);
			dR13_da = (GC_3.w * tmp10) + (b_inv * tmp11);

			dR21_da = (GC_3.z * tmp10) + (a_inv * tmp11);
			dR22_da = -GC_4.y * (tmp10 + tmp8);
			dR23_da = (GC_4.x * tmp10) - (c_inv * tmp11);

			dR31_da = (GC_3.w * tmp10) - (b_inv * tmp11);
			dR32_da = (GC_4.x * tmp10) + (c_inv * tmp11);
			dR33_da = GC_4.y * (tmp10 + tmp7);

			vecx_tmp2 = ((vecx_tmp * dR11_da) + (vecy_tmp * dR12_da) + (vecz_tmp * dR13_da));
			vecy_tmp2 = ((vecx_tmp * dR21_da) + (vecy_tmp * dR22_da) + (vecz_tmp * dR23_da));
			vecz_tmp2 = ((vecx_tmp * dR31_da) + (vecy_tmp * dR32_da) + (vecz_tmp * dR33_da));

			dot_product_1 = (vecx_tmp2 * Ox) + (vecy_tmp2 * Oy) + (vecz_tmp2 * Oz);
			dot_product_2 = (vecx_tmp2 * vx) + (vecy_tmp2 * vy) + (vecz_tmp2 * vz);

			dL_dparam = (tmp4 * dot_product_2) - dot_product_1;
			dL_dparam = tmp6 * dL_dparam;
			atomicAdd(((REAL *)dev_params.dL_dparams_4) + (GaussInd * 2) + 1, dL_dparam);

			// *** *** *** *** ***

			if (k == 0) {
				dR_dalpha = dR_dalpha * (1.0f - alpha_next);
				dG_dalpha = dG_dalpha * (1.0f - alpha_next);
				dB_dalpha = dB_dalpha * (1.0f - alpha_next);
			}

			alpha_prev = alpha_next;
			R_Gauss_prev = GC_1.x;
			G_Gauss_prev = GC_1.y;
			B_Gauss_prev = GC_1.z;
		}*/
	}
}

// *** *** *** *** ***

__global__ void ComputeGradient(SCUDARenderParams dev_params) {
	int tid = (blockIdx.x * blockDim.x) + threadIdx.x;

	//unsigned i, j;
	//dev_DeinterleaveBits(tid, j, i);

	unsigned i = tid / dev_params.w;
	unsigned j = tid % dev_params.w;

	if ((j < dev_params.w) && (i < dev_params.h)) {
#ifndef USE_DOUBLE_PRECISION
		float wInv = __frcp_rn(dev_params.w);
		float hInv = __frcp_rn(dev_params.h);
#else
		double wInv = 1.0 / dev_params.w;
		double hInv = 1.0 / dev_params.h;
#endif

		REAL dX, dY, dZ;
		if (dev_params.h <= dev_params.w) {
			dX = dev_params.w * hInv * (-0.5f + ((j + 0.5f) * wInv));
			dY = -0.5f + ((i + 0.5f) * hInv);
		}
		else {
			dX = -0.5f + ((j + 0.5f) * wInv);
			dY = dev_params.h * wInv * (-0.5f + ((i + 0.5f) * hInv));
		}
#ifndef USE_DOUBLE_PRECISION
		dZ = 0.5f * __frcp_rn(__tanf(0.5f * dev_params.FOV));
#else
		dZ = 0.5 / (tan(0.5 * dev_params.FOV));
#endif

		REAL vx = (dev_params.Rx * dX) + (dev_params.Dx * dY) + (dev_params.Fx * dZ);
		REAL vy = (dev_params.Ry * dX) + (dev_params.Dy * dY) + (dev_params.Fy * dZ);
		REAL vz = (dev_params.Rz * dX) + (dev_params.Dz * dY) + (dev_params.Fz * dZ);

		//************************************************************************************************
		
		REAL R = dev_params.dev_bitmap_out_R[(i * (dev_params.w + 11 - 1)) + j]; // !!! !!! !!!
		REAL G = dev_params.dev_bitmap_out_G[(i * (dev_params.w + 11 - 1)) + j]; // !!! !!! !!!
		REAL B = dev_params.dev_bitmap_out_B[(i * (dev_params.w + 11 - 1)) + j]; // !!! !!! !!!

		const REAL tmp0 = 1.0 / 255.0; // !!! !!! !!!
		int color_ref = dev_params.bitmap_ref[(dev_params.poseNum * (dev_params.w * dev_params.h)) + (i * dev_params.w) + j];
		REAL B_ref = (color_ref & 255) * tmp0;
		color_ref = color_ref >> 8;
		REAL G_ref = (color_ref & 255) * tmp0;
		color_ref = color_ref >> 8;
		REAL R_ref = color_ref * tmp0;

		// Compute loss
		atomicAdd(
			(double *)dev_params.output_params,
			((R - R_ref) * (R - R_ref)) + ((G - G_ref) * (G - G_ref)) + ((B - B_ref) * (B - B_ref))
		);

		const REAL tmp1 = 1.0 / (2.0 * 11.3449); // !!! !!! !!!
		REAL tmp2 = 1.0 / 11.3449; // !!! !!! !!!

		REAL dR_dalpha = ((REAL)0.0);
		REAL dG_dalpha = ((REAL)0.0);
		REAL dB_dalpha = ((REAL)0.0);

		// OLD
		//REAL d_dR_dalpha = (2.0 * (R - R_ref));
		//REAL d_dG_dalpha = (2.0 * (G - G_ref));
		//REAL d_dB_dalpha = (2.0 * (B - B_ref));

		// NEW
		REAL d_dR_dalpha = ((1.0 - 0.2) * ((2.0 * (R - R_ref)) / (3.0 * dev_params.w * dev_params.h))) + (0.2 * dev_params.dev_mu_bitmap_out_R[((i + 10) * (dev_params.w + 11 - 1)) + (j + 10)]);
		REAL d_dG_dalpha = ((1.0 - 0.2) * ((2.0 * (G - G_ref)) / (3.0 * dev_params.w * dev_params.h))) + (0.2 * dev_params.dev_mu_bitmap_out_G[((i + 10) * (dev_params.w + 11 - 1)) + (j + 10)]);
		REAL d_dB_dalpha = ((1.0 - 0.2) * ((2.0 * (B - B_ref)) / (3.0 * dev_params.w * dev_params.h))) + (0.2 * dev_params.dev_mu_bitmap_out_B[((i + 10) * (dev_params.w + 11 - 1)) + (j + 10)]);

		//REAL d_dR_dalpha = (2.0 * (R - R_ref)) / (3.0 * dev_params.w * dev_params.h);
		//REAL d_dG_dalpha = (2.0 * (G - G_ref)) / (3.0 * dev_params.w * dev_params.h);
		//REAL d_dB_dalpha = (2.0 * (B - B_ref)) / (3.0 * dev_params.w * dev_params.h);

		REAL alpha_prev = ((REAL)0.0);
		REAL alpha_next;

		REAL R_Gauss_prev = R;
		REAL G_Gauss_prev = G;
		REAL B_Gauss_prev = B;

		//******************************************************************************************

		for (int k = 0; k < MAX_RAY_LENGTH; ++k) {
			int GaussInd = dev_params.dump[(k * dev_params.w * dev_params.h) + (i * dev_params.w) + j]; // !!! !!! !!!
			if (GaussInd == -1) return; // !!! !!! !!!

			//**************************************************************************************

			float4 GC_1 = dev_params.GC_part_1[GaussInd];
			float4 GC_2 = dev_params.GC_part_2[GaussInd];
			float4 GC_3 = dev_params.GC_part_3[GaussInd];
			float2 GC_4 = dev_params.GC_part_4[GaussInd];

			REAL aa = ((REAL)GC_3.z) * GC_3.z;
			REAL bb = ((REAL)GC_3.w) * GC_3.w;
			REAL cc = ((REAL)GC_4.x) * GC_4.x;
			REAL dd = ((REAL)GC_4.y) * GC_4.y;
#ifndef USE_DOUBLE_PRECISION
			float s = 2.0f * __frcp_rn(aa + bb + cc + dd);
#else
			double s = 2.0 / (aa + bb + cc + dd);
#endif

			REAL bs = GC_3.w * s;  REAL cs = GC_4.x * s;  REAL ds = GC_4.y * s;
			REAL ab = GC_3.z * bs; REAL ac = GC_3.z * cs; REAL ad = GC_3.z * ds;
			bb = bb * s;           REAL bc = GC_3.w * cs; REAL bd = GC_3.w * ds;
			cc = cc * s;           REAL cd = GC_4.x * ds;      dd = dd * s;

			REAL R11 = 1.0f - cc - dd;
			REAL R12 = bc - ad;
			REAL R13 = bd + ac;

			REAL R21 = bc + ad;
			REAL R22 = 1.0f - bb - dd;
			REAL R23 = cd - ab;

			REAL R31 = bd - ac;
			REAL R32 = cd + ab;
			REAL R33 = 1.0f - bb - cc;

			REAL Ox = ((REAL)dev_params.Ox) - GC_2.x;
			REAL Oy = ((REAL)dev_params.Oy) - GC_2.y;
			REAL Oz = ((REAL)dev_params.Oz) - GC_2.z;

#ifndef USE_DOUBLE_PRECISION
			float sXInvMinusOne = __expf(-GC_2.w); // !!! !!! !!!
			float sYInvMinusOne = __expf(-GC_3.x); // !!! !!! !!!
			float sZInvMinusOne = __expf(-GC_3.y); // !!! !!! !!!
#else
			double sXInvMinusOne = exp(-GC_2.w); // !!! !!! !!!
			double sYInvMinusOne = exp(-GC_3.x); // !!! !!! !!!
			double sZInvMinusOne = exp(-GC_3.y); // !!! !!! !!!
#endif

			REAL sXInv = 1.0f + sXInvMinusOne;
			REAL Ox_prim = ((R11 * Ox) + (R21 * Oy) + (R31 * Oz)) * sXInv;
			REAL vx_prim = ((R11 * vx) + (R21 * vy) + (R31 * vz)) * sXInv;

			REAL sYInv = 1.0f + sYInvMinusOne;
			REAL Oy_prim = ((R12 * Ox) + (R22 * Oy) + (R32 * Oz)) * sYInv;
			REAL vy_prim = ((R12 * vx) + (R22 * vy) + (R32 * vz)) * sYInv;

			REAL sZInv = 1.0f + sZInvMinusOne;
			REAL Oz_prim = ((R13 * Ox) + (R23 * Oy) + (R33 * Oz)) * sZInv;
			REAL vz_prim = ((R13 * vx) + (R23 * vy) + (R33 * vz)) * sZInv;

			REAL v_dot_v = (vx_prim * vx_prim) + (vy_prim * vy_prim) + (vz_prim * vz_prim);
			REAL O_dot_O = (Ox_prim * Ox_prim) + (Oy_prim * Oy_prim) + (Oz_prim * Oz_prim);
			REAL v_dot_O = (vx_prim * Ox_prim) + (vy_prim * Oy_prim) + (vz_prim * Oz_prim);
#ifndef USE_DOUBLE_PRECISION
			float tmp3 = __frcp_rn(v_dot_v);
#else
			double tmp3 = 1.0 / v_dot_v;
#endif
			REAL tmp4 = v_dot_O * tmp3;
#ifndef USE_DOUBLE_PRECISION
			float tmp5 = __frcp_rn(1.0f + __expf(-GC_1.w));
#else
			double tmp5 = 1.0 / (1.0 + exp((double)-GC_1.w));
#endif
			REAL tmp6;

#ifndef USE_DOUBLE_PRECISION
			alpha_next = tmp5 * fmaxf(1.0f + (((v_dot_O * tmp4) - O_dot_O) / 11.3449f), 0.0f); // !!! !!! !!!
#else
			alpha_next = tmp5 * fmax(1.0 + (((v_dot_O * tmp4) - O_dot_O) / 11.3449), 0.0); // !!! !!! !!!
#endif

			d_dR_dalpha = d_dR_dalpha * (1.0f - alpha_prev);
			d_dG_dalpha = d_dG_dalpha * (1.0f - alpha_prev);
			d_dB_dalpha = d_dB_dalpha * (1.0f - alpha_prev);

			dR_dalpha = dR_dalpha + ((GC_1.x - R_Gauss_prev) * d_dR_dalpha);
			dG_dalpha = dG_dalpha + ((GC_1.y - G_Gauss_prev) * d_dG_dalpha);
			dB_dalpha = dB_dalpha + ((GC_1.z - B_Gauss_prev) * d_dB_dalpha);

#ifndef USE_DOUBLE_PRECISION
			tmp6 = (dR_dalpha + dG_dalpha + dB_dalpha) * __frcp_rn(1.0f - alpha_next);
#else
			tmp6 = (dR_dalpha + dG_dalpha + dB_dalpha) / (1.0 - alpha_next);
#endif

			REAL vecx_tmp = Ox_prim - (vx_prim * tmp4);
			REAL vecy_tmp = Oy_prim - (vy_prim * tmp4);
			REAL vecz_tmp = Oz_prim - (vz_prim * tmp4);

			// *** *** *** *** ***

			REAL dL_dparam;

			// *** *** *** *** ***

			// dL_d[R, G, B, alpha]
			dL_dparam = d_dR_dalpha * alpha_next;
			atomicAdd(((REAL *)dev_params.dL_dparams_1) + (GaussInd * 4), dL_dparam);
			dL_dparam = d_dG_dalpha * alpha_next;
			atomicAdd(((REAL *)dev_params.dL_dparams_1) + (GaussInd * 4) + 1, dL_dparam);
			dL_dparam = d_dB_dalpha * alpha_next;
			atomicAdd(((REAL *)dev_params.dL_dparams_1) + (GaussInd * 4) + 2, dL_dparam);
			dL_dparam = tmp6 * alpha_next * (1.0f - tmp5);
			atomicAdd(((REAL *)dev_params.dL_dparams_1) + (GaussInd * 4) + 3, dL_dparam);

			// *** *** *** *** ***

			#ifndef USE_DOUBLE_PRECISION
			tmp6 = tmp6 * tmp5 * (2.0f / 11.3449f);
			#else
			tmp6 = tmp6 * tmp5 * (2.0 / 11.3449);
			#endif

			// *** *** *** *** ***

			// dL_dmX
			REAL dot_product = ((vecx_tmp * R11 * sXInv) + (vecy_tmp * R12 * sYInv) + (vecz_tmp * R13 * sZInv));
			dL_dparam = tmp6 * dot_product;
			atomicAdd(((REAL *)dev_params.dL_dparams_2) + (GaussInd * 4), dL_dparam);

			// dL_dmY
			dot_product = ((vecx_tmp * R21 * sXInv) + (vecy_tmp * R22 * sYInv) + (vecz_tmp * R23 * sZInv));
			dL_dparam = tmp6 * dot_product;
			atomicAdd(((REAL *)dev_params.dL_dparams_2) + (GaussInd * 4) + 1, dL_dparam);

			// dL_dmZ
			dot_product = ((vecx_tmp * R31 * sXInv) + (vecy_tmp * R32 * sYInv) + (vecz_tmp * R33 * sZInv));
			dL_dparam = tmp6 * dot_product;
			atomicAdd(((REAL *)dev_params.dL_dparams_2) + (GaussInd * 4) + 2, dL_dparam);

			// *** *** *** *** ***

			// dL_dsX
			// tmp4 * vec_tmp.x * ((-vx)*R11*sXInvMinusOne + (-vy)*R21*sXInvMinusOne + (-vz)*R31*sXInvMinusOne) -
			// vec_tmp.x * (-(Ox-mx)*R11*sXInvMinusOne + -(Oy-my)*R21*sXInvMinusOne + -(Oz-mz)*R31*sXInvMinusOne) =

			// vec_tmp.x * sXInvMinusOne * (((Ox-mx)*R11) + ((Oy-my)*R21) + ((Oz-mz)*R31)) -
			// tmp4 * vec_tmp.x * sXInvMinusOne * ((vx*R11) + (vy*R21) + (vz*R31))

			// dot_product_1 = ((Ox-mx)*R11) + ((Oy-my)*R21) + ((Oz-mz)*R31)
			// dot_product_2 = (vx*R11) + (vy*R21) + (vz*R31)
			// vec_tmp.x * sXInvMinusOne * (dot_product_1 - (tmp4 * dot_product_2))

			REAL dot_product_1 = (Ox * R11) + (Oy * R21) + (Oz * R31);
			REAL dot_product_2 = (vx * R11) + (vy * R21) + (vz * R31);
			dL_dparam = vecx_tmp * sXInvMinusOne * (dot_product_1 - (tmp4 * dot_product_2));
			dL_dparam = tmp6 * dL_dparam;
			atomicAdd(((REAL *)dev_params.dL_dparams_2) + (GaussInd * 4) + 3, dL_dparam);

			// dL_dsY
			dot_product_1 = (Ox * R12) + (Oy * R22) + (Oz * R32);
			dot_product_2 = (vx * R12) + (vy * R22) + (vz * R32);
			dL_dparam = vecy_tmp * sYInvMinusOne * (dot_product_1 - (tmp4 * dot_product_2));
			dL_dparam = tmp6 * dL_dparam;
			atomicAdd(((REAL *)dev_params.dL_dparams_3) + (GaussInd * 4), dL_dparam);

			// dL_dsZ
			dot_product_1 = (Ox * R13) + (Oy * R23) + (Oz * R33);
			dot_product_2 = (vx * R13) + (vy * R23) + (vz * R33);
			dL_dparam = vecz_tmp * sZInvMinusOne * (dot_product_1 - (tmp4 * dot_product_2));
			dL_dparam = tmp6 * dL_dparam;
			atomicAdd(((REAL *)dev_params.dL_dparams_3) + (GaussInd * 4) + 1, dL_dparam);

			// *** *** *** *** ***

			REAL tmp7 = s * (1.0f - (aa * s));
			REAL tmp8 = s * (1.0f - bb);
			REAL tmp9 = s * (1.0f - cc);
			REAL tmp10 = s * (1.0f - dd);

			REAL tmp11 = -(ab * cd); // -abcd(s^2)

#ifndef USE_DOUBLE_PRECISION
			REAL a_inv = __frcp_rn(GC_3.z);
			REAL b_inv = __frcp_rn(GC_3.w);
			REAL c_inv = __frcp_rn(GC_4.x);
			REAL d_inv = __frcp_rn(GC_4.y);
#else
			REAL a_inv = 1.0 / GC_3.z;
			REAL b_inv = 1.0 / GC_3.w;
			REAL c_inv = 1.0 / GC_4.x;
			REAL d_inv = 1.0 / GC_4.y;
#endif

			// !!! !!! !!!
			vecx_tmp = vecx_tmp * sXInv;
			vecy_tmp = vecy_tmp * sYInv;
			vecz_tmp = vecz_tmp * sZInv;
			// !!! !!! !!!

			// *** *** *** *** ***

			// dL_da
			REAL dR11_da = GC_3.z * (tmp7 + tmp8);
			REAL dR12_da = (d_inv * tmp11) - (GC_4.y * tmp7);
			REAL dR13_da = (GC_4.x * tmp7) + (c_inv * tmp11);

			REAL dR21_da = (GC_4.y * tmp7) + (d_inv * tmp11);
			REAL dR22_da = GC_3.z * (tmp7 + tmp9);
			REAL dR23_da = (b_inv * tmp11) - (GC_3.w * tmp7);

			REAL dR31_da = (c_inv * tmp11) - (GC_4.x * tmp7);
			REAL dR32_da = (GC_3.w * tmp7) + (b_inv * tmp11);
			REAL dR33_da = GC_3.z * (tmp7 + tmp10);

			REAL vecx_tmp2 = ((vecx_tmp * dR11_da) + (vecy_tmp * dR12_da) + (vecz_tmp * dR13_da));
			REAL vecy_tmp2 = ((vecx_tmp * dR21_da) + (vecy_tmp * dR22_da) + (vecz_tmp * dR23_da));
			REAL vecz_tmp2 = ((vecx_tmp * dR31_da) + (vecy_tmp * dR32_da) + (vecz_tmp * dR33_da));

			dot_product_1 = (vecx_tmp2 * Ox) + (vecy_tmp2 * Oy) + (vecz_tmp2 * Oz);
			dot_product_2 = (vecx_tmp2 * vx) + (vecy_tmp2 * vy) + (vecz_tmp2 * vz);

			dL_dparam = (tmp4 * dot_product_2) - dot_product_1;
			dL_dparam = tmp6 * dL_dparam; 
			atomicAdd(((REAL *)dev_params.dL_dparams_3) + (GaussInd * 4) + 2, dL_dparam);

			// *** *** *** *** ***

			// dL_db
			dR11_da = GC_3.w * (tmp8 + tmp7);
			dR12_da = (GC_4.x * tmp8) - (c_inv * tmp11);
			dR13_da = (GC_4.y * tmp8) + (d_inv * tmp11);

			dR21_da = (GC_4.x * tmp8) + (c_inv * tmp11);
			dR22_da = -GC_3.w * (tmp8 + tmp10);
			dR23_da = (a_inv * tmp11) - (GC_3.z * tmp8);

			dR31_da = (GC_4.y * tmp8) - (d_inv * tmp11);
			dR32_da = (GC_3.z * tmp8) + (a_inv * tmp11);
			dR33_da = -GC_3.w * (tmp8 + tmp9);

			vecx_tmp2 = ((vecx_tmp * dR11_da) + (vecy_tmp * dR12_da) + (vecz_tmp * dR13_da));
			vecy_tmp2 = ((vecx_tmp * dR21_da) + (vecy_tmp * dR22_da) + (vecz_tmp * dR23_da));
			vecz_tmp2 = ((vecx_tmp * dR31_da) + (vecy_tmp * dR32_da) + (vecz_tmp * dR33_da));

			dot_product_1 = (vecx_tmp2 * Ox) + (vecy_tmp2 * Oy) + (vecz_tmp2 * Oz);
			dot_product_2 = (vecx_tmp2 * vx) + (vecy_tmp2 * vy) + (vecz_tmp2 * vz);

			dL_dparam = (tmp4 * dot_product_2) - dot_product_1;
			dL_dparam = tmp6 * dL_dparam; 
			atomicAdd(((REAL *)dev_params.dL_dparams_3) + (GaussInd * 4) + 3, dL_dparam);

			// *** *** *** *** ***

			// dL_dc
			dR11_da = -GC_4.x * (tmp9 + tmp10);
			dR12_da = (GC_3.w * tmp9) - (b_inv * tmp11);
			dR13_da = (GC_3.z * tmp9) + (a_inv * tmp11);

			dR21_da = (GC_3.w * tmp9) + (b_inv * tmp11);
			dR22_da = GC_4.x * (tmp9 + tmp7);
			dR23_da = (GC_4.y * tmp9) - (d_inv * tmp11);

			dR31_da = (a_inv * tmp11) - (GC_3.z * tmp9);
			dR32_da = (GC_4.y * tmp9) + (d_inv * tmp11);
			dR33_da = -GC_4.x * (tmp9 + tmp8);

			vecx_tmp2 = ((vecx_tmp * dR11_da) + (vecy_tmp * dR12_da) + (vecz_tmp * dR13_da));
			vecy_tmp2 = ((vecx_tmp * dR21_da) + (vecy_tmp * dR22_da) + (vecz_tmp * dR23_da));
			vecz_tmp2 = ((vecx_tmp * dR31_da) + (vecy_tmp * dR32_da) + (vecz_tmp * dR33_da));

			dot_product_1 = (vecx_tmp2 * Ox) + (vecy_tmp2 * Oy) + (vecz_tmp2 * Oz);
			dot_product_2 = (vecx_tmp2 * vx) + (vecy_tmp2 * vy) + (vecz_tmp2 * vz);

			dL_dparam = (tmp4 * dot_product_2) - dot_product_1;
			dL_dparam = tmp6 * dL_dparam;
			atomicAdd(((REAL *)dev_params.dL_dparams_4) + (GaussInd * 2), dL_dparam);

			// *** *** *** *** ***

			// dL_dd
			dR11_da = -GC_4.y * (tmp10 + tmp9);
			dR12_da = (a_inv * tmp11) - (GC_3.z * tmp10);
			dR13_da = (GC_3.w * tmp10) + (b_inv * tmp11);

			dR21_da = (GC_3.z * tmp10) + (a_inv * tmp11);
			dR22_da = -GC_4.y * (tmp10 + tmp8);
			dR23_da = (GC_4.x * tmp10) - (c_inv * tmp11);

			dR31_da = (GC_3.w * tmp10) - (b_inv * tmp11);
			dR32_da = (GC_4.x * tmp10) + (c_inv * tmp11);
			dR33_da = GC_4.y * (tmp10 + tmp7);

			vecx_tmp2 = ((vecx_tmp * dR11_da) + (vecy_tmp * dR12_da) + (vecz_tmp * dR13_da));
			vecy_tmp2 = ((vecx_tmp * dR21_da) + (vecy_tmp * dR22_da) + (vecz_tmp * dR23_da));
			vecz_tmp2 = ((vecx_tmp * dR31_da) + (vecy_tmp * dR32_da) + (vecz_tmp * dR33_da));

			dot_product_1 = (vecx_tmp2 * Ox) + (vecy_tmp2 * Oy) + (vecz_tmp2 * Oz);
			dot_product_2 = (vecx_tmp2 * vx) + (vecy_tmp2 * vy) + (vecz_tmp2 * vz);

			dL_dparam = (tmp4 * dot_product_2) - dot_product_1;
			dL_dparam = tmp6 * dL_dparam;
			atomicAdd(((REAL *)dev_params.dL_dparams_4) + (GaussInd * 2) + 1, dL_dparam);

			// *** *** *** *** ***

			alpha_prev = alpha_next;
			R_Gauss_prev = GC_1.x;
			G_Gauss_prev = GC_1.y;
			B_Gauss_prev = GC_1.z;
		}
	}
}

// *** *** *** *** ***

unsigned InterleaveBits(
	unsigned x,
	unsigned y
) {
	x = (x | (x << 8)) & 16711935U;
	x = (x | (x << 4)) & 252645135U;
	x = (x | (x << 2)) & 858993459U;
	x = (x | (x << 1)) & 1431655765U;

	y = (y | (y << 8)) & 16711935U;
	y = (y | (y << 4)) & 252645135U;
	y = (y | (y << 2)) & 858993459U;
	y = (y | (y << 1)) & 1431655765U;

	return x | (y << 1);
}

// *** *** *** *** ***

__global__ void ComputeArraysForGradientComputation(
	REAL *dev_mu1R, REAL *dev_mu2R,
	REAL *dev_sigma12R,
	REAL *dev_sigma1R_square, REAL *dev_sigma2R_square,

	REAL *dev_mu1G, REAL *dev_mu2G,
	REAL *dev_sigma12G,
	REAL *dev_sigma1G_square, REAL *dev_sigma2G_square,

	REAL *dev_mu1B, REAL *dev_mu2B,
	REAL *dev_sigma12B,
	REAL *dev_sigma1B_square, REAL *dev_sigma2B_square,

	REAL *dev_tmp1R, REAL *dev_tmp2R, REAL *dev_tmp3R,
	REAL *dev_tmp1G, REAL *dev_tmp2G, REAL *dev_tmp3G,
	REAL *dev_tmp1B, REAL *dev_tmp2B, REAL *dev_tmp3B,

	int width, int height, int kernel_radius
) {
	int tid = (blockIdx.x * blockDim.x) + threadIdx.x;
	int x = tid % width;
	int y = tid / width;
	int size = width * height;
	if (tid < size) {
		if ((x >= kernel_radius) && (x < width - kernel_radius) && (y >= kernel_radius) && (y < height - kernel_radius)) {
			REAL mu1R = dev_mu1R[tid] / size;
			REAL mu2R = dev_mu2R[tid] / size;
			REAL sigma12R =  dev_sigma12R[tid] / size;
			REAL sigma1R_square = dev_sigma1R_square[tid] / size;
			REAL sigma2R_square = dev_sigma2R_square[tid] / size;

			REAL mu1G = dev_mu1G[tid] / size;
			REAL mu2G = dev_mu2G[tid] / size;
			REAL sigma12G =  dev_sigma12G[tid] / size;
			REAL sigma1G_square = dev_sigma1G_square[tid] / size;
			REAL sigma2G_square = dev_sigma2G_square[tid] / size;

			REAL mu1B = dev_mu1B[tid] / size;
			REAL mu2B = dev_mu2B[tid] / size;
			REAL sigma12B =  dev_sigma12B[tid] / size;
			REAL sigma1B_square = dev_sigma1B_square[tid] / size;
			REAL sigma2B_square = dev_sigma2B_square[tid] / size;

			REAL c1 = ((REAL)(0.01 * 0.01));
			REAL c2 = ((REAL)(0.03 * 0.03));

			REAL AR = (((REAL)2.0) * mu1R * mu2R) + c1;
			REAL BR = (((REAL)2.0) * (sigma12R - (mu1R * mu2R))) + c2;
			REAL CR = ((mu1R * mu1R) + (mu2R * mu2R) + c1);
			REAL DR = ((sigma1R_square - (mu1R * mu1R)) + (sigma2R_square - (mu2R * mu2R)) + c2);

			REAL AG = (((REAL)2.0) * mu1G * mu2G) + c1;
			REAL BG = (((REAL)2.0) * (sigma12G - (mu1G * mu2G))) + c2;
			REAL CG = ((mu1G * mu1G) + (mu2G * mu2G) + c1);
			REAL DG = ((sigma1G_square - (mu1G * mu1G)) + (sigma2G_square - (mu2G * mu2G)) + c2);

			REAL AB = (((REAL)2.0) * mu1B * mu2B) + c1;
			REAL BB = (((REAL)2.0) * (sigma12B - (mu1B * mu2B))) + c2;
			REAL CB = ((mu1B * mu1B) + (mu2B * mu2B) + c1);
			REAL DB = ((sigma1B_square - (mu1B * mu1B)) + (sigma2B_square - (mu2B * mu2B)) + c2);

			REAL tmp1R = (((REAL)2.0) * ((CR * DR * mu2R * (BR - AR)) - (AR * BR * mu1R * (DR - CR)))) / (CR * CR * DR * DR);
			REAL tmp2R = (((REAL)2.0) * AR * CR * DR) / (CR * CR * DR * DR);
			REAL tmp3R = (((REAL)2.0) * AR * BR * CR) / (CR * CR * DR * DR);

			REAL tmp1G = (((REAL)2.0) * ((CG * DG * mu2G * (BG - AG)) - (AG * BG * mu1G * (DG - CG)))) / (CG * CG * DG * DG);
			REAL tmp2G = (((REAL)2.0) * AG * CG * DG) / (CG * CG * DG * DG);
			REAL tmp3G = (((REAL)2.0) * AG * BG * CG) / (CG * CG * DG * DG);

			REAL tmp1B = (((REAL)2.0) * ((CB * DB * mu2B * (BB - AB)) - (AB * BB * mu1B * (DB - CB)))) / (CB * CB * DB * DB);
			REAL tmp2B = (((REAL)2.0) * AB * CB * DB) / (CB * CB * DB * DB);
			REAL tmp3B = (((REAL)2.0) * AB * BB * CB) / (CB * CB * DB * DB);

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
			dev_tmp1R[tid] = ((REAL)0.0);
			dev_tmp2R[tid] = ((REAL)0.0);
			dev_tmp3R[tid] = ((REAL)0.0);

			dev_tmp1G[tid] = ((REAL)0.0);
			dev_tmp2G[tid] = ((REAL)0.0);
			dev_tmp3G[tid] = ((REAL)0.0);

			dev_tmp1B[tid] = ((REAL)0.0);
			dev_tmp2B[tid] = ((REAL)0.0);
			dev_tmp3B[tid] = ((REAL)0.0);
		}
	}
}

// *** *** *** *** ***

__global__ void ComputeGradientSSIM(
	REAL *dev_conv1R,
	REAL *dev_conv2R, REAL *dev_img2R,
	REAL *dev_conv3R, REAL *dev_img1R,

	REAL *dev_conv1G,
	REAL *dev_conv2G, REAL *dev_img2G,
	REAL *dev_conv3G, REAL *dev_img1G,

	REAL *dev_conv1B,
	REAL *dev_conv2B, REAL *dev_img2B,
	REAL *dev_conv3B, REAL *dev_img1B,

	REAL *dev_gradR, REAL *dev_gradG, REAL *dev_gradB,

	int width, int height, int kernel_size
) {
	int tid = (blockIdx.x * blockDim.x) + threadIdx.x;
	int x = tid % width;
	int y = tid / width;
	int size = width * height;
	if ((tid < size) && (x >= kernel_size - 1) && (y >= kernel_size - 1)) {
		int ind = ((y - (kernel_size - 1)) * width) + (x - (kernel_size - 1));

		REAL conv1R = dev_conv1R[tid] / size;
		REAL conv2R = dev_conv2R[tid] / size;
		REAL img2R = dev_img2R[ind];
		REAL conv3R = dev_conv3R[tid] / size;
		REAL img1R = dev_img1R[ind];

		REAL conv1G = dev_conv1G[tid] / size;
		REAL conv2G = dev_conv2G[tid] / size;
		REAL img2G = dev_img2G[ind];
		REAL conv3G = dev_conv3G[tid] / size;
		REAL img1G = dev_img1G[ind];

		REAL conv1B = dev_conv1B[tid] / size;
		REAL conv2B = dev_conv2B[tid] / size;
		REAL img2B = dev_img2B[ind];
		REAL conv3B = dev_conv3B[tid] / size;
		REAL img1B = dev_img1B[ind];

		REAL gradR = ((conv3R * img1R) - conv1R - (conv2R * img2R)) / (((REAL)2.0 * 3.0) * (width - (kernel_size - ((REAL)1.0))) * (height - (kernel_size - ((REAL)1.0))));
		REAL gradG = ((conv3G * img1G) - conv1G - (conv2G * img2G)) / (((REAL)2.0 * 3.0) * (width - (kernel_size - ((REAL)1.0))) * (height - (kernel_size - ((REAL)1.0))));
		REAL gradB = ((conv3B * img1B) - conv1B - (conv2B * img2B)) / (((REAL)2.0 * 3.0) * (width - (kernel_size - ((REAL)1.0))) * (height - (kernel_size - ((REAL)1.0))));

		dev_gradR[tid] = gradR;
		dev_gradG[tid] = gradG;
		dev_gradB[tid] = gradB;
	}
}

// *** *** *** *** ***

extern bool RenderCUDA(SRenderParams& params, SCUDARenderParams& dev_params) {
	cudaError_t cudaStatus;

	/*cudaFuncSetAttribute(RenderCUDAKernel, cudaFuncAttributeMaxDynamicSharedMemorySize, 99 * 1024);
	cudaStatus = cudaGetLastError();
	if (cudaStatus != cudaSuccess)
		goto Error;*/

	// 1
	/*RenderCUDAKernel<<<((params.w * params.h) + 63) >> 6, 64, 0>>>(dev_params);
	cudaStatus = cudaGetLastError();
	if (cudaStatus != cudaSuccess)
		goto Error;*/

	// 2
	unsigned code = InterleaveBits(params.w - 1, params.h - 1);
	/*if (!dev_params.render) RenderCUDAKernel <<<(code + 63) >> 6, 64, ((48 << 5) << 2) * 8>>> (dev_params);
	else
		RenderCUDAKernel <<<(code + 63) >> 6, 64, 0>>> (dev_params);*/
	RenderCUDAKernel<<<(code + 63) >> 6, 64, 0>>>(dev_params);
	cudaStatus = cudaGetLastError();
	if (cudaStatus != cudaSuccess)
		goto Error;

	cudaStatus = cudaDeviceSynchronize();
	if (cudaStatus != cudaSuccess)
		goto Error;

	// *** *** *** *** ***

	if (dev_params.render) { // !!! !!! !!!
		cudaStatus = cudaMemcpy(params.bitmap, dev_params.bitmap, 4 * params.w * params.h, cudaMemcpyDeviceToHost);
		if (cudaStatus != cudaSuccess)
			goto Error;
	} else {
		//******************************************************************************************
		// SSIM                                                                                    *
		//******************************************************************************************

		const int kernel_size = 11;
		const int kernel_radius = kernel_size >> 1;
		const REAL sigma = ((REAL)1.5);

		int arraySizeReal = (params.w + (kernel_size - 1)) * (params.h + (kernel_size - 1)); // !!! !!! !!!
		int arraySizeComplex = (((params.w + (kernel_size - 1)) >> 1) + 1) * (params.h + (kernel_size - 1)); // !!! !!! !!!

		cufftResult error_CUFFT;

		//********************************
		// Compute mu's for output image *
		//********************************

		// R channel
		error_CUFFT = DFFT(params.planr2c, dev_params.dev_bitmap_out_R, params.dev_F_2);
		if (error_CUFFT != CUFFT_SUCCESS) goto Error;

		MultiplyPointwiseComplex<<<(arraySizeComplex + 255) >> 8, 256>>>(params.dev_F_2, params.dev_F_1, params.dev_F_2, arraySizeComplex);
		cudaStatus = cudaGetLastError();
		if (cudaStatus != cudaSuccess) goto Error;

		error_CUFFT = IDFFT(params.planc2r, params.dev_F_2, params.dev_mu_bitmap_out_R);
		if (error_CUFFT != CUFFT_SUCCESS) goto Error;

		// *** *** *** *** ***

		// G channel
		error_CUFFT = DFFT(params.planr2c, dev_params.dev_bitmap_out_G, params.dev_F_2);
		if (error_CUFFT != CUFFT_SUCCESS) goto Error;

		MultiplyPointwiseComplex<<<(arraySizeComplex + 255) >> 8, 256>>>(params.dev_F_2, params.dev_F_1, params.dev_F_2, arraySizeComplex);
		cudaStatus = cudaGetLastError();
		if (cudaStatus != cudaSuccess) goto Error;

		error_CUFFT = IDFFT(params.planc2r, params.dev_F_2, params.dev_mu_bitmap_out_G);
		if (error_CUFFT != CUFFT_SUCCESS) goto Error;

		// *** *** *** *** ***

		// B channel
		error_CUFFT = DFFT(params.planr2c, dev_params.dev_bitmap_out_B, params.dev_F_2);
		if (error_CUFFT != CUFFT_SUCCESS) goto Error;

		MultiplyPointwiseComplex<<<(arraySizeComplex + 255) >> 8, 256>>>(params.dev_F_2, params.dev_F_1, params.dev_F_2, arraySizeComplex);
		cudaStatus = cudaGetLastError();
		if (cudaStatus != cudaSuccess) goto Error;

		error_CUFFT = IDFFT(params.planc2r, params.dev_F_2, params.dev_mu_bitmap_out_B);
		if (error_CUFFT != CUFFT_SUCCESS) goto Error;

		// *** *** *** *** ***

		//***************************************
		// Compute mu's for output image square *
		//***************************************

		// R channel
		// dev_mu_bitmap_out_bitmap_ref_R = dev_bitmap_out_R * dev_bitmap_out_R
		MultiplyPointwiseReal<<<(arraySizeReal + 255) >> 8, 256>>>(
			dev_params.dev_bitmap_out_R,
			dev_params.dev_bitmap_out_R,
			params.dev_mu_bitmap_out_bitmap_ref_R,
			arraySizeReal
		);
		cudaStatus = cudaGetLastError();
		if (cudaStatus != cudaSuccess) goto Error;

		error_CUFFT = DFFT(params.planr2c, params.dev_mu_bitmap_out_bitmap_ref_R, params.dev_F_2);
		if (error_CUFFT != CUFFT_SUCCESS) goto Error;

		MultiplyPointwiseComplex<<<(arraySizeComplex + 255) >> 8, 256>>>(params.dev_F_2, params.dev_F_1, params.dev_F_2, arraySizeComplex);
		cudaStatus = cudaGetLastError();
		if (cudaStatus != cudaSuccess) goto Error;

		error_CUFFT = IDFFT(params.planc2r, params.dev_F_2, params.dev_mu_bitmap_out_R_square);
		if (error_CUFFT != CUFFT_SUCCESS) goto Error;

		// *** *** *** *** ***

		// G channel
		// dev_mu_bitmap_out_bitmap_ref_R = dev_bitmap_out_G * dev_bitmap_out_G
		MultiplyPointwiseReal<<<(arraySizeReal + 255) >> 8, 256>>>(
			dev_params.dev_bitmap_out_G,
			dev_params.dev_bitmap_out_G,
			params.dev_mu_bitmap_out_bitmap_ref_R,
			arraySizeReal
			);
		cudaStatus = cudaGetLastError();
		if (cudaStatus != cudaSuccess) goto Error;

		error_CUFFT = DFFT(params.planr2c, params.dev_mu_bitmap_out_bitmap_ref_R, params.dev_F_2);
		if (error_CUFFT != CUFFT_SUCCESS) goto Error;

		MultiplyPointwiseComplex<<<(arraySizeComplex + 255) >> 8, 256>>>(params.dev_F_2, params.dev_F_1, params.dev_F_2, arraySizeComplex);
		cudaStatus = cudaGetLastError();
		if (cudaStatus != cudaSuccess) goto Error;

		error_CUFFT = IDFFT(params.planc2r, params.dev_F_2, params.dev_mu_bitmap_out_G_square);
		if (error_CUFFT != CUFFT_SUCCESS) goto Error;

		// *** *** *** *** ***

		// B channel
		// dev_mu_bitmap_out_bitmap_ref_R = dev_bitmap_out_B * dev_bitmap_out_B
		MultiplyPointwiseReal<<<(arraySizeReal + 255) >> 8, 256>>>(
			dev_params.dev_bitmap_out_B,
			dev_params.dev_bitmap_out_B,
			params.dev_mu_bitmap_out_bitmap_ref_R,
			arraySizeReal
			);
		cudaStatus = cudaGetLastError();
		if (cudaStatus != cudaSuccess) goto Error;

		error_CUFFT = DFFT(params.planr2c, params.dev_mu_bitmap_out_bitmap_ref_R, params.dev_F_2);
		if (error_CUFFT != CUFFT_SUCCESS) goto Error;

		MultiplyPointwiseComplex<<<(arraySizeComplex + 255) >> 8, 256>>>(params.dev_F_2, params.dev_F_1, params.dev_F_2, arraySizeComplex);
		cudaStatus = cudaGetLastError();
		if (cudaStatus != cudaSuccess) goto Error;

		error_CUFFT = IDFFT(params.planc2r, params.dev_F_2, params.dev_mu_bitmap_out_B_square);
		if (error_CUFFT != CUFFT_SUCCESS) goto Error;

		// *** *** *** *** ***

		//***************************************************************
		// Compute mu's for product of output image and reference image *
		//***************************************************************

		// R channel
		// dev_mu_bitmap_out_bitmap_ref_R = dev_bitmap_out_R * dev_bitmap_ref_R
		MultiplyPointwiseReal<<<(arraySizeReal + 255) >> 8, 256>>>(
			dev_params.dev_bitmap_out_R,
			params.dev_bitmap_ref_R + (params.poseNum * arraySizeReal), // !!! !!! !!!
			params.dev_mu_bitmap_out_bitmap_ref_R,
			arraySizeReal
		);
		cudaStatus = cudaGetLastError();
		if (cudaStatus != cudaSuccess) goto Error;

		error_CUFFT = DFFT(params.planr2c, params.dev_mu_bitmap_out_bitmap_ref_R, params.dev_F_2);
		if (error_CUFFT != CUFFT_SUCCESS) goto Error;

		MultiplyPointwiseComplex<<<(arraySizeComplex + 255) >> 8, 256>>>(params.dev_F_2, params.dev_F_1, params.dev_F_2, arraySizeComplex);
		cudaStatus = cudaGetLastError();
		if (cudaStatus != cudaSuccess) goto Error;

		error_CUFFT = IDFFT(params.planc2r, params.dev_F_2, params.dev_mu_bitmap_out_bitmap_ref_R);
		if (error_CUFFT != CUFFT_SUCCESS) goto Error;

		// *** *** *** *** ***

		// G channel
		// dev_mu_bitmap_out_bitmap_ref_G = dev_bitmap_out_G * dev_bitmap_ref_G
		MultiplyPointwiseReal<<<(arraySizeReal + 255) >> 8, 256>>>(
			dev_params.dev_bitmap_out_G,
			params.dev_bitmap_ref_G + (params.poseNum * arraySizeReal), // !!! !!! !!!
			params.dev_mu_bitmap_out_bitmap_ref_G,
			arraySizeReal
		);
		cudaStatus = cudaGetLastError();
		if (cudaStatus != cudaSuccess) goto Error;

		error_CUFFT = DFFT(params.planr2c, params.dev_mu_bitmap_out_bitmap_ref_G, params.dev_F_2);
		if (error_CUFFT != CUFFT_SUCCESS) goto Error;

		MultiplyPointwiseComplex<<<(arraySizeComplex + 255) >> 8, 256>>>(params.dev_F_2, params.dev_F_1, params.dev_F_2, arraySizeComplex);
		cudaStatus = cudaGetLastError();
		if (cudaStatus != cudaSuccess) goto Error;

		error_CUFFT = IDFFT(params.planc2r, params.dev_F_2, params.dev_mu_bitmap_out_bitmap_ref_G);
		if (error_CUFFT != CUFFT_SUCCESS) goto Error;

		// *** *** *** *** ***

		// B channel
		// dev_mu_bitmap_out_bitmap_ref_B = dev_bitmap_out_B * dev_bitmap_ref_B
		MultiplyPointwiseReal<<<(arraySizeReal + 255) >> 8, 256>>>(
			dev_params.dev_bitmap_out_B,
			params.dev_bitmap_ref_B + (params.poseNum * arraySizeReal), // !!! !!! !!!
			params.dev_mu_bitmap_out_bitmap_ref_B,
			arraySizeReal
		);
		cudaStatus = cudaGetLastError();
		if (cudaStatus != cudaSuccess) goto Error;

		error_CUFFT = DFFT(params.planr2c, params.dev_mu_bitmap_out_bitmap_ref_B, params.dev_F_2);
		if (error_CUFFT != CUFFT_SUCCESS) goto Error;

		MultiplyPointwiseComplex<<<(arraySizeComplex + 255) >> 8, 256>>>(params.dev_F_2, params.dev_F_1, params.dev_F_2, arraySizeComplex);
		cudaStatus = cudaGetLastError();
		if (cudaStatus != cudaSuccess) goto Error;

		error_CUFFT = IDFFT(params.planc2r, params.dev_F_2, params.dev_mu_bitmap_out_bitmap_ref_B);
		if (error_CUFFT != CUFFT_SUCCESS) goto Error;

		// *** *** *** *** ***

		/*
		ComputeArraysForGradientComputation<<<(arraySizeReal + 255) >> 8, 256>>>(
			dev_mu1R, dev_mu2R,
			dev_sigma12R,
			dev_sigma1R_square, dev_sigma2R_square,

			dev_mu1G, dev_mu2G,
			dev_sigma12G,
			dev_sigma1G_square, dev_sigma2G_square,

			dev_mu1B, dev_mu2B,
			dev_sigma12B,
			dev_sigma1B_square, dev_sigma2B_square,

			// !!! !!! !!!
			dev_mu1R, dev_sigma12R, dev_sigma1R_square,
			dev_mu1G, dev_sigma12G, dev_sigma1G_square,
			dev_mu1B, dev_sigma12B, dev_sigma1B_square,
			// !!! !!! !!!

			width + (kernel_size - 1), height + (kernel_size - 1), kernel_radius
		);
		*/

		ComputeArraysForGradientComputation<<<(arraySizeReal + 255) >> 8, 256>>>(
			params.dev_mu_bitmap_out_R, params.dev_mu_bitmap_ref_R + (params.poseNum * arraySizeReal),
			params.dev_mu_bitmap_out_bitmap_ref_R,
			params.dev_mu_bitmap_out_R_square, params.dev_mu_bitmap_ref_R_square + (params.poseNum * arraySizeReal),

			params.dev_mu_bitmap_out_G, params.dev_mu_bitmap_ref_G + (params.poseNum * arraySizeReal),
			params.dev_mu_bitmap_out_bitmap_ref_G,
			params.dev_mu_bitmap_out_G_square, params.dev_mu_bitmap_ref_G_square + (params.poseNum * arraySizeReal),

			params.dev_mu_bitmap_out_B, params.dev_mu_bitmap_ref_B + (params.poseNum * arraySizeReal),
			params.dev_mu_bitmap_out_bitmap_ref_B,
			params.dev_mu_bitmap_out_B_square, params.dev_mu_bitmap_ref_B_square + (params.poseNum * arraySizeReal),

			// !!! !!! !!!
			params.dev_mu_bitmap_out_R, params.dev_mu_bitmap_out_bitmap_ref_R, params.dev_mu_bitmap_out_R_square,
			params.dev_mu_bitmap_out_G, params.dev_mu_bitmap_out_bitmap_ref_G, params.dev_mu_bitmap_out_G_square,
			params.dev_mu_bitmap_out_B, params.dev_mu_bitmap_out_bitmap_ref_B, params.dev_mu_bitmap_out_B_square,
			// !!! !!! !!!

			params.w + (kernel_size - 1), params.h + (kernel_size - 1), kernel_radius
		);
		cudaStatus = cudaGetLastError();
		if (cudaStatus != cudaSuccess) goto Error;

		cudaStatus = cudaDeviceSynchronize();
		if (cudaStatus != cudaSuccess) goto Error;

		// *** *** *** *** ***

		//**********************************************************
		// Compute auxiliary convolutions for gradient computation */
		//**********************************************************

		// convolution_1_R
		error_CUFFT = DFFT(params.planr2c, params.dev_mu_bitmap_out_R, params.dev_F_2);
		if (error_CUFFT != CUFFT_SUCCESS) goto Error;

		MultiplyPointwiseComplex<<<(arraySizeComplex + 255) >> 8, 256>>>(params.dev_F_2, params.dev_F_1, params.dev_F_2, arraySizeComplex);
		cudaStatus = cudaGetLastError();
		if (cudaStatus != cudaSuccess) goto Error;

		error_CUFFT = IDFFT(params.planc2r, params.dev_F_2, params.dev_mu_bitmap_out_R);
		if (error_CUFFT != CUFFT_SUCCESS) goto Error;

		// *** *** *** *** ***

		// convolution_1_G
		error_CUFFT = DFFT(params.planr2c, params.dev_mu_bitmap_out_G, params.dev_F_2);
		if (error_CUFFT != CUFFT_SUCCESS) goto Error;

		MultiplyPointwiseComplex<<<(arraySizeComplex + 255) >> 8, 256>>>(params.dev_F_2, params.dev_F_1, params.dev_F_2, arraySizeComplex);
		cudaStatus = cudaGetLastError();
		if (cudaStatus != cudaSuccess) goto Error;

		error_CUFFT = IDFFT(params.planc2r, params.dev_F_2, params.dev_mu_bitmap_out_G);
		if (error_CUFFT != CUFFT_SUCCESS) goto Error;

		// *** *** *** *** ***

		// convolution_1_B
		error_CUFFT = DFFT(params.planr2c, params.dev_mu_bitmap_out_B, params.dev_F_2);
		if (error_CUFFT != CUFFT_SUCCESS) goto Error;

		MultiplyPointwiseComplex<<<(arraySizeComplex + 255) >> 8, 256>>>(params.dev_F_2, params.dev_F_1, params.dev_F_2, arraySizeComplex);
		cudaStatus = cudaGetLastError();
		if (cudaStatus != cudaSuccess) goto Error;

		error_CUFFT = IDFFT(params.planc2r, params.dev_F_2, params.dev_mu_bitmap_out_B);
		if (error_CUFFT != CUFFT_SUCCESS) goto Error;

		// *** *** *** *** ***

		// convolution_2_R
		error_CUFFT = DFFT(params.planr2c, params.dev_mu_bitmap_out_bitmap_ref_R, params.dev_F_2);
		if (error_CUFFT != CUFFT_SUCCESS) goto Error;

		MultiplyPointwiseComplex<<<(arraySizeComplex + 255) >> 8, 256>>>(params.dev_F_2, params.dev_F_1, params.dev_F_2, arraySizeComplex);
		cudaStatus = cudaGetLastError();
		if (cudaStatus != cudaSuccess) goto Error;

		error_CUFFT = IDFFT(params.planc2r, params.dev_F_2, params.dev_mu_bitmap_out_bitmap_ref_R);
		if (error_CUFFT != CUFFT_SUCCESS) goto Error;

		// *** *** *** *** ***

		// convolution_2_G
		error_CUFFT = DFFT(params.planr2c, params.dev_mu_bitmap_out_bitmap_ref_G, params.dev_F_2);
		if (error_CUFFT != CUFFT_SUCCESS) goto Error;

		MultiplyPointwiseComplex<<<(arraySizeComplex + 255) >> 8, 256>>>(params.dev_F_2, params.dev_F_1, params.dev_F_2, arraySizeComplex);
		cudaStatus = cudaGetLastError();
		if (cudaStatus != cudaSuccess) goto Error;

		error_CUFFT = IDFFT(params.planc2r, params.dev_F_2, params.dev_mu_bitmap_out_bitmap_ref_G);
		if (error_CUFFT != CUFFT_SUCCESS) goto Error;

		// *** *** *** *** ***

		// convolution_2_B
		error_CUFFT = DFFT(params.planr2c, params.dev_mu_bitmap_out_bitmap_ref_B, params.dev_F_2);
		if (error_CUFFT != CUFFT_SUCCESS) goto Error;

		MultiplyPointwiseComplex<<<(arraySizeComplex + 255) >> 8, 256>>>(params.dev_F_2, params.dev_F_1, params.dev_F_2, arraySizeComplex);
		cudaStatus = cudaGetLastError();
		if (cudaStatus != cudaSuccess) goto Error;

		error_CUFFT = IDFFT(params.planc2r, params.dev_F_2, params.dev_mu_bitmap_out_bitmap_ref_B);
		if (error_CUFFT != CUFFT_SUCCESS) goto Error;

		// *** *** *** *** ***

		// convolution_3_R
		error_CUFFT = DFFT(params.planr2c, params.dev_mu_bitmap_out_R_square, params.dev_F_2);
		if (error_CUFFT != CUFFT_SUCCESS) goto Error;

		MultiplyPointwiseComplex<<<(arraySizeComplex + 255) >> 8, 256>>>(params.dev_F_2, params.dev_F_1, params.dev_F_2, arraySizeComplex);
		cudaStatus = cudaGetLastError();
		if (cudaStatus != cudaSuccess) goto Error;

		error_CUFFT = IDFFT(params.planc2r, params.dev_F_2, params.dev_mu_bitmap_out_R_square);
		if (error_CUFFT != CUFFT_SUCCESS) goto Error;

		// *** *** *** *** ***

		// convolution_3_G
		error_CUFFT = DFFT(params.planr2c, params.dev_mu_bitmap_out_G_square, params.dev_F_2);
		if (error_CUFFT != CUFFT_SUCCESS) goto Error;

		MultiplyPointwiseComplex<<<(arraySizeComplex + 255) >> 8, 256>>>(params.dev_F_2, params.dev_F_1, params.dev_F_2, arraySizeComplex);
		cudaStatus = cudaGetLastError();
		if (cudaStatus != cudaSuccess) goto Error;

		error_CUFFT = IDFFT(params.planc2r, params.dev_F_2, params.dev_mu_bitmap_out_G_square);
		if (error_CUFFT != CUFFT_SUCCESS) goto Error;

		// *** *** *** *** ***

		// convolution_3_B
		error_CUFFT = DFFT(params.planr2c, params.dev_mu_bitmap_out_B_square, params.dev_F_2);
		if (error_CUFFT != CUFFT_SUCCESS) goto Error;

		MultiplyPointwiseComplex<<<(arraySizeComplex + 255) >> 8, 256>>>(params.dev_F_2, params.dev_F_1, params.dev_F_2, arraySizeComplex);
		cudaStatus = cudaGetLastError();
		if (cudaStatus != cudaSuccess) goto Error;

		error_CUFFT = IDFFT(params.planc2r, params.dev_F_2, params.dev_mu_bitmap_out_B_square);
		if (error_CUFFT != CUFFT_SUCCESS) goto Error;

		// *** *** *** *** ***

		ComputeGradientSSIM<<<(arraySizeReal + 255) >> 8, 256>>>(
			params.dev_mu_bitmap_out_R,
			params.dev_mu_bitmap_out_bitmap_ref_R, params.dev_bitmap_ref_R + (params.poseNum * arraySizeReal),
			params.dev_mu_bitmap_out_R_square, dev_params.dev_bitmap_out_R,

			params.dev_mu_bitmap_out_G,
			params.dev_mu_bitmap_out_bitmap_ref_G, params.dev_bitmap_ref_G + (params.poseNum * arraySizeReal),
			params.dev_mu_bitmap_out_G_square, dev_params.dev_bitmap_out_G,

			params.dev_mu_bitmap_out_B,
			params.dev_mu_bitmap_out_bitmap_ref_B, params.dev_bitmap_ref_B + (params.poseNum * arraySizeReal),
			params.dev_mu_bitmap_out_B_square, dev_params.dev_bitmap_out_B,

			// !!! !!! !!!
			params.dev_mu_bitmap_out_R,	params.dev_mu_bitmap_out_G, params.dev_mu_bitmap_out_B,
			// !!! !!! !!!

			params.w + (kernel_size - 1), params.h + (kernel_size - 1), kernel_size
		);
		cudaStatus = cudaGetLastError();
		if (cudaStatus != cudaSuccess) goto Error;

		// *** *** *** *** ***

		// TEST 1
		/*if (params.epoch == 1) {
			REAL *buf = NULL;
			buf = (REAL *)malloc(sizeof(REAL) * arraySizeReal);
			if (buf == NULL) goto Error;

			// R channel
			cudaStatus = cudaMemcpy(buf, params.dev_mu_bitmap_out_R, sizeof(REAL) * arraySizeReal, cudaMemcpyDeviceToHost);
			if (cudaStatus != cudaSuccess) goto Error;

			for (int i = 0; i < params.h; ++i) {
				for (int j = 0; j < params.w; ++j) {
					REAL Rf = buf[((kernel_radius + i) * (params.w + (kernel_size - 1))) + (kernel_radius + j)] / arraySizeReal;
					if (Rf < ((REAL)0.0)) Rf = ((REAL)0.0);
					if (Rf > ((REAL)1.0)) Rf = ((REAL)1.0);
					unsigned char Ri = Rf * ((REAL)255.0);
					params.bitmap_ref[(i * params.w) + j] = (((unsigned) Ri) << 16);
				}
			}

			// G channel
			cudaStatus = cudaMemcpy(buf, params.dev_mu_bitmap_out_G, sizeof(REAL) *  arraySizeReal, cudaMemcpyDeviceToHost);
			if (cudaStatus != cudaSuccess) goto Error;

			for (int i = 0; i < params.h; ++i) {
				for (int j = 0; j < params.w; ++j) {
					REAL Gf = buf[((kernel_radius + i) * (params.w + (kernel_size - 1))) + (kernel_radius + j)] / arraySizeReal;
					if (Gf < ((REAL)0.0)) Gf = ((REAL)0.0);
					if (Gf > ((REAL)1.0)) Gf = ((REAL)1.0);
					unsigned char Gi = Gf * ((REAL)255.0);
					params.bitmap_ref[(i * params.w) + j] |= (((unsigned) Gi) << 8);
				}
			}

			// B channel
			cudaStatus = cudaMemcpy(buf, params.dev_mu_bitmap_out_B, sizeof(REAL) *  arraySizeReal, cudaMemcpyDeviceToHost);
			if (cudaStatus != cudaSuccess) goto Error;

			for (int i = 0; i < params.h; ++i) {
				for (int j = 0; j < params.w; ++j) {
					REAL Bf = buf[((kernel_radius + i) * (params.w + (kernel_size - 1))) + (kernel_radius + j)] / arraySizeReal;
					if (Bf < ((REAL)0.0)) Bf = ((REAL)0.0);
					if (Bf > ((REAL)1.0)) Bf = ((REAL)1.0);
					unsigned char Bi = Bf * ((REAL)255.0);
					params.bitmap_ref[(i * params.w) + j] |= ((unsigned)Bi);
				}
			}

			// Copy to bitmap on hdd
			unsigned char *foo = (unsigned char *)malloc(3 * params.w * params.h);
			for (int i = 0; i < params.h; ++i) {
				for (int j = 0; j < params.w; ++j) {
					unsigned char R = params.bitmap_ref[(i * params.w) + j] >> 16;
					unsigned char G = (params.bitmap_ref[(i * params.w) + j] >> 8) & 255;
					unsigned char B = params.bitmap_ref[(i * params.w) + j] & 255;
					foo[((((params.h - 1 - i) * params.w) + j) * 3) + 2] = R;
					foo[((((params.h - 1 - i) * params.w) + j) * 3) + 1] = G;
					foo[(((params.h - 1 - i) * params.w) + j) * 3] = B;		
				}
			}

			FILE *f = fopen("test.bmp", "rb+");
			fseek(f, 54, SEEK_SET);
			fwrite(foo, sizeof(int) * params.w * params.h, 1, f);
			fclose(f);

			free(buf);
		}*/

		// *** *** *** *** ***

		// TEST 2
		//if (params.epoch == 1) {
		if (params.epoch == 46) {
			REAL *buf = NULL;
			buf = (REAL *)malloc(sizeof(REAL) * arraySizeReal);
			if (buf == NULL) goto Error;

			cudaStatus = cudaMemcpy(buf, params.dev_mu_bitmap_out_B, sizeof(REAL) * arraySizeReal, cudaMemcpyDeviceToHost);
			if (cudaStatus != cudaSuccess) goto Error;

			FILE *f = fopen("output.txt", "wt");
			for (int i = 0; i < 32; ++i) {
				//REAL value = buf[((kernel_radius + 399) * (params.w + (kernel_size - 1))) + (kernel_radius + 400 - 16 + i)] / arraySizeReal;
				//REAL value = buf[((kernel_radius * 2 + 399) * (params.w + (kernel_size - 1))) + (kernel_radius * 2 + 400 - 16 + i)] / arraySizeReal;
				REAL value = buf[((kernel_radius * 2 + 399) * (params.w + (kernel_size - 1))) + (kernel_radius * 2 + 400 - 16 + i)];

				char buffer[256];
				#ifndef USE_DOUBLE_PRECISION
				sprintf(buffer, "%.20f\n", value);
				#else
				sprintf(buffer, "%.20lf\n", value);
				#endif
				fprintf(f, "%s", buffer);
			}
			fclose(f);

			free(buf);
		}

		//******************************************************************************************
		// UPDATE GRADIENT                                                                         *
		//******************************************************************************************

		//ComputeGradient<<<(code + 63) >> 6, 64, 0>>>(dev_params);
		ComputeGradient<<<((params.w * params.h) + 63) >> 6, 64, 0>>>(dev_params);
		cudaStatus = cudaGetLastError();
		if (cudaStatus != cudaSuccess)
			goto Error;

		cudaStatus = cudaDeviceSynchronize();
		if (cudaStatus != cudaSuccess)
			goto Error;
	}

	// *** *** *** *** ***

	// !!! !!! !!!
	cudaStatus = cudaMemcpy(params.dump, dev_params.dump, sizeof(int) * MAX_RAY_LENGTH * params.w * params.h, cudaMemcpyDeviceToHost);
	if (cudaStatus != cudaSuccess)
		goto Error;
	// !!! !!! !!!
	
	return true;
Error:
	return false;
}

// *** *** *** *** ***

extern bool DestroyCUDARendererAS(SCUDARenderParams& dev_params) {
	cudaError_t cudaStatus;

	cudaStatus = cudaFree(dev_params.tree_part_1);
	if (cudaStatus != cudaSuccess)
		goto Error;
	
	cudaStatus = cudaFree(dev_params.tree_part_2);
	if (cudaStatus != cudaSuccess)
		goto Error;
	
	cudaStatus = cudaFree(dev_params.GC_part_1);
	if (cudaStatus != cudaSuccess)
		goto Error;

	cudaStatus = cudaFree(dev_params.GC_part_2);
	if (cudaStatus != cudaSuccess)
		goto Error;
	cudaStatus = cudaFree(dev_params.GC_part_3);
	if (cudaStatus != cudaSuccess)
		goto Error;

	cudaStatus = cudaFree(dev_params.GC_part_4);
	if (cudaStatus != cudaSuccess)
		goto Error;

	cudaStatus = cudaFree(dev_params.d);
	if (cudaStatus != cudaSuccess)
		goto Error;

	return true;
Error:
	return false;
}