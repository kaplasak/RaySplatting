#define _USE_MATH_DEFINES
#include <conio.h>
#include <intrin.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "Renderer.h"

// *** *** *** *** ***

unsigned seed_float = 0;

float RandomFloat() {
	float result;

	*((unsigned*)&result) = (127 << 23) | (seed_float & ((1 << 23) - 1));
	seed_float = (1664525 * seed_float) + 1013904223;
	return result - 1.0f;
}

// *** *** *** *** ***

unsigned RandomInteger() {
	unsigned result;

	result = seed_float;
	seed_float = (1664525 * seed_float) + 1013904223;
	return result;
}

// *** *** *** *** ***

unsigned long long seed_double = 0;

double RandomDouble() {
	double result;

	*((unsigned long long*) & result) = (1023ULL << 52) | (seed_double & ((1ULL << 52) - 1ULL));
	seed_double = (6364136223846793005ULL * seed_double) + 1442695040888963407ULL;
	return result - 1.0;
}

// *** *** *** *** ***

void RandomNormalFloat(float& Z1, float& Z2) {
	float U1 = RandomFloat();
	float U2 = RandomFloat();
	float tmp1 = sqrt(-2.0f * log(U1));
	float tmp2 = 2.0f * M_PI * U2;
	Z1 = tmp1 * cos(tmp2);
	Z2 = tmp1 * sin(tmp2);
}

// *** *** *** *** ***

void RandomNormalDouble(double& Z1, double& Z2) {
	double U1;
	double tmp1;
	do {
		U1 = RandomDouble();
		tmp1 = log(U1);
	} while (!(isfinite(tmp1)));
	double U2 = RandomDouble();
	tmp1 = sqrt(-2.0 * tmp1);
	double tmp2 = 2.0 * M_PI * U2;
	Z1 = tmp1 * cos(tmp2);
	Z2 = tmp1 * sin(tmp2);
}

// *** *** *** *** ***

unsigned long long InterleaveBits(
	unsigned long long x,
	unsigned long long y,
	unsigned long long z
) {
	x = (x | (x << 32)) & 18446462598732906495ULL;
	x = (x | (x << 16)) & 71776123339407615ULL;
	x = (x | (x << 8)) & 17298045724797235215ULL;
	x = (x | (x << 4)) & 3513665537849438403ULL;
	x = (x | (x << 2)) & 10540996613548315209ULL;

	y = (y | (y << 32)) & 18446462598732906495ULL;
	y = (y | (y << 16)) & 71776123339407615ULL;
	y = (y | (y << 8)) & 17298045724797235215ULL;
	y = (y | (y << 4)) & 3513665537849438403ULL;
	y = (y | (y << 2)) & 10540996613548315209ULL;

	z = (z | (z << 32)) & 18446462598732906495ULL;
	z = (z | (z << 16)) & 71776123339407615ULL;
	z = (z | (z << 8)) & 17298045724797235215ULL;
	z = (z | (z << 4)) & 3513665537849438403ULL;
	z = (z | (z << 2)) & 10540996613548315209ULL;

	return x | (y << 1) | (z << 2);
}

// *** *** *** *** ***

void GetGaussianComponentAABB(
	SGaussianComponent& GC,
	float threshold,
	float& lB, float& rB,
	float& uB, float& dB,
	float& bB, float& fB
) {
	float aa = GC.qr * GC.qr;
	float bb = GC.qi * GC.qi;
	float cc = GC.qj * GC.qj;
	float dd = GC.qk * GC.qk;
	float s = 2.0f / (aa + bb + cc + dd);

	float bs = GC.qi * s;  float cs = GC.qj * s;  float ds = GC.qk * s;
	float ab = GC.qr * bs; float ac = GC.qr * cs; float ad = GC.qr * ds;
	bb = bb * s;		   float bc = GC.qi * cs; float bd = GC.qi * ds;
	cc = cc * s;           float cd = GC.qj * ds;       dd = dd * s;

	float Q11 = 1.0f - cc - dd;
	float Q12 = bc - ad;
	float Q13 = bd + ac;

	float Q21 = bc + ad;
	float Q22 = 1.0f - bb - dd;
	float Q23 = cd - ab;

	float Q31 = bd - ac;
	float Q32 = cd + ab;
	float Q33 = 1.0f - bb - cc;

	float tmp;

	float sX = 1.0f / (1.0f + expf(-GC.sX));
	float sY = 1.0f / (1.0f + expf(-GC.sY));
	float sZ = 1.0f / (1.0f + expf(-GC.sZ));

	tmp = sqrtf(11.3449f * ((sX * sX * Q11 * Q11) + (sY * sY * Q12 * Q12) + (sZ * sZ * Q13 * Q13)));
	lB = GC.mX - tmp;
	rB = GC.mX + tmp;
	tmp = sqrtf(11.3449f * ((sX * sX * Q21 * Q21) + (sY * sY * Q22 * Q22) + (sZ * sZ * Q23 * Q23)));
	uB = GC.mY - tmp;
	dB = GC.mY + tmp;
	tmp = sqrtf(11.3449f * ((sX * sX * Q31 * Q31) + (sY * sY * Q32 * Q32) + (sZ * sZ * Q33 * Q33)));
	bB = GC.mZ - tmp;
	fB = GC.mZ + tmp;
}

// *** *** *** *** ***

template<typename T>
bool RadixSort(
	T* array,
	int size,
	int numberOfBits,
	int* indices
) {
	int* indicesTmp = (int*)malloc(sizeof(int) * size);
	if (indicesTmp == NULL)
		return false;

	int* indices_in;
	int* indices_out;
	if (((((sizeof(T) << 3) + (numberOfBits - 1)) / numberOfBits) & 1) == 0) {
		indices_in = indices;
		indices_out = indicesTmp;
	}
	else {
		indices_in = indicesTmp;
		indices_out = indices;
	}

	int* histogram = (int*)malloc(sizeof(int) * (1 << numberOfBits));
	if (histogram == NULL)
		return false;

	for (int i = 0; i < size; ++i)
		indices_in[i] = i;

	T mask = (((T)1) << numberOfBits) - 1;
	int position = 0;
	while (position < (sizeof(T) << 3) - 1) {
		for (int i = 0; i < (1 << numberOfBits); ++i) histogram[i] = 0;
		for (int i = 0; i < size; ++i) {
			int index_in = indices_in[i];
			int index_out = (array[index_in] & mask) >> position;
			++histogram[index_out];
		}
		int previous = histogram[0];
		histogram[0] = 0;
		for (int i = 1; i < (1 << numberOfBits); ++i) {
			int tmp = histogram[i];
			histogram[i] = histogram[i - 1] + previous;
			previous = tmp;
		}
		for (int i = 0; i < size; ++i) {
			int index_in = indices_in[i];
			int index_out = (array[index_in] & mask) >> position;
			indices_out[histogram[index_out]++] = index_in;
		}
		mask = mask << numberOfBits;
		position = position + numberOfBits;

		int* tmp = indices_in;
		indices_in = indices_out;
		indices_out = tmp;
	}

	indices = indices_in; // !!! !!! !!!

	free(indicesTmp);
	free(histogram);
	return true;
}

// *** *** *** *** ***

int GCD(int a, int b) {
	while (b != 0) {
		int tmp = a;
		a = b;
		b = tmp % b;
	}
	return a;
}

// *** *** *** *** ***

void GenerateCodes(
	SLBVHTreeNode* LBVHTree,
	unsigned long long* codes,
	int nodeInd,
	unsigned long long code
) {
	SLBVHTreeNode node = LBVHTree[nodeInd];
	codes[nodeInd] = code;
	if (node.info != 3) {
		GenerateCodes(LBVHTree, codes, node.lNode, code << 1);
		GenerateCodes(LBVHTree, codes, node.rNode, (code << 1) + 1ULL);
	}
}

// *** *** *** *** ***

bool BuildLBVHTree(
	SGaussianComponent* GC, int size, int numberOfBits,
	SLBVHTreeNode*& LBVHTree,
	int*& d,
	int& D, int& H
) {
	wchar_t consoleBuffer[256];
	//swprintf(consoleBuffer, 256, L"STARTING BUILDING BVH TREE... .\n");
	//WriteConsole(GetStdHandle(STD_OUTPUT_HANDLE), consoleBuffer, wcslen(consoleBuffer), NULL, NULL);

	int* indices = (int*)malloc(sizeof(int) * size);
	if (indices == NULL)
		return false;

	unsigned long long* codes = (unsigned long long*)malloc(sizeof(unsigned long long) * size);
	if (codes == NULL) return false;

	float XMin = INFINITY, XMax = -INFINITY;
	float YMin = INFINITY, YMax = -INFINITY;
	float ZMin = INFINITY, ZMax = -INFINITY;
	for (int i = 0; i < size; ++i) {
		if (GC[i].mX < XMin) XMin = GC[i].mX;
		if (GC[i].mX > XMax) XMax = GC[i].mX;
		if (GC[i].mY < YMin) YMin = GC[i].mY;
		if (GC[i].mY > YMax) YMax = GC[i].mY;
		if (GC[i].mZ < ZMin) ZMin = GC[i].mZ;
		if (GC[i].mZ > ZMax) ZMax = GC[i].mZ;
	}

	float tmp1;
	*((unsigned*)&tmp1) = (127 << 23) | ((1 << 23) - 1);
	tmp1 = tmp1 - 1.0f;

	float tmp2 = tmp1 / (XMax - XMin);
	float tmp3 = tmp1 / (YMax - YMin);
	float tmp4 = tmp1 / (ZMax - ZMin);
	for (int i = 0; i < size; ++i) {
		float XTransformed = 1.0f + ((GC[i].mX - XMin) * tmp2);
		float YTransformed = 1.0f + ((GC[i].mY - YMin) * tmp3);
		float ZTransformed = 1.0f + ((GC[i].mZ - ZMin) * tmp4);
		unsigned XBits = (*((unsigned*)&XTransformed) & ((1 << 23) - 1)) >> 2;
		unsigned YBits = (*((unsigned*)&YTransformed) & ((1 << 23) - 1)) >> 2;
		unsigned ZBits = (*((unsigned*)&ZTransformed) & ((1 << 23) - 1)) >> 2;
		codes[i] = InterleaveBits(XBits, YBits, ZBits);
	}

	// *** *** *** *** ***

	// Sorting means of the Gaussians
	if (!RadixSort<unsigned long long>(codes, size, numberOfBits, indices))
		return false;

	SGaussianComponent* GCTmp = (SGaussianComponent*)malloc(sizeof(SGaussianComponent) * size);
	for (int i = 0; i < size; ++i)
		GCTmp[i] = GC[indices[i]];
	for (int i = 0; i < size; ++i)
		GC[i] = GCTmp[i];
	free(GCTmp);

	unsigned long long* codesTmp = (unsigned long long*)malloc(sizeof(unsigned long long) * size);
	for (int i = 0; i < size; ++i)
		codesTmp[i] = codes[indices[i]];
	for (int i = 0; i < size; ++i)
		codes[i] = codesTmp[i];
	free(codesTmp);

	// *** *** *** *** ***

	int numberOfLeafs = 1;
	for (int i = 1; i < size; ++i) {
		if (codes[i] != codes[i - 1]) ++numberOfLeafs;
	}
	int numberOfNodes = (numberOfLeafs << 1) - 1;

	LBVHTree = (SLBVHTreeNode*)malloc(sizeof(SLBVHTreeNode) * numberOfNodes);
	if (LBVHTree == NULL)
		return false;

	// LBVH tree construction
	LBVHTree[0].lNode = 0;
	LBVHTree[0].rNode = size - 1;

	int i1 = 0, i2 = 1, i3 = 1;
	while (i1 < i2) {
		printf("%d %d %d\n", i1, i2, i3);

		while (i1 < i2) {
			SLBVHTreeNode node = LBVHTree[i1];

			if (codes[node.lNode] != codes[node.rNode]) {
				unsigned long index;
				_BitScanReverse64(&index, codes[node.lNode] ^ codes[node.rNode]);
				unsigned long long mask = 1ULL << index;

				int lB = node.lNode;
				int uB = node.rNode;
				while (uB - lB > 1) {
					int M = (lB + uB) >> 1;
					if ((codes[M] & mask) != 0)
						uB = M;
					else
						lB = M;
				}

				LBVHTree[i3].lNode = node.lNode;
				LBVHTree[i3].rNode = lB;

				LBVHTree[i3 + 1].lNode = uB;
				LBVHTree[i3 + 1].rNode = node.rNode;

				LBVHTree[i1].info = index % 3;
				LBVHTree[i1].lNode = i3;
				LBVHTree[i1].rNode = i3 + 1;

				i3 += 2;
			}
			else
				LBVHTree[i1].info = 3;

			++i1;
		}
		i1 = i2;
		i2 = i3;
	}

	// Computing LBVH tree bounding boxes
	for (int i = numberOfNodes - 1; i >= 0; --i) {
		SLBVHTreeNode node = LBVHTree[i];

		if (node.info == 3) {
			node.lB = INFINITY; node.rB = -INFINITY;
			node.uB = INFINITY; node.dB = -INFINITY;
			node.bB = INFINITY; node.fB = -INFINITY;

			for (int j = node.lNode; j <= node.rNode; ++j) {
				float lB, rB;
				float uB, dB;
				float bB, fB;

				GetGaussianComponentAABB(GC[j], 0.01, lB, rB, uB, dB, bB, fB);

				node.lB = (lB < node.lB) ? lB : node.lB;
				node.rB = (rB > node.rB) ? rB : node.rB;
				node.uB = (uB < node.uB) ? uB : node.uB;
				node.dB = (dB > node.dB) ? dB : node.dB;
				node.bB = (bB < node.bB) ? bB : node.bB;
				node.fB = (fB > node.fB) ? fB : node.fB;

				LBVHTree[i].lB = node.lB;
				LBVHTree[i].rB = node.rB;
				LBVHTree[i].uB = node.uB;
				LBVHTree[i].dB = node.dB;
				LBVHTree[i].bB = node.bB;
				LBVHTree[i].fB = node.fB;
			}
		}
		else {
			SLBVHTreeNode lNode = LBVHTree[node.lNode];
			SLBVHTreeNode rNode = LBVHTree[node.rNode];

			LBVHTree[i].lB = (lNode.lB <= rNode.lB) ? lNode.lB : rNode.lB;
			LBVHTree[i].rB = (lNode.rB >= rNode.rB) ? lNode.rB : rNode.rB;
			LBVHTree[i].uB = (lNode.uB <= rNode.uB) ? lNode.uB : rNode.uB;
			LBVHTree[i].dB = (lNode.dB >= rNode.dB) ? lNode.dB : rNode.dB;
			LBVHTree[i].bB = (lNode.bB <= rNode.bB) ? lNode.bB : rNode.bB;
			LBVHTree[i].fB = (lNode.fB >= rNode.fB) ? lNode.fB : rNode.fB;
		}
	}

	// *** *** *** *** ***

	if (numberOfNodes > size)
		indices = (int*)realloc(indices, sizeof(int) * numberOfNodes);
	codes = (unsigned long long*)realloc(codes, sizeof(unsigned long long) * numberOfNodes);
	GenerateCodes(LBVHTree, codes, 0, 1ULL);

	D = 1;
	while ((D << 2) < numberOfNodes) D <<= 1;

	H = (numberOfNodes << 1) + 1;
	while (GCD(H, D) != 1) ++H;

	int* histogram = (int*)malloc(sizeof(int) * D);
	for (int i = 0; i < D; ++i) histogram[i] = 0;
	for (int i = 0; i < numberOfNodes; ++i)
		--histogram[codes[i] & (D - 1)];

	unsigned long long* array = (unsigned long long*)malloc(sizeof(unsigned long long) * numberOfNodes);
	for (int i = 0; i < numberOfNodes; ++i)
		array[i] = (((unsigned long long)histogram[codes[i] & (D - 1)]) << 32) + (codes[i] & (D - 1));

	RadixSort<unsigned long long>(array, numberOfNodes, 16, indices);

	/*for (int i = 0; i < 100; ++i) {
		swprintf(consoleBuffer, 256, L"%d : %d %d %d\n", i, array[indices[i]] >> 32, array[indices[i]] & (unsigned)-1, indices[i]);
		WriteConsole(GetStdHandle(STD_OUTPUT_HANDLE), consoleBuffer, wcslen(consoleBuffer),NULL,NULL);
	}*/

	d = (int*)malloc(sizeof(int) * D);

	bool* occurrences = (bool*)malloc(sizeof(bool) * H);
	for (int i = 0; i < H; ++i) occurrences[i] = false;

	int i = 0;
	int displacement = -1;
	while (i < numberOfNodes) {
		int nodeIndex = indices[i];
		int count = -(int)(array[nodeIndex] >> 32);
		int dIndex = array[nodeIndex] & (unsigned)-1;

		bool failed;
		++displacement;
		do {
			int j = 0;
			failed = false;
			while ((j < count) && (!failed)) {
				nodeIndex = indices[i + j];
				if (!occurrences[(displacement + codes[nodeIndex]) % H]) {
					occurrences[(displacement + codes[nodeIndex]) % H] = true;
					++j;
				}
				else
					failed = true;
			}

			if (failed) {
				/*swprintf(consoleBuffer, 256, L"Failed... .\n");
				WriteConsole(GetStdHandle(STD_OUTPUT_HANDLE), consoleBuffer, wcslen(consoleBuffer),NULL,NULL);*/

				for (int k = 0; k < j; ++k) {
					nodeIndex = indices[i + k];
					occurrences[(displacement + codes[nodeIndex]) % H] = false;
				}
				++displacement;
			}
		} while (failed);

		d[dIndex] = displacement;
		i += count;

		/*swprintf(consoleBuffer, 256, L"%d : %d %d %d [%d]\n", i, nodeIndex, count, dIndex, displacement);
		WriteConsole(GetStdHandle(STD_OUTPUT_HANDLE), consoleBuffer, wcslen(consoleBuffer),NULL,NULL);*/
	}
	/*swprintf(consoleBuffer, 256, L"%d\n", H);
	WriteConsole(GetStdHandle(STD_OUTPUT_HANDLE), consoleBuffer, wcslen(consoleBuffer),NULL,NULL);*/

	SLBVHTreeNode* LBVHTreeTmp = (SLBVHTreeNode*)malloc(sizeof(SLBVHTreeNode) * H);
	for (int i = 0; i < numberOfNodes; ++i) {
		SLBVHTreeNode node = LBVHTree[i];

		int nodeIndex = (codes[i] + d[codes[i] & (D - 1)]) % H;
		int lNodeIndex = (codes[node.lNode] + d[codes[node.lNode] & (D - 1)]) % H;
		int rNodeIndex = (codes[node.rNode] + d[codes[node.rNode] & (D - 1)]) % H;

		if (node.info != 3) {
			node.lNode = lNodeIndex;
			node.rNode = rNodeIndex;
		}
		LBVHTreeTmp[nodeIndex] = node;

		if (i == 0) {
			swprintf(consoleBuffer, 256, L"%d\n", nodeIndex);
			WriteConsole(GetStdHandle(STD_OUTPUT_HANDLE), consoleBuffer, wcslen(consoleBuffer), NULL, NULL);
		}
	}
	free(LBVHTree); // !!! !!! !!!
	LBVHTree = LBVHTreeTmp;

	// CHECK IF HASH MAP CORRECT
	for (int i = 0; i < H; ++i) occurrences[i] = false;
	for (int i = 0; i < numberOfNodes; ++i) {
		int index = (codes[i] + d[codes[i] & (D - 1)]) % H;
		if (!occurrences[index])
			occurrences[index] = true;
		else {
			swprintf(consoleBuffer, 256, L"ERROR\n");
			WriteConsole(GetStdHandle(STD_OUTPUT_HANDLE), consoleBuffer, wcslen(consoleBuffer), NULL, NULL);
		}
	}

	// *********************************************************************************************

	int maxPrimitives = 0;
	for (int i = 0; i < numberOfNodes; ++i) {
		SLBVHTreeNode node = LBVHTree[i];
		if (node.info == 3) {
			if (((node.rNode - node.lNode) + 1) >= maxPrimitives)
				maxPrimitives = (node.rNode - node.lNode) + 1;
		}
	}
	swprintf(consoleBuffer, 256, L"MAX GAUSSIANS: %d\n", maxPrimitives);
	WriteConsole(GetStdHandle(STD_OUTPUT_HANDLE), consoleBuffer, wcslen(consoleBuffer), NULL, NULL);

	// *********************************************************************************************
	
	// !!! !!! !!!
	free(indices);
	free(codes);
	free(histogram);
	free(array);
	free(occurrences);

	// *** *** *** *** ***

	return true;
}

// *** *** *** *** ***

bool IntersectRayAABB(
	float Ox, float Oy, float Oz,
	float vx, float vy, float vz,
	float lB, float rB, float uB, float dB, float bB, float fB,
	float& tHit1, float& tHit2, float tStart
) {
	float vxInv = 1.0f / vx;
	float vyInv = 1.0f / vy;
	float vzInv = 1.0f / vz;
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

// *** *** *** *** ***

bool IntersectRayGaussianComponent(
	float Ox, float Oy, float Oz,
	float vx, float vy, float vz,
	SGaussianComponent& GC,
	float& tHit1, float& tHit2, float tStart
) {
	Ox = Ox - GC.mX;
	Oy = Oy - GC.mY;
	Oz = Oz - GC.mZ;

	double Ox_prim = (GC.A11 * Ox) + (GC.A21 * Oy) + (GC.A31 * Oz);
	double Oy_prim = (GC.A12 * Ox) + (GC.A22 * Oy) + (GC.A32 * Oz);
	double Oz_prim = (GC.A13 * Ox) + (GC.A23 * Oy) + (GC.A33 * Oz);

	double vx_prim = (GC.A11 * vx) + (GC.A21 * vy) + (GC.A31 * vz);
	double vy_prim = (GC.A12 * vx) + (GC.A22 * vy) + (GC.A32 * vz);
	double vz_prim = (GC.A13 * vx) + (GC.A23 * vy) + (GC.A33 * vz);

	double a = (vx_prim * vx_prim) + (vy_prim * vy_prim) + (vz_prim * vz_prim);
	double b = 2.0 * ((Ox_prim * vx_prim) + (Oy_prim * vy_prim) + (Oz_prim * vz_prim));
	double c = ((Ox_prim * Ox_prim) + (Oy_prim * Oy_prim) + (Oz_prim * Oz_prim)) - 11.3449f;
	double delta = (b * b) - (4.0 * a * c);
	if (delta >= 0.0f) {
		/*__m128 tmp1 = _mm_set_ss(delta);
		tmp1 = _mm_rsqrt_ss(tmp1);
		tmp1 = _mm_rcp_ss(tmp1);

		__m128 tmp2 = _mm_set_ss(2.0f * a);
		tmp2 = _mm_rcp_ss(tmp2);

		tHit1 = (-b - tmp1.m128_f32[0]) * tmp2.m128_f32[0];
		tHit2 = (-b + tmp1.m128_f32[0]) * tmp2.m128_f32[0];
		*/

		double tmp1 = sqrt(delta);
		double tmp2 = 1.0 / (2.0 * a);

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

bool IntersectRayLBVHTree(
	float Ox, float Oy, float Oz,
	float vx, float vy, float vz,
	SLBVHTreeNode* tree,
	SGaussianComponent* GC,
	int nodeInd,
	float& tHit,
	int& GaussInd,
	float tStart,
	float tEnd
) {
	SLBVHTreeNode node = tree[nodeInd];

	float t1, t2;
	bool nodeIntersection = IntersectRayAABB(Ox, Oy, Oz, vx, vy, vz, node.lB, node.rB, node.uB, node.dB, node.bB, node.fB, t1, t2, tStart);

	int nodeInd1, nodeInd2;

	if ((nodeIntersection) && (t1 < tEnd)) {
		//if (nodeIntersection) {
		switch (node.info) {
		case 0: {
			if (vx <= 0.0f) {
				nodeInd1 = node.rNode;
				nodeInd2 = node.lNode;
			}
			else {
				nodeInd1 = node.lNode;
				nodeInd2 = node.rNode;
			}
			break;
		}
		case 1: {
			if (vy <= 0.0f) {
				nodeInd1 = node.rNode;
				nodeInd2 = node.lNode;
			}
			else {
				nodeInd1 = node.lNode;
				nodeInd2 = node.rNode;
			}
			break;
		}
		case 2: {
			if (vz <= 0.0f) {
				nodeInd1 = node.rNode;
				nodeInd2 = node.lNode;
			}
			else {
				nodeInd1 = node.lNode;
				nodeInd2 = node.rNode;
			}
			break;
		}
		}

		if (node.info != 3) {
			float tHitLocal1, tHitLocal2;
			int GaussIndLocal1, GaussIndLocal2;

			bool nodeIntersection1 = IntersectRayLBVHTree(Ox, Oy, Oz, vx, vy, vz, tree, GC, nodeInd1, tHitLocal1, GaussIndLocal1, tStart, tEnd);

			if (nodeIntersection1) {
				bool nodeIntersection2 = IntersectRayLBVHTree(Ox, Oy, Oz, vx, vy, vz, tree, GC, nodeInd2, tHitLocal2, GaussIndLocal2, tStart, tHitLocal1);
				if (nodeIntersection2) {
					if (tHitLocal1 <= tHitLocal2) {
						tHit = tHitLocal1;
						GaussInd = GaussIndLocal1;
					}
					else {
						tHit = tHitLocal2;
						GaussInd = GaussIndLocal2;
					}
				}
				else {
					tHit = tHitLocal1;
					GaussInd = GaussIndLocal1;
				}
				return true;
			}
			else {
				bool nodeIntersection2 = IntersectRayLBVHTree(Ox, Oy, Oz, vx, vy, vz, tree, GC, nodeInd2, tHitLocal2, GaussIndLocal2, tStart, tEnd);
				if (nodeIntersection2) {
					tHit = tHitLocal2;
					GaussInd = GaussIndLocal2;
					return true;
				}
				else
					return false;
			}
		}
		else {
			float tHitMin = INFINITY;
			int GaussIndMin = -1;

			for (int i = node.lNode; i <= node.rNode; ++i) {
				float tHit1, tHit2;

				SGaussianComponent Gauss = GC[i];
				bool intersection = IntersectRayGaussianComponent(
					Ox, Oy, Oz,
					vx, vy, vz,
					Gauss,
					tHit1, tHit2,
					tStart // !!! !!! !!!
				);
				if (intersection) {
					if (tHit1 <= tHitMin) {
						tHitMin = tHit1;
						GaussIndMin = i;
					}
				}
			}

			if (GaussIndMin != -1) {
				tHit = tHitMin;
				GaussInd = GaussIndMin;
				return true;
			}
			else
				return false;
		}
	}
	else
		return false;
}

// *** *** *** *** ***

bool IntersectRayLBVHTreeStackless(
	float Ox, float Oy, float Oz,
	float vx, float vy, float vz,
	SLBVHTreeNode* tree,
	SGaussianComponent* GC,
	unsigned long long& nodeTrail,
	unsigned long long& notCompletedMask,
	float& tHit,
	int& GaussInd,
	float tStart,
	float tEnd,
	int* d,
	int D, int H
) {
	tHit = INFINITY;
	GaussInd = -1;
	unsigned long long nodeTrailTmp = nodeTrail;
	unsigned long long notCompletedMaskTmp = notCompletedMask;
	unsigned long long nodeInd = (nodeTrailTmp + d[nodeTrailTmp & (D - 1)]) % H;
	bool firstIntersection = true;

	do {
		SLBVHTreeNode node = tree[nodeInd];

		float t1, t2;
		bool nodeIntersection = IntersectRayAABB(Ox, Oy, Oz, vx, vy, vz, node.lB, node.rB, node.uB, node.dB, node.bB, node.fB, t1, t2, tStart);

		if ((nodeIntersection) && (t1 < tHit)) {
			switch (node.info) {
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

			if (node.info != 3) {
				notCompletedMaskTmp = (notCompletedMaskTmp << 1) + 1;
				if ((nodeTrailTmp & 1) == 0)
					nodeInd = node.lNode;
				else
					nodeInd = node.rNode;
			}
			else {
				for (int i = node.lNode; i <= node.rNode; ++i) {
					float tHit1, tHit2;

					SGaussianComponent Gauss = GC[i];
					bool intersection = IntersectRayGaussianComponent(
						Ox, Oy, Oz,
						vx, vy, vz,
						Gauss,
						tHit1, tHit2,
						tStart // !!! !!! !!!
					);
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

				unsigned long index;
				_BitScanForward64(&index, notCompletedMaskTmp);
				nodeTrailTmp = (nodeTrailTmp >> index) ^ 1;
				notCompletedMaskTmp = (notCompletedMaskTmp >> index) ^ 1;

				nodeInd = (nodeTrailTmp + d[nodeTrailTmp & (D - 1)]) % H;
			}
		}
		else {
			unsigned long index;
			_BitScanForward64(&index, notCompletedMaskTmp);
			nodeTrailTmp = (nodeTrailTmp >> index) ^ 1;
			notCompletedMaskTmp = (notCompletedMaskTmp >> index) ^ 1;

			nodeInd = (nodeTrailTmp + d[nodeTrailTmp & (D - 1)]) % H;
		}
	} while (notCompletedMaskTmp != 0);

	return (GaussInd != -1);
}

// *** *** *** *** ***

DWORD WINAPI RenderRoutine(LPVOID paramsPtr) {
	SRenderParams& params = *((SRenderParams*)paramsPtr);

	for (int i = params.threadId; i < params.h; i += params.threadsNum) {
		for (int j = 0; j < params.w; ++j) {
			float wInv = 1.0f / params.w;
			float hInv = 1.0f / params.h;
			float dX, dY;
			if (params.h <= params.w) {
				dX = params.w * hInv * (-0.5f + ((j + 0.5f) * wInv));
				dY = -0.5f + ((i + 0.5f) * hInv);
			}
			else {
				dX = -0.5f + ((j + 0.5f) * wInv);
				dY = params.h * wInv * (-0.5f + ((i + 0.5f) * hInv));
			}

			float vx = (params.Rx * dX) + (params.Dx * dY) + (params.Fx * 0.5f);
			float vy = (params.Ry * dX) + (params.Dy * dY) + (params.Fy * 0.5f);
			float vz = (params.Rz * dX) + (params.Dz * dY) + (params.Fz * 0.5f);

			float T = 1.0f;
			float tHit = 0.0f;
			float R = 0.0f;
			float G = 0.0f;
			float B = 0.0f;
			bool intersection;

			unsigned long long nodeTrail = 1;
			unsigned long long notCompletedMask = 1;
			do {
				int GaussInd;

				intersection = IntersectRayLBVHTreeStackless(
					params.Ox, params.Oy, params.Oz,
					vx, vy, vz,
					params.tree,
					params.GC,

					nodeTrail, // !!! !!! !!!
					notCompletedMask, // !!! !!! !!!

					tHit,
					GaussInd,
					tHit, INFINITY, // !!! !!! !!!
					params.d,
					params.D, params.H
				);
				if (intersection) {
					SGaussianComponent Gauss = params.GC[GaussInd];
					if (params.volumetric) {
						R = R + (Gauss.R * Gauss.alpha * T);
						G = G + (Gauss.G * Gauss.alpha * T);
						B = B + (Gauss.B * Gauss.alpha * T);
						T = T * (1.0f - Gauss.alpha);
					}
					else {
						R = Gauss.R;
						G = Gauss.G;
						B = Gauss.B;
						break;
					}
				}
			} while (intersection && (T >= 1.0f / 255.0f));

			if (R > 1.0f) R = 1.0f;
			if (G > 1.0f) G = 1.0f;
			if (B > 1.0f) B = 1.0f;
			int Ri = R * 255.0f;
			int Gi = G * 255.0f;
			int Bi = B * 255.0f;
			((int*)params.bitmap)[(i * params.w) + j] = (Ri << 16) + (Gi << 8) + Bi;
		}
	}
	return 0;
}

// *** *** *** *** ***

void Render(
	SRenderParams* params
) {
	HANDLE* threads = new HANDLE[params[0].threadsNum]; // 24        
	for (int i = 0; i < params[0].threadsNum; ++i) threads[i] = CreateThread(NULL, 0, RenderRoutine, &params[i], 0, NULL);
	WaitForMultipleObjects(params[0].threadsNum, threads, true, INFINITE);
	for (int i = 0; i < params[0].threadsNum; ++i) CloseHandle(threads[i]);
	delete[] threads;
}