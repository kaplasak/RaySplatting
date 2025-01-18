#include <optix_device.h>

#include "Header.cuh"

// Dla pliku shaders.cu nale¿y ustawiæ typ kompilacji na "Generate device-only .ptx file (-ptx)" ustawiaj¹c:
// CUDA C/C++ -> Common -> NVCC Compilation Type na "Generate device-only .ptx file (-ptx)", a nastêpnie ustawiæ opcjê:
// CUDA C/C++ -> Common -> Compilaer Output (obj/cubin) na "$(CudaIntDirFullPath)\%(Filename)%(Extension).ptx", aby zapobiec skasowaniu
// pliku w procesie kompilacji. Utworzony plik *.ptx powinien pojawiæ siê w folderze <nazwa projektu>\<nazwa projektu>\x64\Release
// (Uwaga! Nie w g³ównym folderze zawieraj¹cym wyjœcie kompilatora: <nazwa projektu>\x64\Release).

// *************************************************************************************************

struct LaunchParams {
	unsigned *bitmap;
	unsigned width;
	unsigned height;

	float3 O;
	float3 R, D, F;
	float double_tan_half_fov_x;
	float double_tan_half_fov_y;

	OptixTraversableHandle traversable;

	float4 *GC_part_1;
	float4 *GC_part_2;
	float4 *GC_part_3;
	float2 *GC_part_4;

	int *Gaussians_indices;
	REAL_G *bitmap_out_R, *bitmap_out_G, *bitmap_out_B;

	float ray_termination_T_threshold;
	float last_significant_Gauss_alpha_gradient_precision;
	float chi_square_squared_radius;
	int max_Gaussians_per_ray;
};

// *************************************************************************************************

extern "C" __constant__ LaunchParams optixLaunchParams;

// *************************************************************************************************

// 3115 2078 "fy": 3240.8079303073573, "fx": 3231.5814007282897

// !!! !!! !!!
// Dodanie extern "C" zapobiega name manglingowi, który utrudni³by znalezienie w za³adowanym pliku *.ptx funkcji po ich oryginalnych nazwach.
// !!! !!! !!!
extern "C" __global__ void __raygen__renderFrame() {
	int x = optixGetLaunchIndex().x;
	int y = optixGetLaunchIndex().y;
	int pixel_ind = (y * optixLaunchParams.width) + x;
	
	REAL3_R d = make_REAL3_R(
		(((REAL_R)-0.5) + ((x + ((REAL_R)0.5)) / optixLaunchParams.width)) * optixLaunchParams.double_tan_half_fov_x,
		(((REAL_R)-0.5) + ((y + ((REAL_R)0.5)) / optixLaunchParams.height)) * optixLaunchParams.double_tan_half_fov_y,
		1,
	);
	/*REAL3_R d = make_REAL3_R(
		((REAL_R)-0.5) + ((x + ((REAL_R)0.5)) / optixLaunchParams.width),
		((REAL_R)-0.5) + ((y + ((REAL_R)0.5)) / optixLaunchParams.height),
		((REAL_R)0.5) / TAN_R(((REAL_R)0.5) * optixLaunchParams.FOV),
	);*/
	REAL3_R v = make_REAL3_R(
		MAD_R(optixLaunchParams.R.x, d.x, MAD_R(optixLaunchParams.D.x, d.y, optixLaunchParams.F.x * d.z)),
		MAD_R(optixLaunchParams.R.y, d.x, MAD_R(optixLaunchParams.D.y, d.y, optixLaunchParams.F.y * d.z)),
		MAD_R(optixLaunchParams.R.z, d.x, MAD_R(optixLaunchParams.D.z, d.y, optixLaunchParams.F.z * d.z))
	);

	float tMin = 0.0f;

	REAL_R R = 0;
	REAL_R G = 0;
	REAL_R B = 0;

	REAL_R T = 1;
	REAL_R TLastGrad = 1;

	bool belowThreshold = false;

	for (int i = 0; i < optixLaunchParams.max_Gaussians_per_ray; ++i) {
		unsigned Gauss_ind = -1;
		unsigned t_as_uint;
		
		optixTrace(
			optixLaunchParams.traversable,
			optixLaunchParams.O,
			#ifndef RENDERER_OPTIX_USE_DOUBLE_PRECISION
				v,
			#else
				make_float3(v.x, v.y, v.z),
			#endif
			tMin,     // tmin
			INFINITY, // tmax
			0.0f,     // rayTime
			OptixVisibilityMask(255),
			OPTIX_RAY_FLAG_NONE,
			0,
			1,
			0,

			Gauss_ind,
			t_as_uint
		);
		
		optixLaunchParams.Gaussians_indices[(i * optixLaunchParams.width * optixLaunchParams.height) + pixel_ind] = Gauss_ind;
		if (Gauss_ind != -1) {
			float4 GC_1 = optixLaunchParams.GC_part_1[Gauss_ind];
			float4 GC_2 = optixLaunchParams.GC_part_2[Gauss_ind];
			float4 GC_3 = optixLaunchParams.GC_part_3[Gauss_ind];
			float2 GC_4 = optixLaunchParams.GC_part_4[Gauss_ind];

			// *************************************************************************************

			REAL_R aa = ((REAL_R)GC_3.z) * GC_3.z;
			REAL_R bb = ((REAL_R)GC_3.w) * GC_3.w;
			REAL_R cc = ((REAL_R)GC_4.x) * GC_4.x;
			REAL_R dd = ((REAL_R)GC_4.y) * GC_4.y;
			REAL_R s = ((REAL_R)0.5) * (aa + bb + cc + dd);

			REAL_R ab = GC_3.z * GC_3.w; REAL_R ac = GC_3.z * GC_4.x; REAL_R ad = GC_3.z * GC_4.y;
										 REAL_R bc = GC_3.w * GC_4.x; REAL_R bd = GC_3.w * GC_4.y;
											    					  REAL_R cd = GC_4.x * GC_4.y;

			REAL_R R11 = s - cc - dd;
			REAL_R R12 = bc - ad;
			REAL_R R13 = bd + ac;

			REAL_R R21 = bc + ad;
			REAL_R R22 = s - bb - dd;
			REAL_R R23 = cd - ab;

			REAL_R R31 = bd - ac;
			REAL_R R32 = cd + ab;
			REAL_R R33 = s - bb - cc;

			// *************************************************************************************

			REAL3_R O = make_REAL3_R(
				optixLaunchParams.O.x - GC_2.x,
				optixLaunchParams.O.y - GC_2.y,
				optixLaunchParams.O.z - GC_2.z
			);

			REAL3_R O_prim;
			REAL3_R v_prim;

			REAL_R sXInv = ((REAL_R)1.0) + EXP_R(-GC_2.w);
			O_prim.x = MAD_R(R11, O.x, MAD_R(R21, O.y, R31 * O.z)) * sXInv;
			v_prim.x = MAD_R(R11, v.x, MAD_R(R21, v.y, R31 * v.z)) * sXInv;

			REAL_R sYInv = ((REAL_R)1.0) + EXP_R(-GC_3.x);
			O_prim.y = MAD_R(R12, O.x, MAD_R(R22, O.y, R32 * O.z)) * sYInv;
			v_prim.y = MAD_R(R12, v.x, MAD_R(R22, v.y, R32 * v.z)) * sYInv;

			REAL_R sZInv = ((REAL_R)1.0) + EXP_R(-GC_3.y);
			O_prim.z = MAD_R(R13, O.x, MAD_R(R23, O.y, R33 * O.z)) * sZInv;
			v_prim.z = MAD_R(R13, v.x, MAD_R(R23, v.y, R33 * v.z)) * sZInv;

			// *************************************************************************************

			REAL_R v_dot_v = MAD_R(v_prim.x, v_prim.x, MAD_R(v_prim.y, v_prim.y, v_prim.z * v_prim.z));
			REAL_R O_dot_O = MAD_R(O_prim.x, O_prim.x, MAD_R(O_prim.y, O_prim.y, O_prim.z * O_prim.z));
			REAL_R v_dot_O = MAD_R(v_prim.x, O_prim.x, MAD_R(v_prim.y, O_prim.y, v_prim.z * O_prim.z));
			// OLD
			//REAL_R alpha = EXP_R(((REAL_R)0.5) * ((((v_dot_O * v_dot_O) / v_dot_v) - O_dot_O) / (s * s)));
			// NEW (MORE NUMERICALLY STABLE)
			REAL tmp_ = v_dot_O / v_dot_v;
			REAL_G O_perp_x = MAD_G(-v_prim.x, tmp_, O_prim.x); // !!! !!! !!!
			REAL_G O_perp_y = MAD_G(-v_prim.y, tmp_, O_prim.y); // !!! !!! !!!
			REAL_G O_perp_z = MAD_G(-v_prim.z, tmp_, O_prim.z); // !!! !!! !!!
			REAL_R alpha = EXP_R(-((REAL_R)0.5) * (MAD_R(O_perp_x, O_perp_x, MAD_R(O_perp_y, O_perp_y, O_perp_z * O_perp_z)) / (s * s)));
			#ifndef RENDERER_OPTIX_USE_DOUBLE_PRECISION
				alpha = __saturatef(alpha) / (((REAL_R)1.0) + EXP_R(-GC_1.w));
			#else
				alpha = (alpha < 0) ? 0 : alpha;
				alpha = (alpha > 1) ? 1 : alpha;
				alpha = alpha / (((REAL_R)1.0) + EXP_R(-GC_1.w));
			#endif

			REAL_R tmp = T * alpha;

			R = R + (GC_1.x * tmp);
			G = G + (GC_1.y * tmp);
			B = B + (GC_1.z * tmp);
			T = T - tmp; // T = T * (1 - alpha)

			if (T < ((REAL_R)optixLaunchParams.ray_termination_T_threshold)) {
				if (!belowThreshold) belowThreshold = true;
				else
					TLastGrad = TLastGrad * (1 - alpha);
			}
			if (TLastGrad >= ((REAL_R)optixLaunchParams.last_significant_Gauss_alpha_gradient_precision)) {					
				float t = __uint_as_float(t_as_uint);
				tMin = nextafter(t, INFINITY);
			} else {
				if (i < optixLaunchParams.max_Gaussians_per_ray - 1)
					optixLaunchParams.Gaussians_indices[((i + 1) * optixLaunchParams.width * optixLaunchParams.height) + pixel_ind] = -1;
				break;
			}
		} else
			break;
	}

	#ifndef RENDERER_OPTIX_USE_DOUBLE_PRECISION
		R = __saturatef(R);
		G = __saturatef(G);
		B = __saturatef(B);
	#else
		//R = MIN_R(MAX_R(((REAL_R)R), ((REAL_R)0)), ((REAL_R)1));
		R = (R < 0.0) ? 0.0 : R;
		R = (R > 1.0) ? 1.0 : R;
		//G = MIN_R(MAX_R(((REAL_R)G), ((REAL_R)0)), ((REAL_R)1));
		G = (G < 0.0) ? 0.0 : G;
		G = (G > 1.0) ? 1.0 : G;
		//B = MIN_R(MAX_R(((REAL_R)B), ((REAL_R)0)), ((REAL_R)1));
		B = (B < 0.0) ? 0.0 : B;
		B = (B > 1.0) ? 1.0 : B;
	#endif
	int Ri = RINT_R(R * 255);
	int Gi = RINT_R(G * 255);
	int Bi = RINT_R(B * 255);

	optixLaunchParams.bitmap[pixel_ind] = (Ri << 16) + (Gi << 8) + Bi;
	optixLaunchParams.bitmap_out_R[(y * (optixLaunchParams.width + 11 - 1)) + x] = R; // !!! !!! !!!
	optixLaunchParams.bitmap_out_G[(y * (optixLaunchParams.width + 11 - 1)) + x] = G; // !!! !!! !!!
	optixLaunchParams.bitmap_out_B[(y * (optixLaunchParams.width + 11 - 1)) + x] = B; // !!! !!! !!!
}

// *************************************************************************************************

extern "C" __global__ void __anyhit__radiance() {
}

// *************************************************************************************************

extern "C" __global__ void __closesthit__radiance() {
	unsigned Gauss_ind = optixGetPrimitiveIndex();
	optixSetPayload_0(Gauss_ind);
	optixSetPayload_1(__float_as_uint(optixGetRayTmax()));
}

// *************************************************************************************************

extern "C" __global__ void __intersection__is() {
	int index = optixGetPrimitiveIndex();

	float4 GC_1 = optixLaunchParams.GC_part_1[index];
	float4 GC_2 = optixLaunchParams.GC_part_2[index];
	float4 GC_3 = optixLaunchParams.GC_part_3[index];
	float2 GC_4 = optixLaunchParams.GC_part_4[index];

	// *********************************************************************************************

	REAL_R aa = ((REAL_R)GC_3.z) * GC_3.z;
	REAL_R bb = ((REAL_R)GC_3.w) * GC_3.w;
	REAL_R cc = ((REAL_R)GC_4.x) * GC_4.x;
	REAL_R dd = ((REAL_R)GC_4.y) * GC_4.y;
	REAL_R s = ((REAL_R)0.5) * (aa + bb + cc + dd);

	REAL_R ab = ((REAL_R)GC_3.z) * GC_3.w; REAL_R ac = ((REAL_R)GC_3.z) * GC_4.x; REAL_R ad = ((REAL_R)GC_3.z) * GC_4.y;
										   REAL_R bc = ((REAL_R)GC_3.w) * GC_4.x; REAL_R bd = ((REAL_R)GC_3.w) * GC_4.y;
																				  REAL_R cd = ((REAL_R)GC_4.x) * GC_4.y;

	REAL_R R11 = s - cc - dd;
	REAL_R R12 = bc - ad;
	REAL_R R13 = bd + ac;

	REAL_R R21 = bc + ad;
	REAL_R R22 = s - bb - dd;
	REAL_R R23 = cd - ab;

	REAL_R R31 = bd - ac;
	REAL_R R32 = cd + ab;
	REAL_R R33 = s - bb - cc;
	
	// *********************************************************************************************

	float3 O = optixGetObjectRayOrigin();
	float3 v = optixGetObjectRayDirection();

	REAL_R Ox = ((REAL_R)O.x) - GC_2.x;
	REAL_R Oy = ((REAL_R)O.y) - GC_2.y;
	REAL_R Oz = ((REAL_R)O.z) - GC_2.z;

	REAL_R sXInv = 1 + EXP_R(-GC_2.w);
	REAL_R Ox_prim = MAD_R(R11, Ox, MAD_R(R21, Oy, R31 * Oz)) * sXInv;
	REAL_R vx_prim = MAD_R(R11, v.x, MAD_R(R21, v.y, R31 * v.z)) * sXInv;

	REAL_R sYInv = 1 + EXP_R(-GC_3.x);
	REAL_R Oy_prim = MAD_R(R12, Ox, MAD_R(R22, Oy, R32 * Oz)) * sYInv;
	REAL_R vy_prim = MAD_R(R12, v.x, MAD_R(R22, v.y, R32 * v.z)) * sYInv;

	REAL_R sZInv = 1 + EXP_R(-GC_3.y);
	REAL_R Oz_prim = MAD_R(R13, Ox, MAD_R(R23, Oy, R33 * Oz)) * sZInv;
	REAL_R vz_prim = MAD_R(R13, v.x, MAD_R(R23, v.y, R33 * v.z)) * sZInv;
	
	// *********************************************************************************************

	// OLD
	/*REAL_R a = MAD_R(vx_prim, vx_prim, MAD_R(vy_prim, vy_prim, vz_prim * vz_prim));
	REAL_R b = 2 * MAD_R(Ox_prim, vx_prim, MAD_R(Oy_prim, vy_prim, Oz_prim * vz_prim));
	REAL_R c = MAD_R(Ox_prim, Ox_prim, MAD_R(Oy_prim, Oy_prim, MAD_R(Oz_prim, Oz_prim, -((REAL)optixLaunchParams.chi_square_squared_radius) * s * s)));
	REAL_R delta = MAD_R(b, b, -4 * a * c);
	if (delta >= 0) {
		REAL_R t1 = (-b - SQRT_R(delta)) / (2 * a);
		optixReportIntersection(
			((float)t1),
			0
		);
	}*/

	// NEW (MORE NUMERICALLY STABLE)
	REAL_R v_dot_v = MAD_R(vx_prim, vx_prim, MAD_R(vy_prim, vy_prim, vz_prim * vz_prim));
	REAL_R O_dot_v = MAD_R(Ox_prim, vx_prim, MAD_R(Oy_prim, vy_prim, Oz_prim * vz_prim));
	REAL_R O_dot_O = MAD_R(Ox_prim, Ox_prim, MAD_R(Oy_prim, Oy_prim, Oz_prim * Oz_prim));

	REAL_R tmp1 = 1 / v_dot_v;
	REAL_R tmp2 = O_dot_v * tmp1;

	REAL_R O_perp_x = MAD_R(-vx_prim, tmp2, Ox_prim);
	REAL_R O_perp_y = MAD_R(-vy_prim, tmp2, Oy_prim);
	REAL_R O_perp_z = MAD_R(-vz_prim, tmp2, Oz_prim);

	REAL_R delta = -MAD_R(O_perp_x, O_perp_x, MAD_R(O_perp_y, O_perp_y, MAD_R(O_perp_z, O_perp_z, -((REAL)optixLaunchParams.chi_square_squared_radius) * s * s))); // !!! !!! !!!
	if (delta >= 0) {
		REAL_R t = -(O_dot_v + COPYSIGN_R(SQRT_R(v_dot_v * delta), O_dot_v)) * tmp1;
		REAL_R t1 = ((O_dot_v <= 0) ? (O_dot_O - (((REAL)optixLaunchParams.chi_square_squared_radius) * s * s)) / (v_dot_v * t) : t);
		optixReportIntersection(
			((float)t1),
			0
		);
	}
}

// *************************************************************************************************

struct FloatFloat {
	float x, y;

	__device__ FloatFloat(): x(0.0f), y(0.0f) {};

	__device__ FloatFloat(float x): x(x), y(0.0f) {};

	__device__ FloatFloat(double x): x(x), y(0.0f) {};
};

/*__device__ FloatFloat operator+(const FloatFloat& value1, const float value2) {
	float tmp1, tmp2;
	FloatFloat result;

	tmp1 = __fadd_rn(value1.x, value2);
	if (fabsf(value1.x) > fabsf(value2))
		tmp2 = __fadd_rn(__fadd_rn(__fsub_rn(value1.x, tmp1), value2), value1.y);
	else
		tmp2 = __fadd_rn(__fadd_rn(__fsub_rn(value2, tmp1), value1.x), value1.y);
	result.x = __fadd_rn(tmp1, tmp2);
	result.y = __fadd_rn(__fsub_rn(tmp1, result.x), tmp2);
	return result;
}

__device__ FloatFloat operator-(const FloatFloat& value1, const float value2) {
	float tmp1, tmp2;
	FloatFloat result;

	tmp1 = __fsub_rn(value1.x, value2);
	if (fabsf(value1.x) > fabsf(value2))
		tmp2 = __fadd_rn(__fsub_rn(__fsub_rn(value1.x, tmp1), value2), value1.y);
	else
		tmp2 = __fadd_rn(__fadd_rn(__fsub_rn(-value2, tmp1), value1.x), value1.y);
	result.x = __fadd_rn(tmp1, tmp2);
	result.y = __fadd_rn(__fsub_rn(tmp1, result.x), tmp2);
	return result;
}*/

__device__ FloatFloat operator+(const FloatFloat& value1, const FloatFloat &value2) {
float tmp1, tmp2;
FloatFloat result;

tmp1 = __fadd_rn(value1.x, value2.x);
if (fabsf(value1.x) > fabsf(value2.x))
tmp2 = __fadd_rn(__fadd_rn(__fadd_rn(__fsub_rn(value1.x, tmp1), value2.x), value2.y), value1.y);
else
tmp2 = __fadd_rn(__fadd_rn(__fadd_rn(__fsub_rn(value2.x, tmp1), value1.x), value1.y), value2.y);
result.x = __fadd_rn(tmp1, tmp2);
result.y = __fadd_rn(__fsub_rn(tmp1, result.x), tmp2);
return result;
}

__device__ FloatFloat operator-(const FloatFloat& value) {
FloatFloat result;

result.x = -value.x;
result.y = -value.y;
return result;
}


__device__ FloatFloat operator-(const FloatFloat& value1, const FloatFloat &value2) {
float tmp1, tmp2;
FloatFloat result;

tmp1 = __fsub_rn(value1.x, value2.x);
if (fabsf(value1.x) > fabsf(value2.x))
tmp2 = __fadd_rn(__fsub_rn(__fsub_rn(__fsub_rn(value1.x, tmp1), value2.x), value2.y), value1.y);
else
tmp2 = __fsub_rn(__fadd_rn(__fadd_rn(__fsub_rn(-value2.x, tmp1), value1.x), value1.y), value2.y);
result.x = __fadd_rn(tmp1, tmp2);
result.y = __fadd_rn(__fsub_rn(tmp1, result.x), tmp2);
return result;
}

__device__ FloatFloat operator*(const FloatFloat& value1, const FloatFloat& value2) {
float tmp1, tmp2;
FloatFloat result;

tmp1 = __fmul_rn(value1.x, value2.x);
tmp2 = __fmaf_rn(value1.x, value2.y, __fmaf_rn(value1.y, value2.x, __fmaf_rn(value1.x, value2.x, -tmp1)));
result.x = __fadd_rn(tmp1, tmp2);
result.y = __fadd_rn(__fsub_rn(tmp1, result.x), tmp2);
return result;
}

__device__ FloatFloat operator/(const FloatFloat& value1, const FloatFloat& value2) {
FloatFloat value2Corrected;
float tmp1, tmp2, tmp3, tmp4;
FloatFloat result;

if (value2.x != 0.0f)
value2Corrected = value2;
else {
value2Corrected.x = value2.y;
value2Corrected.y = 0.0f;
}
tmp1 = __fdiv_rn(value1.x, value2Corrected.x);
tmp2 = __fmul_rn(tmp1, value2Corrected.x);
tmp3 = __fmaf_rn(tmp1, value2Corrected.x, -tmp2);
tmp4 = __fdiv_rn(__fmaf_rn(-tmp1, value2Corrected.y, __fadd_rn(__fsub_rn(__fsub_rn(value1.x, tmp2), tmp3), value1.y)), value2Corrected.x); 	
result.x = __fadd_rn(tmp1, tmp4);
result.y = __fadd_rn(__fsub_rn(tmp1, result.x), tmp4);
return result;
}

