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