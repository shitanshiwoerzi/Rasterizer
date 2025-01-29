#pragma once

#include <iostream>
#include <vector>
#include "vec4.h"
#include <xmmintrin.h>  // SSE
#include <smmintrin.h>  // SSE 4.1

// Matrix class for 4x4 transformation matrices
class matrix {
	alignas(16) float a[16];

public:
	// Default constructor initializes the matrix as an identity matrix
	matrix() {
		identity();
	}

	// Access matrix elements by row and column
	inline float& operator()(unsigned row, unsigned col) {
		return a[row * 4 + col];
	}
	inline float operator()(unsigned row, unsigned col) const {
		return a[row * 4 + col];
	}

	// Display the matrix elements in a readable format
	void display() {
		for (unsigned int i = 0; i < 4; i++) {
			for (unsigned int j = 0; j < 4; j++)
				std::cout << a[i * 4 + j] << '\t';
			std::cout << std::endl;
		}
	}

	// Multiply the matrix by a 4D vector
	// Input Variables:
	// - v: vec4 object to multiply with the matrix
	// Returns the resulting transformed vec4
	vec4 operator * (const vec4& v) const {
		vec4 result;

		// load the vector and 4 rows of matrix
		__m128 vec = _mm_loadu_ps(v.v);
		__m128 row0 = _mm_loadu_ps(&a[0]);
		__m128 row1 = _mm_loadu_ps(&a[4]);
		__m128 row2 = _mm_loadu_ps(&a[8]);
		__m128 row3 = _mm_loadu_ps(&a[12]);

		// dot product
		// put result to the register x channel
		__m128 dp0 = _mm_dp_ps(row0, vec, 0xF1);
		__m128 dp1 = _mm_dp_ps(row1, vec, 0xF1);
		__m128 dp2 = _mm_dp_ps(row2, vec, 0xF1);
		__m128 dp3 = _mm_dp_ps(row3, vec, 0xF1);

		// write dp0,1,2,3 to result
		result[0] = _mm_cvtss_f32(dp0);
		result[1] = _mm_cvtss_f32(dp1);
		result[2] = _mm_cvtss_f32(dp2);
		result[3] = _mm_cvtss_f32(dp3);

		return result;
	}

	// Multiply the matrix by another matrix
	// Input Variables:
	// - mx: Another matrix to multiply with
	// Returns the resulting matrix
	matrix operator * (const matrix& mx) const {
		matrix ret;

		// load left matrix
		__m128 row0 = _mm_loadu_ps(&a[0]);
		__m128 row1 = _mm_loadu_ps(&a[4]);
		__m128 row2 = _mm_loadu_ps(&a[8]);
		__m128 row3 = _mm_loadu_ps(&a[12]);

		// 4 columns of the right matrix 'mx'
		for (int col = 0; col < 4; ++col) {
			__m128 mxCol = _mm_set_ps(
				mx.a[3 * 4 + col],
				mx.a[2 * 4 + col],
				mx.a[1 * 4 + col],
				mx.a[0 * 4 + col]);

			// dot product
			__m128 dp0 = _mm_dp_ps(row0, mxCol, 0xF1);
			__m128 dp1 = _mm_dp_ps(row1, mxCol, 0xF1);
			__m128 dp2 = _mm_dp_ps(row2, mxCol, 0xF1);
			__m128 dp3 = _mm_dp_ps(row3, mxCol, 0xF1);

			// store results to ret
			ret.a[0 * 4 + col] = _mm_cvtss_f32(dp0);
			ret.a[1 * 4 + col] = _mm_cvtss_f32(dp1);
			ret.a[2 * 4 + col] = _mm_cvtss_f32(dp2);
			ret.a[3 * 4 + col] = _mm_cvtss_f32(dp3);
		}

		return ret;
	}

	// Create a perspective projection matrix
	// Input Variables:
	// - fov: Field of view in radians
	// - aspect: Aspect ratio of the viewport
	// - n: Near clipping plane
	// - f: Far clipping plane
	// Returns the perspective matrix
	static matrix makePerspective(float fov, float aspect, float n, float f) {
		matrix m;
		m.zero();
		float tanHalfFov = std::tan(fov / 2.0f);

		m.a[0] = 1.0f / (aspect * tanHalfFov);
		m.a[5] = 1.0f / tanHalfFov;
		m.a[10] = -f / (f - n);
		m.a[11] = -(f * n) / (f - n);
		m.a[14] = -1.0f;
		return m;
	}

	// Create a translation matrix
	// Input Variables:
	// - tx, ty, tz: Translation amounts along the X, Y, and Z axes
	// Returns the translation matrix
	static matrix makeTranslation(float tx, float ty, float tz) {
		matrix m;
		m.identity();
		m.a[3] = tx;
		m.a[7] = ty;
		m.a[11] = tz;
		return m;
	}

	// Create a rotation matrix around the Z-axis
	// Input Variables:
	// - aRad: Rotation angle in radians
	// Returns the rotation matrix
	static matrix makeRotateZ(float aRad) {
		matrix m;
		m.identity();
		m.a[0] = std::cos(aRad);
		m.a[1] = -std::sin(aRad);
		m.a[4] = std::sin(aRad);
		m.a[5] = std::cos(aRad);
		return m;
	}

	// Create a rotation matrix around the X-axis
	// Input Variables:
	// - aRad: Rotation angle in radians
	// Returns the rotation matrix
	static matrix makeRotateX(float aRad) {
		matrix m;
		m.identity();
		m.a[5] = std::cos(aRad);
		m.a[6] = -std::sin(aRad);
		m.a[9] = std::sin(aRad);
		m.a[10] = std::cos(aRad);
		return m;
	}

	// Create a rotation matrix around the Y-axis
	// Input Variables:
	// - aRad: Rotation angle in radians
	// Returns the rotation matrix
	static matrix makeRotateY(float aRad) {
		matrix m;
		m.identity();
		m.a[0] = std::cos(aRad);
		m.a[2] = std::sin(aRad);
		m.a[8] = -std::sin(aRad);
		m.a[10] = std::cos(aRad);
		return m;
	}

	// Create a composite rotation matrix from X, Y, and Z rotations
	// Input Variables:
	// - x, y, z: Rotation angles in radians around each axis
	// Returns the composite rotation matrix
	static matrix makeRotateXYZ(float x, float y, float z) {
		return matrix::makeRotateX(x) * matrix::makeRotateY(y) * matrix::makeRotateZ(z);
	}

	// Create a scaling matrix
	// Input Variables:
	// - s: Scaling factor
	// Returns the scaling matrix
	static matrix makeScale(float s) {
		matrix m;
		s = max(s, 0.01f); // Ensure scaling factor is not too small
		m.identity();
		m.a[0] = s;
		m.a[5] = s;
		m.a[10] = s;
		return m;
	}

	// Create an identity matrix
	// Returns an identity matrix
	static matrix makeIdentity() {
		matrix m;
		for (int i = 0; i < 4; ++i) {
			for (int j = 0; j < 4; ++j) {
				m(i, j) = (i == j) ? 1.0f : 0.0f;
			}
		}
		return m;
	}

private:
	// Set all elements of the matrix to 0
	void zero() {
		for (unsigned int i = 0; i < 16; i++)
			a[i] = 0.f;
	}

	// Set the matrix as an identity matrix
	void identity() {
		for (int i = 0; i < 4; ++i) {
			for (int j = 0; j < 4; ++j) {
				a[i * 4 + j] = (i == j) ? 1.0f : 0.0f;
			}
		}
	}
};


