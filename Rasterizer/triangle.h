#pragma once

#include "mesh.h"
#include "colour.h"
#include "renderer.h"
#include "light.h"
#include <iostream>
#include <immintrin.h>
#include <xmmintrin.h>
#include <smmintrin.h>   // SSE4.1

struct LineFunc {
	float A, B, C;
	// for any (x,y): value = A*x + B*y + C
};

// Simple support class for a 2D vector
class vec2D {
public:
	float x, y;

	// Default constructor initializes both components to 0
	vec2D() { x = y = 0.f; };

	// Constructor initializes components with given values
	vec2D(float _x, float _y) : x(_x), y(_y) {}

	// Constructor initializes components from a vec4
	vec2D(vec4 v) {
		x = v[0];
		y = v[1];
	}

	// Display the vector components
	void display() { std::cout << x << '\t' << y << std::endl; }

	// Overloaded subtraction operator for vector subtraction
	vec2D operator- (vec2D& v) {
		vec2D q;
		q.x = x - v.x;
		q.y = y - v.y;
		return q;
	}
};

// Class representing a triangle for rendering purposes
class triangle {
	Vertex v[3];       // Vertices of the triangle
	float area;        // Area of the triangle
	colour col[3];     // Colors for each vertex of the triangle

public:
	// Constructor initializes the triangle with three vertices
	// Input Variables:
	// - v1, v2, v3: Vertices defining the triangle
	triangle(const Vertex& v1, const Vertex& v2, const Vertex& v3) {
		v[0] = v1;
		v[1] = v2;
		v[2] = v3;

		// Calculate the 2D area of the triangle
		vec2D e1 = vec2D(v[1].p - v[0].p);
		vec2D e2 = vec2D(v[2].p - v[0].p);
		area = abs(e1.x * e2.y - e1.y * e2.x);
	}

	inline LineFunc getArguments(vec2D v1, vec2D v2) {
		// f(x,y) = (y - v1.y)*(v2.x - v1.x)-(x - v1.x)*(v2.y - v1.y) 
		// same as 'e-q' in getC
		LineFunc lf;
		lf.A = v1.y - v2.y;
		lf.B = v2.x - v1.x;
		lf.C = v1.x * v2.y - v1.y * v2.x;

		return lf;
	}

	// Helper function to compute the cross product for barycentric coordinates
	// Input Variables:
	// - v1, v2: Edges defining the vector
	// - p: Point for which coordinates are being calculated
	inline float getC(vec2D v1, vec2D v2, vec2D p) {
		vec2D e = v2 - v1;
		vec2D q = p - v1;
		return q.y * e.x - q.x * e.y;
	}

	inline __m256 getC_avx(vec2D v1, vec2D v2, __m256 px, __m256 py) {
		__m256 e_x = _mm256_set1_ps(v2.x - v1.x);
		__m256 e_y = _mm256_set1_ps(v2.y - v1.y);
		__m256 q_x = _mm256_sub_ps(px, _mm256_set1_ps(v1.x));
		__m256 q_y = _mm256_sub_ps(py, _mm256_set1_ps(v1.y));

		return _mm256_fmsub_ps(q_y, e_x, _mm256_mul_ps(q_x, e_y));
	}

	// Compute barycentric coordinates for a given point
	// Input Variables:
	// - p: Point to check within the triangle
	// Output Variables:
	// - alpha, beta, gamma: Barycentric coordinates of the point
	// Returns true if the point is inside the triangle, false otherwise
	inline bool getCoordinates(vec2D p, float& alpha, float& beta, float& gamma) {
		alpha = getC(vec2D(v[0].p), vec2D(v[1].p), p) / area;
		beta = getC(vec2D(v[1].p), vec2D(v[2].p), p) / area;
		gamma = getC(vec2D(v[2].p), vec2D(v[0].p), p) / area;

		if (alpha < 0.f || beta < 0.f || gamma < 0.f) return false;
		return true;
	}

	// Template function to interpolate values using barycentric coordinates
	// Input Variables:
	// - alpha, beta, gamma: Barycentric coordinates
	// - a1, a2, a3: Values to interpolate
	// Returns the interpolated value
	template <typename T>
	T interpolate(float alpha, float beta, float gamma, T a1, T a2, T a3) {
		return (a1 * alpha) + (a2 * beta) + (a3 * gamma);
	}

	inline __m256 interpolate_avx(__m256 alpha, __m256 beta, __m256 gamma, float a1, float a2, float a3) {
		__m256 v1 = _mm256_set1_ps(a1);
		__m256 v2 = _mm256_set1_ps(a2);
		__m256 v3 = _mm256_set1_ps(a3);

		//alpha * a1 + beta * a2 + gamma * a3
		return _mm256_add_ps(
			_mm256_mul_ps(alpha, v1),
			_mm256_add_ps(_mm256_mul_ps(beta, v2), _mm256_mul_ps(gamma, v3))
		);
	}

	inline void interpolate_avx_color(__m256 alpha, __m256 beta, __m256 gamma,
		colour c1, colour c2, colour c3,
		__m256& r, __m256& g, __m256& b) {
		__m256 r1 = _mm256_set1_ps(c1.r);
		__m256 g1 = _mm256_set1_ps(c1.g);
		__m256 b1 = _mm256_set1_ps(c1.b);

		__m256 r2 = _mm256_set1_ps(c2.r);
		__m256 g2 = _mm256_set1_ps(c2.g);
		__m256 b2 = _mm256_set1_ps(c2.b);

		__m256 r3 = _mm256_set1_ps(c3.r);
		__m256 g3 = _mm256_set1_ps(c3.g);
		__m256 b3 = _mm256_set1_ps(c3.b);

		// r = alpha * r1 + beta * r2 + gamma * r3
		r = _mm256_add_ps(_mm256_mul_ps(alpha, r1),
			_mm256_add_ps(_mm256_mul_ps(beta, r2), _mm256_mul_ps(gamma, r3)));

		g = _mm256_add_ps(_mm256_mul_ps(alpha, g1),
			_mm256_add_ps(_mm256_mul_ps(beta, g2), _mm256_mul_ps(gamma, g3)));

		b = _mm256_add_ps(_mm256_mul_ps(alpha, b1),
			_mm256_add_ps(_mm256_mul_ps(beta, b2), _mm256_mul_ps(gamma, b3)));
	}

	// Draw the triangle on the canvas
	// Input Variables:
	// - renderer: Renderer object for drawing
	// - L: Light object for shading calculations
	// - ka, kd: Ambient and diffuse lighting coefficients
	void draw(Renderer& renderer, Light& L, float ka, float kd) {
		vec2D minV, maxV;

		// normalize the light direction
		vec4 normalizedOmega = L.omega_i;
		normalizedOmega.normalise();
		__m256 omega_x = _mm256_set1_ps(normalizedOmega.x);
		__m256 omega_y = _mm256_set1_ps(normalizedOmega.y);
		__m256 omega_z = _mm256_set1_ps(normalizedOmega.z);

		// Get the screen-space bounds of the triangle
		getBoundsWindow(renderer.canvas, minV, maxV);

		// Skip very small triangles
		if (area < 1.f) return;

		LineFunc lf0 = getArguments(vec2D(v[0].p), vec2D(v[1].p));
		LineFunc lf1 = getArguments(vec2D(v[1].p), vec2D(v[2].p));
		LineFunc lf2 = getArguments(vec2D(v[2].p), vec2D(v[0].p));

		// bounding Box
		int startX = (int)minV.x;
		int startY = (int)minV.y;
		int endX = (int)ceil(maxV.x);
		int endY = (int)ceil(maxV.y);

		// AVX segements
		int width = endX - startX;
		if (width <= 0) return;
		int aligned = (width / 8) * 8; // divisible part
		int leftover = width - aligned;

		// compute function of 3 lines in the start position
		float f0row = lf0.A * startX + lf0.B * startY + lf0.C;
		float f1row = lf1.A * startX + lf1.B * startY + lf1.C;
		float f2row = lf2.A * startX + lf2.B * startY + lf2.C;

		// increment x,y of lf0,1,2
		float dfx0 = lf0.A, dfy0 = lf0.B;
		float dfx1 = lf1.A, dfy1 = lf1.B;
		float dfx2 = lf2.A, dfy2 = lf2.B;

		__m256 x_offset = _mm256_setr_ps(0, 1, 2, 3, 4, 5, 6, 7);
		__m256 step0 = _mm256_mul_ps(x_offset, _mm256_set1_ps(dfx0));
		__m256 step1 = _mm256_mul_ps(x_offset, _mm256_set1_ps(dfx1));
		__m256 step2 = _mm256_mul_ps(x_offset, _mm256_set1_ps(dfx2));

		// Iterate over the bounding box and check each pixel
		for (int y = startY; y < endY; y++) {
			float f0 = f0row;
			float f1 = f1row;
			float f2 = f2row;

			// pixels in avx
			for (int offset = 0; offset < aligned; offset += 8) {
				int x = startX + offset;
				__m256 vf0 = _mm256_add_ps(_mm256_set1_ps(f0), step0);
				__m256 vf1 = _mm256_add_ps(_mm256_set1_ps(f1), step1);
				__m256 vf2 = _mm256_add_ps(_mm256_set1_ps(f2), step2);

				__m256 zero = _mm256_set1_ps(-1e-6);
				__m256 inTri = _mm256_and_ps(_mm256_and_ps(_mm256_cmp_ps(vf0, zero, _CMP_GE_OQ),
					_mm256_cmp_ps(vf1, zero, _CMP_GE_OQ)),
					_mm256_cmp_ps(vf2, zero, _CMP_GE_OQ));

				int mask = _mm256_movemask_ps(inTri);

				__m256 vx = _mm256_add_ps(_mm256_set1_ps(x), _mm256_setr_ps(0, 1, 2, 3, 4, 5, 6, 7));
				__m256 vy = _mm256_set1_ps(y);

				__m256 inv_area = _mm256_set1_ps(1.0f / area);

				__m256 c0 = getC_avx(vec2D(v[0].p), vec2D(v[1].p), vx, vy);
				__m256 c1 = getC_avx(vec2D(v[1].p), vec2D(v[2].p), vx, vy);
				__m256 c2 = getC_avx(vec2D(v[2].p), vec2D(v[0].p), vx, vy);

				__m256 alpha = _mm256_mul_ps(c0, inv_area);
				__m256 beta = _mm256_mul_ps(c1, inv_area);
				__m256 gamma = _mm256_mul_ps(c2, inv_area);

				alignas(32) float alpha_vals[8], beta_vals[8], gamma_vals[8];
				_mm256_storeu_ps(alpha_vals, alpha);
				_mm256_storeu_ps(beta_vals, beta);
				_mm256_storeu_ps(gamma_vals, gamma);

				if (mask != 0) {
					for (int i = 0; i < 8; i++) {
						if ((mask & (1 << i)) == 0) continue;

						// compute colour, depth and normal
						__m256 r, g, b;
						interpolate_avx_color(beta, gamma, alpha, v[0].rgb, v[1].rgb, v[2].rgb, r, g, b);

						// clamp
						r = _mm256_max_ps(_mm256_set1_ps(0.0f), _mm256_min_ps(_mm256_set1_ps(1.0f), r));
						g = _mm256_max_ps(_mm256_set1_ps(0.0f), _mm256_min_ps(_mm256_set1_ps(1.0f), g));
						b = _mm256_max_ps(_mm256_set1_ps(0.0f), _mm256_min_ps(_mm256_set1_ps(1.0f), b));

						__m256 depth = interpolate_avx(beta, gamma, alpha, v[0].p[2], v[1].p[2], v[2].p[2]);

						__m256 norm_x = interpolate_avx(beta, gamma, alpha, v[0].normal.x, v[1].normal.x, v[2].normal.x);
						__m256 norm_y = interpolate_avx(beta, gamma, alpha, v[0].normal.y, v[1].normal.y, v[2].normal.y);
						__m256 norm_z = interpolate_avx(beta, gamma, alpha, v[0].normal.z, v[1].normal.z, v[2].normal.z);

						// normalize
						__m256 length_squared = _mm256_add_ps(
							_mm256_add_ps(_mm256_mul_ps(norm_x, norm_x),
								_mm256_mul_ps(norm_y, norm_y)),
							_mm256_mul_ps(norm_z, norm_z));

						// Avoiding zero vector normalisation leading to NaN
						__m256 epsilon = _mm256_set1_ps(1e-8);
						__m256 inv_length = _mm256_div_ps(_mm256_set1_ps(1.0f), _mm256_sqrt_ps(_mm256_max_ps(length_squared, epsilon)));

						norm_x = _mm256_mul_ps(norm_x, inv_length);
						norm_y = _mm256_mul_ps(norm_y, inv_length);
						norm_z = _mm256_mul_ps(norm_z, inv_length);

						// 提取 'depth' 标量数据
						alignas(32) float depth_vals[8];
						_mm256_storeu_ps(depth_vals, depth);

						//if (mask & (1 << i)) {
						int px = x + i;
						if (renderer.zbuffer(px, y) > depth_vals[i] && depth_vals[i] > 0.01f) {
							// compute light
							__m256 dot = _mm256_max_ps(zero, _mm256_add_ps(_mm256_add_ps(_mm256_mul_ps(omega_x, norm_x),
								_mm256_mul_ps(omega_y, norm_y)),
								_mm256_mul_ps(omega_z, norm_z)));

							// light colour
							__m256 Lr = _mm256_set1_ps(L.L.r);
							__m256 Lg = _mm256_set1_ps(L.L.g);
							__m256 Lb = _mm256_set1_ps(L.L.b);

							// ambient colour
							__m256 Ar = _mm256_set1_ps(L.ambient.r);
							__m256 Ag = _mm256_set1_ps(L.ambient.g);
							__m256 Ab = _mm256_set1_ps(L.ambient.b);

							// diffuse
							__m256 kd_vec = _mm256_set1_ps(kd);

							// shading = kd * (L.L * dot + L.ambient * kd)
							__m256 shading_r = _mm256_mul_ps(kd_vec, _mm256_add_ps(_mm256_mul_ps(Lr, dot), _mm256_mul_ps(Ar, kd_vec)));
							__m256 shading_g = _mm256_mul_ps(kd_vec, _mm256_add_ps(_mm256_mul_ps(Lg, dot), _mm256_mul_ps(Ag, kd_vec)));
							__m256 shading_b = _mm256_mul_ps(kd_vec, _mm256_add_ps(_mm256_mul_ps(Lb, dot), _mm256_mul_ps(Ab, kd_vec)));

							// a = c * shading				
							r = _mm256_mul_ps(r, shading_r);
							g = _mm256_mul_ps(g, shading_g);
							b = _mm256_mul_ps(b, shading_b);

							r = _mm256_mul_ps(r, _mm256_set1_ps(255));
							g = _mm256_mul_ps(g, _mm256_set1_ps(255));
							b = _mm256_mul_ps(b, _mm256_set1_ps(255));

							//__m256i r_int = _mm256_cvtps_epi32(r);
							//__m256i g_int = _mm256_cvtps_epi32(g);
							//__m256i b_int = _mm256_cvtps_epi32(b);

							// 提取 `r, g, b` 标量数据
							alignas(32) float r_vals[8], g_vals[8], b_vals[8];
							_mm256_storeu_ps(r_vals, r);
							_mm256_storeu_ps(g_vals, g);
							_mm256_storeu_ps(b_vals, b);

							// 写入颜色
							unsigned char cr = static_cast<unsigned char>(std::floor(r_vals[i]));
							unsigned char cg = static_cast<unsigned char>(std::floor(g_vals[i]));
							unsigned char cb = static_cast<unsigned char>(std::floor(b_vals[i]));
							renderer.canvas.draw(px, y, cr, cg, cb);
							//renderer.canvas.draw_avx(px, y, r_int, g_int, b_int);

							// 更新 Z-buffer
							renderer.zbuffer(px, y) = depth_vals[i];
							//}
						}
					}
				}


				// move to next pixel with increment x and y
				f0 += 8 * dfx0;
				f1 += 8 * dfx1;
				f2 += 8 * dfx2;
			}

			// processing remaining pixels
			int xLeft = startX + aligned;
			for (int i = 0; i < leftover; i++) {
				int xx = xLeft + i;
				float f0_cur = f0 + i * dfx0;
				float f1_cur = f1 + i * dfx1;
				float f2_cur = f2 + i * dfx2;
				if (f0_cur >= 0 && f1_cur >= 0 && f2_cur >= 0) {
					float alpha, beta, gamma;
					if (getCoordinates(vec2D((float)xx, (float)y), alpha, beta, gamma)) {
						// Interpolate color, depth, and normals
						colour c = interpolate(beta, gamma, alpha, v[0].rgb, v[1].rgb, v[2].rgb);
						c.clampColour();
						float depth = interpolate(beta, gamma, alpha, v[0].p[2], v[1].p[2], v[2].p[2]);
						vec4 normal = interpolate(beta, gamma, alpha, v[0].normal, v[1].normal, v[2].normal);
						normal.normalise();

						// Perform Z-buffer test and apply shading
						if (renderer.zbuffer(xx, y) > depth && depth > 0.01f) {
							// typical shader begin
							L.omega_i.normalise();
							float dot = max(vec4::dot(L.omega_i, normal), 0.0f);
							colour a = (c * kd) * (L.L * dot + (L.ambient * kd));
							// typical shader end
							unsigned char r, g, b;
							a.toRGB(r, g, b);
							renderer.canvas.draw(xx, y, r, g, b);
							renderer.zbuffer(xx, y) = depth;
						}
					}
				}
			}

			// end this row and to the next row
			f0row += dfy0;
			f1row += dfy1;
			f2row += dfy2;
		}
	}

	// Compute the 2D bounds of the triangle
	// Output Variables:
	// - minV, maxV: Minimum and maximum bounds in 2D space
	void getBounds(vec2D& minV, vec2D& maxV) {
		minV = vec2D(v[0].p);
		maxV = vec2D(v[0].p);
		for (unsigned int i = 1; i < 3; i++) {
			minV.x = min(minV.x, v[i].p[0]);
			minV.y = min(minV.y, v[i].p[1]);
			maxV.x = max(maxV.x, v[i].p[0]);
			maxV.y = max(maxV.y, v[i].p[1]);
		}
	}

	// Compute the 2D bounds of the triangle, clipped to the canvas
	// Input Variables:
	// - canvas: Reference to the rendering canvas
	// Output Variables:
	// - minV, maxV: Clipped minimum and maximum bounds
	void getBoundsWindow(GamesEngineeringBase::Window& canvas, vec2D& minV, vec2D& maxV) {
		getBounds(minV, maxV);
		minV.x = max(minV.x, 0);
		minV.y = max(minV.y, 0);
		maxV.x = min(maxV.x, canvas.getWidth());
		maxV.y = min(maxV.y, canvas.getHeight());
	}

	// Debugging utility to display the triangle bounds on the canvas
	// Input Variables:
	// - canvas: Reference to the rendering canvas
	void drawBounds(GamesEngineeringBase::Window& canvas) {
		vec2D minV, maxV;
		getBounds(minV, maxV);

		for (int y = (int)minV.y; y < (int)maxV.y; y++) {
			for (int x = (int)minV.x; x < (int)maxV.x; x++) {
				canvas.draw(x, y, 255, 0, 0);
			}
		}
	}

	// Debugging utility to display the coordinates of the triangle vertices
	void display() {
		for (unsigned int i = 0; i < 3; i++) {
			v[i].p.display();
		}
		std::cout << std::endl;
	}
};
