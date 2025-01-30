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

	LineFunc getArguments(vec2D v1, vec2D v2) {
		// f(x,y) = (y - v1.y)*(v2.x - v1.x)-(x - v1.x)*(v2.y - v1.y) 
		// same as 'e-q' in getC
		LineFunc lf;
		lf.A = -(v2.y - v1.y);
		lf.B = (v2.x - v1.x);
		lf.C = (v1.x * v2.y - v1.y * v2.x);

		return lf;
	}

	// Helper function to compute the cross product for barycentric coordinates
	// Input Variables:
	// - v1, v2: Edges defining the vector
	// - p: Point for which coordinates are being calculated
	float getC(vec2D v1, vec2D v2, vec2D p) {
		vec2D e = v2 - v1;
		vec2D q = p - v1;
		return q.y * e.x - q.x * e.y;
	}

	// Compute barycentric coordinates for a given point
	// Input Variables:
	// - p: Point to check within the triangle
	// Output Variables:
	// - alpha, beta, gamma: Barycentric coordinates of the point
	// Returns true if the point is inside the triangle, false otherwise
	bool getCoordinates(vec2D p, float& alpha, float& beta, float& gamma) {
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

	// Draw the triangle on the canvas
	// Input Variables:
	// - renderer: Renderer object for drawing
	// - L: Light object for shading calculations
	// - ka, kd: Ambient and diffuse lighting coefficients
	void draw(Renderer& renderer, Light& L, float ka, float kd) {
		vec2D minV, maxV;

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

		// Iterate over the bounding box and check each pixel
		for (int y = startY; y < endY; y++) {
			float f0 = f0row;
			float f1 = f1row;
			float f2 = f2row;

			// pixels in avx
			for (int offset = 0; offset < aligned; offset += 8) {
				int x = startX + offset;
				__m256 vf0 = _mm256_setr_ps(
					f0 + 0 * dfx0, f0 + 1 * dfx0, f0 + 2 * dfx0, f0 + 3 * dfx0,
					f0 + 4 * dfx0, f0 + 5 * dfx0, f0 + 6 * dfx0, f0 + 7 * dfx0);

				__m256 vf1 = _mm256_setr_ps(
					f1 + 0 * dfx1, f1 + 1 * dfx1, f1 + 2 * dfx1, f1 + 3 * dfx1,
					f1 + 4 * dfx1, f1 + 5 * dfx1, f1 + 6 * dfx1, f1 + 7 * dfx1);

				__m256 vf2 = _mm256_setr_ps(
					f2 + 0 * dfx2, f2 + 1 * dfx2, f2 + 2 * dfx2, f2 + 3 * dfx2,
					f2 + 4 * dfx2, f2 + 5 * dfx2, f2 + 6 * dfx2, f2 + 7 * dfx2);

				// compare with 0
				__m256 zero = _mm256_set1_ps(0.0f);
				// in SSE4, cmpge_ps => _CMP_GE_OQ
				__m256 mask0 = _mm256_cmp_ps(vf0, zero, _CMP_GE_OQ);
				__m256 mask1 = _mm256_cmp_ps(vf1, zero, _CMP_GE_OQ);
				__m256 mask2 = _mm256_cmp_ps(vf2, zero, _CMP_GE_OQ);

				// inTri = mask0 & mask1 & mask2
				__m256 inTri = _mm256_and_ps(_mm256_and_ps(mask0, mask1), mask2);

				int bits = _mm256_movemask_ps(inTri);

				// Check if one of 8 pixels lies inside the triangle
				if (bits != 0) {
					for (int i = 0; i < 8; i++)
					{
						int mask = (1 << i);
						// if this one pixel is in the triangle
						if ((bits & mask) != 0)
						{
							float alpha, beta, gamma;
							if (getCoordinates(vec2D((float)(x + i), (float)y), alpha, beta, gamma)) {
								// Interpolate color, depth, and normals
								colour c = interpolate(beta, gamma, alpha, v[0].rgb, v[1].rgb, v[2].rgb);
								c.clampColour();
								float depth = interpolate(beta, gamma, alpha, v[0].p[2], v[1].p[2], v[2].p[2]);
								vec4 normal = interpolate(beta, gamma, alpha, v[0].normal, v[1].normal, v[2].normal);
								normal.normalise();

								// Perform Z-buffer test and apply shading
								if (renderer.zbuffer(x + i, y) > depth && depth > 0.01f) {
									// typical shader begin
									L.omega_i.normalise();
									float dot = max(vec4::dot(L.omega_i, normal), 0.0f);
									colour a = (c * kd) * (L.L * dot + (L.ambient * kd));
									// typical shader end
									unsigned char r, g, b;
									a.toRGB(r, g, b);
									renderer.canvas.draw(x + i, y, r, g, b);
									renderer.zbuffer(x + i, y) = depth;
								}
							}
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
