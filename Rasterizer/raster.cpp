#include <iostream>
#define _USE_MATH_DEFINES
#include <cmath>

#include "GamesEngineeringBase.h" // Include the GamesEngineeringBase header
#include <algorithm>
#include <chrono>
#include <thread>

#include <cmath>
#include "matrix.h"
#include "colour.h"
#include "mesh.h"
#include "zbuffer.h"
#include "renderer.h"
#include "RNG.h"
#include "light.h"
#include "triangle.h"

const int NUM_THREADS = 11;

struct Tile {
	int x, y;           // tile postion
	int width, height;  // tile width & height
	std::vector<triangle> triangles;
};

std::vector<Tile> createTiles(int screenWidth, int screenHeight, int tileSize) {
	std::vector<Tile> tiles;
	for (int y = 0; y < screenHeight; y += tileSize) {
		for (int x = 0; x < screenWidth; x += tileSize) {
			Tile tile;
			tile.x = x;
			tile.y = y;
			tile.width = min(tileSize, screenWidth - x);
			tile.height = min(tileSize, screenHeight - y);
			tiles.push_back(tile);
		}
	}
	return tiles;
}

void assignTrianglesToTiles(const std::vector<triangle>& triangles, std::vector<Tile>& tiles, int screenWidth, int screenHeight, int tileSize) {
	int numTilesX = (screenWidth + tileSize - 1) / tileSize;
	for (const auto& tri : triangles) {
		// 假设 tri 中的顶点位置为 tri.v[0].p, tri.v[1].p, tri.v[2].p，且 p[0]、p[1] 为屏幕坐标
		float minX = min(min(tri.v[0].p[0], tri.v[1].p[0]), tri.v[2].p[0]);
		float maxX = max(max(tri.v[0].p[0], tri.v[1].p[0]), tri.v[2].p[0]);
		float minY = min(min(tri.v[0].p[1], tri.v[1].p[1]), tri.v[2].p[1]);
		float maxY = max(max(tri.v[0].p[1], tri.v[1].p[1]), tri.v[2].p[1]);

		int tileStartX = max(0, static_cast<int>(minX) / tileSize);
		int tileEndX = min((screenWidth - 1) / tileSize, static_cast<int>(maxX) / tileSize);
		int tileStartY = max(0, static_cast<int>(minY) / tileSize);
		int tileEndY = min((screenHeight - 1) / tileSize, static_cast<int>(maxY) / tileSize);

		for (int ty = tileStartY; ty <= tileEndY; ty++) {
			for (int tx = tileStartX; tx <= tileEndX; tx++) {
				int index = ty * numTilesX + tx;
				if (index < tiles.size()) {
					tiles[index].triangles.push_back(tri);
				}
			}
		}
	}
}

void rasterizeTile(Tile& tile, Renderer& renderer, Light& L) {
	for (auto& tri : tile.triangles) {
		tri.draw(renderer, L, 1, 1);  // kd & ka = 1
	}
}

void renderTiles(Renderer& renderer, std::vector<Tile>& tiles, Light& L) {
	unsigned int numThreads = NUM_THREADS;
	if (numThreads == 0)
		numThreads = 2;
	int numTiles = tiles.size();
	int tilesPerThread = (numTiles + numThreads - 1) / numThreads;
	std::vector<std::thread> workers;
	for (unsigned int t = 0; t < numThreads; t++) {
		int startTile = t * tilesPerThread;
		int endTile = min(startTile + tilesPerThread, numTiles);
		if (startTile >= endTile)
			break;
		workers.emplace_back([startTile, endTile, &tiles, &renderer, &L]() {
			for (int i = startTile; i < endTile; i++) {
				rasterizeTile(tiles[i], renderer, L);
			}
			});
	}
	for (auto& worker : workers) {
		worker.join();
	}
}

void processMesh(Renderer& renderer, Mesh* mesh, matrix& camera, Light& L, std::vector<triangle>& triangles) {
	// Combine perspective, camera, and world transformations for the mesh
	matrix p = renderer.perspective * camera * mesh->world;

	// Iterate through all triangles in the mesh
	for (triIndices& ind : mesh->triangles) {
		Vertex t[3]; // Temporary array to store transformed triangle vertices

		// Transform each vertex of the triangle
		for (unsigned int i = 0; i < 3; i++) {
			t[i].p = p * mesh->vertices[ind.v[i]].p; // Apply transformations
			t[i].p.divideW(); // Perspective division to normalize coordinates

			// Transform normals into world space for accurate lighting
			// no need for perspective correction as no shearing or non-uniform scaling
			t[i].normal = mesh->world * mesh->vertices[ind.v[i]].normal;
			t[i].normal.normalise();

			// Map normalized device coordinates to screen space
			t[i].p[0] = (t[i].p[0] + 1.f) * 0.5f * static_cast<float>(renderer.canvas.getWidth());
			t[i].p[1] = (t[i].p[1] + 1.f) * 0.5f * static_cast<float>(renderer.canvas.getHeight());
			t[i].p[1] = renderer.canvas.getHeight() - t[i].p[1]; // Invert y-axis

			// Copy vertex colours
			t[i].rgb = mesh->vertices[ind.v[i]].rgb;
		}

		// Clip triangles with Z-values outside [-1, 1]
		if (fabs(t[0].p[2]) > 1.0f || fabs(t[1].p[2]) > 1.0f || fabs(t[2].p[2]) > 1.0f) continue;

		// Create a triangle object and render it
		triangle tri(t[0], t[1], t[2]);
		triangles.push_back(tri);
		//tri.draw(renderer, L, mesh->ka, mesh->kd);
	}
}

// Main rendering function that processes a mesh, transforms its vertices, applies lighting, and draws triangles on the canvas.
// Input Variables:
// - renderer: The Renderer object used for drawing.
// - mesh: Pointer to the Mesh object containing vertices and triangles to render.
// - camera: Matrix representing the camera's transformation.
// - L: Light object representing the lighting parameters.
void render(Renderer& renderer, Mesh* mesh, matrix& camera, Light& L) {
	// Combine perspective, camera, and world transformations for the mesh
	matrix p = renderer.perspective * camera * mesh->world;

	// Iterate through all triangles in the mesh
	for (triIndices& ind : mesh->triangles) {
		Vertex t[3]; // Temporary array to store transformed triangle vertices

		// Transform each vertex of the triangle
		for (unsigned int i = 0; i < 3; i++) {
			t[i].p = p * mesh->vertices[ind.v[i]].p; // Apply transformations
			t[i].p.divideW(); // Perspective division to normalize coordinates

			// Transform normals into world space for accurate lighting
			// no need for perspective correction as no shearing or non-uniform scaling
			t[i].normal = mesh->world * mesh->vertices[ind.v[i]].normal;
			t[i].normal.normalise();

			// Map normalized device coordinates to screen space
			t[i].p[0] = (t[i].p[0] + 1.f) * 0.5f * static_cast<float>(renderer.canvas.getWidth());
			t[i].p[1] = (t[i].p[1] + 1.f) * 0.5f * static_cast<float>(renderer.canvas.getHeight());
			t[i].p[1] = renderer.canvas.getHeight() - t[i].p[1]; // Invert y-axis

			// Copy vertex colours
			t[i].rgb = mesh->vertices[ind.v[i]].rgb;
		}

		// face culling
		vec2D p0(t[0].p), p1(t[1].p), p2(t[2].p);
		float cross = (p1.x - p0.x) * (p2.y - p0.y) - (p1.y - p0.y) * (p2.x - p0.x);
		if (cross < 0) continue;

		// Clip triangles with Z-values outside [-1, 1]
		if (fabs(t[0].p[2]) > 1.0f || fabs(t[1].p[2]) > 1.0f || fabs(t[2].p[2]) > 1.0f) continue;

		// Create a triangle object and render it
		triangle tri(t[0], t[1], t[2]);
		tri.draw(renderer, L, mesh->ka, mesh->kd);
	}
}

void renderMt(Renderer& renderer, Mesh* mesh, matrix& camera, Light& L) {
	// Combine perspective, camera, and world transformations for the mesh
	matrix p = renderer.perspective * camera * mesh->world;

	// get all triangles
	size_t totalTriangles = mesh->triangles.size();

	// ensure thread number
	unsigned int numThreads = 2;

	// Calculate the number of triangles each thread needs to process
	size_t chunkSize = (totalTriangles + numThreads - 1) / numThreads;

	// store threads
	std::vector<std::thread> threads;

	// Allocate tasks to each thread
	for (unsigned int t = 0; t < numThreads; t++) {
		size_t startIndex = t * chunkSize;
		size_t endIndex = min(startIndex + chunkSize, totalTriangles);
		if (startIndex >= endIndex) break;

		threads.emplace_back([startIndex, endIndex, &renderer, mesh, &camera, &L, &p]() {
			for (size_t i = startIndex; i < endIndex; i++) {
				triIndices& ind = mesh->triangles[i];

				Vertex t[3];

				// Transform each vertex of the triangle
				for (unsigned int j = 0; j < 3; j++) {
					t[j].p = p * mesh->vertices[ind.v[j]].p; // Apply transformations
					t[j].p.divideW(); // Perspective division to normalize coordinates

					// Transform normals into world space for accurate lighting
					// no need for perspective correction as no shearing or non-uniform scaling
					t[j].normal = mesh->world * mesh->vertices[ind.v[j]].normal;
					t[j].normal.normalise();

					// Map normalized device coordinates to screen space
					t[j].p[0] = (t[j].p[0] + 1.f) * 0.5f * static_cast<float>(renderer.canvas.getWidth());
					t[j].p[1] = (t[j].p[1] + 1.f) * 0.5f * static_cast<float>(renderer.canvas.getHeight());
					t[j].p[1] = renderer.canvas.getHeight() - t[j].p[1]; // Invert y-axis

					// Copy vertex colours
					t[j].rgb = mesh->vertices[ind.v[j]].rgb;
				}

				// Clip triangles with Z-values outside [-1, 1]
				if (fabs(t[0].p[2]) > 1.0f || fabs(t[1].p[2]) > 1.0f || fabs(t[2].p[2]) > 1.0f) continue;

				// Create a triangle object and render it
				triangle tri(t[0], t[1], t[2]);
				tri.draw(renderer, L, mesh->ka, mesh->kd);
			}
			});
	}

	// wait all done
	for (auto& thread : threads) {
		thread.join();
	}
}

// Test scene function to demonstrate rendering with user-controlled transformations
// No input variables
void sceneTest() {
	Renderer renderer;
	// create light source {direction, diffuse intensity, ambient intensity}
	Light L{ vec4(0.f, 1.f, 1.f, 0.f), colour(1.0f, 1.0f, 1.0f), colour(0.1f, 0.1f, 0.1f) };
	// camera is just a matrix
	matrix camera = matrix::makeIdentity(); // Initialize the camera with identity matrix

	bool running = true; // Main loop control variable

	std::vector<Mesh*> scene; // Vector to store scene objects

	// Create a sphere and a rectangle mesh
	Mesh mesh = Mesh::makeSphere(1.0f, 10, 20);
	//Mesh mesh2 = Mesh::makeRectangle(-2, -1, 2, 1);

	// add meshes to scene
	scene.push_back(&mesh);
	// scene.push_back(&mesh2); 

	float x = 0.0f, y = 0.0f, z = -4.0f; // Initial translation parameters
	mesh.world = matrix::makeTranslation(x, y, z);
	//mesh2.world = matrix::makeTranslation(x, y, z) * matrix::makeRotateX(0.01f);

	// Main rendering loop
	while (running) {
		renderer.canvas.checkInput(); // Handle user input
		renderer.clear(); // Clear the canvas for the next frame

		// Apply transformations to the meshes
	 //   mesh2.world = matrix::makeTranslation(x, y, z) * matrix::makeRotateX(0.01f);
		mesh.world = matrix::makeTranslation(x, y, z);

		// Handle user inputs for transformations
		if (renderer.canvas.keyPressed(VK_ESCAPE)) break;
		if (renderer.canvas.keyPressed('A')) x += -0.1f;
		if (renderer.canvas.keyPressed('D')) x += 0.1f;
		if (renderer.canvas.keyPressed('W')) y += 0.1f;
		if (renderer.canvas.keyPressed('S')) y += -0.1f;
		if (renderer.canvas.keyPressed('Q')) z += 0.1f;
		if (renderer.canvas.keyPressed('E')) z += -0.1f;

		// Render each object in the scene
		for (auto& m : scene)
			render(renderer, m, camera, L);

		renderer.present(); // Display the rendered frame
	}
}

// Utility function to generate a random rotation matrix
// No input variables
matrix makeRandomRotation() {
	RandomNumberGenerator& rng = RandomNumberGenerator::getInstance();
	unsigned int r = rng.getRandomInt(0, 3);

	switch (r) {
	case 0: return matrix::makeRotateX(rng.getRandomFloat(0.f, 2.0f * M_PI));
	case 1: return matrix::makeRotateY(rng.getRandomFloat(0.f, 2.0f * M_PI));
	case 2: return matrix::makeRotateZ(rng.getRandomFloat(0.f, 2.0f * M_PI));
	default: return matrix::makeIdentity();
	}
}

// Function to render a scene with multiple objects and dynamic transformations
// No input variables
void scene1() {
	Renderer renderer;
	matrix camera;
	Light L{ vec4(0.f, 1.f, 1.f, 0.f), colour(1.0f, 1.0f, 1.0f), colour(0.1f, 0.1f, 0.1f) };

	bool running = true;

	std::vector<Mesh*> scene;

	// Create a scene of 40 cubes with random rotations
	for (unsigned int i = 0; i < 20; i++) {
		Mesh* m = new Mesh();
		*m = Mesh::makeCube(1.f);
		m->world = matrix::makeTranslation(-2.0f, 0.0f, (-3 * static_cast<float>(i))) * makeRandomRotation();
		scene.push_back(m);
		m = new Mesh();
		*m = Mesh::makeCube(1.f);
		m->world = matrix::makeTranslation(2.0f, 0.0f, (-3 * static_cast<float>(i))) * makeRandomRotation();
		scene.push_back(m);
	}

	float zoffset = 8.0f; // Initial camera Z-offset
	float step = -0.1f;  // Step size for camera movement

	auto start = std::chrono::high_resolution_clock::now();
	std::chrono::time_point<std::chrono::high_resolution_clock> end;
	int cycle = 0;

	// Main rendering loop
	while (running) {
		renderer.canvas.checkInput();
		renderer.clear();

		camera = matrix::makeTranslation(0, 0, -zoffset); // Update camera position

		// Rotate the first two cubes in the scene
		scene[0]->world = scene[0]->world * matrix::makeRotateXYZ(0.1f, 0.1f, 0.0f);
		scene[1]->world = scene[1]->world * matrix::makeRotateXYZ(0.0f, 0.1f, 0.2f);

		if (renderer.canvas.keyPressed(VK_ESCAPE)) break;

		zoffset += step;
		if (zoffset < -60.f || zoffset > 8.f) {
			step *= -1.f;
			if (++cycle % 2 == 0) {
				end = std::chrono::high_resolution_clock::now();
				std::cout << cycle / 2 << " :" << std::chrono::duration<double, std::milli>(end - start).count() << "ms\n";
				start = std::chrono::high_resolution_clock::now();
			}
		}

		for (auto& m : scene)
			render(renderer, m, camera, L);
		renderer.present();
	}

	for (auto& m : scene)
		delete m;
}

// Scene with a grid of cubes and a moving sphere
// No input variables
void scene2() {
	Renderer renderer;
	matrix camera = matrix::makeIdentity();
	Light L{ vec4(0.f, 1.f, 1.f, 0.f), colour(1.0f, 1.0f, 1.0f), colour(0.1f, 0.1f, 0.1f) };

	std::vector<Mesh*> scene;

	struct rRot { float x; float y; float z; }; // Structure to store random rotation parameters
	std::vector<rRot> rotations;

	RandomNumberGenerator& rng = RandomNumberGenerator::getInstance();

	// Create a grid of cubes with random rotations
	for (unsigned int y = 0; y < 6; y++) {
		for (unsigned int x = 0; x < 8; x++) {
			Mesh* m = new Mesh();
			*m = Mesh::makeCube(1.f);
			scene.push_back(m);
			m->world = matrix::makeTranslation(-7.0f + (static_cast<float>(x) * 2.f), 5.0f - (static_cast<float>(y) * 2.f), -8.f);
			rRot r{ rng.getRandomFloat(-.1f, .1f), rng.getRandomFloat(-.1f, .1f), rng.getRandomFloat(-.1f, .1f) };
			rotations.push_back(r);
		}
	}

	// Create a sphere and add it to the scene
	Mesh* sphere = new Mesh();
	*sphere = Mesh::makeSphere(1.0f, 10, 20);
	scene.push_back(sphere);
	float sphereOffset = -6.f;
	float sphereStep = 0.1f;
	sphere->world = matrix::makeTranslation(sphereOffset, 0.f, -6.f);

	auto start = std::chrono::high_resolution_clock::now();
	std::chrono::time_point<std::chrono::high_resolution_clock> end;
	int cycle = 0;

	bool running = true;
	while (running) {
		renderer.canvas.checkInput();
		renderer.clear();

		// Rotate each cube in the grid
		for (unsigned int i = 0; i < rotations.size(); i++)
			scene[i]->world = scene[i]->world * matrix::makeRotateXYZ(rotations[i].x, rotations[i].y, rotations[i].z);

		// Move the sphere back and forth
		sphereOffset += sphereStep;
		sphere->world = matrix::makeTranslation(sphereOffset, 0.f, -6.f);
		if (sphereOffset > 6.0f || sphereOffset < -6.0f) {
			sphereStep *= -1.f;
			if (++cycle % 2 == 0) {
				end = std::chrono::high_resolution_clock::now();
				std::cout << cycle / 2 << " :" << std::chrono::duration<double, std::milli>(end - start).count() << "ms\n";
				start = std::chrono::high_resolution_clock::now();
			}
		}

		if (renderer.canvas.keyPressed(VK_ESCAPE)) break;

		for (auto& m : scene)
			render(renderer, m, camera, L);
		renderer.present();
	}

	for (auto& m : scene)
		delete m;
}

void scene3() {
	Renderer renderer;
	matrix camera = matrix::makeIdentity();

	Light L{ vec4(0.f, 1.f, 1.f, 0.f),
			 colour(1.0f, 1.0f, 1.0f),
			 colour(0.1f, 0.1f, 0.1f) };

	std::vector<Mesh*> scene;
	struct rRot { float x; float y; float z; }; // Structure to store random rotation parameters
	std::vector<rRot> rotations;

	RandomNumberGenerator& rng = RandomNumberGenerator::getInstance();

	for (int row = 0; row < 200; row++) {
		for (int col = 0; col < 200; col++) {
			Mesh* m = new Mesh();
			*m = Mesh::makeCube(1.f);
			scene.push_back(m);
			// put in a location of grid
			float px = -20.f + (col * 2.0f);
			float py = 20.f - (row * 2.0f);
			float pz = -20.f - row * 0.2f;
			m->world = matrix::makeTranslation(px, py, pz);
			// random rotation
			rRot r{ rng.getRandomFloat(-.1f, .1f), rng.getRandomFloat(-.1f, .1f), rng.getRandomFloat(-.1f, .1f) };
			rotations.push_back(r);

		}
	}

	// Create a sphere and add it to the scene
	Mesh* sphere = new Mesh();
	*sphere = Mesh::makeSphere(1.0f, 10, 20);
	scene.push_back(sphere);
	float sphereOffset = -6.f;
	float sphereStep = 0.1f;
	sphere->world = matrix::makeTranslation(sphereOffset, 0.f, -6.f);

	auto start = std::chrono::high_resolution_clock::now();
	std::chrono::time_point<std::chrono::high_resolution_clock> end;
	int cycle = 0;

	bool running = true;
	while (running) {
		renderer.canvas.checkInput();
		renderer.clear();

		for (size_t i = 0; i < rotations.size(); i++) {
			scene[i]->world = scene[i]->world *
				matrix::makeRotateXYZ(rotations[i].x,
					rotations[i].y,
					rotations[i].z);
		}

		// Move the sphere back and forth
		sphereOffset += sphereStep;
		sphere->world = matrix::makeTranslation(sphereOffset, 0.f, -6.f);
		if (sphereOffset > 6.0f || sphereOffset < -6.0f) {
			sphereStep *= -1.f;
			if (++cycle % 2 == 0) {
				end = std::chrono::high_resolution_clock::now();
				std::cout << cycle / 2 << " :" << std::chrono::duration<double, std::milli>(end - start).count() << "ms\n";
				start = std::chrono::high_resolution_clock::now();
			}
		}

		if (renderer.canvas.keyPressed(VK_ESCAPE)) break;

		// single thread
		for (auto& m : scene)
			render(renderer, m, camera, L);
		renderer.present();
	}

	for (auto& m : scene) {
		delete m;
	}
}

void renderChunk(Renderer& renderer, const std::vector<Mesh*>& chunk, matrix& camera, Light& L) {
	for (auto& m : chunk) {
		render(renderer, m, camera, L);
	}
}

void multithreadedRender(Renderer& renderer, const std::vector<Mesh*>& scene, matrix& camera,  Light& L, int numThreads) {
	std::vector<std::thread> threads;
	size_t totalMeshes = scene.size();
	size_t meshesPerThread = (totalMeshes + numThreads - 1) / numThreads;

	for (int i = 0; i < numThreads; i++) {
		size_t startIdx = i * meshesPerThread;
		size_t endIdx = min(startIdx + meshesPerThread, totalMeshes);
		if (startIdx >= endIdx)
			break;
		std::vector<Mesh*> threadMeshes(scene.begin() + startIdx, scene.begin() + endIdx);
		threads.emplace_back(renderChunk, std::ref(renderer), threadMeshes, std::ref(camera), std::ref(L));
	}

	// wait all threads done
	for (auto& t : threads) {
		t.join();
	}
}

void scene1Mt() {
	Renderer renderer;
	matrix camera;
	Light L{ vec4(0.f, 1.f, 1.f, 0.f), colour(1.0f, 1.0f, 1.0f), colour(0.1f, 0.1f, 0.1f) };

	bool running = true;

	std::vector<Mesh*> scene;

	// Create a scene of 40 cubes with random rotations
	for (unsigned int i = 0; i < 20; i++) {
		Mesh* m = new Mesh();
		*m = Mesh::makeCube(1.f);
		m->world = matrix::makeTranslation(-2.0f, 0.0f, (-3 * static_cast<float>(i))) * makeRandomRotation();
		scene.push_back(m);
		m = new Mesh();
		*m = Mesh::makeCube(1.f);
		m->world = matrix::makeTranslation(2.0f, 0.0f, (-3 * static_cast<float>(i))) * makeRandomRotation();
		scene.push_back(m);
	}

	float zoffset = 8.0f; // Initial camera Z-offset
	float step = -0.1f;  // Step size for camera movement

	auto start = std::chrono::high_resolution_clock::now();
	std::chrono::time_point<std::chrono::high_resolution_clock> end;
	int cycle = 0;

	// Main rendering loop
	while (running) {
		renderer.canvas.checkInput();
		renderer.clear();

		camera = matrix::makeTranslation(0, 0, -zoffset); // Update camera position

		// Rotate the first two cubes in the scene
		scene[0]->world = scene[0]->world * matrix::makeRotateXYZ(0.1f, 0.1f, 0.0f);
		scene[1]->world = scene[1]->world * matrix::makeRotateXYZ(0.0f, 0.1f, 0.2f);

		if (renderer.canvas.keyPressed(VK_ESCAPE)) break;

		zoffset += step;
		if (zoffset < -60.f || zoffset > 8.f) {
			step *= -1.f;
			if (++cycle % 2 == 0) {
				end = std::chrono::high_resolution_clock::now();
				std::cout << cycle / 2 << " :" << std::chrono::duration<double, std::milli>(end - start).count() << "ms\n";
				start = std::chrono::high_resolution_clock::now();
			}
		}

		multithreadedRender(renderer, scene, camera, L, NUM_THREADS);

		renderer.present();
	}

	for (auto& m : scene)
		delete m;
}

void scene2Mt() {
	Renderer renderer;
	matrix camera = matrix::makeIdentity();
	Light L{ vec4(0.f, 1.f, 1.f, 0.f), colour(1.0f, 1.0f, 1.0f), colour(0.1f, 0.1f, 0.1f) };

	std::vector<Mesh*> scene;

	struct rRot { float x; float y; float z; }; // Structure to store random rotation parameters
	std::vector<rRot> rotations;

	RandomNumberGenerator& rng = RandomNumberGenerator::getInstance();

	// Create a grid of cubes with random rotations
	for (unsigned int y = 0; y < 6; y++) {
		for (unsigned int x = 0; x < 8; x++) {
			Mesh* m = new Mesh();
			*m = Mesh::makeCube(1.f);
			scene.push_back(m);
			m->world = matrix::makeTranslation(-7.0f + (static_cast<float>(x) * 2.f), 5.0f - (static_cast<float>(y) * 2.f), -8.f);
			rRot r{ rng.getRandomFloat(-.1f, .1f), rng.getRandomFloat(-.1f, .1f), rng.getRandomFloat(-.1f, .1f) };
			rotations.push_back(r);
		}
	}

	// Create a sphere and add it to the scene
	Mesh* sphere = new Mesh();
	*sphere = Mesh::makeSphere(1.0f, 10, 20);
	scene.push_back(sphere);
	float sphereOffset = -6.f;
	float sphereStep = 0.1f;
	sphere->world = matrix::makeTranslation(sphereOffset, 0.f, -6.f);

	auto start = std::chrono::high_resolution_clock::now();
	std::chrono::time_point<std::chrono::high_resolution_clock> end;
	int cycle = 0;

	bool running = true;
	while (running) {
		renderer.canvas.checkInput();
		renderer.clear();

		// Rotate each cube in the grid
		for (unsigned int i = 0; i < rotations.size(); i++)
			scene[i]->world = scene[i]->world * matrix::makeRotateXYZ(rotations[i].x, rotations[i].y, rotations[i].z);

		// Move the sphere back and forth
		sphereOffset += sphereStep;
		sphere->world = matrix::makeTranslation(sphereOffset, 0.f, -6.f);
		if (sphereOffset > 6.0f || sphereOffset < -6.0f) {
			sphereStep *= -1.f;
			if (++cycle % 2 == 0) {
				end = std::chrono::high_resolution_clock::now();
				std::cout << cycle / 2 << " :" << std::chrono::duration<double, std::milli>(end - start).count() << "ms\n";
				start = std::chrono::high_resolution_clock::now();
			}
		}

		if (renderer.canvas.keyPressed(VK_ESCAPE)) break;

		multithreadedRender(renderer, scene, camera, L, NUM_THREADS);
		renderer.present();
	}

	for (auto& m : scene)
		delete m;
}

void scene3Mt() {
	Renderer renderer;
	matrix camera = matrix::makeIdentity();
	Light L{ vec4(0.f, 1.f, 1.f, 0.f),
			 colour(1.0f, 1.0f, 1.0f),
			 colour(0.1f, 0.1f, 0.1f) };

	std::vector<Mesh*> scene;
	struct rRot { float x; float y; float z; }; // Structure to store random rotation parameters
	std::vector<rRot> rotations;

	RandomNumberGenerator& rng = RandomNumberGenerator::getInstance();

	for (int row = 0; row < 200; row++) {
		for (int col = 0; col < 200; col++) {
			Mesh* m = new Mesh();
			*m = Mesh::makeCube(1.f);
			scene.push_back(m);
			// put in a location of grid
			float px = -20.f + (col * 2.0f);
			float py = 20.f - (row * 2.0f);
			float pz = -20.f - row * 0.2f;
			m->world = matrix::makeTranslation(px, py, pz);
			// random rotation
			rRot r{ rng.getRandomFloat(-.1f, .1f), rng.getRandomFloat(-.1f, .1f), rng.getRandomFloat(-.1f, .1f) };
			rotations.push_back(r);

		}
	}

	// Create a sphere and add it to the scene
	Mesh* sphere = new Mesh();
	*sphere = Mesh::makeSphere(1.0f, 10, 20);
	scene.push_back(sphere);
	float sphereOffset = -6.f;
	float sphereStep = 0.1f;
	sphere->world = matrix::makeTranslation(sphereOffset, 0.f, -6.f);

	auto start = std::chrono::high_resolution_clock::now();
	std::chrono::time_point<std::chrono::high_resolution_clock> end;
	int cycle = 0;

	bool running = true;
	while (running) {
		renderer.canvas.checkInput();
		renderer.clear();

		// update rotation world matrix
		for (size_t i = 0; i < rotations.size(); i++) {
			scene[i]->world = scene[i]->world *
				matrix::makeRotateXYZ(rotations[i].x,
					rotations[i].y,
					rotations[i].z);
		}

		// Move the sphere back and forth
		sphereOffset += sphereStep;
		sphere->world = matrix::makeTranslation(sphereOffset, 0.f, -6.f);
		if (sphereOffset > 6.0f || sphereOffset < -6.0f) {
			sphereStep *= -1.f;
			if (++cycle % 2 == 0) {
				end = std::chrono::high_resolution_clock::now();
				std::cout << cycle / 2 << " :" << std::chrono::duration<double, std::milli>(end - start).count() << "ms\n";
				start = std::chrono::high_resolution_clock::now();
			}
		}

		if (renderer.canvas.keyPressed(VK_ESCAPE)) break;

		multithreadedRender(renderer, scene, camera, L, NUM_THREADS);
		renderer.present();
	}

	for (auto& m : scene) {
		delete m;
	}
}

void scene3TriMt() {
	Renderer renderer;
	matrix camera = matrix::makeIdentity();
	Light L{ vec4(0.f, 1.f, 1.f, 0.f),
			 colour(1.0f, 1.0f, 1.0f),
			 colour(0.1f, 0.1f, 0.1f) };

	std::vector<Mesh*> scene;
	struct rRot { float x; float y; float z; }; // Structure to store random rotation parameters
	std::vector<rRot> rotations;

	RandomNumberGenerator& rng = RandomNumberGenerator::getInstance();

	for (int row = 0; row < 200; row++) {
		for (int col = 0; col < 200; col++) {
			Mesh* m = new Mesh();
			*m = Mesh::makeCube(1.f);
			scene.push_back(m);
			// put in a location of grid
			float px = -20.f + (col * 2.0f);
			float py = 20.f - (row * 2.0f);
			float pz = -20.f - row * 0.2f;
			m->world = matrix::makeTranslation(px, py, pz);
			// random rotation
			rRot r{ rng.getRandomFloat(-.1f, .1f), rng.getRandomFloat(-.1f, .1f), rng.getRandomFloat(-.1f, .1f) };
			rotations.push_back(r);

		}
	}

	// Create a sphere and add it to the scene
	Mesh* sphere = new Mesh();
	*sphere = Mesh::makeSphere(1.0f, 10, 20);
	scene.push_back(sphere);
	float sphereOffset = -6.f;
	float sphereStep = 0.1f;
	sphere->world = matrix::makeTranslation(sphereOffset, 0.f, -6.f);

	auto start = std::chrono::high_resolution_clock::now();
	std::chrono::time_point<std::chrono::high_resolution_clock> end;
	int cycle = 0;

	bool running = true;
	while (running) {
		renderer.canvas.checkInput();
		renderer.clear();

		// update rotation world matrix
		for (size_t i = 0; i < rotations.size(); i++) {
			scene[i]->world = scene[i]->world *
				matrix::makeRotateXYZ(rotations[i].x,
					rotations[i].y,
					rotations[i].z);
		}

		// Move the sphere back and forth
		sphereOffset += sphereStep;
		sphere->world = matrix::makeTranslation(sphereOffset, 0.f, -6.f);
		if (sphereOffset > 6.0f || sphereOffset < -6.0f) {
			sphereStep *= -1.f;
			if (++cycle % 2 == 0) {
				end = std::chrono::high_resolution_clock::now();
				std::cout << cycle / 2 << " :" << std::chrono::duration<double, std::milli>(end - start).count() << "ms\n";
				start = std::chrono::high_resolution_clock::now();
			}
		}

		if (renderer.canvas.keyPressed(VK_ESCAPE)) break;

		for (auto& m : scene)
			renderMt(renderer, m, camera, L);
		renderer.present();
	}

	for (auto& m : scene) {
		delete m;
	}
}

void scene3TileMt() {
	Renderer renderer;
	matrix camera = matrix::makeIdentity();

	Light L{ vec4(0.f, 1.f, 1.f, 0.f),
			 colour(1.0f, 1.0f, 1.0f),
			 colour(0.1f, 0.1f, 0.1f) };

	std::vector<Mesh*> scene;
	struct rRot { float x; float y; float z; }; // Structure to store random rotation parameters
	std::vector<rRot> rotations;
	std::vector<triangle> globalTriangles;

	RandomNumberGenerator& rng = RandomNumberGenerator::getInstance();

	for (int row = 0; row < 200; row++) {
		for (int col = 0; col < 200; col++) {
			Mesh* m = new Mesh();
			*m = Mesh::makeCube(1.f);
			scene.push_back(m);
			// put in a location of grid
			float px = -20.f + (col * 2.0f);
			float py = 20.f - (row * 2.0f);
			float pz = -20.f - row * 0.2f;
			m->world = matrix::makeTranslation(px, py, pz);
			// random rotation
			rRot r{ rng.getRandomFloat(-.1f, .1f), rng.getRandomFloat(-.1f, .1f), rng.getRandomFloat(-.1f, .1f) };
			rotations.push_back(r);

		}
	}

	// Create a sphere and add it to the scene
	Mesh* sphere = new Mesh();
	*sphere = Mesh::makeSphere(1.0f, 10, 20);
	scene.push_back(sphere);
	float sphereOffset = -6.f;
	float sphereStep = 0.1f;
	sphere->world = matrix::makeTranslation(sphereOffset, 0.f, -6.f);

	auto start = std::chrono::high_resolution_clock::now();
	std::chrono::time_point<std::chrono::high_resolution_clock> end;
	int cycle = 0;

	bool running = true;
	while (running) {
		renderer.canvas.checkInput();
		renderer.clear();
		globalTriangles.clear();

		for (size_t i = 0; i < rotations.size(); i++) {
			scene[i]->world = scene[i]->world *
				matrix::makeRotateXYZ(rotations[i].x,
					rotations[i].y,
					rotations[i].z);
		}

		// Move the sphere back and forth
		sphereOffset += sphereStep;
		sphere->world = matrix::makeTranslation(sphereOffset, 0.f, -6.f);
		if (sphereOffset > 6.0f || sphereOffset < -6.0f) {
			sphereStep *= -1.f;
			if (++cycle % 2 == 0) {
				end = std::chrono::high_resolution_clock::now();
				std::cout << cycle / 2 << " :" << std::chrono::duration<double, std::milli>(end - start).count() << "ms\n";
				start = std::chrono::high_resolution_clock::now();
			}
		}

		if (renderer.canvas.keyPressed(VK_ESCAPE)) break;

		for (auto& m : scene)
			processMesh(renderer, m, camera, L, globalTriangles);

		int screenWidth = renderer.canvas.getWidth();
		int screenHeight = renderer.canvas.getHeight();
		int tileSize = 32;
		std::vector<Tile> tiles = createTiles(screenWidth, screenHeight, tileSize);
		assignTrianglesToTiles(globalTriangles, tiles, screenWidth, screenHeight, tileSize);
		renderTiles(renderer, tiles, L);
		renderer.present();
	}

	for (auto& m : scene) {
		delete m;
	}
}

// Entry point of the application
// No input variables
int main() {
	// Uncomment the desired scene function to run
	scene1();
	//scene2();
	//scene3();
	//scene1Mt();
	//scene2Mt();
	//scene3Mt();
	//scene3TriMt();
	//scene3TileMt();


	return 0;
}