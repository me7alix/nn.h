#include <stdbool.h>
#include <stddef.h>
#include <stdlib.h>
#include <time.h>
#include <raylib.h>
#define NN_IMPLEMENTATION
#include "../nn.h"

#define TCAP 256
#define ADAM_OPT

bool is_learning_started = false;
Mat inp, outp;

void draw_lines(NN nn, int sw, int sh) {
	Vector2 prev = {-100, 0};
	for (int i = 0; i < sw; i+=2) {
		Vector2 res = {(float)(i)/(sw-1), 0};
		MAT_AT(NN_INPUT(nn), 0, 0) = res.x;
		nn_forward(nn);
		res.y = MAT_AT(NN_OUTPUT(nn), 0, 0);
		res.x *= (float)sw;
		res.y *= (float)sh;
		DrawLineEx(res, prev, 2, GREEN);
		prev = res;
	}
}

void draw_points(Vector2 points[TCAP], int size, Color clr,
				 int sw, int sh) {
	for (int i = 0; i < size; i++) {
		Vector2 res = points[i];
		res.x *= (float)sw;
		res.y *= (float)sh;
		DrawCircleV(res, 5, clr);
	}
}

int main(void) {
	srand(time(0));
	const int screenWidth = 800;
	const int screenHeight = 800;
	InitWindow(screenWidth, screenHeight, "Window");
	SetTargetFPS(60);

	Vector2 points[TCAP];
	size_t points_num = 0;

	Layer layers[] = {
		(Layer){
			.size = 1,
		},
		(Layer){
			.actf = ACT_SIGM,
			.size = 5,
			.randf = glorot_randf,
		},
		(Layer){
			.actf = ACT_SIGM,
			.size = 5,
			.randf = glorot_randf,
		},
		(Layer){
			.actf = ACT_SIGM,
			.size = 1,
			.randf = glorot_randf,
		},
	};

	NN nn = nn_alloc(layers, ARR_LEN(layers)); 
	NN g = nn_alloc(layers, ARR_LEN(layers)); 

#ifdef ADAM_OPT
	AdamOptimizer adam = adam_alloc(nn);
#endif

	nn_rand(nn);

	float learning_rate = 0.002;
	inp = mat_alloc(1, 1);
	outp = mat_alloc(1, 1);

	int iter = 0;

	while (!WindowShouldClose()) {
		BeginDrawing();
		ClearBackground(BLACK);

		if (IsMouseButtonPressed(MOUSE_BUTTON_LEFT)) { 
			Vector2 n = GetMousePosition();
			n.x /= (float)screenWidth;
			n.y /= (float)screenHeight;
			points[points_num++] = n;
		}

		if (IsKeyPressed(KEY_ENTER)) {
			is_learning_started = !is_learning_started;
		}

		for (int o = 0; o < 100; o++) {
			if (!(points_num != 0 && is_learning_started)) continue;

			MAT_AT(inp, 0, 0) = points[iter%points_num].x; 
			MAT_AT(outp, 0, 0) = points[iter%points_num].y;

			nn_backprop(nn, g, inp, outp);

#ifdef ADAM_OPT
			adam_update(nn, &adam, g, learning_rate, 0.9, 0.999, 1e-8);
#else 
			nn_learn(nn, g, learning_rate); 
#endif

			iter++;
		}
	
		draw_lines(nn, screenWidth, screenHeight);
		draw_points(points, points_num, LIGHTGRAY, screenWidth, screenHeight);
		DrawText("Press ENTER to toggle learning process", 20, 20, 20, LIGHTGRAY);

		EndDrawing();
	}

	CloseWindow();
	return 0;
}
