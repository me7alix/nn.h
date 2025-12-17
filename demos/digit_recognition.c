#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdbool.h>
#include <time.h>
#include <raylib.h>
#include "../parsers/csv_parser.c"

#define NN_IMPLEMENTATION
#include "../nn.h"

void paint(NN nn);

bool wait_for_keypress() {
	fd_set fds;
	struct timeval timeout = {0, 0};
	FD_ZERO(&fds);
	FD_SET(0, &fds);

	if (select(1, &fds, NULL, NULL, &timeout) > 0) {
		char input = getchar();
		if (input == 'S' || input == 's') {
			return 1;
		}
	}

	return 0;
}

int main() {
	srand(time(0));
	Mat mat = parse_csv_to_mat("./dataset/train.csv", ",");

	Layer layers[] = {
		(Layer){
			.size = 28*28,
			.randf = glorot_randf,
		}, 
		(Layer){
			.size = 128,
			.actf = ACT_RELU,
			.randf = glorot_randf,
		},
		(Layer){
			.size = 64,
			.actf = ACT_RELU,
			.randf = glorot_randf,
		}, 
		(Layer){
			.size = 10,
			.actf = ACT_SOFTMAX,
			.randf = glorot_randf,
		}
	};

	NN nn = nn_alloc(layers, ARR_LEN(layers));
	NN g = nn_alloc(layers, ARR_LEN(layers));
	AdamOptimizer adam = adam_alloc(nn);
	nn_rand(nn);

	// initializing the data
	Mat nto = mat_submatrix(mat, 0, 0, 0, mat.rows - 1);
	Mat ti  = mat_submatrix(mat, 1, 0, mat.cols - 1, mat.rows - 1);
	Mat to  = mat_alloc(mat.rows, 10);
	Mat cti = mat_submatrix(mat, 1, 0, mat.cols - 1, 1000);
	Mat cto = mat_submatrix(to, 0, 0, to.cols - 1, 1000);

	mat_zero(to);

	// preparing the data
	for (size_t i = 0; i < mat.rows; i++) {
		MAT_AT(to, i, (int)MAT_AT(nto, i, 0)) = 1.0;
	}

	for (size_t i = 0; i < ti.rows; i++) {
		for (size_t j = 0; j < ti.cols; j++) {
			MAT_AT(ti, i, j) /= 255.0; 
		}
	}

	printf("Press S+ENTER to stop learning process\n");
	printf("cost before training = %f\n", nn_cost(nn, cti, cto));

	// learning process
	size_t batch_size = 32;
	float learning_rate = 0.001;

	for (size_t i = 0; true; i++) { 
		size_t pos = (rand()) % (ti.rows - batch_size);
		Mat gti = mat_submatrix(ti, 0, pos, ti.cols - 1, pos + batch_size);
		Mat gto = mat_submatrix(to, 0, pos, to.cols - 1, pos + batch_size);

		nn_backprop(nn, g, gti, gto);
		adam_update(nn, &adam, g, learning_rate, 0.9, 0.999, 1e-8);

		if (i % 5000 == 0) {
			if (wait_for_keypress()) break;

			float tc = nn_cost(nn, cti, cto);
			printf("cost %zu - %f\n", i, tc);
		}
	}

	printf("cost after training = %f\n", nn_cost(nn, cti, cto));

	paint(nn);
	return 0;
}

Mat ConvertToMatrix(float pixels[28][28]) {
	Mat matrix = mat_alloc(1, 28 * 28);
	for (int y = 0; y < 28; y++) {
		for (int x = 0; x < 28; x++) {
			MAT_AT(matrix, 0, x + y * 28) = pixels[y][x];
		}
	}
	return matrix;
}

void paint(NN nn) {
	const int tiles = 28;
	const int tile = 20;
	const int screenWidth = tile * tiles;
	const int screenHeight = tile * tiles + 200;

	InitWindow(screenWidth, screenHeight,
			"Drawing with cross brush on 28x28 canvas");

	float pixels[28][28];

	SetTargetFPS(60);
	int cnt = 0;

	while (!WindowShouldClose()) {
		Vector2 mousePosition = GetMousePosition();
		if (IsMouseButtonDown(MOUSE_LEFT_BUTTON)) {
			int x = mousePosition.x / tile;
			int y = mousePosition.y / tile;
			if (x >= 0 && x < tiles && y >= 0 && y < tiles) {
				float b = 0.4;
				pixels[y][x] += b;
				if (x > 0)
					pixels[y][x - 1] += b / 2.0;
				if (x < tiles - 1)
					pixels[y][x + 1] += b / 2.0;
				if (y > 0)
					pixels[y - 1][x] += b / 2.0;
				if (y < tiles - 1)
					pixels[y + 1][x] += b / 2.0;
			}
			for (int i = 0; i < tiles; i++) {
				for (int j = 0; j < tiles; j++) {
					pixels[i][j] = pixels[i][j] > 1.0 ? 1.0 : pixels[i][j];
				}
			}
		}

		if (IsMouseButtonPressed(MOUSE_RIGHT_BUTTON)) {
			for (int y = 0; y < tiles; y++) {
				for (int x = 0; x < tiles; x++) {
					pixels[y][x] = 0.0;
				}
			}
		}

		char buf[128];
		char buf2[128];
		if (cnt++ % 6 == 0) {
			Mat img = ConvertToMatrix(pixels);
			NN_SET_INPUT(nn, img);
			nn_forward(nn);
			float max = 0.0;
			int mval = 0;
			for (size_t j = 0; j < 10; j++) {
				if (MAT_AT(NN_OUTPUT(nn), 0, j) > max) {
					max = MAT_AT(NN_OUTPUT(nn), 0, j);
					mval = j;
				}
			}
			sprintf(buf, "Number - %i", mval);
			mat_free(img);
		}

		BeginDrawing();
		ClearBackground(RAYWHITE);

		DrawText("Click RMB to clear the screen", 228, 580, 20, DARKGRAY);
		DrawText(buf, 20, 580, 20, DARKGRAY);

		for (int i = 0; i < 10; i++) {
			float k = MAT_AT(NN_OUTPUT(nn), 0, i);
			int t = tile * tiles + 160;
			float l = (tile * tiles - 60) / 9.0;
			DrawRectangle(20 + i * l, t - k * 117, 20, k * 117, DARKGRAY);
			sprintf(buf2, "%d", i);
			DrawText(buf2, 20 + i * l + 5, t + 10, 20, DARKGRAY);
		}

		for (int y = 0; y < tiles; y++) {
			for (int x = 0; x < tiles; x++) {
				float v = (int)(pixels[y][x] * 255.0f);
				Color c = {v, v, v, 255};
				DrawRectangle(x * tile, y * tile, tile, tile, c);
			}
		}

		EndDrawing();
	}

	CloseWindow();
}
