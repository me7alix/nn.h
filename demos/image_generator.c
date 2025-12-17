#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <raylib.h>
#include "../parsers/csv_parser.c"

#define NN_IMPLEMENTATION
#include "../nn.h"

// Visualize an image from matrix data
void draw_mat(Mat m, Vector2 pos, int tile){
	for (int i = 0; i < 28 * 28; i++) {
		float c = MAT_AT(m, 0, i) * 255.0;
		DrawRectangle(
			pos.x + (i % 28) * tile,
			pos.y + (int)(i / 28) * tile,
			tile, tile,
			(Color){c, c, c, 255}
		);
	}
}

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

void add_noise_to_batch(Mat batch, float noise_level) {
	for (size_t i = 0; i < batch.rows; i++) {
		for (size_t j = 0; j < batch.cols; j++) {
			float noise = ((float)rand() / RAND_MAX * 2.0f - 1.0f) * noise_level;
			float pixel = MAT_AT(batch, i, j) + noise;
			// Clamp values between 0 and 1
			if (pixel < 0) pixel = 0;
			if (pixel > 1) pixel = 1;
			MAT_AT(batch, i, j) = pixel;
		}
	}
}

int main() {
	srand(time(0));
	Mat mat = parse_csv_to_mat("./dataset/train.csv", ",");

	// Use more validation samples
	Mat cti = mat_submatrix(mat, 1, 0, mat.cols - 1, 2000);

	// Increased latent space dimensions
	const int latent_dims = 2;

	// Improved architecture with more layers and neurons
	Layer layers[] = {
		// Input layer
		(Layer){
			.size = 28*28,
		},
		// Encoder layers
		(Layer){
			.size = 128,
			.actf = ACT_RELU,
			.randf = glorot_randf,
		},
		(Layer){
			.size = 32,
			.actf = ACT_RELU,
			.randf = glorot_randf,
		},
		(Layer){
			.size = 8,
			.actf = ACT_RELU,
			.randf = glorot_randf,
		},
		// Latent space layer (bottleneck)
		(Layer){
			.size = latent_dims,
			.actf = ACT_SIGM,  // Using sigmoid to constrain values between 0-1
			.randf = glorot_randf,
		},
		// Decoder layers
		(Layer){
			.size = 8,
			.actf = ACT_RELU,
			.randf = glorot_randf,
		},
		(Layer){
			.size = 32,
			.actf = ACT_RELU,
			.randf = glorot_randf,
		},
		(Layer){
			.size = 128,
			.actf = ACT_RELU,
			.randf = glorot_randf,
		},
		// Output layer (reconstruction)
		(Layer){
			.size = 28*28,
			.actf = ACT_SIGM,
			.randf = glorot_randf,
		},
	};

	NN nn = nn_alloc(layers, ARR_LEN(layers));
	NN g = nn_alloc(layers, ARR_LEN(layers));
	AdamOptimizer adam = adam_alloc(nn);
	nn_rand(nn);

	// Normalizing the dataset
	Mat imgs = mat_submatrix(mat, 1, 0, mat.cols - 1, mat.rows - 1);

	for (size_t i = 0; i < imgs.rows; i++) {
		for (size_t j = 0; j < imgs.cols; j++) {
			MAT_AT(imgs, i, j) /= 255.0;
		}
	}

	printf("Press S+ENTER to stop learning process\n");

	// Improved learning parameters
	size_t batch_size = 32;  // Increased batch size
	float learning_rate = 0.001;  // Adjusted learning rate
	float noise_level = 0.01f;  // Noise for data augmentation
	size_t max_iterations = 100000;  // Set a reasonable maximum
	float best_cost = 1000.0f;  // Keep track of best model

	// Save best weights
	NN best_nn = nn_alloc(layers, ARR_LEN(layers));
	nn_copy(best_nn, nn);

	// Training loop
	for (size_t i = 0; i < max_iterations; i++) { 
		// Randomly select a batch
		size_t pos = (rand()) % (imgs.rows - batch_size);
		Mat gti = mat_submatrix(imgs, 0, pos, imgs.cols - 1, pos + batch_size);
		Mat gto = mat_submatrix(imgs, 0, pos, imgs.cols - 1, pos + batch_size);

		// Apply data augmentation occasionally
		if (rand() % 5 == 0) {
			add_noise_to_batch(gti, noise_level);
		}

		// Backpropagate and update
		nn_backprop(nn, g, gti, gto);
		adam_update(nn, &adam, g, learning_rate, 0.9, 0.999, 1e-8);

		// Evaluate and report periodically
		if (i % 100 == 0) {
			if (wait_for_keypress()) break;

			float tc = nn_cost(nn, cti, cti);
			printf("Iteration %zu - Cost: %f\n", i, tc);

			// Save the best model
			if (tc < best_cost) {
				best_cost = tc;
				nn_copy(best_nn, nn);
				printf("New best model saved! Cost: %f\n", best_cost);
			}

			// Learning rate decay
			if (i % 5000 == 0 && i > 0) {
				learning_rate *= 0.9f;
				printf("Reducing learning rate to %f\n", learning_rate);
			}
		}
	}

	// Use the best model for further operations
	nn_copy(nn, best_nn);
	printf("Training completed. Best cost: %f\n", best_cost);

	// Create decoder only neural network
	Layer decoder_layers[] = {
		// Input layer (latent space)
		(Layer){
			.size = latent_dims,
		},
		// Decoder layers
		(Layer){
			.size = 8,
			.actf = ACT_RELU,
			.randf = glorot_randf,
		}, 
		(Layer){
			.size = 32,
			.actf = ACT_RELU,
			.randf = glorot_randf,
		},
		(Layer){
			.size = 128,
			.actf = ACT_RELU,
			.randf = glorot_randf,
		},
		// Output layer
		(Layer){
			.size = 28*28,
			.actf = ACT_SIGM,
			.randf = glorot_randf,
		},
	};

	NN decoder = nn_alloc(decoder_layers, ARR_LEN(decoder_layers));

	// Copy weights from the trained autoencoder to the decoder
	size_t decoder_start = 4;  // Index where decoder starts in the full autoencoder
	for (size_t i = 0; i < decoder.count; i++) {
		mat_copy(decoder.ws[i], nn.ws[i + decoder_start]);
		mat_copy(decoder.bs[i], nn.bs[i + decoder_start]);
	}

	// Setup window for visualization
	const int screenWidth = 800;
	const int screenHeight = 650;
	InitWindow(screenWidth, screenHeight, "MNIST Generator");
	SetTargetFPS(60);

	// Latent vector for exploration
	Mat latent = mat_alloc(1, latent_dims);
	mat_rand_between(latent, 0.3, 0.7);  // Initialize with reasonable values

	// For interpolation between two points
	Mat latent_start = mat_alloc(1, latent_dims);
	Mat latent_end = mat_alloc(1, latent_dims);
	bool interpolating = false;
	float interp_progress = 0.0f;

	// Animation speed
	float interp_speed = 0.02f;

	// UI state
	int selected_dim = 0;
	float slider_value = 0.5f;
	bool slider_active = false;

	while (!WindowShouldClose()) {
		// Input handling
		if (IsKeyPressed(KEY_ENTER)) {
			// Generate completely random latent vector
			mat_rand_between(NN_INPUT(decoder), 0.0, 1.0);
			nn_forward(decoder);
		}

		if (IsMouseButtonDown(MOUSE_BUTTON_LEFT)) {
			MAT_AT(NN_INPUT(decoder), 0, 0) = ((float)GetMouseX() / screenWidth);
			MAT_AT(NN_INPUT(decoder), 0, 1) = ((float)GetMouseY() / screenHeight);

			nn_forward(decoder);
		}

		if (IsKeyPressed(KEY_SPACE)) {
			// Start interpolation
			mat_copy(latent_start, NN_INPUT(decoder));
			mat_rand_between(latent_end, 0.0, 1.0);
			interpolating = true;
			interp_progress = 0.0f;
		}

		// Handle dimension selection
		if (IsKeyPressed(KEY_RIGHT)) {
			selected_dim = (selected_dim + 1) % latent_dims;
		}
		if (IsKeyPressed(KEY_LEFT)) {
			selected_dim = (selected_dim - 1 + latent_dims) % latent_dims;
		}

		// Handle slider for the selected dimension
		if (IsKeyDown(KEY_UP)) {
			slider_value = fminf(slider_value + 0.01f, 1.0f);
			MAT_AT(NN_INPUT(decoder), 0, selected_dim) = slider_value;
			nn_forward(decoder);
		}
		if (IsKeyDown(KEY_DOWN)) {
			slider_value = fmaxf(slider_value - 0.01f, 0.0f);
			MAT_AT(NN_INPUT(decoder), 0, selected_dim) = slider_value;
			nn_forward(decoder);
		}

		// Update interpolation
		if (interpolating) {
			interp_progress += interp_speed;
			if (interp_progress >= 1.0f) {
				interp_progress = 1.0f;
				interpolating = false;
			}

			for (int d = 0; d < latent_dims; d++) {
				float start_val = MAT_AT(latent_start, 0, d);
				float end_val = MAT_AT(latent_end, 0, d);
				float current = start_val + (end_val - start_val) * interp_progress;
				MAT_AT(NN_INPUT(decoder), 0, d) = current;
			}

			nn_forward(decoder);
		}

		// Update slider_value to match the current dimension
		slider_value = MAT_AT(NN_INPUT(decoder), 0, selected_dim);

		// Drawing
		BeginDrawing();
		ClearBackground(DARKGRAY);

		draw_mat(NN_OUTPUT(decoder), (Vector2){50, 50}, 10);

		DrawText("LATENT SPACE EXPLORER", 400, 50, 20, WHITE);

		DrawText("Controls:", 400, 80, 15, WHITE);
		DrawText("- ENTER: Random image", 400, 100, 13, WHITE);
		DrawText("- SPACE: Animate between random points", 400, 120, 13, WHITE);

		// Draw slider for currently selected dimension
		DrawText(TextFormat("Dimension %d:", selected_dim), 50, 450, 15, WHITE);
		DrawRectangle(50, 470, 200, 20, DARKGRAY);
		DrawRectangle(50, 470, (int)(slider_value * 200), 20, BLUE);
		DrawRectangleLines(50, 470, 200, 20, WHITE);
		DrawText(TextFormat("%.2f", slider_value), 260, 470, 15, WHITE);

		// Interpolation status
		if (interpolating) {
			DrawText(TextFormat("Interpolating: %.0f%%", interp_progress * 100), 400, 200, 15, GREEN);
		}

		EndDrawing();
	}

	// Cleanup
	nn_free(nn);
	nn_free(g);
	nn_free(decoder);
	nn_free(best_nn);
	mat_free(mat);
	mat_free(latent);
	mat_free(latent_start);
	mat_free(latent_end);
	adam_free(adam);

	CloseWindow();
	return 0;
}
