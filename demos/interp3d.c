#include <raylib.h>
#include <raymath.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include "../vox_parser.c"

#define NN_IMPLEMENTATION
#include "../nn.h"

void draw_voxels(Voxel *model) {
  for (size_t i = 0; i < CUBE_SIZE; i++) {
    for (size_t j = 0; j < CUBE_SIZE; j++) {
      for (size_t k = 0; k < CUBE_SIZE; k++) {
        if (model[index3D(j, i, k)].colorIndex != -1) {
          DrawCubeV((Vector3){j, k, i}, (Vector3){1, 1, 1}, BLACK);
          float t = 0.05;
          DrawCubeWiresV((Vector3){j + t / 2.0, k + t / 2.0, i + t / 2.0},
                         (Vector3){1 + t, 1 + t, 1 + t}, WHITE);
        }
      }
    }
  }
}

int main(void) {
  srand(time(0));

  Layer layers[] = {
    (Layer){.size = 4},
    (Layer){.size = 10, .actf = ACT_RELU, .randf = glorot_randf},
    (Layer){.size = 10, .actf = ACT_RELU, .randf = glorot_randf},
    (Layer){.size = 10, .actf = ACT_RELU, .randf = glorot_randf},
    (Layer){.size = 10, .actf = ACT_RELU, .randf = glorot_randf},
    (Layer){.size = 1, .actf = ACT_SIGM, .randf = glorot_randf}
  };

  NN nn = nn_alloc(layers, ARR_LEN(layers));
  NN g = nn_alloc(layers, ARR_LEN(layers));
  AdamOptimizer adam = adam_alloc(nn);
  
  nn_rand(nn);

  Voxel *torus = load_vox_model("models/torus_small.vox");
  Voxel *apple = load_vox_model("models/apple_small.vox");

  Mat input = mat_alloc(1, 4);
  Mat output = mat_alloc(1, 1);

  const float learning_rate = 0.001;

  for (int iter = 0; iter < 3000; iter ++) {
    for (size_t i = 0; i < CUBE_SIZE; i++) {
      for (size_t j = 0; j < CUBE_SIZE; j++) {
        for (size_t k = 0; k < CUBE_SIZE; k++) {
          MAT_AT(input, 0, 0) = i / (CUBE_SIZE-1.0);
          MAT_AT(input, 0, 1) = j / (CUBE_SIZE-1.0);
          MAT_AT(input, 0, 2) = k / (CUBE_SIZE-1.0);
          MAT_AT(input, 0, 3) = 0;
          MAT_AT(output, 0, 0) = (torus[index3D(j, i, k)].colorIndex != -1); 
          nn_backprop(nn, g, input, output);
          nn_learn(nn, g, 0.005);
          //adam_update(nn, &adam, g, learning_rate, 0.9, 0.999, 1e-8);

          MAT_AT(input, 0, 0) = i / (CUBE_SIZE-1.0);
          MAT_AT(input, 0, 1) = j / (CUBE_SIZE-1.0);
          MAT_AT(input, 0, 2) = k / (CUBE_SIZE-1.0);
          MAT_AT(input, 0, 3) = 1;
          MAT_AT(output, 0, 0) = (apple[index3D(j, i, k)].colorIndex != -1); 
          nn_backprop(nn, g, input, output);
          nn_learn(nn, g, 0.005);
          //adam_update(nn, &adam, g, learning_rate, 0.9, 0.999, 1e-8);
        }
      }
    }

    if(iter % 100 == 0) 
      printf("iter - %d\n", iter);
  }

  scanf("\n");

  const int screenWidth = 800;
  const int screenHeight = 600;
  InitWindow(screenWidth, screenHeight, "Interp 3D");

  Camera camera = {0};
  camera.position = (Vector3){
      30.0f, 30.0f, 30.0f}; // Camera position (offset for a good view)
  camera.target = (Vector3){CUBE_SIZE / 2.0f, CUBE_SIZE / 2.0f,
                            CUBE_SIZE / 2.0f}; // Look at center of cube
  camera.up =
      (Vector3){0.0f, 1.0f, 0.0f}; // Up vector (rotation towards target)
  camera.fovy = 70.0f;             // Field-of-view Y
  camera.projection = CAMERA_PERSPECTIVE;

  float slider = 0.0;

  while (!WindowShouldClose()) {
    BeginDrawing();
    ClearBackground(BLACK);

    UpdateCamera(&camera, CAMERA_PERSPECTIVE);
    BeginMode3D(camera);

    if (IsMouseButtonDown(MOUSE_BUTTON_LEFT)) {
      slider = GetMouseX() / (float)screenWidth;
    }

    for (size_t i = 0; i < CUBE_SIZE; i++) {
      for (size_t j = 0; j < CUBE_SIZE; j++) {
        for (size_t k = 0; k < CUBE_SIZE; k++) {
          MAT_AT(NN_INPUT(nn), 0, 0) = i / (CUBE_SIZE-1.0);
          MAT_AT(NN_INPUT(nn), 0, 1) = j / (CUBE_SIZE-1.0);
          MAT_AT(NN_INPUT(nn), 0, 2) = k / (CUBE_SIZE-1.0);
          MAT_AT(NN_INPUT(nn), 0, 3) = slider;
          nn_forward(nn);

          //printf("%f\n", MAT_AT(NN_OUTPUT(nn), 0, 0));

          if (MAT_AT(NN_OUTPUT(nn), 0, 0) > 0.5) {
            DrawCubeV((Vector3){j, k, i}, (Vector3){1, 1, 1}, BLACK);
            float t = 0.05;
            DrawCubeWiresV((Vector3){j + t / 2.0, k + t / 2.0, i + t / 2.0},
                         (Vector3){1 + t, 1 + t, 1 + t}, WHITE); 
          }
        }
      }
    }

    EndMode3D();
    EndDrawing();
  }
  //NN_PRINT(nn);

  CloseWindow();
  return 0;
}
