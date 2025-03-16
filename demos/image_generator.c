#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <raylib.h>
#include "../csv_parser.c"

#define NN_IMPLEMENTATION
#include "../nn.h"

void draw_mat(Mat m, Vector2 pos, int tile){
  for (int i = 0; i < 28 * 28; i++) {
    float c = MAT_AT(m, 0, i) * 255.0;
    DrawRectangle(pos.x + (i % 28) * tile, pos.y + (int)(i / 28) * tile, tile, tile, (Color){c, c, c, 255});
  }
}

int main() {
  srand(time(0));
  Mat mat = parse_csv_to_mat("./dataset/train.csv");
  Mat cti = mat_submatrix(mat, 1, 0, mat.cols - 1, 1000);

  Layer layers[] = {
    (Layer){
      .size = 28*28,
      .randf = glorot_randf,
    },
    (Layer){
      .size = 64,
      .actf = ACT_RELU,
      .randf = glorot_randf,
    },
    (Layer){
      .size = 16,
      .actf = ACT_SIGM,
      .randf = glorot_randf,
    }, 
    (Layer){
      .size = 4,
      .actf = ACT_SIGM,
      .randf = glorot_randf,
    },
    (Layer){
      .size = 16,
      .actf = ACT_RELU,
      .randf = glorot_randf,
    },
    (Layer){
      .size = 64,
      .actf = ACT_RELU,
      .randf = glorot_randf,
    },
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

  // initializing the data
  Mat imgs = mat_submatrix(mat, 1, 0, mat.cols - 1, mat.rows - 1);

  for (size_t i = 0; i < imgs.rows; i++) {
    for (size_t j = 0; j < imgs.cols; j++) {
      MAT_AT(imgs, i, j) /= 255.0;
    }
  }
 
  // learning process
  size_t batch_size = 16;
  float learning_rate = 0.002;

  for (size_t i = 0; true; i++) { 
    size_t pos = (rand()) % (imgs.rows - batch_size);
    Mat gti = mat_submatrix(imgs, 0, pos, imgs.cols - 1, pos + batch_size);
    Mat gto = mat_submatrix(imgs, 0, pos, imgs.cols - 1, pos + batch_size);

    nn_backprop(nn, g, gti, gto);
    //nn_learn(nn, g, learning_rate);
    adam_update(nn, &adam, g, learning_rate, 0.9, 0.999, 1e-8);

    if (i % 600 == 0) {
      fd_set fds;
      struct timeval timeout = {0, 0};
      FD_ZERO(&fds);
      FD_SET(0, &fds);

      if (select(1, &fds, NULL, NULL, &timeout) > 0) {
        char input = getchar();
        if (input == 'S' || input == 's') {
          break; 
        }
      }

      float tc = nn_cost(nn, cti, cti);
      printf("cost %zu - %f\n", i, tc);
    }
  }

  Layer layers_2[] = {
    (Layer){
      .size = 4,
      .actf = ACT_SIGM,
      .randf = glorot_randf,
    },
    (Layer){
      .size = 16,
      .actf = ACT_RELU,
      .randf = glorot_randf,
    },
    (Layer){
      .size = 64,
      .actf = ACT_RELU,
      .randf = glorot_randf,
    },
    (Layer){
      .size = 28*28,
      .actf = ACT_SIGM,
      .randf = glorot_randf,
    },
  };

  NN nn_2 = nn_alloc(layers_2, ARR_LEN(layers_2));
  size_t from = 3;

  for (size_t i = from; i < nn.count; i++) {
    mat_copy(nn_2.ws[i-from], nn.ws[i]);
    mat_copy(nn_2.bs[i-from], nn.bs[i]);
  }

  const int screenWidth = 800;
  const int screehHeight = 600;
  InitWindow(screenWidth, screehHeight, "Image generator");
  SetTargetFPS(60);

  while (!WindowShouldClose()) {
    BeginDrawing();
    ClearBackground(BLACK);

    Mat inp = mat_alloc(1, 3);

    if (IsKeyPressed(KEY_ENTER)) {
      mat_rand_between(NN_INPUT(nn_2), 0, 1);
      nn_forward(nn_2);
    }

    if (IsKeyDown(KEY_ONE)) {
      MAT_AT(inp, 0, 0) = fmin(1.0, fmaxf(GetMouseX()/(float)screenWidth, 0));
      mat_copy(NN_INPUT(nn_2), inp);
      nn_forward(nn_2);
    }

    if (IsKeyDown(KEY_TWO)) {
      MAT_AT(inp, 0, 1) = fmin(1.0, fmaxf(GetMouseX()/(float)screenWidth, 0));
      mat_copy(NN_INPUT(nn_2), inp);
      nn_forward(nn_2);
    }

    if (IsKeyDown(KEY_THREE)) {
      MAT_AT(inp, 0, 2) = fmin(1.0, fmaxf(GetMouseX()/(float)screenWidth, 0));
      mat_copy(NN_INPUT(nn_2), inp);
      nn_forward(nn_2);
    }


    draw_mat(NN_OUTPUT(nn_2), (Vector2){10, 10}, 10);
    draw_mat(NN_OUTPUT(nn_2), (Vector2){290, 10}, 10);
    draw_mat(NN_OUTPUT(nn_2), (Vector2){10, 290}, 10);
    draw_mat(NN_OUTPUT(nn_2), (Vector2){290, 290}, 10);

    EndDrawing();
  } 

  CloseWindow();
  return 0;
}
