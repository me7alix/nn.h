#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <raylib.h>
#include "../parsers/csv_parser.c"

#define NN_IMPLEMENTATION
#include "../nn.h"

void draw_mat(Mat m, Vector2 pos, int tile){
  for (int i = 0; i < 28 * 28; i++) {
    float c = MAT_AT(m, 0, i) * 255.0;
    DrawRectangle(pos.x + (i % 28) * tile, pos.y + (int)(i / 28) * tile, tile, tile, (Color){c, c, c, 255});
  }
}

void mat_print_2(Mat m) {
  for (int i = 0; i < m.rows; i++) {
    printf("\t");
    for (int j = 0; j < m.cols; j++) {
      printf("%f, ", MAT_AT(m, i, j));
    }
    printf("\n");
  }
}

int main() {
  srand(time(0));
  Mat mat = parse_csv_to_mat("./dataset/train.csv");

  Layer layers[] = {
    (Layer){ .size = 3 },
    (Layer){ .size = 10, .actf = ACT_TANH, .randf = glorot_randf }, 
    (Layer){ .size = 10, .actf = ACT_TANH, .randf = glorot_randf }, 
    (Layer){ .size = 10, .actf = ACT_SIGM, .randf = glorot_randf }, 
    (Layer){ .size = 1, .actf = ACT_SIGM, .randf = glorot_randf }
  };


  NN nn = nn_alloc(layers, ARR_LEN(layers));
  NN g = nn_alloc(layers, ARR_LEN(layers));
  nn_rand(nn);

  // initializing the data
  Mat imgs = mat_submatrix(mat, 1, 0, mat.cols - 1, mat.rows - 1);

  for (size_t i = 0; i < imgs.rows; i++) {
    for (size_t j = 0; j < imgs.cols; j++) {
      MAT_AT(imgs, i, j) /= 255.0;
    }
  }

  Mat img1 = mat_submatrix(imgs, 0, 78, 28*28-1, 78);
  Mat img2 = mat_submatrix(imgs, 0, 8, 28*28-1, 8);
  Mat img3 = mat_submatrix(imgs, 0, 5, 28*28-1, 5);

  Mat ti = mat_alloc(1, 3);
  Mat to = mat_alloc(1, 1);

  float input_slider = 0.0;
  float learning_rate_slider = 0.6;

  // init window 
  const int screen_width = 920;
  const int screen_height = 640;

  int iter = 0;

  InitWindow(screen_width, screen_height, "Image learning");
  SetTargetFPS(60);

  while (!WindowShouldClose()) {
    if (IsMouseButtonDown(MOUSE_BUTTON_LEFT))
      input_slider = fmin(1.0, fmaxf(GetMouseX()/(float)screen_width, 0));

    if (IsMouseButtonDown(MOUSE_BUTTON_RIGHT))
      learning_rate_slider = fmin(1.0, fmaxf(GetMouseX()/(float)screen_width, 0));

    if (IsKeyPressed(KEY_ENTER)) {
      for (int i = 0; i < nn.count; i++) {
        printf("ws[%i]\n", i);
        mat_print_2(nn.ws[i]);

        printf("bs[%i]\n", i);
        mat_print_2(nn.bs[i]);
      }
    }

    if (iter % 100 == 0) 
      printf("iter - %i\n", iter);
    iter++;

    BeginDrawing();
    ClearBackground(DARKGRAY);

    for (int i = 0; i < 28; i++) {
      for (int j = 0; j < 28; j++) { 
        MAT_AT(ti, 0, 0) = (j / 28.0 - 0.5);
        MAT_AT(ti, 0, 1) = (i / 28.0 - 0.5);
        MAT_AT(ti, 0, 2) = -0.5;
        MAT_AT(to, 0, 0) = MAT_AT(img1, 0, i*28+j);

        nn_backprop(nn, g, ti, to);
        nn_learn(nn, g, 0.02 * learning_rate_slider);

        MAT_AT(ti, 0, 0) = (j / 28.0 - 0.5);
        MAT_AT(ti, 0, 1) = (i / 28.0 - 0.5);
        MAT_AT(ti, 0, 2) = 0;
        MAT_AT(to, 0, 0) = MAT_AT(img2, 0, i*28+j);

        nn_backprop(nn, g, ti, to);
        nn_learn(nn, g, 0.02 * learning_rate_slider);      

        MAT_AT(ti, 0, 0) = (j / 28.0 - 0.5);
        MAT_AT(ti, 0, 1) = (i / 28.0 - 0.5);
        MAT_AT(ti, 0, 2) = 0.5;
        MAT_AT(to, 0, 0) = MAT_AT(img3, 0, i*28+j);

        nn_backprop(nn, g, ti, to);
        nn_learn(nn, g, 0.02 * learning_rate_slider);
      }
    } 
    
    Vector2 pos = {320, 320};
    float scale = 3;
    float pixels = 280.0/scale;
    for (int i = 0; i < pixels; i++) {
      for (int j = 0; j < pixels; j++) { 
        MAT_AT(NN_INPUT(nn), 0, 0) = (j / pixels - 0.5);
        MAT_AT(NN_INPUT(nn), 0, 1) = (i / pixels - 0.5);
        MAT_AT(NN_INPUT(nn), 0, 2) = input_slider - 0.5;
        nn_forward(nn);

        float c = MAT_AT(NN_OUTPUT(nn), 0, 0) * 255.0;
        DrawRectangle(pos.x + j * scale, pos.y + i * scale, scale, scale, (Color){c, c, c, 255});
      }
    }

    draw_mat(img1, (Vector2){20, 20}, 10);
    draw_mat(img2, (Vector2){320, 20}, 10);
    draw_mat(img3, (Vector2){620, 20}, 10);

    DrawRectangle(learning_rate_slider * screen_width - 10, screen_height - 20, 20, 20, RED);
    DrawRectangle(input_slider * screen_width - 10, screen_height - 20, 20, 20, BLUE);
    DrawText("Use LMB to move \nblue slider (input value)\n\nUse RMB to move \nred slider (learning rate)", 20, 320, 20, WHITE);

    EndDrawing();
  }

  nn_free(nn);
  nn_free(g);
  mat_free(mat);

  CloseWindow();
  return 0;
}
