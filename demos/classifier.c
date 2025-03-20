#include <stdbool.h>
#include <stdlib.h>
#include <time.h>
#include <raylib.h>
#define NN_IMPLEMENTATION
#include "../nn.h"

//#define ADAM_OPT

#define TCAP 256

bool is_learning_started = false;

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

  Vector2 type1[TCAP];
  int type1_size = 0;

  Vector2 type2[TCAP];
  int type2_size = 0;

  Layer layers[] = {
    (Layer){
      .size = 2,
    },
    (Layer){
      .actf = ACT_SIGM,
      .size = 6,
      .randf = glorot_randf,
    },
    (Layer){
      .actf = ACT_SIGM,
      .size = 6,
      .randf = glorot_randf,
    },
    (Layer){
      .actf = ACT_SIGM,
      .size = 6,
      .randf = glorot_randf,
    },
    (Layer){
      .actf = ACT_SIGM,
      .size = 6,
      .randf = glorot_randf,
    },
    (Layer){
      .actf = ACT_SIGM,
      .size = 2,
      .randf = glorot_randf,
    },
  };

  NN nn = nn_alloc(layers, ARR_LEN(layers)); 
  NN g = nn_alloc(layers, ARR_LEN(layers)); 

#ifdef ADAM_OPT
  AdamOptimizer adam = adam_alloc(nn);
#endif

  nn_rand(nn);

  float learning_rate = 0.01;
  Mat inp = mat_alloc(1, 2);
  Mat outp = mat_alloc(1, 2);

  int iter = 0;

  while (!WindowShouldClose()) {
    BeginDrawing();
    ClearBackground(BLACK);

    if (IsMouseButtonPressed(MOUSE_BUTTON_LEFT)) { 
      Vector2 n = GetMousePosition();
      n.x /= (float)screenWidth;
      n.y /= (float)screenHeight;
      type1[type1_size++] = n;
    }

    if (IsMouseButtonPressed(MOUSE_BUTTON_RIGHT)) { 
      Vector2 n = GetMousePosition();
      n.x /= (float)screenWidth;
      n.y /= (float)screenHeight;
      type2[type2_size++] = n;
    }

    if (IsKeyPressed(KEY_ENTER)) {
      is_learning_started = true;
    }

    for (int o = 0; o < 80; o++) {
      if (type1_size == 0 || type2_size == 0 || !is_learning_started) continue;

      MAT_AT(inp, 0, 0) = type1[iter%type1_size].x; 
      MAT_AT(inp, 0, 1) = type1[iter%type1_size].y;
      MAT_AT(outp, 0, 0) = 1;
      MAT_AT(outp, 0, 1) = 0;

      nn_backprop(nn, g, inp, outp);

#ifdef ADAM_OPT
      adam_update(nn, &adam, g, learning_rate, 0.9, 0.999, 1e-8);
#else 
      nn_learn(nn, g, learning_rate); 
#endif

      MAT_AT(inp, 0, 0) = type2[iter%type2_size].x; 
      MAT_AT(inp, 0, 1) = type2[iter%type2_size].y;
      MAT_AT(outp, 0, 0) = 0;
      MAT_AT(outp, 0, 1) = 1;

      nn_backprop(nn, g, inp, outp);

#ifdef ADAM_OPT
      adam_update(nn, &adam, g, learning_rate, 0.9, 0.999, 1e-8);
#else 
      nn_learn(nn, g, learning_rate); 
#endif      

      iter++;
    }

    const int scale = 8;
    for (float i = 0; i < screenHeight; i+=scale) {
      for (float j = 0; j < screenWidth; j+=scale) {
        MAT_AT(NN_INPUT(nn), 0, 0) = j/screenWidth;
        MAT_AT(NN_INPUT(nn), 0, 1) = i/screenHeight;
        nn_forward(nn);
        float calc = (MAT_AT(NN_OUTPUT(nn), 0, 0) - MAT_AT(NN_OUTPUT(nn), 0, 1) + 1.0) / 2.0;
        float coef = calc * 255; 

        DrawRectangle(
          j, i, scale, scale,
          (Color){coef, 255-coef, 255, 255}
        );
      }
    }

    draw_points(type1, type1_size, LIGHTGRAY, screenWidth, screenHeight);
    draw_points(type2, type2_size, PURPLE, screenWidth, screenHeight);
    DrawText("Press ENTER to start learning process", 20, 20, 20, BLACK);

    EndDrawing();
  }

  CloseWindow();
  return 0;
}
