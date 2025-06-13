#include <raylib.h>
#include <raymath.h>
#include <stdbool.h>
#include <stdio.h>

#define NN_IMPLEMENTATION
#include "../nn.h"

#define POPUL 50

#define CAR_W 30
#define CAR_H 50
#define RAY_CNT 5
#define RAY_FOV PI * 0.75
#define RAY_LEN 100

typedef struct {
  Vector2 p1, p2;
} Wall;

typedef struct {
  NN nn;
  Vector2 pos;
  float ang;
  int cur_cp;
  bool isDead;
} Car;

Wall walls[200];
int walls_cnt = 0;

Vector2 checkpoints[200];
float cp_radius = 20.0;
int cp_cnt = 0;

Car cars[POPUL];
float speed = 60;
Vector2 start_pos;
int dead_cars = 0;

// neural network arch
Layer layers[] = {
    (Layer){
        .size = RAY_CNT,
    },
    (Layer){
        .size = 8,
        .actf = ACT_SIGM,
    },
    (Layer){
        .size = 1,
        .actf = ACT_SIGM,
    },
};

NN best_nn;
int best_cp = 0;
float best_dists[200];
bool editor_mode = true;
int epoch = 0;

bool get_intersection(Vector2 a1, Vector2 a2, Vector2 b1, Vector2 b2,
                      Vector2 *intersection) {
  float a1_x = a1.x, a1_y = a1.y;
  float a2_x = a2.x, a2_y = a2.y;
  float b1_x = b1.x, b1_y = b1.y;
  float b2_x = b2.x, b2_y = b2.y;

  float denominator =
      (a1_x - a2_x) * (b1_y - b2_y) - (a1_y - a2_y) * (b1_x - b2_x);

  if (denominator == 0) {
    return false;
  }

  float t = ((a1_x - b1_x) * (b1_y - b2_y) - (a1_y - b1_y) * (b1_x - b2_x)) /
            denominator;
  float u = -((a1_x - a2_x) * (a1_y - b1_y) - (a1_y - a2_y) * (a1_x - b1_x)) /
            denominator;

  if (t >= 0 && t <= 1 && u >= 0 && u <= 1) {
    intersection->x = a1_x + t * (a2_x - a1_x);
    intersection->y = a1_y + t * (a2_y - a1_y);
    return true;
  }

  return false;
}

void cp_draw() {
  char buf[128];
  for (int i = 0; i < cp_cnt; i++) {
    DrawCircleLinesV(checkpoints[i], cp_radius, YELLOW);
    sprintf(buf, "%i", i);
    DrawText(buf, checkpoints[i].x - 5, checkpoints[i].y - 5, 20, YELLOW);
  }
}

void cp_check(Car *car) {
  for (int i = 0; i < cp_cnt; i++) {
    if (Vector2Distance(checkpoints[i], car->pos) < cp_radius + CAR_H) {
      car->cur_cp = i + 1;
    }
  }
}

void cp_add(Vector2 point) { checkpoints[cp_cnt++] = point; }
void cp_pop() {
  if (cp_cnt > 0)
    cp_cnt--;
}

void walls_add(Vector2 a, Vector2 b) { walls[walls_cnt++] = (Wall){a, b}; }
void walls_pop() {
  if (walls_cnt > 0)
    walls_cnt--;
}

void walls_draw() {   
  for (int i = 0; i < walls_cnt; i++) {
    DrawLineV(walls[i].p1, walls[i].p2, RED);
  }
}

void cars_update() {
  for (int i = 0; i < POPUL; i++) {
    if (cars[i].isDead)
      continue;
    for (int k = 0; k < RAY_CNT; k++) {
      Vector2 r = cars[i].pos;
      float ang =
          cars[i].ang - RAY_FOV / 2 + (float)k / (RAY_CNT - 1) * RAY_FOV;
      r.x += cosf(ang) * RAY_LEN;
      r.y -= sinf(ang) * RAY_LEN;

      MAT_AT(NN_INPUT(cars[i].nn), 0, k) = 0;

      float minDist = 99999;
      Vector2 p, pl;
      for (int j = 0; j < walls_cnt; j++) {
        if (get_intersection(cars[i].pos, r, walls[j].p1, walls[j].p2, &p)) {
          float d = Vector2Distance(cars[i].pos, p);
          if (minDist > d) {
            minDist = d;
            pl = p;
          }
        }
      }

      MAT_AT(NN_INPUT(cars[i].nn), 0, k) = 30 / minDist;

      if (minDist < 99998)
        DrawLineV(cars[i].pos, pl, YELLOW);
      else
        DrawLineV(cars[i].pos, r, YELLOW);
    }

    nn_forward(cars[i].nn);
    float r = MAT_AT(NN_OUTPUT(cars[i].nn), 0, 0);
    float dt = (r - 0.5) * PI * 2 * 10;

    cars[i].ang += dt * GetFrameTime();
    cars[i].pos.x += cosf(cars[i].ang) * speed * GetFrameTime();
    cars[i].pos.y -= sinf(cars[i].ang) * speed * GetFrameTime();

    float dx1 = cosf(cars[i].ang) * CAR_H / 2.0;
    float dy1 = sinf(cars[i].ang) * CAR_H / 2.0;
    float dx2 = cosf(cars[i].ang - PI / 2.0) * CAR_W / 2.0;
    float dy2 = sinf(cars[i].ang - PI / 2.0) * CAR_W / 2.0;

    Vector2 w1 = cars[i].pos;
    w1.x += dx1; w1.y -= dy1;
    w1.x += -dx2; w1.y -= -dy2;

    Vector2 w2 = cars[i].pos;
    w2.x += dx1; w2.y -= dy1;
    w2.x += dx2; w2.y -= dy2;

    Vector2 w3 = cars[i].pos;
    w3.x += -dx1; w3.y -= -dy1;
    w3.x += -dx2; w3.y -= -dy2;

    Vector2 w4 = cars[i].pos;
    w4.x += -dx1; w4.y -= -dy1;
    w4.x += dx2; w4.y -= dy2;

    Vector2 p;
    for (int j = 0; j < walls_cnt; j++) {
      if (get_intersection(w1, w2, walls[j].p1, walls[j].p2, &p) ||
          get_intersection(w1, w3, walls[j].p1, walls[j].p2, &p) ||
          get_intersection(w2, w4, walls[j].p1, walls[j].p2, &p) ||
          get_intersection(w3, w4, walls[j].p1, walls[j].p2, &p)) {
        dead_cars++;
        cars[i].isDead = true;
        if (cars[i].cur_cp >= best_cp) {
          float dist =
              Vector2Distance(cars[i].pos, checkpoints[cars[i].cur_cp]);
          if (dist < best_dists[cars[i].cur_cp]) {
            nn_copy(best_nn, cars[i].nn);
            best_dists[cars[i].cur_cp] = dist;
            best_cp = cars[i].cur_cp;
          }
        }
        break;
      }
    }

    cp_check(cars + i);

    DrawLineV(w1, w2, GREEN);
    DrawLineV(w1, w3, GREEN);
    DrawLineV(w2, w4, GREEN);
    DrawLineV(w3, w4, GREEN);
  }
}

void cars_init(bool parent) {
  for (int i = 0; i < 200; i++) {
    best_dists[i] = 99999.0;
  }

  dead_cars = 0;
  for (int i = 0; i < POPUL; i++) {
    if (!parent) {
      cars[i] = (Car){
          .nn = nn_alloc(layers, ARR_LEN(layers)),
          .pos = start_pos,
          .ang = 0,
          .cur_cp = 0,
          .isDead = false,
      };
    } else {
      cars[i].pos = start_pos;
      cars[i].ang = 0;
      cars[i].cur_cp = 0;
      cars[i].isDead = false;
    }

    if (!parent) {
      nn_rand_between(cars[i].nn, -0.5, 0.5);
    } else {
      if (i < POPUL * 0.5) {
        nn_copy(cars[i].nn, best_nn);
        nn_add_rand_between(cars[i].nn, -0.02, 0.02);
      } else if (i < POPUL * 0.8) {
        nn_copy(cars[i].nn, best_nn);
        nn_add_rand_between(cars[i].nn, -0.07, 0.07);
      } else {
        nn_rand_between(cars[i].nn, -0.5, 0.5);
      }
    }
  }
}

int main(void) {
  const int screen_width = 800;
  const int screen_height = 600;

  InitWindow(screen_width, screen_height, "Car racing");
  SetTargetFPS(144);

  best_nn = nn_alloc(layers, ARR_LEN(layers));

  Vector2 fp = {};
  while (!WindowShouldClose()) {
    BeginDrawing();
    ClearBackground(BLACK);

    walls_draw();
    cp_draw();

    if (editor_mode) {
      if (IsMouseButtonPressed(MOUSE_BUTTON_LEFT))
        fp = GetMousePosition();
      if (IsMouseButtonReleased(MOUSE_BUTTON_LEFT))
        walls_add(GetMousePosition(), fp);
      if (IsMouseButtonPressed(MOUSE_BUTTON_RIGHT))
        walls_pop();
      if (IsKeyDown(KEY_SPACE))
        start_pos = GetMousePosition();
      if (IsKeyPressed(KEY_Z))
        cp_add(GetMousePosition());
      if (IsKeyPressed(KEY_X))
        cp_pop();
      if (IsKeyPressed(KEY_ENTER)) {
        editor_mode = false;
        cars_init(false);
      }
      if (IsMouseButtonDown(MOUSE_BUTTON_LEFT)) {
        DrawLineV(fp, GetMousePosition(), RED);
      }
    } else {
      if (dead_cars == POPUL) {
        best_cp = 0;
        cars_init(true);
        epoch++;
      }

      cars_update();
    }

    DrawRectangleLines(start_pos.x - CAR_W / 2.0, start_pos.y - CAR_W / 2.0,
                       CAR_W, CAR_W, BLUE);

    char buf[128];
    sprintf(buf, "Population: %d/%d\nEpoch: %d",
            POPUL - dead_cars, POPUL, epoch);
    DrawText(buf, 20, 20, 20, (Color){255, 255, 255, 170});
    DrawText("LMB - create a wall\nRMB - remove last wall\nENTER - start "
             "training\nSPACE - set start position\n"
             "Z - place a checkpoint\nX - remove last checkpoint",
             500, 20, 20, (Color){255, 255, 255, 170});
    EndDrawing();
  }

  for (int i = 0; i < POPUL; i++) {
    nn_free(cars[i].nn);
  }
  nn_free(best_nn);

  CloseWindow();
  return 0;
}
