#include <stdbool.h>
#include <stdlib.h>
#include <time.h>

#include "../parsers/csv_parser.c"
#define NN_IMPLEMENTATION
#include "../nn.h"

#define INPUTS        4
#define OUTPUTS       3
#define BATCH_SIZE    16
#define LEARNING_RATE 0.0006

Layer nn_arch[] = {
	(Layer){
		.size = INPUTS,
	},
	(Layer){
		.actf  = ACT_RELU,
		.size  = 8,
		.randf = glorot_randf,
	},
	(Layer){
		.actf  = ACT_SIGM,
		.size  = OUTPUTS,
		.randf = glorot_randf,
	},
};

int main(void) {
	srand(time(0));

	Mat mat = parse_csv_to_mat("./dataset/iris.csv", "\t");

	NN nn = nn_alloc(nn_arch, ARR_LEN(nn_arch));
	NN g  = nn_alloc(nn_arch, ARR_LEN(nn_arch));

#ifdef ADAM_OPT
	AdamOptimizer adam = adam_alloc(nn);
#endif

	nn_rand(nn);

	// initializing the data
	Mat nto = mat_submatrix(mat, INPUTS, 0, INPUTS,   mat.rows - 1);
	Mat ti  = mat_submatrix(mat, 0,      0, INPUTS-1, mat.rows - 1);;
	Mat to  = mat_alloc(mat.rows, OUTPUTS);

	// preparing the data
	mat_zero(to);
	for (size_t i = 0; i < mat.rows; i++) {
		MAT_AT(to, i, (int) MAT_AT(nto, i, 0)) = 1.0;
	}

	for (size_t i = 0; true; i++) {
		size_t pos = (rand()) % (ti.rows - BATCH_SIZE);
		Mat gti = mat_submatrix(ti, 0, pos, ti.cols - 1, pos + BATCH_SIZE);
		Mat gto = mat_submatrix(to, 0, pos, to.cols - 1, pos + BATCH_SIZE);

		nn_backprop(nn, g, gti, gto);
#ifdef ADAM_OPT
		adam_update(nn, &adam, g, LEARNING_RATE, 0.9, 0.999, 1e-8);
#else
		nn_learn(nn, g, LEARNING_RATE);
#endif

		if (i % 5000 == 0) {
			float tc = nn_cost(nn, ti, to);
			printf("cost %zu - %f\n", i, tc);
			if (tc < 0.1) break;
		}
	}

	printf("NN was trained successfully\n");

	char buf[256];
	while (fgets(buf, sizeof buf, stdin)) {
		if (sscanf(buf, "%f %f %f %f",
			 &MAT_AT(NN_INPUT(nn), 0, 0),
			 &MAT_AT(NN_INPUT(nn), 0, 1),
			 &MAT_AT(NN_INPUT(nn), 0, 2),
			 &MAT_AT(NN_INPUT(nn), 0, 3)) != 4)
		{
			puts("Invalid input");
			continue;
		}

		nn_forward(nn);

		float p0 = MAT_AT(NN_OUTPUT(nn), 0, 0);
		float p1 = MAT_AT(NN_OUTPUT(nn), 0, 1);
		float p2 = MAT_AT(NN_OUTPUT(nn), 0, 2);

		printf("probabilities: %f %f %f\n", p0, p1, p2);
		printf("prediction: ");
		if (p0 > p1 && p0 > p2) {
			printf("iris-setosa\n");
		} else if (p1 > p0 && p1 > p2) {
			printf("iris-versicolor\n");
		} else if (p2 > p0 && p2 > p1) {
			printf("iris-verginica\n");
		}
	}

	return 0;
}
