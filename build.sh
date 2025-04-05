mkdir -p build
cc -g demos/image_learning.c -o build/image_learning -lm -lraylib
cc -g demos/digit_recognition.c -o build/digit_recognition -lm -lraylib
cc -g demos/car_racing.c -o build/car_racing -lm -lraylib
cc -g demos/image_generator.c -o build/image_generator -lm -lraylib
cc -g demos/classifier.c -o build/classifier -lm -lraylib
