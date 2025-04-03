#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>

#define CUBE_SIZE 20  // Define the dimensions of the cube (NxNxN)

// Structure for a voxel stored in the cube.
// A colorIndex of -1 indicates an empty voxel.
typedef struct {
    int colorIndex;
} Voxel;

// Temporary structure to read voxel data from the file.
typedef struct {
    uint8_t x, y, z, colorIndex;
} FileVoxel;

// Structure for chunk header.
typedef struct {
    char id[5]; // 4-character ID plus null terminator.
    uint32_t contentSize;
    uint32_t childrenSize;
} ChunkHeader;

// Structure for the SIZE chunk data.
typedef struct {
    int32_t sizeX, sizeY, sizeZ;
} VoxSize;

// Helper function to read a chunk header from file.
static void read_chunk_header(FILE *fp, ChunkHeader *header) {
    fread(header->id, 1, 4, fp);
    header->id[4] = '\0';
    fread(&header->contentSize, sizeof(uint32_t), 1, fp);
    fread(&header->childrenSize, sizeof(uint32_t), 1, fp);
}

// Helper function warnwarnto compute a 1D index for a 3D grid.
static size_t index3D(int x, int y, int z) {
    return (size_t)x + (size_t)y * CUBE_SIZE + (size_t)z * CUBE_SIZE * CUBE_SIZE;
}

// Function that loads a .vox file and returns a pointer to an NxNxN grid of voxels.
// The model contained in the .vox file is centered in the cube.
// Returns: pointer to the grid (allocated dynamically) or NULL on error.
Voxel* load_vox_model(const char *filepath) {
    FILE *fp = fopen(filepath, "rb");
    if (!fp) {
        perror("Error opening file");
        return NULL;
    }

    // Verify header: first 4 bytes should be "VOX "
    char magic[5];
    if (fread(magic, 1, 4, fp) != 4) {
        fprintf(stderr, "Error reading file header.\n");
        fclose(fp);
        return NULL;
    }
    magic[4] = '\0';
    if (strcmp(magic, "VOX ") != 0) {
        fprintf(stderr, "Not a valid VOX file.\n");
        fclose(fp);
        return NULL;
    }

    // Read version number.
    uint32_t version;
    fread(&version, sizeof(uint32_t), 1, fp);

    // Read MAIN chunk header.
    ChunkHeader mainChunk;
    read_chunk_header(fp, &mainChunk);
    if (strcmp(mainChunk.id, "MAIN") != 0) {
        fprintf(stderr, "Expected MAIN chunk, found %s\n", mainChunk.id);
        fclose(fp);
        return NULL;
    }

    // Allocate and initialize the cube grid.
    Voxel *grid = malloc(CUBE_SIZE * CUBE_SIZE * CUBE_SIZE * sizeof(Voxel));
    if (!grid) {
        fprintf(stderr, "Memory allocation error for grid\n");
        fclose(fp);
        return NULL;
    }
    // Initialize all voxels as empty (colorIndex = -1).
    for (int i = 0; i < CUBE_SIZE * CUBE_SIZE * CUBE_SIZE; i++) {
        grid[i].colorIndex = -1;
    }

    // Variables to store model dimensions.
    VoxSize modelSize = {0, 0, 0};
    int modelLoaded = 0;

    // Process subchunks within the MAIN chunk.
    long mainChunkEnd = ftell(fp) + mainChunk.childrenSize;
    while (ftell(fp) < mainChunkEnd) {
        ChunkHeader chunk;
        read_chunk_header(fp, &chunk);

        long chunkContentPos = ftell(fp);

        if (strcmp(chunk.id, "SIZE") == 0) {
            // Read model dimensions.
            fread(&modelSize.sizeX, sizeof(int32_t), 1, fp);
            fread(&modelSize.sizeY, sizeof(int32_t), 1, fp);
            fread(&modelSize.sizeZ, sizeof(int32_t), 1, fp);
            modelLoaded = 1;
        }
        else if (strcmp(chunk.id, "XYZI") == 0) {
            if (!modelLoaded) {
                fprintf(stderr, "XYZI chunk encountered before SIZE chunk.\n");
                free(grid);
                fclose(fp);
                return NULL;
            }

            // Calculate offsets to center the model in the cube.
            int offsetX = (CUBE_SIZE - modelSize.sizeX) / 2;
            int offsetY = (CUBE_SIZE - modelSize.sizeY) / 2;
            int offsetZ = (CUBE_SIZE - modelSize.sizeZ) / 2;

            // Read the number of voxels.
            uint32_t numVoxels;
            fread(&numVoxels, sizeof(uint32_t), 1, fp);

            // Allocate temporary buffer for voxels from the file.
            FileVoxel *fileVoxels = malloc(numVoxels * sizeof(FileVoxel));
            if (!fileVoxels) {
                fprintf(stderr, "Memory allocation error for voxels\n");
                free(grid);
                fclose(fp);
                return NULL;
            }
            fread(fileVoxels, sizeof(FileVoxel), numVoxels, fp);

            // Place each voxel into the cube grid.
            for (uint32_t i = 0; i < numVoxels; i++) {
                int gridX = fileVoxels[i].x + offsetX;
                int gridY = fileVoxels[i].y + offsetY;
                int gridZ = fileVoxels[i].z + offsetZ;
                // Ensure the voxel falls within bounds.
                if (gridX < 0 || gridX >= CUBE_SIZE ||
                    gridY < 0 || gridY >= CUBE_SIZE ||
                    gridZ < 0 || gridZ >= CUBE_SIZE) {
                    fprintf(stderr, "Voxel %u at (%d, %d, %d) is out of bounds after centering.\n",
                            i, gridX, gridY, gridZ);
                    continue;
                }
                size_t idx = index3D(gridX, gridY, gridZ);
                // Store only the color index.
                grid[idx].colorIndex = fileVoxels[i].colorIndex;
            }
            free(fileVoxels);
        }
        else {
            // Skip unknown chunk content.
            fseek(fp, chunk.contentSize, SEEK_CUR);
        }
        // Skip any children of the chunk.
        if (chunk.childrenSize > 0) {
            fseek(fp, chunk.childrenSize, SEEK_CUR);
        }
        // Ensure file pointer is set to the end of the current chunk.
        long expectedPos = chunkContentPos + chunk.contentSize + chunk.childrenSize;
        fseek(fp, expectedPos, SEEK_SET);
    }

    fclose(fp);
    return grid;
}

