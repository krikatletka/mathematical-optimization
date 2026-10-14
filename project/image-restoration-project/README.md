# Image restoration optimization project

C++17 implementation for NTU KhPI, variant 8.

## Build without CMake

```bash
g++ -std=c++17 -O2 -Wall -Wextra -Wpedantic -Isrc main.cpp -o image_restoration
g++ -std=c++17 -O2 -Wall -Wextra -Wpedantic -Isrc tests/project_tests.cpp -o project_tests
./project_tests
```

On Windows with MinGW, run `image_restoration.exe` and `project_tests.exe`.

## Run

```bash
./image_restoration              # preparation only
./image_restoration --part 1     # preparation + Svenn
./image_restoration --part 2     # through dichotomy and golden section
./image_restoration --part 3     # through Nelder-Mead and Powell
./image_restoration --part 4     # through gradient methods
./image_restoration --part 5     # complete project
./image_restoration --all        # same as complete project
```

The program creates `outputs/*.pgm`. PGM files can be opened in GIMP, IrfanView, XnView, or converted with ImageMagick:

```bash
magick outputs/00_clean.pgm outputs/00_clean.png
```

The implementation uses only the C++ standard library. It does not use OpenCV or ready-made optimization solvers.
