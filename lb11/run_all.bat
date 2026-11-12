@echo off
chcp 65001 >nul
cd /d "%~dp0"

where g++ >nul 2>nul
if errorlevel 1 (
    echo g++ was not found. Install MinGW-w64 or build the files in Visual Studio.
    pause
    exit /b 1
)

g++ -std=c++17 -O2 -Wall -Wextra -pedantic optimization.cpp tests.cpp -o lr11_tests.exe || goto error
g++ -std=c++17 -O2 -Wall -Wextra -pedantic optimization.cpp part1.cpp -o lr11_part1.exe || goto error
g++ -std=c++17 -O2 -Wall -Wextra -pedantic optimization.cpp part2.cpp -o lr11_part2.exe || goto error

echo === TESTS ===
lr11_tests.exe || goto error
echo.
echo === PART 1 ===
lr11_part1.exe
echo.
echo === PART 2 ===
lr11_part2.exe
echo.
echo Everything completed successfully.
pause
exit /b 0

:error
echo Build or tests failed.
pause
exit /b 1
