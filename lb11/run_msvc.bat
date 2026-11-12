@echo off
chcp 65001 >nul
cd /d "%~dp0"

where cl >nul 2>nul
if errorlevel 1 (
    echo Run this file from "Developer Command Prompt for Visual Studio".
    pause
    exit /b 1
)

cl /nologo /std:c++17 /EHsc /O2 /W4 optimization.cpp tests.cpp /Fe:lr11_tests.exe || goto error
cl /nologo /std:c++17 /EHsc /O2 /W4 optimization.cpp part1.cpp /Fe:lr11_part1.exe || goto error
cl /nologo /std:c++17 /EHsc /O2 /W4 optimization.cpp part2.cpp /Fe:lr11_part2.exe || goto error

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
