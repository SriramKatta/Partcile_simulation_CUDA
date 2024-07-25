add_executable(juliasetgpu juliaser-gpu.cu)
target_link_libraries(juliasetgpu PRIVATE lodepng)