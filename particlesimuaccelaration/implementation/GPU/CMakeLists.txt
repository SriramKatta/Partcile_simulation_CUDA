add_executable(MDsimu main.cu)

if(nbdacc)
target_compile_definitions(MDsimu PRIVATE nbdacc)
elseif(cutoff)
target_compile_definitions(MDsimu PRIVATE cutoff)
endif()

if(perfana)
target_compile_definitions(MDsimu PRIVATE perfana)
endif()


target_link_libraries(MDsimu PRIVATE Thrust)