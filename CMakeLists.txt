#----------------------------------------------------------------------------

cmake_minimum_required(VERSION 3.20)
project(main)

#----------------------------------------------------------------------------

find_package(OpenMP)

#----------------------------------------------------------------------------

include_directories(${PROJECT_SOURCE_DIR}/include
		    ${PROJECT_SOURCE_DIR}/shared/include
        ${ROOT_INCLUDE_DIRS})
file(GLOB sources ${PROJECT_SOURCE_DIR}/src/*.cc
		  ${PROJECT_SOURCE_DIR}/shared/src/*.cc)
file(GLOB headers ${PROJECT_SOURCE_DIR}/include/*.hh
		  ${PROJECT_SOURCE_DIR}/shared/include/*.hh)

#----------------------------------------------------------------------------

set(CMAKE_CXX_COMPILER g++)

#----------------------------------------------------------------------------

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -O2 -fPIC -ffast-math -mavx")

#----------------------------------------------------------------------------

add_library(KANplus SHARED ${sources} ${headers})
target_link_libraries(KANplus PUBLIC OpenMP::OpenMP_CXX)

#----------------------------------------------------------------------------

add_executable(main main.cc ${headers})
target_link_libraries(main PUBLIC KANplus OpenMP::OpenMP_CXX)

#----------------------------------------------------------------------------

set(main_SCRIPTS )

foreach(_script ${main_SCRIPTS})
  configure_file(
    ${PROJECT_SOURCE_DIR}/${_script}
    ${PROJECT_BINARY_DIR}/${_script}
    COPYONLY
    )
endforeach()

#----------------------------------------------------------------------------

install(TARGETS main DESTINATION ${PROJECT_SOURCE_DIR}/install)
install(TARGETS KANplus DESTINATION ${PROJECT_SOURCE_DIR}/install)

#----------------------------------------------------------------------------

add_custom_target("uninstall" COMMENT "Uninstall installed files")
add_custom_command(
    TARGET "uninstall"
    POST_BUILD
    COMMENT "Uninstall files with install_manifest.txt"
    COMMAND xargs rm -vf < install_manifest.txt || echo Nothing in
            install_manifest.txt to be uninstalled!
)

#----------------------------------------------------------------------------