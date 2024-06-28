# To enable C++ Torch AlphaZero, you will need to set OPEN_SPIEL_BUILD_WITH_LIBTORCH.
if (OPEN_SPIEL_BUILD_WITH_LIBTORCH)
  add_library (alpha_zero_torch_opp OBJECT
    alpha_zero.h
    alpha_zero.cc
    device_manager.h
    model.h
    model.cc
    vpevaluator.h
    vpevaluator.cc
    vpnet.h
    vpnet.cc
  )
  target_include_directories (alpha_zero_torch_opp PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})

  target_link_libraries (alpha_zero_torch_opp ${TORCH_LIBRARIES})
endif ()
