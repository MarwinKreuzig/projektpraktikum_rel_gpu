add_library(relearn_adapter STATIC)
target_sources(
	relearn_adapter
	PRIVATE
			adapter.cpp

			# for visual studio
			${relearn_adapter_additional_files})

target_include_directories(relearn_adapter PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
set_target_properties(relearn_adapter PROPERTIES ENABLE_EXPORTS 1)

target_link_libraries(relearn_adapter PUBLIC project_options project_libraries)
target_link_libraries(relearn_adapter PRIVATE relearn_lib)

get_target_property(relearn_adapter_sources relearn_adapter SOURCES)
source_group(TREE ${CMAKE_SOURCE_DIR} FILES ${relearn_adapter_sources})
