#####################################################################################
#      Copyright 2009-2018 Barcelona Supercomputing Center                          #
#                                                                                   #
#      This file is part of the NANOS++ library.                                    #
#                                                                                   #
#      NANOS++ is free software: you can redistribute it and/or modify              #
#      it under the terms of the GNU Lesser General Public License as published by  #
#      the Free Software Foundation, either version 3 of the License, or            #
#      (at your option) any later version.                                          #
#                                                                                   #
#      NANOS++ is distributed in the hope that it will be useful,                   #
#      but WITHOUT ANY WARRANTY; without even the implied warranty of               #
#      MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the                #
#      GNU Lesser General Public License for more details.                          #
#                                                                                   #
#      You should have received a copy of the GNU Lesser General Public License     #
#      along with NANOS++.  If not, see <https://www.gnu.org/licenses/>.            #
#####################################################################################

include $(top_srcdir)/src/common.am

EXTRA_DIST= \
	$(END)

pkginclude_HEADERS=\
	nanos-gpu.h \
	$(END)

devinclude_HEADERS = \
	gpuconfig.hpp \
	gpudevice_decl.hpp \
	gpudevice.hpp \
	gpudd.hpp \
	gpumemorytransfer_decl.hpp \
	gpumemorytransfer.hpp \
	gpuprocessor_decl.hpp \
	gpuprocessor.hpp \
	gputhread_decl.hpp \
	gputhread.hpp \
	gpuutils.hpp \
	gpuprocessor_fwd.hpp \
	$(END) 

gpu_api_sources = \
	nanos-gpu.h \
	gpu-api.cpp \
	$(END)
		
gpu_sources = \
	$(top_srcdir)/src/arch/common/device_instrumentation.hpp \
	gpucallback.cpp \
	gpucallback.hpp \
	gpuconfig.cpp \
	gpuconfig.hpp \
	gpudevice_decl.hpp \
	gpudevice.hpp \
	gpudevice.cpp \
	gpudd.hpp \
	gpudd.cpp \
	gpuevent_decl.hpp \
	gpuevent.hpp \
	gpumemorytransfer_decl.hpp \
	gpumemorytransfer.hpp \
	gpumemorytransfer.cpp \
	gpuprocessor_fwd.hpp \
	gpuprocessor_decl.hpp \
	gpuprocessor.hpp \
	gpuprocessor.cpp \
	gputhread_decl.hpp \
	gputhread.hpp \
	gputhread.cpp \
	gpuutils.hpp \
	gpuutils.cpp \
	gpumemoryspace_decl.hpp \
	gpumemoryspace_fwd.hpp \
	gpumemoryspace.cpp \
	$(END)

pe_gpu_sources = \
	gpuplugin.cpp \
	$(END)

gpu_cublas_sources = \
	gpucublasplugin.cpp \
	$(END)

gpu_cusparse_sources = \
	gpucusparseplugin.cpp \
	$(END)
	
noinst_LTLIBRARIES =
debug_LTLIBRARIES =
instrumentation_LTLIBRARIES =
instrumentation_debug_LTLIBRARIES =
performance_LTLIBRARIES =

if is_debug_enabled
noinst_LTLIBRARIES += debug/libgpu.la
debug_LTLIBRARIES += debug/libnanox-gpu-api.la debug/libnanox-pe-gpu.la debug/libnanox-gpu-cublas.la debug/libnanox-gpu-cusparse.la

debug_libgpu_la_CPPFLAGS=$(common_debug_CPPFLAGS) @cudainc@
debug_libgpu_la_CXXFLAGS=$(common_debug_CXXFLAGS)
debug_libgpu_la_LDFLAGS=$(AM_LDFLAGS) @cudalib@
debug_libgpu_la_SOURCES=$(gpu_sources)
debug_libgpu_la_LIBADD=@cudalibs@

debug_libnanox_gpu_api_la_CPPFLAGS=$(common_debug_CPPFLAGS) $(api_includes) @cudainc@
debug_libnanox_gpu_api_la_CXXFLAGS=$(common_debug_CXXFLAGS)
debug_libnanox_gpu_api_la_LDFLAGS=$(AM_LDFLAGS) $(version_flags) @cudalib@
debug_libnanox_gpu_api_la_SOURCES=$(gpu_api_sources)
debug_libnanox_gpu_api_la_LIBADD=@cudalibs@

debug_libnanox_pe_gpu_la_CPPFLAGS=$(common_debug_CPPFLAGS) @cudainc@
debug_libnanox_pe_gpu_la_CXXFLAGS=$(common_debug_CXXFLAGS)
debug_libnanox_pe_gpu_la_LDFLAGS=$(AM_LDFLAGS) $(ld_plugin_flags) @cudalib@
debug_libnanox_pe_gpu_la_SOURCES=$(pe_gpu_sources)
debug_libnanox_pe_gpu_la_LIBADD=@cudalibs@

debug_libnanox_gpu_cublas_la_CPPFLAGS=$(common_debug_CPPFLAGS) @cudainc@
debug_libnanox_gpu_cublas_la_CXXFLAGS=$(common_debug_CXXFLAGS)
debug_libnanox_gpu_cublas_la_LDFLAGS=$(AM_LDFLAGS) $(ld_plugin_flags) @cudalib@
debug_libnanox_gpu_cublas_la_SOURCES=$(gpu_cublas_sources)
debug_libnanox_gpu_cublas_la_LIBADD=@cudalibs@

debug_libnanox_gpu_cusparse_la_CPPFLAGS=$(common_debug_CPPFLAGS) @cudainc@
debug_libnanox_gpu_cusparse_la_CXXFLAGS=$(common_debug_CXXFLAGS)
debug_libnanox_gpu_cusparse_la_LDFLAGS=$(AM_LDFLAGS) $(ld_plugin_flags) @cudalib@
debug_libnanox_gpu_cusparse_la_SOURCES=$(gpu_cusparse_sources)
debug_libnanox_gpu_cusparse_la_LIBADD=@cudalibs@
endif

if is_instrumentation_debug_enabled
noinst_LTLIBRARIES += instrumentation-debug/libgpu.la
instrumentation_debug_LTLIBRARIES += instrumentation-debug/libnanox-gpu-api.la instrumentation-debug/libnanox-pe-gpu.la instrumentation-debug/libnanox-gpu-cublas.la instrumentation-debug/libnanox-gpu-cusparse.la

instrumentation_debug_libgpu_la_CPPFLAGS=$(common_instrumentation_debug_CPPFLAGS) @cudainc@
instrumentation_debug_libgpu_la_CXXFLAGS=$(common_instrumentation_debug_CXXFLAGS)
instrumentation_debug_libgpu_la_LDFLAGS=$(AM_LDFLAGS) @cudalib@
instrumentation_debug_libgpu_la_SOURCES=$(gpu_sources)
instrumentation_debug_libgpu_la_LIBADD=@cudalibs@

instrumentation_debug_libnanox_gpu_api_la_CPPFLAGS=$(common_instrumentation_debug_CPPFLAGS) $(api_includes) @cudainc@
instrumentation_debug_libnanox_gpu_api_la_CXXFLAGS=$(common_instrumentation_debug_CXXFLAGS)
instrumentation_debug_libnanox_gpu_api_la_LDFLAGS=$(AM_LDFLAGS) $(version_flags) @cudalib@
instrumentation_debug_libnanox_gpu_api_la_SOURCES=$(gpu_api_sources)
instrumentation_debug_libnanox_gpu_api_la_LIBADD=@cudalibs@

instrumentation_debug_libnanox_pe_gpu_la_CPPFLAGS=$(common_instrumentation_debug_CPPFLAGS) @cudainc@
instrumentation_debug_libnanox_pe_gpu_la_CXXFLAGS=$(common_instrumentation_debug_CXXFLAGS)
instrumentation_debug_libnanox_pe_gpu_la_LDFLAGS=$(AM_LDFLAGS) $(ld_plugin_flags) @cudalib@
instrumentation_debug_libnanox_pe_gpu_la_SOURCES=$(pe_gpu_sources)
instrumentation_debug_libnanox_pe_gpu_la_LIBADD=@cudalibs@

instrumentation_debug_libnanox_gpu_cublas_la_CPPFLAGS=$(common_instrumentation_debug_CPPFLAGS) @cudainc@
instrumentation_debug_libnanox_gpu_cublas_la_CXXFLAGS=$(common_instrumentation_debug_CXXFLAGS)
instrumentation_debug_libnanox_gpu_cublas_la_LDFLAGS=$(AM_LDFLAGS) $(ld_plugin_flags) @cudalib@
instrumentation_debug_libnanox_gpu_cublas_la_SOURCES=$(gpu_cublas_sources)
instrumentation_debug_libnanox_gpu_cublas_la_LIBADD=@cudalibs@

instrumentation_debug_libnanox_gpu_cusparse_la_CPPFLAGS=$(common_instrumentation_debug_CPPFLAGS) @cudainc@
instrumentation_debug_libnanox_gpu_cusparse_la_CXXFLAGS=$(common_instrumentation_debug_CXXFLAGS)
instrumentation_debug_libnanox_gpu_cusparse_la_LDFLAGS=$(AM_LDFLAGS) $(ld_plugin_flags) @cudalib@
instrumentation_debug_libnanox_gpu_cusparse_la_SOURCES=$(gpu_cusparse_sources)
instrumentation_debug_libnanox_gpu_cusparse_la_LIBADD=@cudalibs@
endif

if is_instrumentation_enabled
noinst_LTLIBRARIES += instrumentation/libgpu.la
instrumentation_LTLIBRARIES += instrumentation/libnanox-gpu-api.la instrumentation/libnanox-pe-gpu.la instrumentation/libnanox-gpu-cublas.la instrumentation/libnanox-gpu-cusparse.la

instrumentation_libgpu_la_CPPFLAGS=$(common_instrumentation_CPPFLAGS) @cudainc@
instrumentation_libgpu_la_CXXFLAGS=$(common_instrumentation_CXXFLAGS)
instrumentation_libgpu_la_LDFLAGS=$(AM_LDFLAGS) @cudalib@
instrumentation_libgpu_la_SOURCES=$(gpu_sources)
instrumentation_libgpu_la_LIBADD=@cudalibs@

instrumentation_libnanox_gpu_api_la_CPPFLAGS=$(common_instrumentation_CPPFLAGS) $(api_includes) @cudainc@
instrumentation_libnanox_gpu_api_la_CXXFLAGS=$(common_instrumentation_CXXFLAGS)
instrumentation_libnanox_gpu_api_la_LDFLAGS=$(AM_LDFLAGS) $(version_flags) @cudalib@
instrumentation_libnanox_gpu_api_la_SOURCES=$(gpu_api_sources)
instrumentation_libnanox_gpu_api_la_LIBADD=@cudalibs@

instrumentation_libnanox_pe_gpu_la_CPPFLAGS=$(common_instrumentation_CPPFLAGS) @cudainc@
instrumentation_libnanox_pe_gpu_la_CXXFLAGS=$(common_instrumentation_CXXFLAGS)
instrumentation_libnanox_pe_gpu_la_LDFLAGS=$(AM_LDFLAGS) $(ld_plugin_flags) @cudalib@
instrumentation_libnanox_pe_gpu_la_SOURCES=$(pe_gpu_sources)
instrumentation_libnanox_pe_gpu_la_LIBADD=@cudalibs@

instrumentation_libnanox_gpu_cublas_la_CPPFLAGS=$(common_instrumentation_CPPFLAGS) @cudainc@
instrumentation_libnanox_gpu_cublas_la_CXXFLAGS=$(common_instrumentation_CXXFLAGS)
instrumentation_libnanox_gpu_cublas_la_LDFLAGS=$(AM_LDFLAGS) $(ld_plugin_flags) @cudalib@
instrumentation_libnanox_gpu_cublas_la_SOURCES=$(gpu_cublas_sources)
instrumentation_libnanox_gpu_cublas_la_LIBADD=@cudalibs@

instrumentation_libnanox_gpu_cusparse_la_CPPFLAGS=$(common_instrumentation_CPPFLAGS) @cudainc@
instrumentation_libnanox_gpu_cusparse_la_CXXFLAGS=$(common_instrumentation_CXXFLAGS)
instrumentation_libnanox_gpu_cusparse_la_LDFLAGS=$(AM_LDFLAGS) $(ld_plugin_flags) @cudalib@
instrumentation_libnanox_gpu_cusparse_la_SOURCES=$(gpu_cusparse_sources)
instrumentation_libnanox_gpu_cusparse_la_LIBADD=@cudalibs@
endif

if is_performance_enabled
noinst_LTLIBRARIES += performance/libgpu.la
performance_LTLIBRARIES += performance/libnanox-gpu-api.la performance/libnanox-pe-gpu.la performance/libnanox-gpu-cublas.la performance/libnanox-gpu-cusparse.la

performance_libgpu_la_CPPFLAGS=$(common_performance_CPPFLAGS) @cudainc@
performance_libgpu_la_CXXFLAGS=$(common_performance_CXXFLAGS)
performance_libgpu_la_LDFLAGS=$(AM_LDFLAGS) @cudalib@
performance_libgpu_la_SOURCES=$(gpu_sources)
performance_libgpu_la_LIBADD=@cudalibs@

performance_libnanox_gpu_api_la_CPPFLAGS=$(common_performance_CPPFLAGS) $(api_includes) @cudainc@
performance_libnanox_gpu_api_la_CXXFLAGS=$(common_performance_CXXFLAGS)
performance_libnanox_gpu_api_la_LDFLAGS=$(AM_LDFLAGS) $(version_flags) @cudalib@
performance_libnanox_gpu_api_la_SOURCES=$(gpu_api_sources)
performance_libnanox_gpu_api_la_LIBADD=@cudalibs@

performance_libnanox_pe_gpu_la_CPPFLAGS=$(common_performance_CPPFLAGS) @cudainc@
performance_libnanox_pe_gpu_la_CXXFLAGS=$(common_performance_CXXFLAGS)
performance_libnanox_pe_gpu_la_LDFLAGS=$(AM_LDFLAGS) $(ld_plugin_flags) @cudalib@
performance_libnanox_pe_gpu_la_SOURCES=$(pe_gpu_sources)
performance_libnanox_pe_gpu_la_LIBADD=@cudalibs@

performance_libnanox_gpu_cublas_la_CPPFLAGS=$(common_performance_CPPFLAGS) @cudainc@
performance_libnanox_gpu_cublas_la_CXXFLAGS=$(common_performance_CXXFLAGS)
performance_libnanox_gpu_cublas_la_LDFLAGS=$(AM_LDFLAGS) $(ld_plugin_flags) @cudalib@
performance_libnanox_gpu_cublas_la_SOURCES=$(gpu_cublas_sources)
performance_libnanox_gpu_cublas_la_LIBADD=@cudalibs@

performance_libnanox_gpu_cusparse_la_CPPFLAGS=$(common_performance_CPPFLAGS) @cudainc@
performance_libnanox_gpu_cusparse_la_CXXFLAGS=$(common_performance_CXXFLAGS)
performance_libnanox_gpu_cusparse_la_LDFLAGS=$(AM_LDFLAGS) $(ld_plugin_flags) @cudalib@
performance_libnanox_gpu_cusparse_la_SOURCES=$(gpu_cusparse_sources)
performance_libnanox_gpu_cusparse_la_LIBADD=@cudalibs@
endif

