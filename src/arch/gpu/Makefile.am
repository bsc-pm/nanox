#####################################################################################
#      Copyright 2009 Barcelona Supercomputing Center                               #
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
#      along with NANOS++.  If not, see <http://www.gnu.org/licenses/>.             #
#####################################################################################

include $(top_srcdir)/src/common.am

noinst_LTLIBRARIES = libgpu.la
lib_LTLIBRARIES = libnanox-pe-gpu.la

EXTRA_DIST= \
	$(END)


libgpu_la_CXXFLAGS=$(common_includes) $(common_cxxflags) $(ult_flags) -I @CUDA_INC@
libgpu_la_LDFLAGS = -L@CUDA_LIB@ -lcudart
libgpu_la_SOURCES = \
		gpudevice.hpp \
		gpudevice.cpp \
		gpudd.hpp \
		gpudd.cpp \
		gpumemorytransfer.hpp \
		gpumemorytransfer.cpp \
		gpuprocessor_decl.hpp \
		gpuprocessor.hpp \
		gpuprocessor.cpp \
		gputhread.hpp \
		gputhread.cpp \
		gpuutils.hpp \
		gpuutils.cpp \
		$(END)



libnanox_pe_gpu_la_CXXFLAGS=$(common_includes) $(common_cxxflags) $(ult_flags) -I @CUDA_INC@
libnanox_pe_gpu_la_LDFLAGS = $(ld_plugin_flags) -L@CUDA_LIB@ -lcudart
libnanox_pe_gpu_la_SOURCES = \
		gpuplugin.cpp \
		$(END)

