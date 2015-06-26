#####################################################################################
#      Copyright 2015 Barcelona Supercomputing Center                               #
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

EXTRA_DIST = api-generator.in core-generator.in gpu-generator.in mixed-generator.in nanox-creation-generator.in api-omp-generator.in mcc-openmp-generator.in mcc-ompss-generator.in resiliency-generator.in config.py mpi-offload-generator.in opencl-generator.in

noinst_SCRIPTS = api-generator core-generator gpu-generator mixed-generator nanox-creation-generator api-omp-generator mcc-openmp-generator mcc-ompss-generator resiliency-generator mpi-offload-generator opencl-generator

CLEANFILES = api-generator core-generator gpu-generator mixed-generator nanox-creation-generator api-omp-generator mcc-openmp-generator mcc-ompss-generator resiliency-generator mpi-offload-generator opencl-generator

api-generator: api-generator.in
	rm -f api-generator
	tmp_plugins_perfo=`cat $(abs_top_builddir)/src/plugins-list | grep ^performance\= | cut -d'=' -f2` ;\
	tmp_plugins_debug=`cat $(abs_top_builddir)/src/plugins-list | grep ^debug\= | cut -d'=' -f2` ;\
	tmp_plugins_instr=`cat $(abs_top_builddir)/src/plugins-list | grep ^instrumentation\= | cut -d'=' -f2` ;\
	tmp_plugins_instr_debug=`cat $(abs_top_builddir)/src/plugins-list | grep ^instrumentation_debug\= | cut -d'=' -f2` ;\
	tmp_compile_versions=`cat $(abs_top_builddir)/src/plugins-list | cut -d'=' -f1 | sed -e :a -e '$!N; s/\n/ /; ta'` ;\
	tmp_srcdir=$(abs_top_srcdir) ; \
	tmp_builddir=$(abs_top_builddir) ; \
	tmp_perfo_cflags='"$(perfo_cflags)"' ; \
	tmp_debug_cflags='"$(debug_cflags)"' ; \
	tmp_instr_cflags='"$(instr_cflags)"' ; \
	tmp_instrdebug_cflags='"$(instrdebug_cflags)"' ; \
	tmp_perfo_cxxflags='"$(global_perfo_cxxflags)"' ; \
	tmp_debug_cxxflags='"$(global_debug_cxxflags)"' ; \
	tmp_instr_cxxflags='"$(global_instr_cxxflags)"' ; \
	tmp_instrdebug_cxxflags='"$(global_instrdebug_cxxflags)"' ; \
	tmp_cc='"$(CC)"' ; \
	tmp_cxx='"$(CXX)"' ; \
	sed \
		-e's,@SRCDIR\@,'"$$tmp_srcdir"',g' \
		-e's,@BUILDDIR\@,'"$$tmp_builddir"',g' \
		-e's,@PLUGINS_PERFORMANCE\@,'"$$tmp_plugins_perfo"',g' \
		-e's,@PLUGINS_DEBUG\@,'"$$tmp_plugins_debug"',g' \
		-e's,@PLUGINS_INSTRUMENTATION\@,'"$$tmp_plugins_instr"',g' \
		-e's,@PLUGINS_INSTRUMENTATION_DEBUG\@,'"$$tmp_plugins_instr_debug"',g' \
		-e's,@COMPILE_VERSIONS\@,'"$$tmp_compile_versions"',g' \
		-e's,@PERFO_CFLAGS\@,'"$$tmp_perfo_cflags"',g' \
		-e's,@DEBUG_CFLAGS\@,'"$$tmp_debug_cflags"',g' \
		-e's,@INSTR_CFLAGS\@,'"$$tmp_instr_cflags"',g' \
		-e's,@INSTRDEBUG_CFLAGS\@,'"$$tmp_instrdebug_cflags"',g' \
		-e's,@PERFO_CXXFLAGS\@,'"$$tmp_perfo_cxxflags"',g' \
		-e's,@DEBUG_CXXFLAGS\@,'"$$tmp_debug_cxxflags"',g' \
		-e's,@INSTR_CXXFLAGS\@,'"$$tmp_instr_cxxflags"',g' \
		-e's,@INSTRDEBUG_CXXFLAGS\@,'"$$tmp_instrdebug_cxxflags"',g' \
		-e's,@CONFIG_CC\@,'"$$tmp_cc"',g' \
		-e's,@CONFIG_CXX\@,'"$$tmp_cxx"',g' \
		$(srcdir)/api-generator.in > api-generator.tmp
	mv api-generator.tmp api-generator
	chmod 755 api-generator

core-generator: core-generator.in
	rm -f core-generator
	tmp_plugins_perfo=`cat $(abs_top_builddir)/src/plugins-list | grep ^performance\= | cut -d'=' -f2` ;\
	tmp_plugins_debug=`cat $(abs_top_builddir)/src/plugins-list | grep ^debug\= | cut -d'=' -f2` ;\
	tmp_plugins_instr=`cat $(abs_top_builddir)/src/plugins-list | grep ^instrumentation\= | cut -d'=' -f2` ;\
	tmp_plugins_instr_debug=`cat $(abs_top_builddir)/src/plugins-list | grep ^instrumentation_debug\= | cut -d'=' -f2` ;\
	tmp_compile_versions=`cat $(abs_top_builddir)/src/plugins-list | cut -d'=' -f1 | sed -e :a -e '$!N; s/\n/ /; ta'` ;\
	tmp_srcdir=$(abs_top_srcdir) ; \
	tmp_builddir=$(abs_top_builddir) ; \
	tmp_perfo_cflags='"$(perfo_cflags)"' ; \
	tmp_debug_cflags='"$(debug_cflags)"' ; \
	tmp_instr_cflags='"$(instr_cflags)"' ; \
	tmp_instrdebug_cflags='"$(instrdebug_cflags)"' ; \
	tmp_perfo_cxxflags='"$(global_perfo_cxxflags)"' ; \
	tmp_debug_cxxflags='"$(global_debug_cxxflags)"' ; \
	tmp_instr_cxxflags='"$(global_instr_cxxflags)"' ; \
	tmp_instrdebug_cxxflags='"$(global_instrdebug_cxxflags)"' ; \
	tmp_cc='"$(CC)"' ; \
	tmp_cxx='"$(CXX)"' ; \
	sed \
		-e's,@SRCDIR\@,'"$$tmp_srcdir"',g' \
		-e's,@BUILDDIR\@,'"$$tmp_builddir"',g' \
		-e's,@PLUGINS_PERFORMANCE\@,'"$$tmp_plugins_perfo"',g' \
		-e's,@PLUGINS_DEBUG\@,'"$$tmp_plugins_debug"',g' \
		-e's,@PLUGINS_INSTRUMENTATION\@,'"$$tmp_plugins_instr"',g' \
		-e's,@PLUGINS_INSTRUMENTATION_DEBUG\@,'"$$tmp_plugins_instr_debug"',g' \
		-e's,@COMPILE_VERSIONS\@,'"$$tmp_compile_versions"',g' \
		-e's,@PERFO_CFLAGS\@,'"$$tmp_perfo_cflags"',g' \
		-e's,@DEBUG_CFLAGS\@,'"$$tmp_debug_cflags"',g' \
		-e's,@INSTR_CFLAGS\@,'"$$tmp_instr_cflags"',g' \
		-e's,@INSTRDEBUG_CFLAGS\@,'"$$tmp_instrdebug_cflags"',g' \
		-e's,@PERFO_CXXFLAGS\@,'"$$tmp_perfo_cxxflags"',g' \
		-e's,@DEBUG_CXXFLAGS\@,'"$$tmp_debug_cxxflags"',g' \
		-e's,@INSTR_CXXFLAGS\@,'"$$tmp_instr_cxxflags"',g' \
		-e's,@INSTRDEBUG_CXXFLAGS\@,'"$$tmp_instrdebug_cxxflags"',g' \
		-e's,@CONFIG_CC\@,'"$$tmp_cc"',g' \
		-e's,@CONFIG_CXX\@,'"$$tmp_cxx"',g' \
		$(srcdir)/core-generator.in > core-generator.tmp
	mv core-generator.tmp core-generator
	chmod 755 core-generator

gpu-generator: gpu-generator.in
	rm -f gpu-generator
	tmp_plugins_perfo=`cat $(abs_top_builddir)/src/plugins-list | grep ^performance\= | cut -d'=' -f2` ;\
	tmp_plugins_debug=`cat $(abs_top_builddir)/src/plugins-list | grep ^debug\= | cut -d'=' -f2` ;\
	tmp_plugins_instr=`cat $(abs_top_builddir)/src/plugins-list | grep ^instrumentation\= | cut -d'=' -f2` ;\
	tmp_plugins_instr_debug=`cat $(abs_top_builddir)/src/plugins-list | grep ^instrumentation_debug\= | cut -d'=' -f2` ;\
	tmp_compile_versions=`cat $(abs_top_builddir)/src/plugins-list | cut -d'=' -f1 | sed -e :a -e '$!N; s/\n/ /; ta'` ;\
	tmp_srcdir=$(abs_top_srcdir) ; \
	tmp_builddir=$(abs_top_builddir) ; \
	tmp_perfo_cflags='"$(perfo_cflags)"' ; \
	tmp_debug_cflags='"$(debug_cflags)"' ; \
	tmp_instr_cflags='"$(instr_cflags)"' ; \
	tmp_instrdebug_cflags='"$(instrdebug_cflags)"' ; \
	tmp_perfo_cxxflags='"$(global_perfo_cxxflags)"' ; \
	tmp_debug_cxxflags='"$(global_debug_cxxflags)"' ; \
	tmp_instr_cxxflags='"$(global_instr_cxxflags)"' ; \
	tmp_instrdebug_cxxflags='"$(global_instrdebug_cxxflags)"' ; \
	tmp_gpu_valid='$(gpu_valid)' ; \
	tmp_cuda_dir='$(cuda_dir)' ; \
	tmp_cc='"$(CC)"' ; \
	tmp_cxx='"$(CXX)"' ; \
	sed \
		-e's,@SRCDIR\@,'"$$tmp_srcdir"',g' \
		-e's,@BUILDDIR\@,'"$$tmp_builddir"',g' \
		-e's,@PLUGINS_PERFORMANCE\@,'"$$tmp_plugins_perfo"',g' \
		-e's,@PLUGINS_DEBUG\@,'"$$tmp_plugins_debug"',g' \
		-e's,@PLUGINS_INSTRUMENTATION\@,'"$$tmp_plugins_instr"',g' \
		-e's,@PLUGINS_INSTRUMENTATION_DEBUG\@,'"$$tmp_plugins_instr_debug"',g' \
		-e's,@COMPILE_VERSIONS\@,'"$$tmp_compile_versions"',g' \
		-e's,@PERFO_CFLAGS\@,'"$$tmp_perfo_cflags"',g' \
		-e's,@DEBUG_CFLAGS\@,'"$$tmp_debug_cflags"',g' \
		-e's,@INSTR_CFLAGS\@,'"$$tmp_instr_cflags"',g' \
		-e's,@INSTRDEBUG_CFLAGS\@,'"$$tmp_instrdebug_cflags"',g' \
		-e's,@PERFO_CXXFLAGS\@,'"$$tmp_perfo_cxxflags"',g' \
		-e's,@DEBUG_CXXFLAGS\@,'"$$tmp_debug_cxxflags"',g' \
		-e's,@INSTR_CXXFLAGS\@,'"$$tmp_instr_cxxflags"',g' \
		-e's,@INSTRDEBUG_CXXFLAGS\@,'"$$tmp_instrdebug_cxxflags"',g' \
		-e's,@CONFIG_CC\@,'"$$tmp_cc"',g' \
		-e's,@CONFIG_CXX\@,'"$$tmp_cxx"',g' \
		-e's,@GPU_VALID\@,'"$$tmp_gpu_valid"',g' \
		-e's,@CUDA_DIR\@,'"$$tmp_cuda_dir"',g' \
		$(srcdir)/gpu-generator.in > gpu-generator.tmp
	mv gpu-generator.tmp gpu-generator
	chmod 755 gpu-generator

mixed-generator: mixed-generator.in
	rm -f mixed-generator
	tmp_plugins_perfo=`cat $(abs_top_builddir)/src/plugins-list | grep ^performance\= | cut -d'=' -f2` ;\
	tmp_plugins_debug=`cat $(abs_top_builddir)/src/plugins-list | grep ^debug\= | cut -d'=' -f2` ;\
	tmp_plugins_instr=`cat $(abs_top_builddir)/src/plugins-list | grep ^instrumentation\= | cut -d'=' -f2` ;\
	tmp_plugins_instr_debug=`cat $(abs_top_builddir)/src/plugins-list | grep ^instrumentation_debug\= | cut -d'=' -f2` ;\
	tmp_compile_versions=`cat $(abs_top_builddir)/src/plugins-list | cut -d'=' -f1 | sed -e :a -e '$!N; s/\n/ /; ta'` ;\
	tmp_srcdir=$(abs_top_srcdir) ; \
	tmp_builddir=$(abs_top_builddir) ; \
	tmp_perfo_cflags='"$(perfo_cflags)"' ; \
	tmp_debug_cflags='"$(debug_cflags)"' ; \
	tmp_instr_cflags='"$(instr_cflags)"' ; \
	tmp_instrdebug_cflags='"$(instrdebug_cflags)"' ; \
	tmp_perfo_cxxflags='"$(global_perfo_cxxflags)"' ; \
	tmp_debug_cxxflags='"$(global_debug_cxxflags)"' ; \
	tmp_instr_cxxflags='"$(global_instr_cxxflags)"' ; \
	tmp_instrdebug_cxxflags='"$(global_instrdebug_cxxflags)"' ; \
	tmp_cc='"$(CC)"' ; \
	tmp_cxx='"$(CXX)"' ; \
	sed \
		-e's,@SRCDIR\@,'"$$tmp_srcdir"',g' \
		-e's,@BUILDDIR\@,'"$$tmp_builddir"',g' \
		-e's,@PLUGINS_PERFORMANCE\@,'"$$tmp_plugins_perfo"',g' \
		-e's,@PLUGINS_DEBUG\@,'"$$tmp_plugins_debug"',g' \
		-e's,@PLUGINS_INSTRUMENTATION\@,'"$$tmp_plugins_instr"',g' \
		-e's,@PLUGINS_INSTRUMENTATION_DEBUG\@,'"$$tmp_plugins_instr_debug"',g' \
		-e's,@COMPILE_VERSIONS\@,'"$$tmp_compile_versions"',g' \
		-e's,@PERFO_CFLAGS\@,'"$$tmp_perfo_cflags"',g' \
		-e's,@DEBUG_CFLAGS\@,'"$$tmp_debug_cflags"',g' \
		-e's,@INSTR_CFLAGS\@,'"$$tmp_instr_cflags"',g' \
		-e's,@INSTRDEBUG_CFLAGS\@,'"$$tmp_instrdebug_cflags"',g' \
		-e's,@PERFO_CXXFLAGS\@,'"$$tmp_perfo_cxxflags"',g' \
		-e's,@DEBUG_CXXFLAGS\@,'"$$tmp_debug_cxxflags"',g' \
		-e's,@INSTR_CXXFLAGS\@,'"$$tmp_instr_cxxflags"',g' \
		-e's,@INSTRDEBUG_CXXFLAGS\@,'"$$tmp_instrdebug_cxxflags"',g' \
		-e's,@CONFIG_CC\@,'"$$tmp_cc"',g' \
		-e's,@CONFIG_CXX\@,'"$$tmp_cxx"',g' \
		$(srcdir)/mixed-generator.in > mixed-generator.tmp
	mv mixed-generator.tmp mixed-generator
	chmod 755 mixed-generator

nanox-creation-generator: nanox-creation-generator.in
	rm -f nanox-creation-generator
	tmp_plugins_perfo=`cat $(abs_top_builddir)/src/plugins-list | grep ^performance\= | cut -d'=' -f2` ;\
	tmp_plugins_debug=`cat $(abs_top_builddir)/src/plugins-list | grep ^debug\= | cut -d'=' -f2` ;\
	tmp_plugins_instr=`cat $(abs_top_builddir)/src/plugins-list | grep ^instrumentation\= | cut -d'=' -f2` ;\
	tmp_plugins_instr_debug=`cat $(abs_top_builddir)/src/plugins-list | grep ^instrumentation_debug\= | cut -d'=' -f2` ;\
	tmp_compile_versions=`cat $(abs_top_builddir)/src/plugins-list | cut -d'=' -f1 | sed -e :a -e '$!N; s/\n/ /; ta'` ;\
	tmp_srcdir=$(abs_top_srcdir) ; \
	tmp_builddir=$(abs_top_builddir) ; \
	tmp_perfo_cflags='"$(perfo_cflags)"' ; \
	tmp_debug_cflags='"$(debug_cflags)"' ; \
	tmp_instr_cflags='"$(instr_cflags)"' ; \
	tmp_instrdebug_cflags='"$(instrdebug_cflags)"' ; \
	tmp_perfo_cxxflags='"$(global_perfo_cxxflags)"' ; \
	tmp_debug_cxxflags='"$(global_debug_cxxflags)"' ; \
	tmp_instr_cxxflags='"$(global_instr_cxxflags)"' ; \
	tmp_instrdebug_cxxflags='"$(global_instrdebug_cxxflags)"' ; \
	tmp_cc='"$(CC)"' ; \
	tmp_cxx='"$(CXX)"' ; \
	sed \
		-e's,@SRCDIR\@,'"$$tmp_srcdir"',g' \
		-e's,@BUILDDIR\@,'"$$tmp_builddir"',g' \
		-e's,@PLUGINS_PERFORMANCE\@,'"$$tmp_plugins_perfo"',g' \
		-e's,@PLUGINS_DEBUG\@,'"$$tmp_plugins_debug"',g' \
		-e's,@PLUGINS_INSTRUMENTATION\@,'"$$tmp_plugins_instr"',g' \
		-e's,@PLUGINS_INSTRUMENTATION_DEBUG\@,'"$$tmp_plugins_instr_debug"',g' \
		-e's,@COMPILE_VERSIONS\@,'"$$tmp_compile_versions"',g' \
		-e's,@PERFO_CFLAGS\@,'"$$tmp_perfo_cflags"',g' \
		-e's,@DEBUG_CFLAGS\@,'"$$tmp_debug_cflags"',g' \
		-e's,@INSTR_CFLAGS\@,'"$$tmp_instr_cflags"',g' \
		-e's,@INSTRDEBUG_CFLAGS\@,'"$$tmp_instrdebug_cflags"',g' \
		-e's,@PERFO_CXXFLAGS\@,'"$$tmp_perfo_cxxflags"',g' \
		-e's,@DEBUG_CXXFLAGS\@,'"$$tmp_debug_cxxflags"',g' \
		-e's,@INSTR_CXXFLAGS\@,'"$$tmp_instr_cxxflags"',g' \
		-e's,@INSTRDEBUG_CXXFLAGS\@,'"$$tmp_instrdebug_cxxflags"',g' \
		-e's,@CONFIG_CC\@,'"$$tmp_cc"',g' \
		-e's,@CONFIG_CXX\@,'"$$tmp_cxx"',g' \
		$(srcdir)/nanox-creation-generator.in > nanox-creation-generator.tmp
	mv nanox-creation-generator.tmp nanox-creation-generator
	chmod 755 nanox-creation-generator

api-omp-generator: api-omp-generator.in
	rm -f api-omp-generator
	tmp_plugins_perfo=`cat $(abs_top_builddir)/src/plugins-list | grep ^performance\= | cut -d'=' -f2` ;\
	tmp_plugins_debug=`cat $(abs_top_builddir)/src/plugins-list | grep ^debug\= | cut -d'=' -f2` ;\
	tmp_plugins_instr=`cat $(abs_top_builddir)/src/plugins-list | grep ^instrumentation\= | cut -d'=' -f2` ;\
	tmp_plugins_instr_debug=`cat $(abs_top_builddir)/src/plugins-list | grep ^instrumentation_debug\= | cut -d'=' -f2` ;\
	tmp_compile_versions=`cat $(abs_top_builddir)/src/plugins-list | cut -d'=' -f1 | sed -e :a -e '$!N; s/\n/ /; ta'` ;\
	tmp_srcdir=$(abs_top_srcdir) ; \
	tmp_builddir=$(abs_top_builddir) ; \
	tmp_perfo_cflags='"$(perfo_cflags)"' ; \
	tmp_debug_cflags='"$(debug_cflags)"' ; \
	tmp_instr_cflags='"$(instr_cflags)"' ; \
	tmp_instrdebug_cflags='"$(instrdebug_cflags)"' ; \
	tmp_perfo_cxxflags='"$(global_perfo_cxxflags)"' ; \
	tmp_debug_cxxflags='"$(global_debug_cxxflags)"' ; \
	tmp_instr_cxxflags='"$(global_instr_cxxflags)"' ; \
	tmp_instrdebug_cxxflags='"$(global_instrdebug_cxxflags)"' ; \
	tmp_cc='"$(CC)"' ; \
	tmp_cxx='"$(CXX)"' ; \
	sed \
		-e's,@SRCDIR\@,'"$$tmp_srcdir"',g' \
		-e's,@BUILDDIR\@,'"$$tmp_builddir"',g' \
		-e's,@PLUGINS_PERFORMANCE\@,'"$$tmp_plugins_perfo"',g' \
		-e's,@PLUGINS_DEBUG\@,'"$$tmp_plugins_debug"',g' \
		-e's,@PLUGINS_INSTRUMENTATION\@,'"$$tmp_plugins_instr"',g' \
		-e's,@PLUGINS_INSTRUMENTATION_DEBUG\@,'"$$tmp_plugins_instr_debug"',g' \
		-e's,@COMPILE_VERSIONS\@,'"$$tmp_compile_versions"',g' \
		-e's,@PERFO_CFLAGS\@,'"$$tmp_perfo_cflags"',g' \
		-e's,@DEBUG_CFLAGS\@,'"$$tmp_debug_cflags"',g' \
		-e's,@INSTR_CFLAGS\@,'"$$tmp_instr_cflags"',g' \
		-e's,@INSTRDEBUG_CFLAGS\@,'"$$tmp_instrdebug_cflags"',g' \
		-e's,@PERFO_CXXFLAGS\@,'"$$tmp_perfo_cxxflags"',g' \
		-e's,@DEBUG_CXXFLAGS\@,'"$$tmp_debug_cxxflags"',g' \
		-e's,@INSTR_CXXFLAGS\@,'"$$tmp_instr_cxxflags"',g' \
		-e's,@INSTRDEBUG_CXXFLAGS\@,'"$$tmp_instrdebug_cxxflags"',g' \
		-e's,@CONFIG_CC\@,'"$$tmp_cc"',g' \
		-e's,@CONFIG_CXX\@,'"$$tmp_cxx"',g' \
		$(srcdir)/api-omp-generator.in > api-omp-generator.tmp
	mv api-omp-generator.tmp api-omp-generator
	chmod 755 api-omp-generator

mcc-openmp-generator: mcc-openmp-generator.in
	rm -f mcc-openmp-generator
	tmp_plugins_perfo=`cat $(abs_top_builddir)/src/plugins-list | grep ^performance\= | cut -d'=' -f2` ;\
	tmp_plugins_debug=`cat $(abs_top_builddir)/src/plugins-list | grep ^debug\= | cut -d'=' -f2` ;\
	tmp_plugins_instr=`cat $(abs_top_builddir)/src/plugins-list | grep ^instrumentation\= | cut -d'=' -f2` ;\
	tmp_plugins_instr_debug=`cat $(abs_top_builddir)/src/plugins-list | grep ^instrumentation_debug\= | cut -d'=' -f2` ;\
	tmp_compile_versions=`cat $(abs_top_builddir)/src/plugins-list | cut -d'=' -f1 | sed -e :a -e '$!N; s/\n/ /; ta'` ;\
	tmp_srcdir=$(abs_top_srcdir) ; \
	tmp_builddir=$(abs_top_builddir) ; \
	tmp_perfo_cflags='"$(perfo_cflags)"' ; \
	tmp_debug_cflags='"$(debug_cflags)"' ; \
	tmp_instr_cflags='"$(instr_cflags)"' ; \
	tmp_instrdebug_cflags='"$(instrdebug_cflags)"' ; \
	tmp_perfo_cxxflags='"$(global_perfo_cxxflags)"' ; \
	tmp_debug_cxxflags='"$(global_debug_cxxflags)"' ; \
	tmp_instr_cxxflags='"$(global_instr_cxxflags)"' ; \
	tmp_instrdebug_cxxflags='"$(global_instrdebug_cxxflags)"' ; \
	tmp_cc='"$(CC)"' ; \
	tmp_cxx='"$(CXX)"' ; \
	tmp_mcc_valid='"$(mcc_valid)"' ; \
	tmp_mcc='"$(MCC)"' ; \
	tmp_ldflags='"$(LDFLAGS)"' ; \
	sed \
		-e's,@SRCDIR\@,'"$$tmp_srcdir"',g' \
		-e's,@BUILDDIR\@,'"$$tmp_builddir"',g' \
		-e's,@PLUGINS_PERFORMANCE\@,'"$$tmp_plugins_perfo"',g' \
		-e's,@PLUGINS_DEBUG\@,'"$$tmp_plugins_debug"',g' \
		-e's,@PLUGINS_INSTRUMENTATION\@,'"$$tmp_plugins_instr"',g' \
		-e's,@PLUGINS_INSTRUMENTATION_DEBUG\@,'"$$tmp_plugins_instr_debug"',g' \
		-e's,@COMPILE_VERSIONS\@,'"$$tmp_compile_versions"',g' \
		-e's,@PERFO_CFLAGS\@,'"$$tmp_perfo_cflags"',g' \
		-e's,@DEBUG_CFLAGS\@,'"$$tmp_debug_cflags"',g' \
		-e's,@INSTR_CFLAGS\@,'"$$tmp_instr_cflags"',g' \
		-e's,@INSTRDEBUG_CFLAGS\@,'"$$tmp_instrdebug_cflags"',g' \
		-e's,@PERFO_CXXFLAGS\@,'"$$tmp_perfo_cxxflags"',g' \
		-e's,@DEBUG_CXXFLAGS\@,'"$$tmp_debug_cxxflags"',g' \
		-e's,@INSTR_CXXFLAGS\@,'"$$tmp_instr_cxxflags"',g' \
		-e's,@INSTRDEBUG_CXXFLAGS\@,'"$$tmp_instrdebug_cxxflags"',g' \
		-e's,@CONFIG_CC\@,'"$$tmp_cc"',g' \
		-e's,@CONFIG_CXX\@,'"$$tmp_cxx"',g' \
		-e's,@MCC_SUPPORT\@,'"$$tmp_mcc_valid"',g' \
		-e's,@MCC\@,'"$$tmp_mcc"',g' \
		-e's,@LDFLAGS\@,"$$tmp_ldflags",g' \
		$(srcdir)/mcc-openmp-generator.in > mcc-openmp-generator.tmp
	mv mcc-openmp-generator.tmp mcc-openmp-generator
	chmod 755 mcc-openmp-generator

mcc-ompss-generator: mcc-ompss-generator.in
	rm -f mcc-ompss-generator
	tmp_plugins_perfo=`cat $(abs_top_builddir)/src/plugins-list | grep ^performance\= | cut -d'=' -f2` ;\
	tmp_plugins_debug=`cat $(abs_top_builddir)/src/plugins-list | grep ^debug\= | cut -d'=' -f2` ;\
	tmp_plugins_instr=`cat $(abs_top_builddir)/src/plugins-list | grep ^instrumentation\= | cut -d'=' -f2` ;\
	tmp_plugins_instr_debug=`cat $(abs_top_builddir)/src/plugins-list | grep ^instrumentation_debug\= | cut -d'=' -f2` ;\
	tmp_compile_versions=`cat $(abs_top_builddir)/src/plugins-list | cut -d'=' -f1 | sed -e :a -e '$!N; s/\n/ /; ta'` ;\
	tmp_srcdir=$(abs_top_srcdir) ; \
	tmp_builddir=$(abs_top_builddir) ; \
	tmp_perfo_cflags='"$(perfo_cflags)"' ; \
	tmp_debug_cflags='"$(debug_cflags)"' ; \
	tmp_instr_cflags='"$(instr_cflags)"' ; \
	tmp_instrdebug_cflags='"$(instrdebug_cflags)"' ; \
	tmp_perfo_cxxflags='"$(global_perfo_cxxflags)"' ; \
	tmp_debug_cxxflags='"$(global_debug_cxxflags)"' ; \
	tmp_instr_cxxflags='"$(global_instr_cxxflags)"' ; \
	tmp_instrdebug_cxxflags='"$(global_instrdebug_cxxflags)"' ; \
	tmp_cc='"$(CC)"' ; \
	tmp_cxx='"$(CXX)"' ; \
	tmp_mcc_valid='"$(mcc_valid)"' ; \
	tmp_mcc='"$(MCC)"' ; \
	tmp_ldflags='"$(LDFLAGS)"' ; \
	sed \
		-e's,@SRCDIR\@,'"$$tmp_srcdir"',g' \
		-e's,@BUILDDIR\@,'"$$tmp_builddir"',g' \
		-e's,@PLUGINS_PERFORMANCE\@,'"$$tmp_plugins_perfo"',g' \
		-e's,@PLUGINS_DEBUG\@,'"$$tmp_plugins_debug"',g' \
		-e's,@PLUGINS_INSTRUMENTATION\@,'"$$tmp_plugins_instr"',g' \
		-e's,@PLUGINS_INSTRUMENTATION_DEBUG\@,'"$$tmp_plugins_instr_debug"',g' \
		-e's,@COMPILE_VERSIONS\@,'"$$tmp_compile_versions"',g' \
		-e's,@PERFO_CFLAGS\@,'"$$tmp_perfo_cflags"',g' \
		-e's,@DEBUG_CFLAGS\@,'"$$tmp_debug_cflags"',g' \
		-e's,@INSTR_CFLAGS\@,'"$$tmp_instr_cflags"',g' \
		-e's,@INSTRDEBUG_CFLAGS\@,'"$$tmp_instrdebug_cflags"',g' \
		-e's,@PERFO_CXXFLAGS\@,'"$$tmp_perfo_cxxflags"',g' \
		-e's,@DEBUG_CXXFLAGS\@,'"$$tmp_debug_cxxflags"',g' \
		-e's,@INSTR_CXXFLAGS\@,'"$$tmp_instr_cxxflags"',g' \
		-e's,@INSTRDEBUG_CXXFLAGS\@,'"$$tmp_instrdebug_cxxflags"',g' \
		-e's,@CONFIG_CC\@,'"$$tmp_cc"',g' \
		-e's,@CONFIG_CXX\@,'"$$tmp_cxx"',g' \
		-e's,@MCC_SUPPORT\@,'"$$tmp_mcc_valid"',g' \
		-e's,@MCC\@,'"$$tmp_mcc"',g' \
		-e's,@LDFLAGS\@,"$$tmp_ldflags",g' \
		$(srcdir)/mcc-ompss-generator.in > mcc-ompss-generator.tmp
	mv mcc-ompss-generator.tmp mcc-ompss-generator
	chmod 755 mcc-ompss-generator

resiliency-generator: resiliency-generator.in
	rm -f resiliency-generator
	tmp_plugins_perfo=`cat $(abs_top_builddir)/src/plugins-list | grep ^performance\= | cut -d'=' -f2` ;\
	tmp_plugins_debug=`cat $(abs_top_builddir)/src/plugins-list | grep ^debug\= | cut -d'=' -f2` ;\
	tmp_plugins_instr=`cat $(abs_top_builddir)/src/plugins-list | grep ^instrumentation\= | cut -d'=' -f2` ;\
	tmp_plugins_instr_debug=`cat $(abs_top_builddir)/src/plugins-list | grep ^instrumentation_debug\= | cut -d'=' -f2` ;\
	tmp_compile_versions=`cat $(abs_top_builddir)/src/plugins-list | cut -d'=' -f1 | sed -e :a -e '$!N; s/\n/ /; ta'` ;\
	tmp_srcdir=$(abs_top_srcdir) ; \
	tmp_builddir=$(abs_top_builddir) ; \
	tmp_perfo_cflags='"$(perfo_cflags)"' ; \
	tmp_debug_cflags='"$(debug_cflags)"' ; \
	tmp_instr_cflags='"$(instr_cflags)"' ; \
	tmp_instrdebug_cflags='"$(instrdebug_cflags)"' ; \
	tmp_perfo_cxxflags='"$(global_perfo_cxxflags)"' ; \
	tmp_debug_cxxflags='"$(global_debug_cxxflags)"' ; \
	tmp_instr_cxxflags='"$(global_instr_cxxflags)"' ; \
	tmp_instrdebug_cxxflags='"$(global_instrdebug_cxxflags)"' ; \
	tmp_resiliency='"$(resiliency)"' ; \
	tmp_cc='"$(CC)"' ; \
	tmp_cxx='"$(CXX)"' ; \
	sed \
		-e's,@SRCDIR\@,'"$$tmp_srcdir"',g' \
		-e's,@BUILDDIR\@,'"$$tmp_builddir"',g' \
		-e's,@PLUGINS_PERFORMANCE\@,'"$$tmp_plugins_perfo"',g' \
		-e's,@PLUGINS_DEBUG\@,'"$$tmp_plugins_debug"',g' \
		-e's,@PLUGINS_INSTRUMENTATION\@,'"$$tmp_plugins_instr"',g' \
		-e's,@PLUGINS_INSTRUMENTATION_DEBUG\@,'"$$tmp_plugins_instr_debug"',g' \
		-e's,@COMPILE_VERSIONS\@,'"$$tmp_compile_versions"',g' \
		-e's,@PERFO_CFLAGS\@,'"$$tmp_perfo_cflags"',g' \
		-e's,@DEBUG_CFLAGS\@,'"$$tmp_debug_cflags"',g' \
		-e's,@INSTR_CFLAGS\@,'"$$tmp_instr_cflags"',g' \
		-e's,@INSTRDEBUG_CFLAGS\@,'"$$tmp_instrdebug_cflags"',g' \
		-e's,@PERFO_CXXFLAGS\@,'"$$tmp_perfo_cxxflags"',g' \
		-e's,@DEBUG_CXXFLAGS\@,'"$$tmp_debug_cxxflags"',g' \
		-e's,@INSTR_CXXFLAGS\@,'"$$tmp_instr_cxxflags"',g' \
		-e's,@INSTRDEBUG_CXXFLAGS\@,'"$$tmp_instrdebug_cxxflags"',g' \
		-e's,@CONFIG_CC\@,'"$$tmp_cc"',g' \
		-e's,@CONFIG_CXX\@,'"$$tmp_cxx"',g' \
		-e's,@resiliency\@,'"$$tmp_resiliency"',g' \
		$(srcdir)/resiliency-generator.in > resiliency-generator.tmp
	mv resiliency-generator.tmp resiliency-generator
	chmod 755 resiliency-generator
	
mpi-offload-generator: mpi-offload-generator.in
	rm -f mpi-offload-generator 
	tmp_plugins_perfo=`cat $(abs_top_builddir)/src/plugins-list | grep ^performance\= | cut -d'=' -f2` ;\
	tmp_plugins_debug=`cat $(abs_top_builddir)/src/plugins-list | grep ^debug\= | cut -d'=' -f2` ;\
	tmp_plugins_instr=`cat $(abs_top_builddir)/src/plugins-list | grep ^instrumentation\= | cut -d'=' -f2` ;\
	tmp_plugins_instr_debug=`cat $(abs_top_builddir)/src/plugins-list | grep ^instrumentation_debug\= | cut -d'=' -f2` ;\
	tmp_compile_versions=`cat $(abs_top_builddir)/src/plugins-list | cut -d'=' -f1 | sed -e :a -e '$!N; s/\n/ /; ta'` ;\
	tmp_srcdir=$(abs_top_srcdir) ; \
	tmp_builddir=$(abs_top_builddir) ; \
	tmp_perfo_cflags='"$(perfo_cflags)"' ; \
	tmp_debug_cflags='"$(debug_cflags)"' ; \
	tmp_instr_cflags='"$(instr_cflags)"' ; \
	tmp_instrdebug_cflags='"$(instrdebug_cflags)"' ; \
	tmp_perfo_cxxflags='"$(global_perfo_cxxflags)"' ; \
	tmp_debug_cxxflags='"$(global_debug_cxxflags)"' ; \
	tmp_instr_cxxflags='"$(global_instr_cxxflags)"' ; \
	tmp_instrdebug_cxxflags='"$(global_instrdebug_cxxflags)"' ; \
	tmp_mpi_valid='$(mpi_valid)' ; \
	tmp_prefix='$(NANOX_MPI_PREFIX)' ; \
	tmp_cc='"$(CC)"' ; \
	tmp_cxx='"$(CXX)"' ; \
	sed \
		-e's,@SRCDIR\@,'"$$tmp_srcdir"',g' \
		-e's,@BUILDDIR\@,'"$$tmp_builddir"',g' \
		-e's,@PLUGINS_PERFORMANCE\@,'"$$tmp_plugins_perfo"',g' \
		-e's,@PLUGINS_DEBUG\@,'"$$tmp_plugins_debug"',g' \
		-e's,@PLUGINS_INSTRUMENTATION\@,'"$$tmp_plugins_instr"',g' \
		-e's,@PLUGINS_INSTRUMENTATION_DEBUG\@,'"$$tmp_plugins_instr_debug"',g' \
		-e's,@COMPILE_VERSIONS\@,'"$$tmp_compile_versions"',g' \
		-e's,@PERFO_CFLAGS\@,'"$$tmp_perfo_cflags"',g' \
		-e's,@DEBUG_CFLAGS\@,'"$$tmp_debug_cflags"',g' \
		-e's,@INSTR_CFLAGS\@,'"$$tmp_instr_cflags"',g' \
		-e's,@INSTRDEBUG_CFLAGS\@,'"$$tmp_instrdebug_cflags"',g' \
		-e's,@PERFO_CXXFLAGS\@,'"$$tmp_perfo_cxxflags"',g' \
		-e's,@DEBUG_CXXFLAGS\@,'"$$tmp_debug_cxxflags"',g' \
		-e's,@INSTR_CXXFLAGS\@,'"$$tmp_instr_cxxflags"',g' \
		-e's,@INSTRDEBUG_CXXFLAGS\@,'"$$tmp_instrdebug_cxxflags"',g' \
		-e's,@CONFIG_CC\@,'"$$tmp_cc"',g' \
		-e's,@CONFIG_CXX\@,'"$$tmp_cxx"',g' \
		-e's,@MPI_VALID\@,'"$$tmp_mpi_valid"',g' \
		-e's,@PREFIX\@,'"$$tmp_prefix"',g' \
		$(srcdir)/mpi-offload-generator.in > mpi-offload-generator.tmp
	mv mpi-offload-generator.tmp mpi-offload-generator
	chmod 755 mpi-offload-generator

opencl-generator: opencl-generator.in
	rm -f opencl-generator
	tmp_plugins_perfo=`cat $(abs_top_builddir)/src/plugins-list | grep ^performance\= | cut -d'=' -f2` ;\
	tmp_plugins_debug=`cat $(abs_top_builddir)/src/plugins-list | grep ^debug\= | cut -d'=' -f2` ;\
	tmp_plugins_instr=`cat $(abs_top_builddir)/src/plugins-list | grep ^instrumentation\= | cut -d'=' -f2` ;\
	tmp_plugins_instr_debug=`cat $(abs_top_builddir)/src/plugins-list | grep ^instrumentation_debug\= | cut -d'=' -f2` ;\
	tmp_compile_versions=`cat $(abs_top_builddir)/src/plugins-list | cut -d'=' -f1 | sed -e :a -e '$!N; s/\n/ /; ta'` ;\
	tmp_srcdir=$(abs_top_srcdir) ; \
	tmp_builddir=$(abs_top_builddir) ; \
	tmp_perfo_cflags='"$(perfo_cflags)"' ; \
	tmp_debug_cflags='"$(debug_cflags)"' ; \
	tmp_instr_cflags='"$(instr_cflags)"' ; \
	tmp_instrdebug_cflags='"$(instrdebug_cflags)"' ; \
	tmp_perfo_cxxflags='"$(global_perfo_cxxflags)"' ; \
	tmp_debug_cxxflags='"$(global_debug_cxxflags)"' ; \
	tmp_instr_cxxflags='"$(global_instr_cxxflags)"' ; \
	tmp_instrdebug_cxxflags='"$(global_instrdebug_cxxflags)"' ; \
	tmp_cc='"$(CC)"' ; \
	tmp_cxx='"$(CXX)"' ; \
	tmp_opencl_inc='"$(OPENCL_INC)"' ; \
	tmp_opencl_ld='"$(OPENCL_LD)"' ; \
	sed \
		-e's,@SRCDIR\@,'"$$tmp_srcdir"',g' \
		-e's,@BUILDDIR\@,'"$$tmp_builddir"',g' \
		-e's,@PLUGINS_PERFORMANCE\@,'"$$tmp_plugins_perfo"',g' \
		-e's,@PLUGINS_DEBUG\@,'"$$tmp_plugins_debug"',g' \
		-e's,@PLUGINS_INSTRUMENTATION\@,'"$$tmp_plugins_instr"',g' \
		-e's,@PLUGINS_INSTRUMENTATION_DEBUG\@,'"$$tmp_plugins_instr_debug"',g' \
		-e's,@COMPILE_VERSIONS\@,'"$$tmp_compile_versions"',g' \
		-e's,@PERFO_CFLAGS\@,'"$$tmp_perfo_cflags"',g' \
		-e's,@DEBUG_CFLAGS\@,'"$$tmp_debug_cflags"',g' \
		-e's,@INSTR_CFLAGS\@,'"$$tmp_instr_cflags"',g' \
		-e's,@INSTRDEBUG_CFLAGS\@,'"$$tmp_instrdebug_cflags"',g' \
		-e's,@PERFO_CXXFLAGS\@,'"$$tmp_perfo_cxxflags"',g' \
		-e's,@DEBUG_CXXFLAGS\@,'"$$tmp_debug_cxxflags"',g' \
		-e's,@INSTR_CXXFLAGS\@,'"$$tmp_instr_cxxflags"',g' \
		-e's,@INSTRDEBUG_CXXFLAGS\@,'"$$tmp_instrdebug_cxxflags"',g' \
		-e's,@CONFIG_CC\@,'"$$tmp_cc"',g' \
		-e's,@CONFIG_CXX\@,'"$$tmp_cxx"',g' \
		-e's,@CONFIG_OPENCL_INC\@,'"$$tmp_opencl_inc"',g' \
		-e's,@CONFIG_OPENCL_LD\@,'"$$tmp_opencl_ld"',g' \
		$(srcdir)/opencl-generator.in > opencl-generator.tmp
	mv opencl-generator.tmp opencl-generator
	chmod 755 opencl-generator
