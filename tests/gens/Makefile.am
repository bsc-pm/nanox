EXTRA_DIST = api-generator.in core-generator.in mixed-generator.in nanox-creation-generator.in api-omp-generator.in config.py

noinst_SCRIPTS = api-generator core-generator mixed-generator nanox-creation-generator api-omp-generator

CLEANFILES = api-generator core-generator mixed-generator nanox-creation-generator api-omp-generator

instrumentation_cxxflags=-DNANOS_INSTRUMENTATION_ENABLED

USED_VERSION=@USED_VERSION@

performance:
	rm -f api-generator core-generator mixed-generator nanox-creation-generator
	$(MAKE) -C. USED_VERSION=performance api-generator core-generator mixed-generator nanox-creation-generator

debug:
	rm -f api-generator core-generator mixed-generator nanox-creation-generator
	$(MAKE) -C. USED_VERSION=debug api-generator core-generator mixed-generator nanox-creation-generator

instrumentation:
	rm -f api-generator core-generator mixed-generator nanox-creation-generator
	$(MAKE) -C. USED_VERSION=instrumentation api-generator core-generator mixed-generator nanox-creation-generator

api-generator: api-generator.in
	rm -f api-generator
	tmp_plugins_dirs=`cat $(abs_top_builddir)/src/plugins-list | grep $(USED_VERSION)\= | cut -d'=' -f2` ;\
	tmp_srcdir=$(abs_top_srcdir) ; \
	tmp_builddir=$(abs_top_builddir) ; \
	tmp_used_version=$(USED_VERSION) ; \
	tmp_instr_cxxflags=$($(USED_VERSION)_cxxflags) ; \
	sed \
		-e's,@SRCDIR\@,'"$$tmp_srcdir"',g' \
		-e's,@BUILDDIR\@,'"$$tmp_builddir"',g' \
		-e's,@PLUGINSDIRS\@,'"$$tmp_plugins_dirs"',g' \
		-e's,@USEDVERSION\@,'"$$tmp_used_version"',g' \
		-e's,@INSTR_CXXFLAGS\@,'"$$tmp_instr_cxxflags"',g' \
		$(srcdir)/api-generator.in > api-generator.tmp
	mv api-generator.tmp api-generator
	chmod 755 api-generator

core-generator: core-generator.in
	rm -f core-generator
	tmp_plugins_dirs=`cat $(abs_top_builddir)/src/plugins-list | grep $(USED_VERSION)\= | cut -d'=' -f2` ;\
	tmp_srcdir=$(abs_top_srcdir) ; \
	tmp_builddir=$(abs_top_builddir) ; \
	tmp_used_version=$(USED_VERSION) ; \
	tmp_instr_cxxflags=$($(USED_VERSION)_cxxflags) ; \
	sed \
		-e's,@SRCDIR\@,'"$$tmp_srcdir"',g' \
		-e's,@BUILDDIR\@,'"$$tmp_builddir"',g' \
		-e's,@PLUGINSDIRS\@,'"$$tmp_plugins_dirs"',g' \
		-e's,@USEDVERSION\@,'"$$tmp_used_version"',g' \
		-e's,@INSTR_CXXFLAGS\@,'"$$tmp_instr_cxxflags"',g' \
		$(srcdir)/core-generator.in > core-generator.tmp
	mv core-generator.tmp core-generator
	chmod 755 core-generator

mixed-generator: mixed-generator.in
	rm -f mixed-generator
	tmp_plugins_dirs=`cat $(abs_top_builddir)/src/plugins-list | grep $(USED_VERSION)\= | cut -d'=' -f2` ;\
	tmp_srcdir=$(abs_top_srcdir) ; \
	tmp_builddir=$(abs_top_builddir) ; \
	tmp_used_version=$(USED_VERSION) ; \
	tmp_instr_cxxflags=$($(USED_VERSION)_cxxflags) ; \
	sed \
		-e's,@SRCDIR\@,'"$$tmp_srcdir"',g' \
		-e's,@BUILDDIR\@,'"$$tmp_builddir"',g' \
		-e's,@PLUGINSDIRS\@,'"$$tmp_plugins_dirs"',g' \
		-e's,@USEDVERSION\@,'"$$tmp_used_version"',g' \
		-e's,@INSTR_CXXFLAGS\@,'"$$tmp_instr_cxxflags"',g' \
		$(srcdir)/mixed-generator.in > mixed-generator.tmp
	mv mixed-generator.tmp mixed-generator
	chmod 755 mixed-generator

nanox-creation-generator: nanox-creation-generator.in
	rm -f nanox-creation-generator
	tmp_plugins_dirs=`cat $(abs_top_builddir)/src/plugins-list | grep $(USED_VERSION)\= | cut -d'=' -f2` ;\
	tmp_srcdir=$(abs_top_srcdir) ; \
	tmp_builddir=$(abs_top_builddir) ; \
	tmp_used_version=$(USED_VERSION) ; \
	tmp_instr_cxxflags=$($(USED_VERSION)_cxxflags) ; \
	sed \
		-e's,@SRCDIR\@,'"$$tmp_srcdir"',g' \
		-e's,@BUILDDIR\@,'"$$tmp_builddir"',g' \
		-e's,@PLUGINSDIRS\@,'"$$tmp_plugins_dirs"',g' \
		-e's,@USEDVERSION\@,'"$$tmp_used_version"',g' \
		-e's,@INSTR_CXXFLAGS\@,'"$$tmp_instr_cxxflags"',g' \
		$(srcdir)/nanox-creation-generator.in > nanox-creation-generator.tmp
	mv nanox-creation-generator.tmp nanox-creation-generator
	chmod 755 nanox-creation-generator

api-omp-generator: api-omp-generator.in
	rm -f api-omp-generator
	tmp_srcdir=$(abs_top_srcdir) ; \
	tmp_builddir=$(abs_top_builddir) ; \
	sed \
		-e's,@SRCDIR\@,'"$$tmp_srcdir"',g' \
		-e's,@BUILDDIR\@,'"$$tmp_builddir"',g' \
		$(srcdir)/api-omp-generator.in > api-omp-generator.tmp
	mv api-omp-generator.tmp api-omp-generator
	chmod 755 api-omp-generator

