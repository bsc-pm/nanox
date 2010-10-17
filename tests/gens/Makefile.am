EXTRA_DIST = api-generator.in core-generator.in mixed-generator.in nanox-creation-generator.in config.py

noinst_SCRIPTS = api-generator core-generator mixed-generator nanox-creation-generator

CLEANFILES = api-generator core-generator mixed-generator nanox-creation-generator

api-generator: api-generator.in
	rm -f api-generator
	tmp_srcdir=$(abs_top_srcdir) ; \
	tmp_builddir=$(abs_top_builddir) ; \
	sed \
		-e's,@SRCDIR\@,'"$$tmp_srcdir"',g' \
		-e's,@BUILDDIR\@,'"$$tmp_builddir"',g' \
		$(srcdir)/api-generator.in > api-generator.tmp
	mv api-generator.tmp api-generator
	chmod 755 api-generator

core-generator: core-generator.in
	rm -f core-generator
	tmp_srcdir=$(abs_top_srcdir) ; \
	tmp_builddir=$(abs_top_builddir) ; \
	sed \
		-e's,@SRCDIR\@,'"$$tmp_srcdir"',g' \
		-e's,@BUILDDIR\@,'"$$tmp_builddir"',g' \
		$(srcdir)/core-generator.in > core-generator.tmp
	mv core-generator.tmp core-generator
	chmod 755 core-generator

mixed-generator: mixed-generator.in
	rm -f mixed-generator
	tmp_srcdir=$(abs_top_srcdir) ; \
	tmp_builddir=$(abs_top_builddir) ; \
	sed \
		-e's,@SRCDIR\@,'"$$tmp_srcdir"',g' \
		-e's,@BUILDDIR\@,'"$$tmp_builddir"',g' \
		$(srcdir)/mixed-generator.in > mixed-generator.tmp
	mv mixed-generator.tmp mixed-generator
	chmod 755 mixed-generator

nanox-creation-generator: nanox-creation-generator.in
	rm -f nanox-creation-generator
	tmp_srcdir=$(abs_top_srcdir) ; \
	tmp_builddir=$(abs_top_builddir) ; \
	sed \
		-e's,@SRCDIR\@,'"$$tmp_srcdir"',g' \
		-e's,@BUILDDIR\@,'"$$tmp_builddir"',g' \
		$(srcdir)/nanox-creation-generator.in > nanox-creation-generator.tmp
	mv nanox-creation-generator.tmp nanox-creation-generator
	chmod 755 nanox-creation-generator

