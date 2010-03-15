

EXTRA_DIST = api-generator.in core-generator.in mixed-generator.in
noinst_PROGRAMS = config-generator

noinst_SCRIPTS = api-generator core-generator mixed-generator

CLEANFILES = api-generator core-generator mixed-generator

config_generator_SOURCES=config-generator.c

api-generator: api-generator.in
	rm -f api-generator
	tmp_srcdir=$(PWD)/$(top_srcdir)/src ; \
	tmp_builddir=$(PWD)/$(top_builddir)/src ; \
	sed \
		-e's,@SRCDIR\@,'"$$tmp_srcdir"',g' \
		-e's,@BUILDDIR\@,'"$$tmp_builddir"',g' \
		$(srcdir)/api-generator.in > api-generator.tmp
	mv api-generator.tmp api-generator
	chmod 755 api-generator

core-generator: core-generator.in
	rm -f core-generator
	tmp_srcdir=$(PWD)/$(top_srcdir)/src ; \
	tmp_builddir=$(PWD)/$(top_builddir)/src ; \
	sed \
		-e's,@SRCDIR\@,'"$$tmp_srcdir"',g' \
		-e's,@BUILDDIR\@,'"$$tmp_builddir"',g' \
		$(srcdir)/core-generator.in > core-generator.tmp
	mv core-generator.tmp core-generator
	chmod 755 core-generator

mixed-generator: mixed-generator.in
	rm -f mixed-generator
	tmp_srcdir=$(PWD)/$(top_srcdir)/src ; \
	tmp_builddir=$(PWD)/$(top_builddir)/src ; \
	sed \
		-e's,@SRCDIR\@,'"$$tmp_srcdir"',g' \
		-e's,@BUILDDIR\@,'"$$tmp_builddir"',g' \
		$(srcdir)/mixed-generator.in > mixed-generator.tmp
	mv mixed-generator.tmp mixed-generator
	chmod 755 mixed-generator

