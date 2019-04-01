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

SUBDIRS = .

LIT_EXTRA = lit/PKG-INFO \
	    lit/README.txt \
	    lit/lit.py \
	    lit/lit/BooleanExpression.py \
	    lit/lit/LitConfig.py \
	    lit/lit/LitTestCase.py \
	    lit/lit/ProgressBar.py \
	    lit/lit/ShCommands.py \
	    lit/lit/ShUtil.py \
	    lit/lit/Test.py \
	    lit/lit/TestRunner.py \
	    lit/lit/TestingConfig.py \
	    lit/lit/__init__.py \
	    lit/lit/discovery.py \
	    lit/lit/main.py \
	    lit/lit/run.py \
	    lit/lit/util.py \
	    lit/lit/builtin_commands/__init__.py \
	    lit/lit/builtin_commands/cat.py \
	    lit/lit/formats/__init__.py \
	    lit/lit/formats/base.py \
	    lit/lit/formats/googletest.py \
	    lit/lit/formats/shtest.py \
	    lit/lit/llvm/__init__.py \
	    lit/lit/llvm/config.py \
	    lit/lit/llvm/subst.py
	    $(END)

EXTRA_DIST=\
	astyle-nanox.sh \
	bets \
	headache.cfg \
	headerfy.sh \
	preamble.txt \
	nanox.spec \
	debian/compat \
	debian/source/format \
	debian/rules \
	debian/control \
	$(LIT_EXTRA) \
	$(END)

dist_bin_SCRIPTS=\
	extrae_post_process.sh \
	offload_slave_launch.sh \
	offload_instrumentation.sh \
	nanox\
	track_deps.sh\
	task_numbers_in_path_to_selected_tasks.REF.cfg\
	tasks_in_path_to_selected.REF.cfg\
	create_mic_nx_hostlist\
	$(END)

rpm:
	@echo "Generating RPM structure"
	mkdir -p SOURCES BUILD RPMS SPECS SRPMS
	cp $(top_build_prefix)*.tar.gz SOURCES
	RELEASE=$${RELEASE:-$(shell date +%Y%m%d)}; \
	THREADS=$$(getconf _NPROCESSORS_ONLN); THREADS=$${THREADS:-1}; \
	rpmbuild -v -bb --clean $(srcdir)/nanox.spec --with extrae --define '_topdir $(CURDIR)' --define '_version $(VERSION)' --define "_release $${RELEASE}" --define '_rpm_prefix $(RPM_PREFIX)' --define '_host $(host_triplet)' --define '_target $(target_triplet)' --define '_build $(build_triplet)' --define "threads $${THREADS}"

clean-rpm:
	-rm -rf BUILDROOT SOURCES BUILD RPMS SPECS SRPMS


deb:
	cp $(top_builddir)/$(PACKAGE)-$(VERSION).tar.gz $(PACKAGE)_$(VERSION).orig.tar.gz
	tar -xf $(PACKAGE)_$(VERSION).orig.tar.gz
	rm -rf upstream_dir
	mv $(PACKAGE)-$(VERSION) upstream_dir/
	cp -rf $(srcdir)/debian upstream_dir/
	cp -rf upstream_dir/COPYING upstream_dir/debian/copyright
	rm -f upstream_dir/debian/changelog
	RELEASE=$${RELEASE:-$(shell date +%Y%m%d)}$(DEB_RELEASE); \
	if test ! -d $(top_srcdir)/.git; \
	then dch --create -c upstream_dir/debian/changelog --empty --package $(PACKAGE) -v "$(VERSION)-$${RELEASE}"; \
	else git --git-dir="$(top_srcdir)/.git" --work-tree="$(top_srcdir)/.git" log -1 --pretty=format:"$(PACKAGE) ($(VERSION)-$${RELEASE}) unstable; urgency=low%x0A%x0A  * %h %s%x0A%x0A -- %an <%ae>  %aD%n" > upstream_dir/debian/changelog; \
	fi
	THREADS=$$(getconf _NPROCESSORS_ONLN); THREADS=$${THREADS:-1}; cd upstream_dir; DEB_BUILD_OPTIONS="nocheck parallel=$${THREADS}" debuild -us -uc
	rm -rf upstream_dir

clean-deb:
	rm -f $(PACKAGE)$(VERSION).orig.tar.gz $(PACKAGE)$(VERSION)*.debian.tar.gz $(PACKAGE)$(VERSION)*.dsc $(PACKAGE)$(DEB_RELEASE)*.build $(PACKAGE)$(DEB_RELEASE)*.changes *.deb; \
	rm -rf upstream_dir

clean-local: clean-rpm clean-deb
