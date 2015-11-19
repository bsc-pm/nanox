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

SUBDIRS = .

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
	$(END)

dist_bin_SCRIPTS=\
	extrae_post_process.sh \
	offload_slave_launch.sh \
	offload_instrumentation.sh \
	nanox\
   track_deps.sh\
   task_numbers_in_path_to_selected_tasks.REF.cfg\
   tasks_in_path_to_selected.REF.cfg\
	$(END)

rpm:
	@echo "Generating RPM structure"
	mkdir -p SOURCES BUILD RPMS SPECS SRPMS
	cp $(top_build_prefix)*.tar.gz SOURCES
        
	THREADS=$$(getconf _NPROCESSORS_ONLN); THREADS=$${THREADS:-1}; rpmbuild -v -bb --clean $(srcdir)/nanox.spec --with extrae --define '_topdir $(CURDIR)' --define 'version $(VERSION)' --define 'release $(shell date +%Y%m%d)' --define '_rpm_prefix $(RPM_PREFIX)' --define '_host $(host_triplet)' --define '_target $(target_triplet)' --define '_build $(build_triplet)' --define "threads $${THREADS}"

clean-rpm:
	-rm -rf BUILDROOT SOURCES BUILD RPMS SPECS SRPMS


deb:
	cp $(top_build_prefix)*.tar.gz nanox_$(VERSION).orig.tar.gz
	tar -xf nanox_$(VERSION).orig.tar.gz
	-rm -rf upstream_dir
	mv nanox-$(VERSION) upstream_dir
	cp -rf $(srcdir)/debian upstream_dir/
	cp -rf upstream_dir/COPYING upstream_dir/debian/copyright
	-rm -f upstream_dir/debian/changelog

# Generate an empty changelog if git is not present
	if test ! -d $(top_srcdir)/.git; \
	then dch --create -c upstream_dir/debian/changelog --empty --package nanox -v "$(VERSION)-$(shell date +%Y%m%d)$(DEB_RELEASE)"; \
	else git --git-dir="$(top_srcdir)/.git" --work-tree="$(top_srcdir)/.git" log -1 --pretty=format:"nanox ($(VERSION)-$(shell date +%Y%m%d)$(DEB_RELEASE)) unstable; urgency=low%x0A%x0A  * %h %s%x0A%x0A -- %an <%ae>  %aD" > upstream_dir/debian/changelog; \
	fi

	THREADS=$$(getconf _NPROCESSORS_ONLN); THREADS=$${THREADS:-1};  cd upstream_dir; DEB_BUILD_OPTIONS="nocheck parallel=$${THREADS}" debuild -us -uc
	-rm -rf upstream_dir

clean-deb:
	-rm -f nanox_$(VERSION).orig.tar.gz nanox_$(VERSION)*.debian.tar.gz nanox_$(VERSION)*.dsc nanox_$(DEB_RELEASE)*.build nanox_$(DEB_RELEASE)*.changes *.deb
	-rm -rf upstream_dir

clean-local: clean-rpm clean-deb
