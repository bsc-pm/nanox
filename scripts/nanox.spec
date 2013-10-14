# Good tutorials:
# http://freshrpms.net/docs/fight/
# http://rpm.org/api/4.4.2.2/conditionalbuilds.html
# http://fedoraproject.org/wiki/How_to_create_an_RPM_package
# http://backreference.org/2011/09/17/some-tips-on-rpm-conditional-macros/
# Repos:
# http://eureka.ykyuen.info/2010/01/06/opensuse-create-your-own-software-repository-1/
# http://en.opensuse.org/SDB:Creating_YaST_installation_sources

# Build line:
# rpmbuild -v -bb --clean SPECS/nanox.spec -with extrae && cp RPMS/x86_64/* ~/source/rpm_repo/x86_64/ && createrepo ~/source/rpm_repo

%{?_with_extrae: %define _with_extrae 1}
%{!?_with_extrae: %define _with_extrae 0}
%define feature		nanox
%define release		2
%define buildroot       %{_topdir}/%{name}-%{version}-root

# Override prefix if _rpm_prefix is given
%{?_rpm_prefix: %define _prefix  %{_rpm_prefix} }


BuildRoot:	        %{buildroot}
Summary: 		Nanos++
License: 		GPL
%if %_with_extrae
Name: 			%{feature}-extrae
%else
Name: 			%{feature}-no-extrae
%endif
Version: 		%{version}
Release: 		%{release}
Source: 		%{feature}-%{version}.tar.gz
Prefix: 		%{_prefix}
Group: 			Development/Tools
Provides:		%{feature}
%if %_with_extrae
#BuildRequires: 		extrae
Requires: 		extrae
Conflicts: 		%{feature}-no-extrae
%else
Conflicts: 		%{feature}-extrae
%endif

%if %_with_extrae
%description
Nanos++ with extrae support.
%else
%description
Nanos++ without extrae support.
%endif

%prep
%setup -q -n %{feature}-%{version}

%build
%if %_with_extrae
%configure --with-extrae=%{_prefix}
%else
%configure
%endif
make -j4

#%check
#make check

%install
%makeinstall

%files
%defattr(-,root,root)
%{_bindir}/*
%{_libdir}/debug/*
%{_libdir}/instrumentation/*
#%{_libdir}/instrumentation-debug/*
%{_libdir}/performance/*
%{_includedir}/*
%{_datarootdir}/doc/nanox/*
