#! /bin/sh
srcdir=$1
distdir=$2

# FIXME: First we would have to check if git command exists. Assumming every dist is from git repository

# Creating ChangeLog file
git --git-dir=$srcdir/.git log  --pretty=format:"[%h] %cd : %s (%an)" --date=short > ${distdir}/ChangeLog

# Creating VERSION file
git_run=`git --git-dir=$srcdir/.git show --pretty=format:"%h %ci" HEAD | head -n 1`
git_version="git $git_run developer version"
echo $git_version > ${distdir}/VERSION
