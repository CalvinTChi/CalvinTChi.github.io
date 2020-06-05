#!/bin/bash

# update master branch
mkdir -p /tmp/workspace
cp -r * /tmp/workspace
git add -A .
git commit -m "update slides"

# update gh-pages branch
git checkout -B gh-pages
cp -r /tmp/workspace/* .
git add -A .
git commit -m "update slides"

# push update
git push origin master gh-pages
git checkout master
rm -rf /tmp/workspace
