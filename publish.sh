#!/bin/bash

# update master branc
mkdir -p /tmp/workspace
cp -r * /tmp/workspace/
git add -A .
git commit -m "update slides"

# update gh-pagesi branch
git checkout gh-pages
cp -r /tmp/workspace/* .
git add -A .
git commit -m "update slides"
git push origin master gh-pages

# push update
git checkout master
rm -rf /tmp/workspace

