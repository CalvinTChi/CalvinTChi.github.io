#!/bin/bash

# update master branch
git add -A .
git commit -m "update slides"

# update gh-pagesi branch
git checkout -B gh-pages
git add -A .
git commit -m "update slides"

# push update
git push origin master gh-pages
git checkout master
