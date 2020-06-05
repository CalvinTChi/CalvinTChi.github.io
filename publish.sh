#!/bin/bash

# update master branch
git add -A .
git commit -m "update slides"
git push origin master

# update gh-pagesi branch
git checkout gh-pages
git add -A .
git commit -m "update slides"

# push update
git checkout master

