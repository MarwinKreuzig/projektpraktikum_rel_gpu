#!/bin/bash

mkdir tidy/
cd source

for filename in ./*
do
clang-tidy-10 -p ../build-docker/ $filename > ../tidy/$filename
done

cd ..
