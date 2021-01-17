#!/bin/bash
for filename in ./source/*
do
clang-format-10 -style=file $filename > $filename.new
#echo $filename
done
