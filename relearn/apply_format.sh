#!/bin/bash



for filename in ./source/*
do
clang-format-10 -style=file $filename
#echo $filename
done
