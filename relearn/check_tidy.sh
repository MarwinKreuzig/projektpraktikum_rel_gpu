#/bin/sh

for f in $(find ../source/ -name '*.cpp'); 
do 
echo "Processing ${f}"; 
clang-tidy $f --checks=*,-clang-analyzer-core.CallAndMessage,-clang-analyzer-core.NonNullParamChecker,-openmp-exception-escape,-llvmlibc-restrict-system-libc-headers,-llvmlibc-callee-namespace,-llvmlibc-implementation-in-namespace,-modernize-use-trailing-return-type,-llvm-header-guard,-fuchsia-default-arguments-calls
done

