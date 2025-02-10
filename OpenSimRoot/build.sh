#!/bin/bash
make clean
make -j8 release
re=$?
exit $re 

