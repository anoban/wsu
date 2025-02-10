#!/bin/bash
make clean
make -j8 win
re=$?
exit $re 

