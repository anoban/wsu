#!/bin/bash
cd OpenSimRoot/tests/modules

./testModules.sh "../../../release_build/OpenSimRoot"
rex=$?
echo finished testing engine with error status $rex
cd ../../..

exit $rex
