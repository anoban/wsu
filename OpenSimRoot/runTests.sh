#!/bin/bash
cd OpenSimRoot/tests/engine
./testEngine.sh "../../../release_build/OpenSimRoot"
rexe=$?
echo finished testing engine with error status $rexe
cd ../modules
./testModules.sh "../../../release_build/OpenSimRoot"
rexm=$?
echo finished testing modules with error status $rexm
cd ../../..
rex=$(($rexe+$rexm))
echo exiting with error status $rex
exit $rex


