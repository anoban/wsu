/*
 Copyright © 2016 Forschungszentrum Jülich GmbH
 All rights reserved.

 Redistribution and use in source and binary forms, with or without modification, are permitted under the GNU General Public License v3
 You should have received the GNU GENERAL PUBLIC LICENSE v3 with this file in license.txt but can also be found at http://www.gnu.org/licenses/gpl-3.0.en.html

 NOTE: The GPL.v3 license requires that all derivative work is distributed under the same license. That means that if you use this source code in any other program, you can only distribute that program with the full source code included and licensed under a GPL license.

 */

#if _WIN32 || _WIN64 || NOPYTHON

//Do not include the python code, as Python.h does not compile with the cross compiler

#else


#include "../../engine/SimulaDynamic.hpp"
#include "../../engine/SimulaTable.hpp"
#include "../../cli/ANSImanipulators.hpp"

// not this requires the python development files, on ubuntu currently sudo apt install python3-dev
#include <Python.h>


class PythonScript: public DerivativeBase {
public:
	PythonScript(SimulaDynamic *const pSD);
	virtual std::string getName() const;
private:
	virtual void calculate(const Time &t, double&);
	std::string scriptName_;
	std::string functionName_;
	bool scriptIsFile_;
	PyObject *pFunc_;
	SimulaBase::List objectsForArguments_;
	std::vector<bool> getRate_;
	std::vector<SimulaTable<double>*> objectsForReturn_;
	std::string returnFormatString_;
};

PythonScript::PythonScript(SimulaDynamic *const pSD) :
		DerivativeBase(pSD), scriptIsFile_(true) {
	// collect the pointers for getting info
	SimulaBase::List vl;
	pSD->getChild("pathOfVariablesToPass")->getAllChildren(vl); // note that the order matters here!!!
	//std::cout<<std::endl<<" number of children "<<vl.size();
	for (auto vlp : vl) {
		std::string mp;
		vlp->get(mp);
		//std::cout<<std::endl<<"adding "<<mp;
		auto sbp = pSD->existingPath(mp);
		if (!sbp) {
			//try rate
			auto mpr = mp;
			if (mpr.size() > 4) {
				mpr.erase(mpr.size() - 4);
			}
			sbp = pSD->existingPath(mpr);
			if (!sbp)
				msg::error("PythonScript: did not find path " + mp);
			getRate_.push_back(true);
			msg::warning("PythonScript: getting rate for "+mp);
		} else {
			getRate_.push_back(false);
		}
		objectsForArguments_.push_back(sbp);
		returnFormatString_+='d';//only support double returns
		//std::cout<<std::endl<<objectsForArguments_.size()<<" python using path "<<mp;
	}

	//get pointers for setting info
	SimulaBase* p=pSD->existingChild("pathOfVariablesToReturn");
	if(p){
		p->getAllChildren(vl);
		//std::cout<<std::endl<<" number of return values "<<vl.size();
		for (auto vlp : vl) {
			std::string mp;
			vlp->get(mp);
			//std::cout<<std::endl<<"adding "<<mp<<" to the return list.";
			auto sbp = dynamic_cast<SimulaTable<double>*> (pSD->getPath(mp));
			if(sbp){
				objectsForReturn_.push_back(sbp);
			}else{
				msg::error("PythonScript: pathOfVariablesToReturn does not point to a table. Path is "+mp+" declared for "+pSD->getName());
			}
		}
	}

	//function name
	p = pSD->existingChild("functionNameInPython");
	if (p) {
		p->get(functionName_);
	}else{
		functionName_ = "OpenSimRootModule";
	}

	//read script name and remove possible .py extension
	p = pSD->existingChild("scriptFileName");
	if (p) {
		p->get(scriptName_);
		size_t lastindex = scriptName_.find_last_of(".");
		if (scriptName_.substr(lastindex) == ".py")
			scriptName_.erase(lastindex);

		PyObject *pName = PyUnicode_FromString(scriptName_.c_str());
		PyObject *pModule = PyImport_Import(pName);
		if (!pModule) {
			std::cout << std::endl << std::endl << ANSI_Red;
			PyErr_Print();
			msg::error("PythonScript: Script " + scriptName_ + "  not found");
		}
		//std::cout<<std::endl<<_PyUnicode_AsString(pModule);
		pFunc_ = PyObject_GetAttrString(pModule, functionName_.c_str());
		if (!pFunc_) {
			std::cout << std::endl << std::endl << ANSI_Red;
			PyErr_Print();
			//PyObject_Print(pModule, stdout, 0);
			msg::error("PythonScript: Function " + functionName_ + " in script " + scriptName_ + " not found");
		}
		Py_DECREF(pName);
		Py_DECREF(pModule);
	} else {
		//read script
		pSD->getChild("script")->get(scriptName_);
		scriptIsFile_ = false;
	}

	//read list of parameters

}
;

std::string PythonScript::getName() const {
	return "usePythonScript";
}
;

void PythonScript::calculate(const Time &t, double &r) {
	//std::cout<<std::endl<<"0000";
	//collect parameters into array
	int numarguments = objectsForArguments_.size() + 1;
	PyObject *pArgs = PyTuple_New(numarguments);
	PyTuple_SetItem(pArgs, 0, PyFloat_FromDouble(t));
	//std::cout<<std::endl<<" of "<<objectsForArguments_.size();
	for (int j = 1; j < numarguments; ++j) {
		double v(0);
		if(getRate_[j-1]){
			objectsForArguments_[j - 1]->getRate(t, v);
		}else{
			objectsForArguments_[j - 1]->get(t, v);
		}
		//std::cout<<std::endl<<j<<" of "<<objectsForArguments_.size()<<"  "<<objectsForArguments_[j-1]->getName()<<" "<<v;
		PyTuple_SetItem(pArgs, j, PyFloat_FromDouble(v));
	}

	//call script

	if (scriptIsFile_) {
		//Run a python function
		//PyObject * pArgs = PyTuple_Pack(2, PyFloat_FromDouble(t), PyFloat_FromDouble(3));
		PyObject *pValue = PyObject_CallObject(pFunc_, pArgs);
		if (!pValue) {
			std::cout << std::endl << std::endl << ANSI_Red;
			PyErr_Print();
			//PyObject_Print(pModule, stdout, 0);
			msg::error("PythonScript: Error executing function " + functionName_ + " in script " + scriptName_ + " .");
		}
		if (objectsForReturn_.empty()) {
			r = PyFloat_AsDouble(pValue);
			//std::cout<<std::endl<<"AAAA";
		} else {
            //PyArg_ParseTuple(ret,"oo",ob1,ob2);  TODO
			//std::cout<<std::endl<<"BAAA";
			if (PyTuple_Check(pValue)) {
				auto n=PyTuple_Size(pValue);
				//std::cout<<std::endl<<"BBAA";
				if (n != (Py_ssize_t) objectsForReturn_.size()+1)
					msg::error(
							"PythonScript: Script " + scriptName_ + " returns " + std::to_string(n) + " but "	+ std::to_string(objectsForReturn_.size()) + " tables defined in XML input file");
				//std::cout<<std::endl<<"BAAA00";
				PyObject *value = PyTuple_GetItem(pValue, 0);
				//std::cout<<std::endl<<"BAAA01";
				r = PyFloat_AsDouble(value);
				//std::cout<<std::endl<<"BAAA02";
				for (Py_ssize_t i = 1; i < n; i++) {
					//std::cout<<std::endl<<"BBBA"<<i;
					PyObject *v = PyTuple_GetItem(pValue, i);
					double vr=PyFloat_AsDouble(v);
					objectsForReturn_[i-1]->set(t, vr);
				}
			} else {
				msg::error("PythonScript:: PyTuple_check(pValue) failed on script which is suppose to return mulitple arguments.");
			}
		}

		Py_DECREF(pValue);//will also decrease the refs it holds, so no need to do this on v and value pointers
	} else {
		msg::warning("PythonScript:: Running script but the return is not implemented, will simply return 0.");
		PyRun_SimpleString(scriptName_.c_str());
		r = 0.;
	}

	// clean up by releasing the pointer.
	Py_DECREF(pArgs);
}


DerivativeBase* newInstantiationPythonScript(SimulaDynamic *const pSD) {
	return new PythonScript(pSD);
}

static class AutoRegisterPythonInstantiationFunctions {
public:
	AutoRegisterPythonInstantiationFunctions() {
		//initialize python
		Py_Initialize();
		PyObject *sysmodule = PyImport_ImportModule("sys");
		PyObject *syspath = PyObject_GetAttrString(sysmodule, "path");
		PyList_Append(syspath, PyUnicode_FromString("."));		//add current path so we can use a relative path for importing scripts
		Py_DECREF(syspath);
		Py_DECREF(sysmodule);

		// register the maker with the factory
		BaseClassesMap::getDerivativeBaseClasses()["usePythonScript"] = newInstantiationPythonScript;
	}
	~ AutoRegisterPythonInstantiationFunctions() {
		Py_Finalize();
	}
} p1974583542536482754;

#endif
