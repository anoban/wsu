/*
 Copyright © 2016, The Pennsylvania State University
 All rights reserved.

 Copyright © 2016 Forschungszentrum Jülich GmbH
 All rights reserved.

 Redistribution and use in source and binary forms, with or without modification, are permitted under the GNU General Public License v3 and provided that the following conditions are met:
 1. Redistributions of source code must retain the above copyright notice, this list of conditions and the following disclaimer.
 2. Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the following disclaimer in the documentation and/or other materials provided with the distribution.
 3. Neither the name of the copyright holder nor the names of its contributors may be used to endorse or promote products derived from this software without specific prior written permission.

 Disclaimer
 THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

 You should have received the GNU GENERAL PUBLIC LICENSE v3 with this file in license.txt but can also be found at http://www.gnu.org/licenses/gpl-3.0.en.html

 NOTE: The GPL.v3 license requires that all derivative work is distributed under the same license. That means that if you use this source code in any other program, you can only distribute that program with the full source code included and licensed under a GPL license.

 */
#include "OptimalNutrientContent.hpp"
#include "../../cli/Messages.hpp"
#include "../PlantType.hpp"

//Respiration rate at any point in the rootsystem in g CO2/cm root/day
UseRootClassAndNutrientSpecificTable::UseRootClassAndNutrientSpecificTable(SimulaDynamic* pSD) :
		DerivativeBase(pSD) {
	msg::error(
			"UseRootClassAndNutrientSpecificTable: This function is deprecated: use instead function=\"UseParameterFromParameterSection\" for "
					+ pSD->getPath());
}
void UseRootClassAndNutrientSpecificTable::calculate(const Time &t, double &result) {
}

std::string UseRootClassAndNutrientSpecificTable::getName() const {
	return "useRootClassAndNutrientSpecificTable";
}
DerivativeBase * newInstantiationUseRootClassAndNutrientSpecificTable(SimulaDynamic* const pSD) {
	return new UseRootClassAndNutrientSpecificTable(pSD);
}

bool OptimalNutrientContent::issueMessage(true);

OptimalNutrientContent::OptimalNutrientContent(SimulaDynamic* pSD) :
		TotalBaseLabeled(pSD) {
	std::string l(pSD->getName().substr(11, 7));
	if (l[0] == 'O')
		l[0] = 'o';
	if (l[0] == 'M')
		l[0] = 'm';
	current = pSD->getSibling(l + "NutrientConcentration");
	//dryweight
	dw = pSD->getParent(2)->getChild("rootSegmentDryWeight");

	//check if RCS effects respiration
	SimulaBase* contr(ORIGIN->getChild("simulationControls")->existingChild("corticalSenescence"));
	if (contr) {
		contr = contr->getChild("remobilizeNutrients");
		bool flag(true);
		if (contr)
			contr->get(flag);
		if (flag) {
			msg::warning("Optimal/Minimal NutrientContent: Reduction due to RCS simulated.");
		} else {
			dw = pSD->getParent(2)->getChild("rootSegmentDryWeightNoRCS");
			msg::warning("Optimal/Minimal NutrientContent: Reduction due to RCS NOT simulated.");
		}
	}

	//RCA reduction
	contr = (ORIGIN->getChild("simulationControls")->existingChild("aerenchyma"));
	if (contr)
		contr = contr->getChild("remobilizeP");
	bool flag = (true);
	if (contr)
		contr->get(flag);
	if (flag) {
		rca = pSD->getParent(2)->existingChild("aerenchymaFormation");
	} else {
		rca = nullptr;
	}
	if (rca && issueMessage) {
		msg::warning("Optimal/Minimal NutrientContent: Reduction due to RCA simulated.");
		OptimalNutrientContent::issueMessage = false;
	}
	scale_ = 1;
}
void OptimalNutrientContent::calculate(const Time &t, double &result) {
	//check based on time what the next point is: growthpoint or datapoint
	std::string l(pSD->getName().substr(11, 7));
	if (l[0] == 'O')
		l[0] = 'o';
	if (l[0] == 'M')
		l[0] = 'm';
	SimulaBase *next = getNext(t)->getSibling(l + "NutrientConcentration");

	//get optimal concentrations at these datapoints
	double concD0(0);
	current->get(t, concD0);
	double concD1(concD0);
	next->get(t, concD1);

	//get dryweight
	double d;
	dw->get(t, d);

	//get aerenchyma
	double a(0), a2(0);
	if (rca) {
		rca->get(t, a);
		//SimulaBase *nextrca=rca->followChain(t);
		SimulaBase *nextrca = getNext(t)->getParent(2)->existingChild("aerenchymaFormation");
		if (nextrca)
			nextrca->get(t, a2);
		a += a2;
		a /= 2;
	}

	//average concentration*dw
	a2 = 1;
	result = (a2 - a) * d * (concD0 + concD1) / 2;
	scale_ = result;
}
bool OptimalNutrientContent::postIntegrationCorrection(SimulaVariable::Table & data) {
	bool r(false);
	//iterators
	SimulaVariable::Table::iterator eit(data.end());
	--eit;
	SimulaVariable::Table::iterator pit(eit);
	--pit;
	Time dt(eit->first - pit->first);
	double rate((scale_ - pit->second.state) / dt);
	if (pit->second.rate != rate) {
		pit->second.rate = rate;
		eit->second.rate = rate;
		r = true;
	}
	return r;
}
std::string OptimalNutrientContent::getName() const {
	return "rootSegmentOptimalNutrientContent";
}

DerivativeBase * newInstantiationOptimalNutrientContent(SimulaDynamic* const pSD) {
	return new OptimalNutrientContent(pSD);
}

ShootOptimalNutrientContent::ShootOptimalNutrientContent(SimulaDynamic* pSD) :
		DerivativeBase(pSD) {	//leafs or stem
	std::string s(pSD->getName());
	std::size_t n(s.size() - 22);
	std::string sp(s.substr(0, n));
	std::string som(s.substr(0, n + 7));
	//dryweight
	dw = pSD->getParent(2)->getChild(sp + "DryWeight", "g");
	//concentration
	conc = pSD->getSibling(som + "NutrientConcentration", pSD->getUnit() / "g");
}
void ShootOptimalNutrientContent::calculate(const Time &t, double &result) {
	dw->get(t, result);
	double c;
	conc->get(t, c);
	result *= c;
}
std::string ShootOptimalNutrientContent::getName() const {
	return "shootOptimalNutrientContent";
}

DerivativeBase * newInstantiationShootOptimalNutrientContent(SimulaDynamic* const pSD) {
	return new ShootOptimalNutrientContent(pSD);
}

ActualNutrientContent::ActualNutrientContent(SimulaDynamic* pSD):TotalBaseLabeled(pSD), organSize(nullptr) {
	std::string s(pSD->getName());
	std::size_t n(s.size() - 21);
	std::string type = s.substr(s.size() - 7, s.size());
	std::string organ;
	if (type == "Content"){
		organ = s.substr(0, n);
		if (!(pSD->getUnit() == "uMol")){
			msg::error("ActualNutrientContent: Unknown unit");
		}
	} else{
		type = s.substr(s.size() - 13, s.size());
		if (type == "Concentration"){
			n = s.size() - 27;
			organ = s.substr(0, n);
			if (pSD->getUnit() == "uMol/cm2"){
				organSize = pSD->getParent()->getSibling(organ + "Area", "cm2");
			} else if (pSD->getUnit() == "uMol/cm3"){
				organSize = pSD->getParent()->getSibling(organ + "Volume", "cm3");
			} else if (pSD->getUnit() == "uMol/g"){
				organSize = pSD->getParent()->getSibling(organ + "DryWeight", "g");
			} else{
				msg::error("ActualNutrientContent: Unknown unit");
			}
		} else{
			msg::error("ActualNutrientContent:: Unexpected tag name");
		}
	}
	organMinimalContent = pSD->getSibling(organ + "MinimalNutrientContent");
	organOptimalContent = pSD->getSibling(organ + "OptimalNutrientContent");
	std::string nutrient = pSD->getParent()->getName();
	SimulaBase* plant(pSD);
	PLANTTOP(plant);
	plantMinimalContent = plant->getChild(nutrient)->getChild("plantMinimalNutrientContent");
	plantOptimalContent = plant->getChild(nutrient)->getChild("plantOptimalNutrientContent");
	totalUptake = plant->getChild(nutrient)->getChild("plantNutrientUptake");
}
void ActualNutrientContent::calculate(const Time &t, double &result) {
	double uptake, plantMin, plantOpt, organMin, organOpt;
	totalUptake->get(t, uptake);
	if (uptake <= 0){
		msg::warning("ActualNutrientContent:: Uptake <= 0, this shouldn't happen.");
		result = 0;
		if (organSize){
			result = 1e9;
		}
		return;
	}
	plantMinimalContent->get(t, plantMin);
	plantOptimalContent->get(t, plantOpt);
	organMinimalContent->get(t, organMin);
	organOptimalContent->get(t, organOpt);
	if (uptake <= plantMin){
		result = uptake*organMin/plantMin;
		msg::warning("ActualNutrientContent: uptake < plantMinimumContent, you should probably adjust nutrient stress effects on growth");
	} else if (uptake >= plantMin){
		result = uptake*organOpt/plantOpt;
	} else{
		double proportion;
		proportion = (uptake - plantMin)/(plantOpt - plantMin);
		result = uptake*((1 - proportion)*organMin + proportion*organOpt)/((1 - proportion)*plantMin + proportion*plantOpt);
	}
	if (organSize){
		double oSize;
		organSize->get(t, oSize);
		if (oSize > 0) {
			result = result/oSize;
		} else{
			result = 1e9;
		}
		if (result == 0) msg::warning("ActualNutrientContent: Concentration = 0, this shouldn't happen");
	}
}
std::string ActualNutrientContent::getName() const {
	return "actualNutrientContent";
}

DerivativeBase * newInstantiationActualNutrientContent(SimulaDynamic* const pSD) {
	return new ActualNutrientContent(pSD);
}

LeafNutrientContent::LeafNutrientContent(SimulaDynamic* pSD):DerivativeBase(pSD), leafSize(nullptr), leafArea(nullptr), leafPartArea(nullptr) {
	std::string s(pSD->getName());
	std::string name = s.substr(0, 6); // name = sunlit or shaded
	if (name == "sunlit" || name == "shaded"){
		leafArea = pSD->getParent()->getSibling("leafAreaIndex");
		leafPartArea = pSD->getParent()->getSibling(name + "LeafAreaIndex");
	}
	std::string type = s.substr(s.size() - 7, s.size());
	if (type == "Content"){
		if (!(pSD->getUnit() == "uMol")){
			msg::error("leafNutrientContent: Unknown unit");
		}
	} else{
		type = s.substr(s.size() - 13, s.size());
		if (type == "Concentration"){
			if (pSD->getUnit() == "uMol/cm2"){
				leafSize = pSD->getParent()->getSibling("leafArea", "cm2");
			} else if (pSD->getUnit() == "uMol/cm3"){
				leafSize = pSD->getParent()->getSibling("leafVolume", "cm3");
			} else if (pSD->getUnit() == "uMol/g"){
				leafSize = pSD->getParent()->getSibling("leafDryWeight", "g");
			} else{
				msg::error("leafNutrientContent: Unknown unit");
			}
		} else{
			msg::error("leafNutrientContent:: Unexpected tag name");
		}
	}
	leafMinimalContent = pSD->getSibling("leafMinimalNutrientContent");
	leafOptimalContent = pSD->getSibling("leafOptimalNutrientContent");
	std::string nutrient = pSD->getParent()->getName();
	SimulaBase* plant(pSD);
	PLANTTOP(plant);
	plantMinimalContent = plant->getChild(nutrient)->getChild("plantMinimalNutrientContent");
	plantOptimalContent = plant->getChild(nutrient)->getChild("plantOptimalNutrientContent");
	totalUptake = plant->getChild(nutrient)->getChild("plantNutrientUptake");
}
void LeafNutrientContent::calculate(const Time &t, double &result) {
	double uptake, plantMin, plantOpt, leafMin, leafOpt;
	totalUptake->get(t, uptake);
	if (uptake <= 0){
		msg::warning("LeafNutrientContent:: Uptake <= 0, this shouldn't happen");
		result = 0;
		if (leafSize){
			result = 1e9;
		}
		return;
	}
	plantMinimalContent->get(t, plantMin);
	plantOptimalContent->get(t, plantOpt);
	leafMinimalContent->get(t, leafMin);
	leafOptimalContent->get(t, leafOpt);
	if (uptake <= plantMin){
		result = uptake*leafMin/plantMin;
		msg::warning("LeafNutrientContent: uptake < plantMinimumContent, you should probably adjust nutrient stress effects on growth");
	} else if (uptake >= plantMin){
		result = uptake*leafOpt/plantOpt;
	} else{
		double proportion;
		proportion = (uptake - plantMin)/(plantOpt - plantMin);
		result = uptake*((1 - proportion)*leafMin + proportion*leafOpt)/((1 - proportion)*plantMin + proportion*plantOpt);
	}
	if (leafPartArea && !leafSize){
		double totalArea, partArea;
		leafArea->get(t, totalArea);
		leafPartArea->get(t, partArea);
		if (totalArea != 0){
			result = result*partArea/totalArea;
		}
	}
	if (leafSize){
		double lSize;
		leafSize->get(t, lSize);
		if (lSize > 0){
			result = result/lSize;
		} else{
			result = 1e9;
		}
		if (result == 0) msg::warning("LeafNutrientContent: Concentration = 0, this shouldn't happen");
	}
}
std::string LeafNutrientContent::getName() const {
	return "leafNutrientContent";
}

DerivativeBase * newInstantiationLeafNutrientContent(SimulaDynamic* const pSD) {
	return new LeafNutrientContent(pSD);
}

//Register the module
static class AutoRegisterUseRootClassAndNutrientSpecificTableInstantiationFunctions {
public:
	AutoRegisterUseRootClassAndNutrientSpecificTableInstantiationFunctions() {
		BaseClassesMap::getDerivativeBaseClasses()["useRootClassAndNutrientSpecificTable"] = newInstantiationUseRootClassAndNutrientSpecificTable;
		BaseClassesMap::getDerivativeBaseClasses()["rootSegmentOptimalNutrientContent"] = newInstantiationOptimalNutrientContent;
		BaseClassesMap::getDerivativeBaseClasses()["stemOptimalNutrientContent"] = newInstantiationShootOptimalNutrientContent;
		BaseClassesMap::getDerivativeBaseClasses()["leafOptimalNutrientContent"] = newInstantiationShootOptimalNutrientContent;
		BaseClassesMap::getDerivativeBaseClasses()["shootOptimalNutrientContent"] = newInstantiationShootOptimalNutrientContent;
		BaseClassesMap::getDerivativeBaseClasses()["rootSegmentMinimalNutrientContent"] = newInstantiationOptimalNutrientContent;
		BaseClassesMap::getDerivativeBaseClasses()["stemMinimalNutrientContent"] = newInstantiationShootOptimalNutrientContent;
		BaseClassesMap::getDerivativeBaseClasses()["leafMinimalNutrientContent"] = newInstantiationShootOptimalNutrientContent;
		BaseClassesMap::getDerivativeBaseClasses()["shootMinimalNutrientContent"] = newInstantiationShootOptimalNutrientContent;
		BaseClassesMap::getDerivativeBaseClasses()["actualNutrientContent"] = newInstantiationActualNutrientContent;
		BaseClassesMap::getDerivativeBaseClasses()["leafNutrientContent"] = newInstantiationLeafNutrientContent;
	}
} p465841235;

