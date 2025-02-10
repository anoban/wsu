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
#include  "Transpiration.hpp"
#include  "../../engine/Origin.hpp"
#include "../PlantType.hpp"
#include "../../cli/Messages.hpp"

PotentialTranspirationCrop::PotentialTranspirationCrop(SimulaDynamic* pSD):DerivativeBase(pSD){
	pSD->checkUnit("cm3");//todo assumed to be integrated

	cropLeafAreaIndex=pSD->existingPath("/plants/meanLeafAreaIndex","cm2/cm2");
	if(cropLeafAreaIndex){
		leafArea=pSD->getSibling("leafArea","cm2");
		cropTranspiration=pSD->getPath("/atmosphere/potentialTranspiration","cm");
	}else{
		//no crop level transpiration computed, use fall back to compute individual plant transpiration
		//relativePotentialTranspiration;
		std::string plantType;
		PLANTTYPE(plantType,pSD)
		relativePotentialTranspiration=GETSHOOTPARAMETERS(plantType);
		relativePotentialTranspiration=relativePotentialTranspiration->getChild("relativePotentialTranspiration");
		//leafArea
		if(relativePotentialTranspiration->getUnit()=="cm3/cm2/day"){
			mode=1;
			leafArea=pSD->getSibling("leafArea","cm2");
			leafSenescence = pSD->existingSibling("senescedLeafArea");
		}else if(relativePotentialTranspiration->getUnit()=="cm3/g"){
			mode=2;
			leafArea=pSD->getSibling("photosynthesis","g");
		}else{
			msg::error("PotentialTranspirationCrop: unsupported unit '"+relativePotentialTranspiration->getUnit().name+"' for relativePotentialTranspiration. Use cm3/cm2/day or cm3/g");
		}

	}
}
std::string PotentialTranspirationCrop::getName()const{
	return "potentialTranspirationCrop";
}
void PotentialTranspirationCrop::calculate(const Time &t,double &trans){
	if(cropLeafAreaIndex){
		double la,lai;
		cropLeafAreaIndex->get(t,lai);
		if(lai<=1e-6) {
			trans=0;
		}else{
			leafArea->get(t,la);
			cropTranspiration->getRate(t,trans);
			trans /= lai; //cm per unit soil to cm per unit leaf area.
			trans *= la;
		}
	}else{
		//get leaf area
		double la;
		if(mode==1){
			leafArea->get(t,la);
			if(leafSenescence){
				double s;
				leafSenescence->get(t,s);
				la-=s;
			}
		}else{
			leafArea->getRate(t,la);
		}
		double r;
		//get transpiration/leaf area
		relativePotentialTranspiration->get(t-pSD->getStartTime(),r);
		//multiply
		trans=la*r;
	}
}
DerivativeBase * newInstantiationPotentialTranspirationCrop(SimulaDynamic* const pSD){
   return new PotentialTranspirationCrop(pSD);
}

SimplePotentialTranspirationRate::SimplePotentialTranspirationRate(SimulaDynamic* pSD):DerivativeBase(pSD), cachedTime(-10){
	std::string name = pSD->getName().substr(0, 6); // name = sunlit or shaded
	pSD->checkUnit("cm3/day");
	if (name == "sunlit" || name == "shaded"){
		std::string name2 = name;
		name2.at(0) = std::toupper(name2.at(0));
		cropLeafAreaIndex=pSD->getPath("/plants/mean" + name2 + "LeafAreaIndex","cm2/cm2");
		leafArea=pSD->getSibling(name + "LeafArea","cm2");
		cropTranspirationRate = pSD->existingPath("/atmosphere/" + name + "PotentialTranspirationRate","cm/day");
		if (!cropTranspirationRate) cropTranspiration=pSD->getPath("/atmosphere/" + name + "PotentialTranspiration","cm");
	} else{
		cropLeafAreaIndex=pSD->getPath("/plants/meanLeafAreaIndex","cm2/cm2");
		leafArea=pSD->getSibling("leafArea","cm2");
		cropTranspirationRate = pSD->existingPath("/atmosphere/potentialTranspirationRate","cm/day");
		if (!cropTranspirationRate) cropTranspiration=pSD->getPath("/atmosphere/potentialTranspiration","cm");
	}
}

std::string SimplePotentialTranspirationRate::getName()const{
	return "simplePotentialTranspirationRate";
}
void SimplePotentialTranspirationRate::calculate(const Time &t,double &trans){
	if (std::abs(t - cachedTime) > TIMEERROR){
		cropLeafAreaIndex->get(t,lai);
		leafArea->get(t,la);
		cachedTime = t;
	}
	if(lai<=1e-6) {
		trans=0;
	}else{
		if (!cropTranspirationRate) cropTranspiration->getRate(t,trans);
		if (cropTranspirationRate) cropTranspirationRate->get(t, trans);
		trans = trans/lai; //cm per unit soil to cm per unit leaf area.
		trans = trans*la;
	}
}
DerivativeBase * newInstantiationSimplePotentialTranspirationRate(SimulaDynamic* const pSD){
   return new SimplePotentialTranspirationRate(pSD);
}


ActualTranspiration::ActualTranspiration(SimulaDynamic* pSD):DerivativeBase(pSD)
{
	//potential transpiration, and potential water uptake
	potentialTranspiration=pSD->getSibling("potentialTranspiration",pSD->getUnit());
	potentialWaterUptake=pSD->getParent(3)->getChild("rootPotentialWaterUptake",pSD->getUnit());
}
void ActualTranspiration::calculate(const Time &t,double &trans){
	std::size_t np=pSD->getPredictorSize();
	potentialTranspiration->getRate(t,trans);
	if(trans>0){
		double uptake;
		potentialWaterUptake->getRate(t,uptake);
		if(trans>uptake*1.01 && uptake>0){
			trans=uptake;
			if(np==pSD->getPredictorSize()) msg::warning("ActualTranspiration: simulating drought");
		}
	}
}
std::string ActualTranspiration::getName()const{
	return "actualTranspiration";
}

DerivativeBase * newInstantiationActualTranspiration(SimulaDynamic* const pSD){
   return new ActualTranspiration(pSD);
}

StomatalConductance::StomatalConductance(SimulaDynamic* pSD):DerivativeBase(pSD), pLeafNitrogenConcentration(nullptr), C4Photo(false), carboxylationDeactivationEnergy(-9999), cachedTime(-9999), cachedLeafTemperature(-9999), nitrogenLimit(1e99){
	std::string name = pSD->getName().substr(0, 6); // name = sunlit or shaded
	pPhotosynthesisRate = pSD->existingSibling(name + "PhotosynthesisRate", "umol/m2/s");
	if (!pPhotosynthesisRate) pPhotosynthesis = pSD->getSibling(name + "Photosynthesis", "g");
	pLeafRespirationRate = pSD->existingSibling(name + "LeafRespirationRate", "g/day");
	if (!pLeafRespirationRate) pLeafRespiration = pSD->getSibling(name + "LeafRespiration", "g");
	pSaturatedVapourPressure = ORIGIN->getPath("/atmosphere/saturatedVaporPressure");
	pActualVaporPressure = ORIGIN->getPath("/atmosphere/actualVaporPressure");
	SimulaBase *probe = ORIGIN->getPath("/environment/atmosphere/referenceVapourPressureDeficit");
	probe->get(vapourPressureDeficitReference);
	std::string plantType;
	PLANTTYPE(plantType,pSD);
	SimulaBase *shootParameters = GETSHOOTPARAMETERS(plantType);
	probe = shootParameters->getChild("residualConductance", "mol/m2/s");
	probe->get(residualConductance);
	probe = shootParameters->getChild("BBSlope");
	probe->get(mConstant);
	probe = ORIGIN->getPath("/environment/atmosphere/CO2Concentration");
	probe->get(CO2Concentration);
	// pLeafTemperature = ORIGIN->existingPath("/environment/atmosphere/leafTemperature");
	pLeafTemperature = pSD->getSibling(name + "LeafTemperature");
	pLeafArea = pSD->getSibling(name + "LeafArea", "cm2");
	pWaterStressFactor = pSD->getParent(3)->getChild("stressFactor:impactOn:stomatalConductance");
	probe = shootParameters->existingChild("C4Photosynthesis");
	if (probe) probe->get(C4Photo);
	if (C4Photo){
		probe = shootParameters->getChild("rubiscoSpecificityAt25C");
        probe->get(referenceRubiscoSpecificity);
        probe = shootParameters->getChild("rubiscoSpecificityActivationEnergy", "J/mol");
        probe->get(rubiscoSpecificityActivationEnergy);
        probe = shootParameters->getChild("MichaelisPEPAt25C", "umol/mol");
		probe->get(michaelisPEPAt25C);
		probe = shootParameters->getChild("MichaelisPEPActivationEnergy");
		probe->get(michaelisPEPActivationEnergy);
		probe = shootParameters->getChild("bundleSheathConductanceAt25C", "mol/m2/s");
		probe->get(sheathConductanceAt25C);
		probe = shootParameters->getChild("bundleSheathConductanceActivationEnergy", "J/mol");
		probe->get(sheathConductanceActivationEnergy);
		probe = shootParameters->getChild("bundleSheathConductanceDeactivationEnergy", "J/mol");
		probe->get(sheathConductanceDeactivationEnergy);
		probe = shootParameters->getChild("bundleSheathConductanceEntropyTerm", "J/K/mol");
		probe->get(sheathConductanceEntropyTerm);
		probe = shootParameters->getChild("dayRespirationMesophyllFraction");
		probe->get(dayRespirationMesophyllFraction);
		pMaxPEPCarboxylationAt25C = shootParameters->getChild("maxPEPCarboxylationAt25C", "umol/m2/s");
		probe = shootParameters->getChild("PEPCarboxylationActivationEnergy", "J/mol");
		probe->get(pepCarboxylationActivationEnergy);
		probe = shootParameters->getChild("PEPCarboxylationDeactivationEnergy", "J/mol");
		probe->get(pepCarboxylationDeactivationEnergy);
		probe = shootParameters->getChild("PEPCarboxylationEntropyTerm", "J/K/mol");
		probe->get(pepCarboxylationEntropyTerm);
	} else{
		probe = shootParameters->getChild("CO2CompensationPointWithoutDayRespirationAt25C", "umol/mol");
		probe->get(CO2CompensationPointNoDayRespirationref);
		probe = shootParameters->getChild("CO2CompensationPointActivationEnergy", "J/mol");
		probe->get(activationEnergyCO2CompensationPointNoDayRespiration);
	}
	pMesophyllO = pSD->getSibling(name + "MesophyllO2Concentration", "mmol/mol");

	if (!pWaterStressFactor) msg::error("StomatalConductance: water/waterStressFactor not found");
	probe = shootParameters->getChild("RubiscoCO2MichaelisConstantAt25C", "umol/mol");
	probe->get(referenceMichaelisCO2);
	probe = shootParameters->getChild("RubiscoO2MichaelisConstantAt25C", "mmol/mol");
	probe->get(referenceMichaelisO2);
	probe = shootParameters->existingChild("relativeRespirationRateLeafs");
	if (probe){
		double respRate;
		probe->get(respRate);
		if (respRate > 0.){
			msg::warning("StomatalConductance: The relative respiration rate of leaves is not equal to 0. This will lead to inaccurate results if the temperature is not set to a contant value of 298.15 K.");
			probe->get(referenceDayRespiration);
		}
	}
	probe = shootParameters->getChild("RubiscoCO2MichaelisActivationEnergy", "J/mol");
	probe->get(activationEnergyCO2);
	probe = shootParameters->getChild("RubiscoO2MichaelisActivationEnergy", "J/mol");
	probe->get(activationEnergyO2);
	pReferenceMaxCarboxylationEfficiency = shootParameters->getChild("maxCarboxylationEfficiencyAt25C", "umol/m2/s");
	probe = shootParameters->getChild("carboxylationActivationEnergy", "J/mol");
	probe->get(carboxylationActivationEnergy);
	probe = shootParameters->existingChild("carboxylationDeactivationEnergy", "J/mol");
	if (probe) probe->get(carboxylationDeactivationEnergy);
	probe = shootParameters->existingChild("carboxylationEntropyTerm", "J/K/mol");
	if (probe) probe->get(carboxylationEntropyTerm);
	probe = shootParameters->existingChild("maxCarboxylationNitrogenProportionalityConstant", "umol/umol/s");
	if (probe){
		pLeafNitrogenConcentration = pSD->getSibling("nitrate")->getChild(name + "LeafActualNutrientConcentration", "umol/cm2");
		probe->get(maxCarboxylationNitrogenProportionalityConstant);
	}
}
void StomatalConductance::calculate(const Time &t,double &sc){
	if (std::abs(t - cachedTime) > TIMEERROR){
		pLeafArea->get(t, leafArea);
		leafArea = leafArea/10000; // convert from cm2 to m2
	}
	if (leafArea < 1e-6){
		sc = residualConductance;
		return;
	}
	double waterStress, photosynthesisRate, dayRespiration;
	pWaterStressFactor->get(t, waterStress);
	if (!pPhotosynthesisRate){
		pPhotosynthesis->getRate(t, photosynthesisRate);
		photosynthesisRate = photosynthesisRate*1000000/(12.0111*60*60*24*leafArea); // convert from g/day to umol/m2/s
	}
	if (!pLeafRespirationRate) pLeafRespiration->getRate(t, dayRespiration);
	double leafTemperature, mesophyllO, vapourPressureSaturated, vapourPressureActual, CO2CompensationPoint;
	pLeafTemperature->get(t, leafTemperature);
	leafTemperature = leafTemperature + 273.15; // Need leaf temperature in Kelvin
	if (pPhotosynthesisRate) pPhotosynthesisRate->get(t, photosynthesisRate);
	if (pLeafRespirationRate) pLeafRespirationRate->get(t, dayRespiration);
	dayRespiration = dayRespiration*1000000/(12.0111*60*60*24*leafArea); // convert from g/day to umol/m2/s
	photosynthesisRate = photosynthesisRate - dayRespiration;
	pMesophyllO->get(t, mesophyllO);
	if (std::abs(t - cachedTime) > TIMEERROR){
		pSaturatedVapourPressure->get(t, vapourPressureSaturated);
		pActualVaporPressure->get(t, vapourPressureActual);
		vapourPressureDeficit = vapourPressureSaturated - vapourPressureActual;
		pReferenceMaxCarboxylationEfficiency->get(t, referenceMaxCarboxylationEfficiency);
		if (C4Photo){
			pMaxPEPCarboxylationAt25C->get(t, maxRefPEP);
		}
		if (pLeafNitrogenConcentration){
			double leafNitrogen;
			pLeafNitrogenConcentration->get(t, leafNitrogen);
			leafNitrogen = leafNitrogen*10000.; // Convert from umol/cm2 to umol/m2
			nitrogenLimit = leafNitrogen*maxCarboxylationNitrogenProportionalityConstant;
			if (nitrogenLimit == 0) msg::error("StomatalConductance: Nitrogen limited carboxylation rate = 0");
		}
		cachedTime = t;
	}
	if (C4Photo){
		if (std::abs(leafTemperature - cachedLeafTemperature) > 0.01){
			cachedLeafTemperature = leafTemperature;
			double universalGasConstant = 8.3144598; // J/(mol*K)
			double refTemp = 298.15; // K
			michaelisCO2 = referenceMichaelisCO2*exp((leafTemperature-refTemp)*activationEnergyCO2/(leafTemperature*universalGasConstant*refTemp));
			michaelisO2 = referenceMichaelisO2*exp((leafTemperature-refTemp)*activationEnergyO2/(leafTemperature*universalGasConstant*refTemp));
			if (carboxylationDeactivationEnergy < 0){
				temperatureScalingFactor = exp((leafTemperature-refTemp)*carboxylationActivationEnergy/(leafTemperature*universalGasConstant*refTemp));
			} else{
				temperatureScalingFactor = exp((leafTemperature-refTemp)*carboxylationActivationEnergy/(refTemp*universalGasConstant*leafTemperature))*((1+exp((refTemp*carboxylationEntropyTerm-carboxylationDeactivationEnergy)/(refTemp*universalGasConstant)))/(1+exp((carboxylationEntropyTerm*leafTemperature-carboxylationDeactivationEnergy)/(universalGasConstant*leafTemperature))));
			}
			rubiscoSpecificityReciprocal = (1/(2*referenceRubiscoSpecificity))*exp((leafTemperature-refTemp)*rubiscoSpecificityActivationEnergy/(refTemp*universalGasConstant*leafTemperature));
			michaelisPEP = michaelisPEPAt25C*exp((leafTemperature - refTemp)*michaelisPEPActivationEnergy/(refTemp*universalGasConstant*leafTemperature));
			sheathConductance = sheathConductanceAt25C*exp((leafTemperature - refTemp)*sheathConductanceActivationEnergy/(refTemp*universalGasConstant*leafTemperature))*(1 + exp((sheathConductanceEntropyTerm - sheathConductanceDeactivationEnergy/refTemp)/universalGasConstant))/(1 + exp((sheathConductanceEntropyTerm - sheathConductanceDeactivationEnergy/leafTemperature)/universalGasConstant));
			maxPEPCarboxylation = maxRefPEP*exp((leafTemperature - refTemp)*pepCarboxylationActivationEnergy/(refTemp*universalGasConstant*leafTemperature))*(1 + exp((pepCarboxylationEntropyTerm - pepCarboxylationDeactivationEnergy/refTemp)/universalGasConstant))/(1 + exp((pepCarboxylationEntropyTerm - pepCarboxylationDeactivationEnergy/leafTemperature)/universalGasConstant));
		}
		double maxCarboxylationEfficiency = referenceMaxCarboxylationEfficiency*temperatureScalingFactor;
		if (maxCarboxylationEfficiency > nitrogenLimit) maxCarboxylationEfficiency = nitrogenLimit;
		double sheathCompensationPoint = (mesophyllO*rubiscoSpecificityReciprocal + michaelisCO2*(1 + mesophyllO/michaelisO2)*dayRespiration/maxCarboxylationEfficiency)/(1 + dayRespiration/maxCarboxylationEfficiency);
		double mesophyllRespiration = dayRespirationMesophyllFraction*dayRespiration;
		CO2CompensationPoint = michaelisPEP*(sheathConductance*sheathCompensationPoint + mesophyllRespiration)/maxPEPCarboxylation;
	} else{
	//	From farquhar et al 1980
		if (std::abs(leafTemperature - cachedLeafTemperature) > 0.01){
			cachedLeafTemperature = leafTemperature;
			double universalGasConstant = 8.3144598; // J/(mol*K)
			double refTemp = 298.15; // K
			michaelisCO2 = referenceMichaelisCO2*exp((leafTemperature-refTemp)*activationEnergyCO2/(leafTemperature*universalGasConstant*refTemp));
			michaelisO2 = referenceMichaelisO2*exp((leafTemperature-refTemp)*activationEnergyO2/(leafTemperature*universalGasConstant*refTemp));
			if (carboxylationDeactivationEnergy < 0){
				temperatureScalingFactor = exp((leafTemperature-refTemp)*carboxylationActivationEnergy/(leafTemperature*universalGasConstant*refTemp));
			} else{
				temperatureScalingFactor = exp((leafTemperature-refTemp)*carboxylationActivationEnergy/(refTemp*universalGasConstant*leafTemperature))*((1+exp((refTemp*carboxylationEntropyTerm-carboxylationDeactivationEnergy)/(refTemp*universalGasConstant)))/(1+exp((carboxylationEntropyTerm*leafTemperature-carboxylationDeactivationEnergy)/(universalGasConstant*leafTemperature))));
			}
			CO2CompensationPointNoDayRespiration = CO2CompensationPointNoDayRespirationref *exp((leafTemperature-refTemp)*activationEnergyCO2CompensationPointNoDayRespiration/(refTemp*universalGasConstant*leafTemperature));
		}
		double maxCarboxylationEfficiency = referenceMaxCarboxylationEfficiency*temperatureScalingFactor;
		if (maxCarboxylationEfficiency > nitrogenLimit) maxCarboxylationEfficiency = nitrogenLimit;
		CO2CompensationPoint = (CO2CompensationPointNoDayRespiration + michaelisCO2*dayRespiration*(1 + mesophyllO/michaelisO2)/maxCarboxylationEfficiency)/(1 - dayRespiration/maxCarboxylationEfficiency);
	}
//	This is the Ball-Berry-Leuning model, see Xu 2014: Coupled model of stomatal conductance-photosynthesis-transpiration for paddy rice under water-saving irrigation
	sc = mConstant*waterStress*photosynthesisRate/((1+vapourPressureDeficit/vapourPressureDeficitReference)*(CO2Concentration - CO2CompensationPoint)) + residualConductance;
	if(std::isnan(sc)) msg::error("conductance: sc is NaN ");
	sc = std::max(sc, residualConductance);
	// Unit of sc = mol/(m^2*s)
}

void StomatalConductance::getDefaultValue(const Time &t, double &var){
	var = residualConductance;
}
std::string StomatalConductance::getName()const{
	return "stomatalConductance";
}

DerivativeBase * newInstantiationStomatalConductance(SimulaDynamic* const pSD){
   return new StomatalConductance(pSD);
}

MeanStomatalConductance::MeanStomatalConductance(SimulaDynamic* pSD):DerivativeBase(pSD){
	std::string name = pSD->getName().substr(4, 6); // name = sunlit or shaded
	double splitBySunStatus(false);
	if (name == "Sunlit" || name == "Shaded"){
		name.at(0) = std::tolower(name.at(0));
		splitBySunStatus = true;
	}
	SimulaBase::List plants;
	ORIGIN->getChild("plants")->getAllChildren(plants);
	for(auto & it:plants){
		SimulaBase*	probe=(it)->existingChild("plantingTime");
		if(probe) {
			Time pt;
			probe->get(pt);
			plantingTimes.push_back(pt);
			probe=(it)->getChild("plantPosition")->getChild("shoot");
			if (splitBySunStatus){
				leafAreas.push_back(probe->getChild(name + "LeafAreaIndex"));
				conductances.push_back(probe->getChild(name + "StomatalConductance"));
			} else{
				leafAreas.push_back(probe->getChild("leafArea"));
				conductances.push_back(probe->getChild("stomatalConductance"));
			}
		}//else ignore, this is not a plant
	}
}
void MeanStomatalConductance::calculate(const Time &t,double &meanSC){
	meanSC = 0.;
	double totalLeafArea = 0.;
	for (unsigned int i = 0; i < leafAreas.size(); i++){
		if (t >= plantingTimes[i]){
			double leafArea;
			leafAreas[i]->get(t, leafArea);
			totalLeafArea = totalLeafArea + leafArea;
			double stomatalConductance;
			conductances[i]->get(t, stomatalConductance);
			meanSC = meanSC + leafArea*stomatalConductance;
		}
	}
	if (totalLeafArea != 0.) meanSC = meanSC/totalLeafArea; // mol/(m^2*s)
	else meanSC = 0.04;
}
std::string MeanStomatalConductance::getName()const{
	return "meanStomatalConductance";
}

DerivativeBase * newInstantiationMeanStomatalConductance(SimulaDynamic* const pSD){
   return new MeanStomatalConductance(pSD);
}

//Register the module
class AutoRegisterTranspirationInstantiationFunctions {
public:
   AutoRegisterTranspirationInstantiationFunctions() {
 		BaseClassesMap::getDerivativeBaseClasses()["simplePotentialTranspiration"] = newInstantiationPotentialTranspirationCrop;//for backward compatibility
 		BaseClassesMap::getDerivativeBaseClasses()["simplePotentialTranspirationRate"] = newInstantiationSimplePotentialTranspirationRate;//for backward compatibility
 		BaseClassesMap::getDerivativeBaseClasses()["actualTranspiration"] = newInstantiationActualTranspiration;
		BaseClassesMap::getDerivativeBaseClasses()["potentialTranspirationCrop"] = newInstantiationPotentialTranspirationCrop;
		BaseClassesMap::getDerivativeBaseClasses()["stomatalConductance"] = newInstantiationStomatalConductance;
		BaseClassesMap::getDerivativeBaseClasses()["meanStomatalConductance"] = newInstantiationMeanStomatalConductance;
  };
};

// our one instance of the proxy
static AutoRegisterTranspirationInstantiationFunctions l654645648435135753;

