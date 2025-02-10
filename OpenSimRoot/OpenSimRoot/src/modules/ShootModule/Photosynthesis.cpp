/*
Copyright © 2016, The Pennsylvania State University
All rights reserved.

Copyright © 2016 Forschungszentrum Jülich GmbH
All rights reserved.

Copyright © 2022 Ernst Schäfer, Ishan Ajmera
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
#include "Photosynthesis.hpp"
#include "../../cli/Messages.hpp"
#include "../../engine/Origin.hpp"
#include "../PlantType.hpp"
#include <math.h>

//lintul based shoot simulation
PhotosynthesisLintul::PhotosynthesisLintul(SimulaDynamic* pSD):DerivativeBase(pSD)
{
	//planting Time
	pSD->getParent(3)->getChild("plantingTime")->get(plantingTime);
	//simulators
	lightInterceptionSimulator=pSD->getSibling("lightInterception");
	//plant type
	std::string plantType;
	PLANTTYPE(plantType,pSD)
	SimulaBase * param(ORIGIN->getChild("rootTypeParameters")->getChild(plantType)->getChild("shoot"));
	//get Light Use Efficiency note, this can be carbon or drymatter based, in other words in default lintul version respiration costs are taken care off in the lue, however in the orginal sucros model this was not the case
	lightUseEfficiencySimulator=param->getChild("lightUseEfficiency");
	//get area per plant
	areaSimulator=param->getChild("areaPerPlant");
	//unit checks
	// unit of CO2 (most likely g)
	if(pSD->getUnit().element(1)!=lightUseEfficiencySimulator->getUnit().element(1))
		msg::error("PhotosynthesisLintul: CO2 units differ: photosyntesis in "+pSD->getUnit().element(1)+" while LUE in "+lightUseEfficiencySimulator->getUnit().element(1));
	//energy unit Mj or the like
	if(lightUseEfficiencySimulator->getUnit().element(2)!=lightInterceptionSimulator->getUnit().element(1))
		msg::error("PhotosynthesisLintul: energy units differ: LUE in "+lightUseEfficiencySimulator->getUnit().element(2)+" while lightInterception in "+lightInterceptionSimulator->getUnit().element(1));
	//surface area unit (most likely cm2)
	if(areaSimulator->getUnit()!=lightInterceptionSimulator->getUnit().element(2))
		msg::error("PhotosynthesisLintul: area units differ: areaPerPlant in "+areaSimulator->getUnit().name+", while light interception in "+lightInterceptionSimulator->getUnit().element(2));

	//stress
	stress=pSD->getParent(3)->existingChild("stressFactor");
	if(stress) {
		adjust=param->getChild("photosynthesisStressResponse");
		msg::warning("PhotosynthesisLintul: Including stress factor.");
	}
	//RCA
	SimulaBase* contr(ORIGIN->getChild("simulationControls")->existingChild("aerenchyma"));
	if(contr) contr=contr->getChild("includePhotosynthesisEffects");
	bool flag(false);
	if(contr) contr->get(flag);
	if(flag){
		rca=param->existingChild("aerenchymaPhotosynthesisMitigation");
		if(rca) msg::warning("PhotosynthesisLintul: including RCA mitigation factor");
	}else{
		rca=nullptr;
	}
}
void PhotosynthesisLintul::calculate(const Time &t, double &photosynthesis){
	//localTime
	Time localTime=t-plantingTime;
	//area per plant
	double areaPerPlant;
	areaSimulator->get(localTime,areaPerPlant);
	//LUE
	double LUE;
	lightUseEfficiencySimulator->get(localTime,LUE);
	//calculate Intercepted Photosynthetic Available Radiation
	double PARINT;//Mj/cm2/day
	lightInterceptionSimulator->get(t,PARINT);
	//calculate drymatter production rate. note LUE may or may not have been compensated for respiration (g/J/cm2/day)
	photosynthesis= LUE*PARINT*areaPerPlant;//(g/MJ)*(MJ/cm2/day)*cm2 = g/day

	//stress
	if(stress){
		double s;
		stress->get(t,s);
		double sa;
		adjust->get(s,sa);
		if(rca){
			double a;
			rca->get(localTime,a);
			//1-((1-sa)*(1-a))=sa+a-sa*a
			sa+=(a*(1-sa));
		}
		photosynthesis*=sa;
	}

	//check
	if (photosynthesis<0) msg::error("PhotosynthesisLintul: photosynthesis<0");
}
std::string PhotosynthesisLintul::getName()const{
	return "photosynthesisLintul";
}

DerivativeBase * newInstantiationPhotosynthesisLintul(SimulaDynamic* const pSD){
   return new PhotosynthesisLintul(pSD);
}


//lintul based shoot simulation
PhotosynthesisLintulV2::PhotosynthesisLintulV2(SimulaDynamic* pSD):DerivativeBase(pSD), conversionFactor(1)
{
	//planting Time
	pSD->getParent(3)->getChild("plantingTime")->get(plantingTime);
	//simulators
	lightInterceptionSimulator=pSD->getSibling("lightInterception");
	//plant type
	std::string plantType;
	PLANTTYPE(plantType,pSD)
	SimulaBase * param(ORIGIN->getChild("rootTypeParameters")->getChild(plantType)->getChild("shoot"));
	//get Light Use Efficiency note, this can be carbon or drymatter based, in other words in default lintul version respiration costs are taken care off in the lue, however in the orginal sucros model this was not the case
	lightUseEfficiencySimulator=param->getChild("lightUseEfficiency");
	//get area per plant
	areaSimulator=param->getChild("areaPerPlant");
	//unit checks
	// unit of CO2 (most likely g)
	if(pSD->getUnit().element(1)!=lightUseEfficiencySimulator->getUnit().element(1))
		msg::error("PhotosynthesisLintulV2: CO2 units differ: photosyntesis in "+pSD->getUnit().element(1)+" while LUE in "+lightUseEfficiencySimulator->getUnit().element(1));
	//energy unit Mj or the like
	if(lightUseEfficiencySimulator->getUnit().element(2)!=lightInterceptionSimulator->getUnit().element(1)){
		if (lightUseEfficiencySimulator->getUnit().element(2) == "J" && lightInterceptionSimulator->getUnit().element(1) == "W"){
			conversionFactor = conversionFactor*60.*60.*24.;
		} else if (lightUseEfficiencySimulator->getUnit().element(2) == "MJ" && lightInterceptionSimulator->getUnit().element(1) == "W"){
			conversionFactor = conversionFactor*60.*60.*24./1000000.;
		} else msg::error("PhotosynthesisLintulV2: energy units differ: LUE in "+lightUseEfficiencySimulator->getUnit().element(2)+" while lightInterception in "+lightInterceptionSimulator->getUnit().element(1));
	}
	if (lightUseEfficiencySimulator->getUnit().element(2) == "W") msg::error("PhotosynthesisLintulV2: Change energy unit of lightUseEfficiency from W to J");
	//surface area unit (most likely cm2)
	if (areaSimulator->getUnit() != lightInterceptionSimulator->getUnit().element(2)){
		if (areaSimulator->getUnit() == "cm2" && lightInterceptionSimulator->getUnit().element(2) == "m2"){
			conversionFactor = conversionFactor/10000.;
		} else if (areaSimulator->getUnit() == "m2" && lightInterceptionSimulator->getUnit().element(2) == "cm2"){
			conversionFactor = conversionFactor*10000.;
		} else msg::error("PhotosynthesisLintulV2: area units differ: areaPerPlant in "+areaSimulator->getUnit().name+", while light interception in "+lightInterceptionSimulator->getUnit().element(2));
	}
	//stress
	stress=pSD->getParent(3)->existingChild("stressFactor:impactOn:photosynthesis");
	if(!stress)	msg::warning("PhotosynthesisLintulV2: no stress impact factor found");
}
void PhotosynthesisLintulV2::calculate(const Time &t, double &photosynthesis){
	//localTime
	Time localTime=t-plantingTime;
	//area per plant
	double areaPerPlant;
	areaSimulator->get(localTime,areaPerPlant);
	//LUE
	double LUE;
	lightUseEfficiencySimulator->get(localTime,LUE);
	//calculate Intercepted Photosynthetic Available Radiation
	double PARINT;//Mj/cm2/day
	lightInterceptionSimulator->get(t,PARINT);
	//calculate drymatter production rate. note LUE may or may not have been compensated for respiration (g/J/cm2/day)
	photosynthesis= LUE*PARINT*areaPerPlant;//(g/MJ)*(MJ/cm2/day)*cm2 = g/day
	photosynthesis *= conversionFactor;
	//stress
	if(stress){
		double s;
		stress->get(t,s);
		photosynthesis*=s;
	}

	//check
	if (photosynthesis<0) msg::error("PhotosynthesisLintulV2: photosynthesis<0");
}
std::string PhotosynthesisLintulV2::getName()const{
	return "photosynthesisLintulV2";
}
DerivativeBase * newInstantiationPhotosynthesisLintulV2(SimulaDynamic* const pSD){
   return new PhotosynthesisLintulV2(pSD);
}

CarbonLimitedPhotosynthesisRate::CarbonLimitedPhotosynthesisRate(SimulaDynamic* pSD):DerivativeBase(pSD), pLeafNitrogenConcentration(nullptr), C4Photosynthesis(false), carboxylationDeactivationEnergy(-999), cachedTime(-9999), cachedLeafTemperature(-9999), nitrogenLimit(1e99){
	std::string name = pSD->getName().substr(0, 6); // name = sunlit or shaded
	std::string plantType;
	PLANTTYPE(plantType,pSD);
	SimulaBase *shootParameters = GETSHOOTPARAMETERS(plantType);
	SimulaBase *probe = shootParameters->existingChild("C4Photosynthesis");
	if (probe) probe->get(C4Photosynthesis);
	if (C4Photosynthesis){
		pInternalCO2Concentration = pSD->getSibling(name + "BundleSheathCO2Concentration", "umol/mol");
		pInternalO2Concentration = pSD->getSibling(name + "BundleSheathO2Concentration", "mmol/mol");
	} else{
		pInternalCO2Concentration = pSD->getSibling(name + "MesophyllCO2Concentration", "umol/mol");
		pInternalO2Concentration = pSD->getSibling(name + "MesophyllO2Concentration", "mmol/mol");
		probe = shootParameters->getChild("CO2CompensationPointWithoutDayRespirationAt25C", "umol/mol");
		probe->get(CO2CompensationPointNoDayRespirationref);
		probe = shootParameters->getChild("CO2CompensationPointActivationEnergy", "J/mol");
		probe->get(activationEnergyCO2CompensationPointNoDayRespiration);
	}
	pLeafTemperature = pSD->getSibling(name + "LeafTemperature");
	probe = shootParameters->getChild("RubiscoCO2MichaelisConstantAt25C", "umol/mol");
	probe->get(referenceMichaelisCO2);
	probe = shootParameters->getChild("RubiscoO2MichaelisConstantAt25C", "mmol/mol");
	probe->get(referenceMichaelisO2);
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
	if (C4Photosynthesis){
        probe = shootParameters->getChild("rubiscoSpecificityAt25C");
        probe->get(referenceRubiscoSpecificity);
        probe = shootParameters->getChild("rubiscoSpecificityActivationEnergy", "J/mol");
        probe->get(rubiscoSpecificityActivationEnergy);
	}
	probe = shootParameters->existingChild("maxCarboxylationNitrogenProportionalityConstant", "umol/umol/s");
	if (probe){
		pLeafNitrogenConcentration = pSD->getSibling("nitrate")->getChild(name + "LeafActualNutrientConcentration", "umol/cm2");
		probe->get(maxCarboxylationNitrogenProportionalityConstant);
	}
}

void CarbonLimitedPhotosynthesisRate::calculate(const Time &t, double &photosynthesis){
	double leafTemperature, internalCO2, internalO2;
	pLeafTemperature->get(t, leafTemperature);
	leafTemperature = leafTemperature + 273.15; // Need leaf temperature in Kelvin
	pInternalCO2Concentration->get(t,internalCO2);
	pInternalO2Concentration->get(t, internalO2);
	if (std::abs(t - cachedTime) > TIMEERROR){
		cachedTime = t;
		pReferenceMaxCarboxylationEfficiency->get(t, referenceMaxCarboxylationEfficiency);
		if (pLeafNitrogenConcentration){
			double leafNitrogen;
			pLeafNitrogenConcentration->get(t, leafNitrogen);
			leafNitrogen = leafNitrogen*10000.; // Convert from umol/cm2 to umol/m2
			nitrogenLimit = leafNitrogen*maxCarboxylationNitrogenProportionalityConstant;
		}
	}
	if (C4Photosynthesis){
		if (std::abs(leafTemperature - cachedLeafTemperature) > 0.01){
			cachedLeafTemperature = leafTemperature;
			double universalGasConstant = 8.3144598; // J/(mol*K)
			double refTemp = 298.15;
			michaelisCO2 = referenceMichaelisCO2 *exp((leafTemperature-refTemp)*activationEnergyCO2/(refTemp*universalGasConstant*leafTemperature));
			michaelisO2 = referenceMichaelisO2 *exp((leafTemperature-refTemp)*activationEnergyO2/(refTemp*universalGasConstant*leafTemperature));
			if (carboxylationDeactivationEnergy < 0){
				temperatureScalingFactor = exp((leafTemperature-refTemp)*carboxylationActivationEnergy/(leafTemperature*universalGasConstant*refTemp));
			} else{
				temperatureScalingFactor = exp((leafTemperature-refTemp)*carboxylationActivationEnergy/(refTemp*universalGasConstant*leafTemperature))*((1+exp((refTemp*carboxylationEntropyTerm-carboxylationDeactivationEnergy)/(refTemp*universalGasConstant)))/(1+exp((carboxylationEntropyTerm*leafTemperature-carboxylationDeactivationEnergy)/(universalGasConstant*leafTemperature))));
			}
			rubiscoSpecificityReciprocal = (1/(2*referenceRubiscoSpecificity))*exp((leafTemperature-refTemp)*rubiscoSpecificityActivationEnergy/(refTemp*universalGasConstant*leafTemperature));
		}
		double maxCarboxylationEfficiency = referenceMaxCarboxylationEfficiency*temperatureScalingFactor;
		if (maxCarboxylationEfficiency > nitrogenLimit) maxCarboxylationEfficiency = nitrogenLimit;
		photosynthesis = (internalCO2 - internalO2*1000*rubiscoSpecificityReciprocal)*maxCarboxylationEfficiency/(internalCO2 + michaelisCO2*(1 + internalO2/michaelisO2));
	} else{
		if (std::abs(leafTemperature - cachedLeafTemperature) > 0.01){
			cachedLeafTemperature = leafTemperature;
			double universalGasConstant = 8.3144598; // J/(mol*K)
			double refTemp = 298.15;
			michaelisCO2 = referenceMichaelisCO2 *exp((leafTemperature-refTemp)*activationEnergyCO2/(refTemp*universalGasConstant*leafTemperature));
			michaelisO2 = referenceMichaelisO2 *exp((leafTemperature-refTemp)*activationEnergyO2/(refTemp*universalGasConstant*leafTemperature));
			if (carboxylationDeactivationEnergy < 0){
				temperatureScalingFactor = exp((leafTemperature-refTemp)*carboxylationActivationEnergy/(leafTemperature*universalGasConstant*refTemp));
			} else{
				temperatureScalingFactor = exp((leafTemperature-refTemp)*carboxylationActivationEnergy/(refTemp*universalGasConstant*leafTemperature))*((1+exp((refTemp*carboxylationEntropyTerm-carboxylationDeactivationEnergy)/(refTemp*universalGasConstant)))/(1+exp((carboxylationEntropyTerm*leafTemperature-carboxylationDeactivationEnergy)/(universalGasConstant*leafTemperature))));
			}
			CO2CompensationPointNoDayRespiration = CO2CompensationPointNoDayRespirationref *exp((leafTemperature-refTemp)*activationEnergyCO2CompensationPointNoDayRespiration/(refTemp*universalGasConstant*leafTemperature));
		}
		double maxCarboxylationEfficiency = referenceMaxCarboxylationEfficiency*temperatureScalingFactor;
		if (maxCarboxylationEfficiency > nitrogenLimit) maxCarboxylationEfficiency = nitrogenLimit;
		photosynthesis = (internalCO2 - CO2CompensationPointNoDayRespiration)*maxCarboxylationEfficiency/(internalCO2 + michaelisCO2*(1 + internalO2/michaelisO2));
	}
	if(std::isnan(photosynthesis)) msg::error("CarbonLimitedPhotosynthesisRate: photosynthesis is NaN ");
	photosynthesis = std::max(photosynthesis, 0.);
}

std::string CarbonLimitedPhotosynthesisRate::getName()const{
	return "carbonLimitedPhotosynthesisRate";
}

DerivativeBase * newInstantiationCarbonLimitedPhotosynthesisRate(SimulaDynamic* const pSD){
   return new CarbonLimitedPhotosynthesisRate(pSD);
}

LightLimitedPhotosynthesisRate::LightLimitedPhotosynthesisRate(SimulaDynamic* pSD):DerivativeBase(pSD), pLeafNitrogenConcentration(nullptr), C4Photosynthesis(false), cachedTime(-9999), cachedLeafTemperature(-9999), nitrogenLimit(1e99){
	std::string name = pSD->getName().substr(0, 6); // name = sunlit or shaded
	std::string plantType;
	PLANTTYPE(plantType,pSD);
	SimulaBase *shootParameters = GETSHOOTPARAMETERS(plantType);
	SimulaBase *probe = shootParameters->existingChild("C4Photosynthesis");
	if (probe) probe->get(C4Photosynthesis);
	if (C4Photosynthesis){
		pInternalCO2Concentration = pSD->getSibling(name + "BundleSheathCO2Concentration", "umol/mol");
		pInternalO2Concentration = pSD->getSibling(name + "BundleSheathO2Concentration", "mmol/mol");
	} else{
		pInternalCO2Concentration = pSD->getSibling(name + "MesophyllCO2Concentration", "umol/mol");
		pInternalO2Concentration = pSD->getSibling(name + "MesophyllO2Concentration", "mmol/mol");
		probe = shootParameters->getChild("CO2CompensationPointWithoutDayRespirationAt25C", "umol/mol");
		probe->get(CO2CompensationPointNoDayRespirationref);
		probe = shootParameters->getChild("CO2CompensationPointActivationEnergy", "J/mol");
		probe->get(activationEnergyCO2CompensationPointNoDayRespiration);
	}
	double photoPeriod = 24;
	pPhotoPeriod = ORIGIN->existingPath("/environment/atmosphere/photoPeriod", "hour");
	if (pPhotoPeriod) pPhotoPeriod->get(photoPeriod);
	pIrradiation = pSD->getSibling(name + "LeafIrradiation");
//	pIrradiation=ORIGIN->getChild("environment")->getChild("atmosphere")->getChild("irradiation");
	Unit u = pIrradiation->getUnit();
	if (u == "umol/m2/s"){
		conversionFactor = photoPeriod;
		if (pPhotoPeriod) msg::error("LightLimitedPhotosynthesisRate: Incoming radiation is in units of /s and photoPeriod is given. Either change units or remove photoPeriod!");
	} else if (u == "umol/cm2/day") conversionFactor = 10000./(60.*60.);
	else if (u == "umol/m2/day") conversionFactor = 1./(60.*60.);
	else if (u == "umol/cm2/s"){
		conversionFactor = 10000.*photoPeriod;
		if (pPhotoPeriod) msg::error("LightLimitedPhotosynthesisRate: Incoming radiation is in units of /s and photoPeriod is given. Either change units or remove photoPeriod!");
	} else if (u == "W/m2"){
	// 1 W of solar radiation (over the whole spectrum, NOT just 400-700 nm) is equal to ~2.1 uMole/s, see https://www.researchgate.net/post/Can_I_convert_PAR_photo_active_radiation_value_of_micro_mole_M2_S_to_Solar_radiation_in_Watt_m22
		conversionFactor = photoPeriod*2.1;
		if (pPhotoPeriod) msg::error("LightLimitedPhotosynthesisRate: Incoming radiation is in units of /s and photoPeriod is given. Either change units or remove photoPeriod!");
	} else if (u == "W/cm2"){
		conversionFactor = 10000.*photoPeriod*2.1;
		if (pPhotoPeriod) msg::error("LightLimitedPhotosynthesisRate: Incoming radiation is in units of /s and photoPeriod is given. Either change units or remove photoPeriod!");
	} else if (u == "MJ/m2/day"){
		conversionFactor = 1000000.*2.1/(60.*60.);
	} else msg::error("LightLimitedPhotosynthesisRate: Unknown unit for " + name + "LeafIrradiation!");
	pLeafTemperature = pSD->getSibling(name + "LeafTemperature");
	probe = shootParameters->getChild("leafAbsorptance");
	probe->get(absorptance);
	probe = shootParameters->getChild("spectralQuality");
	probe->get(spectralQuality);
	pReferenceJmax = shootParameters->getChild("maximumElectronTransportRateAt25C", "umol/m2/s");
	probe = shootParameters->getChild("maximumElectronTransportRateActivationEnergy", "J/mol");
	probe->get(JmaxActivationEnergy);
	probe = shootParameters->getChild("maximumElectronTransportRateDeactivationEnergy", "J/mol");
	probe->get(JmaxDeactivationEnergy);
	probe = shootParameters->getChild("maximumElectronTransportRateEntropyTerm", "J/K/mol");
	probe->get(JmaxEntropyTerm);
	probe = shootParameters->getChild("irradianceCurvatureFactor");
	probe->get(irradianceCurvatureFactor);
	if (C4Photosynthesis){
        probe = shootParameters->getChild("rubiscoSpecificityAt25C");
        probe->get(referenceRubiscoSpecificity);
        probe = shootParameters->getChild("rubiscoSpecificityActivationEnergy", "J/mol");
        probe->get(rubiscoSpecificityActivationEnergy);
        probe = shootParameters->getChild("electronTransportPartitioningFactor");
        probe->get(electronTransportPartitioningFactor);
	}
	probe = shootParameters->existingChild("maxElectronTransportRateNitrogenProportionalityConstant", "umol/umol/s");
	if (probe){
		pLeafNitrogenConcentration = pSD->getSibling("nitrate")->getChild(name + "LeafActualNutrientConcentration", "umol/cm2");
		probe->get(JmaxNitrogenProportionalityConstant);
	}
}

void LightLimitedPhotosynthesisRate::calculate(const Time &t, double &photosynthesis){
	double leafTemperature, internalCO2, internalO2, II;
	pLeafTemperature->get(t, leafTemperature);
	leafTemperature = leafTemperature + 273.15; // Need leaf temperature in Kelvin
	pInternalCO2Concentration -> get(t,internalCO2);
	if (C4Photosynthesis){
		pInternalO2Concentration->get(t, internalO2);
	}
	pIrradiation->get(t, II);
	double photoPeriod = 24;
	if (pPhotoPeriod) pPhotoPeriod->get(t, photoPeriod);
	II = II*conversionFactor/photoPeriod; // convert to umol/(m^2*s)
	double I2 = II*absorptance*(1 - spectralQuality)/2; // eq 9.16
	if (std::abs(t - cachedTime) > TIMEERROR){
		cachedTime = t;
		pReferenceJmax->get(t, referenceJmax);
		if (pLeafNitrogenConcentration){
			double leafNitrogen;
			pLeafNitrogenConcentration->get(t, leafNitrogen);
			leafNitrogen = leafNitrogen*10000.; // Convert from umol/cm2 to umol/m2
			nitrogenLimit = leafNitrogen*JmaxNitrogenProportionalityConstant;
		}
	}
	if (C4Photosynthesis){
		if (std::abs(leafTemperature - cachedLeafTemperature) > 0.01){
			cachedLeafTemperature = leafTemperature;
			double universalGasConstant = 8.3144598; // J/(mol*K)
			double refTemp = 298.15;
			temperatureScalingFactor = exp((leafTemperature-refTemp)*JmaxActivationEnergy/(refTemp*universalGasConstant*leafTemperature))*((1+exp((refTemp*JmaxEntropyTerm-JmaxDeactivationEnergy)/(refTemp*universalGasConstant)))/(1+exp((JmaxEntropyTerm*leafTemperature-JmaxDeactivationEnergy)/(universalGasConstant*leafTemperature))));
			rubiscoSpecificityReciprocal = (1/(2*referenceRubiscoSpecificity))*exp((leafTemperature-refTemp)*rubiscoSpecificityActivationEnergy/(refTemp*universalGasConstant*leafTemperature));
		}
		double Jmax = referenceJmax*temperatureScalingFactor;
		if (Jmax > nitrogenLimit) Jmax = nitrogenLimit;
		double potETransportRate = (I2 + Jmax - sqrt((I2 + Jmax)*(I2 + Jmax) - 4*irradianceCurvatureFactor*I2*Jmax))/(2*irradianceCurvatureFactor); // potential electron transport rate, J in the paper
        photosynthesis = (internalCO2 - internalO2*1000*rubiscoSpecificityReciprocal)*(1 - electronTransportPartitioningFactor)*potETransportRate/(3*internalCO2 + 7*internalO2*1000*rubiscoSpecificityReciprocal);
	} else{
		if (std::abs(leafTemperature - cachedLeafTemperature) > 0.01){
			cachedLeafTemperature = leafTemperature;
			double universalGasConstant = 8.3144598; // J/(mol*K)
			double refTemp = 298.15;
			temperatureScalingFactor = exp((leafTemperature-refTemp)*JmaxActivationEnergy/(refTemp*universalGasConstant*leafTemperature))*((1+exp((refTemp*JmaxEntropyTerm-JmaxDeactivationEnergy)/(refTemp*universalGasConstant)))/(1+exp((JmaxEntropyTerm*leafTemperature-JmaxDeactivationEnergy)/(universalGasConstant*leafTemperature))));
			CO2CompensationPointNoDayRespiration = CO2CompensationPointNoDayRespirationref *exp((leafTemperature-refTemp)*activationEnergyCO2CompensationPointNoDayRespiration/(refTemp*universalGasConstant*leafTemperature));
		}
		double Jmax = referenceJmax*temperatureScalingFactor;
		if (Jmax > nitrogenLimit) Jmax = nitrogenLimit;
		double potETransportRate = (I2 + Jmax - sqrt((I2 + Jmax)*(I2 + Jmax) - 4*irradianceCurvatureFactor*I2*Jmax))/(2*irradianceCurvatureFactor); // potential electron transport rate, J in the paper
		photosynthesis = (internalCO2 - CO2CompensationPointNoDayRespiration)*potETransportRate/(4*internalCO2 + 8*CO2CompensationPointNoDayRespiration);
	}
	if(std::isnan(photosynthesis)) msg::error("LightLimitedPhotosynthesisRate: photosynthesis is NaN ");
	photosynthesis = std::max(photosynthesis, 0.);
}

std::string LightLimitedPhotosynthesisRate::getName()const{
	return "lightLimitedPhotosynthesisRate";
}

DerivativeBase * newInstantiationLightLimitedPhotosynthesisRate(SimulaDynamic* const pSD){
   return new LightLimitedPhotosynthesisRate(pSD);
}

PhosphorusLimitedPhotosynthesisRate::PhosphorusLimitedPhotosynthesisRate(SimulaDynamic* pSD):DerivativeBase(pSD){
}

void PhosphorusLimitedPhotosynthesisRate::calculate(const Time &t, double &photosynthesis){
	double TP = 10000000; // rate of inorganic phosphate supply to the chloroplast.
	//Should be able to calculate this from the plant phosphorus status.
	photosynthesis = 3*TP;
}

std::string PhosphorusLimitedPhotosynthesisRate::getName()const{
	return "phosphorusLimitedPhotosynthesisRate";
}

DerivativeBase * newInstantiationPhosphorusLimitedPhotosynthesisRate(SimulaDynamic* const pSD){
   return new PhosphorusLimitedPhotosynthesisRate(pSD);
}

PhotosynthesisFarquhar::PhotosynthesisFarquhar(SimulaDynamic* pSD):DerivativeBase(pSD){
	std::string name = pSD->getName().substr(0, 6); // name = sunlit or shaded
	// The 3 photosynthesis models we're taking the minimum of.
	if (name == "sunlit" || name == "shaded"){
		pPhotosynthesisC = pSD->getSibling(name + "CarbonLimitedPhotosynthesisRate", "umol/m2/s");
		pPhotosynthesisJ = pSD->getSibling(name + "LightLimitedPhotosynthesisRate", "umol/m2/s");
		pPhotosynthesisP = pSD->existingSibling(name + "PhosphorusLimitedPhotosynthesisRate", "umol/m2/s");
		pLeafArea = pSD->getSibling(name + "LeafArea", "cm2");
	} else{
		pPhotosynthesisC = pSD->getSibling("carbonLimitedPhotosynthesisRate", "umol/m2/s");
		pPhotosynthesisJ = pSD->getSibling("lightLimitedPhotosynthesisRate", "umol/m2/s");
		pPhotosynthesisP = pSD->existingSibling("phosphorusLimitedPhotosynthesisRate", "umol/m2/s");
		pLeafArea = pSD->getSibling("leafArea", "cm2");
	}
	std::string plantType;
	PLANTTYPE(plantType,pSD);
	SimulaBase *shootParameters = GETSHOOTPARAMETERS(plantType);
	SimulaBase *pointer = shootParameters->existingChild("relativeRespirationRateLeafs");
	if (pointer){
		double respRate;
		pointer->get(respRate);
		if (respRate > 0.){
			msg::error("PhotosynthesisFarquhar: The relative respiration rate of leaves is not equal to 0 but it should be when using the Farquhar model.");
		}
	}
	pPhotoPeriod = ORIGIN->existingPath("/environment/atmosphere/photoPeriod", "hour");
	if (pPhotoPeriod) msg::warning("PhotosynthesisFarquhar: Using photoPeriod to calculate daily photosynthesis production. Make sure irradiation values are given in units of something/s.");
	if (!pPhotoPeriod) msg::warning("PhotosynthesisFarquhar: photoPeriod not given, defaulting to 24 hours, so give as irradiation values, the actual amount received between sunrise and sunset, converted into appropriate units.");
}

void PhotosynthesisFarquhar::calculate(const Time &t, double &photosynthesis){
	double photoC, photoJ, leafArea;
	pLeafArea->get(t, leafArea);
	double photoPeriod = 24;
	if (pPhotoPeriod) pPhotoPeriod->get(t, photoPeriod);
	pPhotosynthesisC->get(t, photoC);
	pPhotosynthesisJ->get(t, photoJ);
	double photoP = 1e20;
	if (pPhotosynthesisP) pPhotosynthesisP->get(t, photoP);
	photosynthesis = std::min(photoC, std::min(photoJ, photoP)); // umol/m2/s
	photosynthesis = photosynthesis*60*60*photoPeriod*12.0107*leafArea/(1e6*1e4); // convert from umol/m2/s to g/(cm^2*day)
}

std::string PhotosynthesisFarquhar::getName()const{
	return "photosynthesisFarquhar";
}

DerivativeBase * newInstantiationPhotosynthesisFarquhar(SimulaDynamic* const pSD){
   return new PhotosynthesisFarquhar(pSD);
}

PhotosynthesisRateFarquhar::PhotosynthesisRateFarquhar(SimulaDynamic* pSD):DerivativeBase(pSD){
	std::string name = pSD->getName().substr(0, 6); // name = sunlit or shaded
	// The 3 photosynthesis models we're taking the minimum of.
	pPhotosynthesisC = pSD->getSibling(name + "CarbonLimitedPhotosynthesisRate", "umol/m2/s");
	pPhotosynthesisJ = pSD->getSibling(name + "LightLimitedPhotosynthesisRate", "umol/m2/s");
	pPhotosynthesisP = pSD->existingSibling(name + "PhosphorusLimitedPhotosynthesisRate", "umol/m2/s");
}

void PhotosynthesisRateFarquhar::calculate(const Time &t, double &photosynthesis){
	double photoC, photoJ;
	pPhotosynthesisC->get(t, photoC);
	pPhotosynthesisJ->get(t, photoJ);
	double photoP = 1e20;
	if (pPhotosynthesisP) pPhotosynthesisP->get(t, photoP);
	photosynthesis = std::min(photoC, std::min(photoJ, photoP)); // umol/m2/s
	photosynthesis = std::max(photosynthesis, 0.);
	if(std::isnan(photosynthesis)) msg::error("PhotoRateFarquhar: photosynthesis is NaN ");
}

void PhotosynthesisRateFarquhar::getDefaultValue(const Time & t, double &var){
	var = 10;
}

std::string PhotosynthesisRateFarquhar::getName()const{
	return "photosynthesisRateFarquhar";
}

DerivativeBase * newInstantiationPhotosynthesisRateFarquhar(SimulaDynamic* const pSD){
   return new PhotosynthesisRateFarquhar(pSD);
}

IntegratePhotosynthesisRate::IntegratePhotosynthesisRate(SimulaDynamic* pSD):DerivativeBase(pSD){
	std::string name = pSD->getName().substr(0, 6); // name = sunlit or shaded
	// The 3 photosynthesis models we're taking the minimum of.
	if (name == "sunlit" || name == "shaded"){
		pPhotosynthesis = pSD->getSibling(name + "PhotosynthesisRate", "umol/m2/s");
		pLeafArea = pSD->getSibling(name + "LeafArea", "cm2");
	} else{
		pPhotosynthesis = pSD->getSibling("photosynthesisRate", "umol/m2/s");
		pLeafArea = pSD->getSibling("leafArea", "cm2");
	}
	std::string plantType;
	PLANTTYPE(plantType,pSD);
	pPhotoPeriod = ORIGIN->existingPath("/environment/atmosphere/photoPeriod", "hour");
	if (pPhotoPeriod) msg::warning("IntegratePhotosynthesisRate: Photoperiod found, it is preferable if you to give irradiation in W/m2 and don't use photoperiod.");
}

void IntegratePhotosynthesisRate::calculate(const Time &t, double &photosynthesis){
	double leafArea;
	pLeafArea->get(t, leafArea);
	double photoPeriod = 24;
	if (pPhotoPeriod) pPhotoPeriod->get(t, photoPeriod);
	pPhotosynthesis->get(t, photosynthesis);
	photosynthesis = photosynthesis*60*60*photoPeriod*12.0107*leafArea/(1e6*1e4); // convert from umol/m2/s to g/day
}

std::string IntegratePhotosynthesisRate::getName()const{
	return "integratePhotosynthesisRate";
}

DerivativeBase * newInstantiationIntegratePhotosynthesisRate(SimulaDynamic* const pSD){
   return new IntegratePhotosynthesisRate(pSD);
}

LightInterception::LightInterception(SimulaDynamic* pSD):DerivativeBase(pSD)
{
	//simulators
	irradiationSimulator=ORIGIN->getChild("environment")->getChild("atmosphere")->getChild("irradiation");
	leafAreaIndexSimulator=pSD->getSibling("leafAreaIndex");
	//plant type
	std::string plantType;
	PLANTTYPE(plantType,pSD)
	//get extinction coefficient (KDF)
	KDF=GETSHOOTPARAMETERS(plantType)->getChild("extinctionCoefficient");
	//correction factor for converting RDD to PAR (NORMALLY 0.5)
	if(irradiationSimulator->existingSibling("PAR/RDD")){
		irradiationSimulator->getSibling("PAR/RDD")->get(RDDPAR);
	}else{
		RDDPAR=1;
	}
	//check if unit given in input file agrees with this function
	if(pSD->getUnit()!=irradiationSimulator->getUnit())
		msg::error("LightInterception: units differ - light inteception in "+pSD->getUnit().name+" while irradiation in "+irradiationSimulator->getUnit().name);
}
void LightInterception::calculate(const Time &t, double &PARINT){
	//get irradiation
	double RDD;
	irradiationSimulator->get(t,RDD);
	//calculate Photosynthetially avialable radiation (PAR), PAR is generally considered 1/2 RDD
	double PAR(RDDPAR*RDD);//KJ/cm2/day*1000=Mj/cm2/day
	//get leaf area index
	double LAI;
	leafAreaIndexSimulator->get(t,LAI);//cm2/cm2
	//calculate Intercepted Photosynthetic Available Radiation
	double k;
	KDF->get(t-pSD->getStartTime(),k);
	PARINT=PAR*(1-exp(-k*LAI));//Mj/cm2/day
}
std::string LightInterception::getName()const{
	return "lightInterception";
}

DerivativeBase * newInstantiationLightInterception(SimulaDynamic* const pSD){
   return new LightInterception(pSD);
}

LeafIrradiation::LeafIrradiation(SimulaDynamic* pSD):DerivativeBase(pSD), cachedTime(-10){
	pBeamIrradiation = ORIGIN->getPath("/environment/atmosphere/beamIrradiation", "W/m2");
	pDiffuseIrradiation = ORIGIN->getPath("/environment/atmosphere/diffuseIrradiation", "W/m2");
	pSolarElevationAngle = ORIGIN->getPath("/environment/atmosphere/sineSolarElevationAngle");
	pLeafAreaIndex = pSD->getSibling("leafAreaIndex");
	std::string plantType;
	PLANTTYPE(plantType,pSD);
	SimulaBase *shootParameters = GETSHOOTPARAMETERS(plantType);
	SimulaBase *probe = shootParameters->getChild("leafAbsorptance");
	probe->get(leafAbsorptanceTerm);
	leafAbsorptanceTerm = sqrt(leafAbsorptanceTerm); // sqrt(1 - sigma) in paper
	horizontalReflection = (1 - leafAbsorptanceTerm)/(1 + leafAbsorptanceTerm);
	diffuseReflection = 0.036;
}

void LeafIrradiation::calculate(const Time &t, double &totalIrradiation){
	if (std::abs(t - cachedTime) < TIMEERROR){
		totalIrradiation = cachedIrradiation;
		return;
	}
	double sinEl, beamPAR, diffusePAR, beamExtinction, diffuseExtinction(0.78), beamScatterExtinction, diffuseScatterExtinction, beamReflection, LAI;
	pSolarElevationAngle->get(t, sinEl);
	pLeafAreaIndex->get(t, LAI);
	if (sinEl <= 0 || LAI < 1e-4){
		totalIrradiation = 0;
		return;
	}
	pBeamIrradiation->get(t, beamPAR);
	pDiffuseIrradiation->get(t, diffusePAR);
	beamExtinction = 0.5/sinEl;
	beamScatterExtinction = beamExtinction*leafAbsorptanceTerm;
	diffuseScatterExtinction = diffuseExtinction*leafAbsorptanceTerm;
	beamReflection = 1 - exp(2*horizontalReflection*beamExtinction/(1 + beamExtinction));
	totalIrradiation = (1 - beamReflection)*beamPAR*(1 - exp(-beamScatterExtinction*LAI)) + (1 - diffuseReflection)*diffusePAR*(1 - exp(-diffuseScatterExtinction*LAI));
	totalIrradiation = totalIrradiation/LAI; // Transform to W/m2 of leaf area, which other models require
	cachedTime = t;
	cachedIrradiation = totalIrradiation;
}
std::string LeafIrradiation::getName()const{
	return "leafIrradiation";
}

DerivativeBase * newInstantiationLeafIrradiation(SimulaDynamic* const pSD){
   return new LeafIrradiation(pSD);
}

SunlitLeafIrradiation::SunlitLeafIrradiation(SimulaDynamic* pSD):DerivativeBase(pSD), cachedTime(-10){
	pBeamIrradiation = ORIGIN->getPath("/environment/atmosphere/beamIrradiation", "W/m2");
	pDiffuseIrradiation = ORIGIN->getPath("/environment/atmosphere/diffuseIrradiation", "W/m2");
	pSolarElevationAngle = ORIGIN->getPath("/environment/atmosphere/sineSolarElevationAngle");
	pLeafAreaIndex = pSD->getSibling("leafAreaIndex");
	pSunlitLeafAreaIndex = pSD->getSibling("sunlitLeafAreaIndex");
	pTotalLeafIrradiation = pSD->getSibling("leafIrradiation");
	std::string plantType;
	PLANTTYPE(plantType,pSD);
	SimulaBase *shootParameters = GETSHOOTPARAMETERS(plantType);
	SimulaBase *probe = shootParameters->getChild("leafAbsorptance");
	probe->get(leafAbsorptance);
	leafAbsorptanceTerm = sqrt(leafAbsorptance);
	horizontalReflection = (1 - leafAbsorptanceTerm)/(1 + leafAbsorptanceTerm);
	diffuseReflection = 0.036;
}

void SunlitLeafIrradiation::calculate(const Time &t, double &sunInterception){
	if (std::abs(t - cachedTime) < TIMEERROR){
		sunInterception = cachedSunlitInterception;
		return;
	}
	double beamPAR, diffusePAR, sinEl, beamExtinction, diffuseExtinction(0.78), beamScatterExtinction, diffuseScatterExtinction, beamReflection, LAI, sunlitLAI, totalIrradiation;
	pSolarElevationAngle->get(t, sinEl);
	pLeafAreaIndex->get(t, LAI);
	pSunlitLeafAreaIndex->get(t, sunlitLAI);
	if (sinEl <= 0 || LAI < 1e-4 || sunlitLAI < 1e-4){
		sunInterception = 0;
		return;
	}
	pBeamIrradiation->get(t, beamPAR);
	pDiffuseIrradiation->get(t, diffusePAR);
	beamExtinction = 0.5/sinEl;
	beamScatterExtinction = beamExtinction*leafAbsorptanceTerm;
	diffuseScatterExtinction = diffuseExtinction*leafAbsorptanceTerm;
	beamReflection = 1 - exp(2*horizontalReflection*beamExtinction/(1 + beamExtinction));
	double directInterception = beamPAR*leafAbsorptance*(1 - exp(-beamExtinction*LAI));
	double diffuseInterception = diffusePAR*(1 - diffuseReflection)*(1 - exp(-(diffuseScatterExtinction + beamExtinction)*LAI))*diffuseScatterExtinction/(diffuseScatterExtinction + beamExtinction);
	double scatterInterception = beamPAR*((1 - beamReflection)*(1 - exp(-(beamExtinction + beamScatterExtinction)*LAI))*beamScatterExtinction/(beamScatterExtinction + beamExtinction) - (1 - exp(-2*beamExtinction*LAI))*leafAbsorptance/2);
	sunInterception = directInterception + diffuseInterception + scatterInterception;
	// Check if sunlit interception is lower than total interception
	pTotalLeafIrradiation->get(t, totalIrradiation);
	if (sunInterception > totalIrradiation*LAI) sunInterception = totalIrradiation*LAI;
	// Have interception in W/m2 units
	sunInterception = sunInterception/sunlitLAI;
	cachedTime = t;
	cachedSunlitInterception = sunInterception;
}
std::string SunlitLeafIrradiation::getName()const{
	return "sunlitLeafIrradiation";
}

DerivativeBase * newInstantiationSunlitLeafIrradiation(SimulaDynamic* const pSD){
   return new SunlitLeafIrradiation(pSD);
}

ShadedLeafIrradiation::ShadedLeafIrradiation(SimulaDynamic* pSD):DerivativeBase(pSD), cachedTime(-10){
	pTotalLeafIrradiation = pSD->getSibling("leafIrradiation");
	pSunlitLeafIrradiation = pSD->getSibling("sunlitLeafIrradiation");
	pLeafAreaIndex = pSD->getSibling("leafAreaIndex");
	pSunlitLeafAreaIndex = pSD->getSibling("sunlitLeafAreaIndex");
	pShadedLeafAreaIndex = pSD->getSibling("shadedLeafAreaIndex");
}
void ShadedLeafIrradiation::calculate(const Time &t, double &shadeInterception){
	if (std::abs(t - cachedTime) < TIMEERROR){
		shadeInterception = cachedShadedLightInterception;
		return;
	}
	double total, sunlit, LAI, sunlitLAI, shadedLAI;
	pLeafAreaIndex->get(t, LAI);
	pShadedLeafAreaIndex->get(t, shadedLAI);
	if (LAI < 1e-4 || shadedLAI < 1e-4){
		shadeInterception = 0;
		return;
	}
	pTotalLeafIrradiation->get(t, total);
	pSunlitLeafIrradiation->get(t, sunlit);
	pSunlitLeafAreaIndex->get(t, sunlitLAI);
	shadeInterception = total*LAI - sunlit*sunlitLAI;
	shadeInterception = shadeInterception/shadedLAI;
	if (shadeInterception < 0){
		shadeInterception = 0;
		msg::warning("ShadedLeafIrradiation: Sunlit leaves light interception larger than total light interception");
	}
	cachedTime = t;
	cachedShadedLightInterception = shadeInterception;
}
std::string ShadedLeafIrradiation::getName()const{
	return "shadedLeafIrradiation";
}

DerivativeBase * newInstantiationShadedLeafIrradiation(SimulaDynamic* const pSD){
   return new ShadedLeafIrradiation(pSD);
}

MeanLightInterception::MeanLightInterception(SimulaDynamic* pSD):DerivativeBase(pSD){
	std::string name(pSD->getName().substr(4, 6));
	bool splitBySunStatus(false);
	if (name == "Sunlit" || name == "Shaded"){
		splitBySunStatus = true;
		name.at(0) = std::tolower(name.at(0));
	}
	SimulaBase::List plants;
	pSD->getParent()->getAllChildren(plants);
	for (auto & it:plants){
		SimulaBase *k = (it)->existingChild("plantingTime");
		if (k){
			if (splitBySunStatus){
				k = (it)->getChild("plantPosition")->getChild("shoot")->getChild(name + "LeafIrradiation");
				lightInterceptions.push_back(k);
				k = k->getSibling(name + "LeafArea");
				leafAreas.push_back(k);
			} else{
				k = (it)->getChild("plantPosition")->getChild("shoot")->getChild("lightInterception");
				lightInterceptions.push_back(k);
				k = k->getSibling("leafArea");
				leafAreas.push_back(k);
			}
		}
	}
}
void MeanLightInterception::calculate(const Time &t, double &meanInterception){
	meanInterception = 0;
	double totalLeafArea = 0;
	for (unsigned int i = 0; i < lightInterceptions.size(); i++){
		double interception, leafArea;
		lightInterceptions[i]->get(t, interception);
		leafAreas[i]->get(t, leafArea);
		meanInterception = meanInterception + interception*leafArea;
		totalLeafArea = totalLeafArea + leafArea;
	}
	if (totalLeafArea == 0){
		meanInterception = 0;
	} else{
		meanInterception = meanInterception/totalLeafArea;
	}
}
std::string MeanLightInterception::getName()const{
	return "meanLightInterception";
}

DerivativeBase * newInstantiationMeanLightInterception(SimulaDynamic* const pSD){
   return new MeanLightInterception(pSD);
}

//==================registration of the classes=================
class AutoRegisterPhotosynthesisInstantiationFunctions {
public:
   AutoRegisterPhotosynthesisInstantiationFunctions() {
		BaseClassesMap::getDerivativeBaseClasses()["photosynthesisLintul"] = newInstantiationPhotosynthesisLintul;
		BaseClassesMap::getDerivativeBaseClasses()["photosynthesisLintulV2"] = newInstantiationPhotosynthesisLintulV2;
		BaseClassesMap::getDerivativeBaseClasses()["carbonLimitedPhotosynthesisRate"] = newInstantiationCarbonLimitedPhotosynthesisRate;
		BaseClassesMap::getDerivativeBaseClasses()["lightLimitedPhotosynthesisRate"] = newInstantiationLightLimitedPhotosynthesisRate;
		BaseClassesMap::getDerivativeBaseClasses()["phosphorusLimitedPhotosynthesisRate"] = newInstantiationPhosphorusLimitedPhotosynthesisRate;
		BaseClassesMap::getDerivativeBaseClasses()["photosynthesisFarquhar"] = newInstantiationPhotosynthesisFarquhar;
		BaseClassesMap::getDerivativeBaseClasses()["photosynthesisRateFarquhar"] = newInstantiationPhotosynthesisRateFarquhar;
		BaseClassesMap::getDerivativeBaseClasses()["integratePhotosynthesisRate"] = newInstantiationIntegratePhotosynthesisRate;
		BaseClassesMap::getDerivativeBaseClasses()["lightInterception"] = newInstantiationLightInterception;
		BaseClassesMap::getDerivativeBaseClasses()["leafIrradiation"] = newInstantiationLeafIrradiation;
		BaseClassesMap::getDerivativeBaseClasses()["sunlitLeafIrradiation"] = newInstantiationSunlitLeafIrradiation;
		BaseClassesMap::getDerivativeBaseClasses()["shadedLeafIrradiation"] = newInstantiationShadedLeafIrradiation;
		BaseClassesMap::getDerivativeBaseClasses()["meanLightInterception"] = newInstantiationMeanLightInterception;
   };
};



// our one instance of the proxy
static AutoRegisterPhotosynthesisInstantiationFunctions p;


