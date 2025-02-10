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
#include "LeafGasExchange.hpp"
#include "../../cli/Messages.hpp"
#include "../../engine/Origin.hpp"
#include "../PlantType.hpp"
#include <math.h>

MesophyllCO2Concentration::MesophyllCO2Concentration(SimulaDynamic* pSD):DerivativeBase(pSD), C4Photosynthesis(false), cachedLeafTemperature(-99999), cachedTime(-10){
	std::string name = pSD->getName().substr(0, 6); // name = sunlit or shaded
	SimulaBase *probe = ORIGIN->getPath("/environment/atmosphere/CO2Concentration", "umol/mol");
	probe->get(atmosphericCO2Concentration);
	pPhotosynthesisRate = pSD->existingSibling(name + "PhotosynthesisRate", "umol/m2/s");
	if (!pPhotosynthesisRate) pPhotosynthesis = pSD->getSibling(name + "Photosynthesis", "g");
	pStomatalConductance = pSD->getSibling(name + "StomatalConductance", "mol/m2/s");
	pLeafArea = pSD->getSibling(name + "LeafArea", "cm2");
	pLeafRespirationRate = pSD->existingSibling(name + "LeafRespirationRate", "g/day");
	if (!pLeafRespirationRate) pLeafRespiration = pSD->getSibling(name + "LeafRespiration", "g");
	pLeafTemperature = pSD->getSibling(name + "LeafTemperature", "degreesC");
	std::string plantType;
	PLANTTYPE(plantType,pSD);
	SimulaBase *shootParameters = GETSHOOTPARAMETERS(plantType);
	probe = shootParameters->existingChild("C4Photosynthesis");
	if (probe) probe->get(C4Photosynthesis);
	if (C4Photosynthesis){
		pSheathC = pSD->getSibling(name + "BundleSheathCO2Concentration", "umol/mol");
		pPEPCarboxylation = pSD->getSibling(name + "PEPCarboxylationRate", "umol/m2/s");
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
	}
}

void MesophyllCO2Concentration::calculate(const Time &t, double &mesophyllC){
	if (std::abs(t - cachedTime) > TIMEERROR){
		pLeafArea->get(t, leafArea);
		leafArea = leafArea/10000; // convert from cm2 to m2
		cachedTime = t;
	}
	if (leafArea < 1e-4){
		mesophyllC = atmosphericCO2Concentration;
		return;
	}
	if (C4Photosynthesis){
		double leafTemperature, leafRespiration, pepCarboxylation, conductance, sheathC;
		pLeafTemperature->get(t, leafTemperature);
		leafTemperature = leafTemperature + 273.15; // Need leaf temperature in Kelvin
		if (!pLeafRespirationRate){
			pLeafRespiration->getRate(t, leafRespiration);
		} else{
			pLeafRespirationRate->get(t, leafRespiration);
		}
		leafRespiration = leafRespiration*1000000/(12.0111*60*60*24*leafArea); // convert from g/day to umol/(m2s)
		pPEPCarboxylation->get(t, pepCarboxylation);
		pStomatalConductance->get(t, conductance);
		pSheathC->get(t, sheathC);
		double mesophyllRespiration = dayRespirationMesophyllFraction*leafRespiration;
		if (std::abs(leafTemperature - cachedLeafTemperature) > 0.01){
			cachedLeafTemperature = leafTemperature;
			double universalGasConstant = 8.3144598; // J/(mol*K)
			double refTemp = 298.15;
			sheathConductance = sheathConductanceAt25C*exp((leafTemperature - refTemp)*sheathConductanceActivationEnergy/(refTemp*universalGasConstant*leafTemperature))*(1 + exp((sheathConductanceEntropyTerm - sheathConductanceDeactivationEnergy/refTemp)/universalGasConstant))/(1 + exp((sheathConductanceEntropyTerm - sheathConductanceDeactivationEnergy/leafTemperature)/universalGasConstant));
			if (sheathConductance == 0.) sheathConductance = 1e-10;
		}
		mesophyllC = (conductance*atmosphericCO2Concentration/1.6 + sheathConductance*sheathC - pepCarboxylation + mesophyllRespiration)/(conductance/1.6 + sheathConductance);
	} else{
		double leafRespiration, photoRate, conductance;
		if (!pLeafRespirationRate){
			pLeafRespiration->getRate(t, leafRespiration);
		} else{
			pLeafRespirationRate->get(t, leafRespiration);
		}
		leafRespiration = leafRespiration*1000000/(12.0111*60*60*24*leafArea); // convert from g/day to umol/(m2s)
		if (!pPhotosynthesisRate){
			pPhotosynthesis->getRate(t, photoRate);
			photoRate = photoRate*1000000/(12.0111*60*60*24*leafArea); // convert from g/day to umol/(m2s)
		} else{
			pPhotosynthesisRate->get(t, photoRate);
		}
		pStomatalConductance->get(t, conductance);
		mesophyllC = atmosphericCO2Concentration - 1.6*(photoRate - leafRespiration)/conductance;
	}
	if(std::isnan(mesophyllC)) msg::error("mesophyllC: mesophyllC is NaN ");
	mesophyllC = std::max(mesophyllC, 0.);
	mesophyllC = std::min(mesophyllC, atmosphericCO2Concentration);
}

void MesophyllCO2Concentration::getDefaultValue(const Time &t, double &var){
	var = 0.75*atmosphericCO2Concentration;
}

std::string MesophyllCO2Concentration::getName()const{
	return "mesophyllCO2Concentration";
}

DerivativeBase * newInstantiationMesophyllCO2Concentration(SimulaDynamic* const pSD){
   return new MesophyllCO2Concentration(pSD);
}

PEPCarboxylationRate::PEPCarboxylationRate(SimulaDynamic* pSD):DerivativeBase(pSD), cachedTime(-9999), cachedLeafTemperature(-99999){
	std::string name = pSD->getName().substr(0, 6); // name = sunlit or shaded
	pLeafTemperature = pSD->getSibling(name + "LeafTemperature", "degreesC");
	SimulaBase *probe = ORIGIN->getPath("/environment/atmosphere/CO2Concentration", "umol/mol");
	pStomatalConductance = pSD->getSibling(name + "StomatalConductance", "mol/m2/s");
	probe->get(atmosphericCO2Concentration);
	pMesophyllC = pSD->getSibling(name + "MesophyllCO2Concentration", "umol/mol");
	std::string plantType;
	PLANTTYPE(plantType,pSD);
	SimulaBase *shootParameters = GETSHOOTPARAMETERS(plantType);
	pMaxPEPCarboxylationAt25C = shootParameters->getChild("maxPEPCarboxylationAt25C", "umol/m2/s");
	probe = shootParameters->getChild("PEPCarboxylationActivationEnergy", "J/mol");
	probe->get(pepCarboxylationActivationEnergy);
	probe = shootParameters->getChild("PEPCarboxylationDeactivationEnergy", "J/mol");
	probe->get(pepCarboxylationDeactivationEnergy);
	probe = shootParameters->getChild("PEPCarboxylationEntropyTerm", "J/K/mol");
	probe->get(pepCarboxylationEntropyTerm);
	probe = shootParameters->getChild("PEPRegeneration", "umol/m2/s");
	probe->get(pepRegeneration);
	probe = shootParameters->getChild("MichaelisPEPAt25C", "umol/mol");
	probe->get(michaelisPEPAt25C);
	probe = shootParameters->getChild("MichaelisPEPActivationEnergy");
	probe->get(michaelisPEPActivationEnergy);
}

void PEPCarboxylationRate::calculate(const Time &t, double &pepCarboxylationRate){
	double leafTemperature, mesophyllC, stomatalConductance;
	pLeafTemperature->get(t, leafTemperature);
	leafTemperature = leafTemperature + 273.15; // Need leaf temperature in Kelvin
	pMesophyllC->get(t, mesophyllC);
	pStomatalConductance->get(t, stomatalConductance);
	if (std::abs(t - cachedTime) > TIMEERROR){
		cachedTime = t;
		pMaxPEPCarboxylationAt25C->get(t, referenceMaxPEPCarboxylation);
	}
	if (std::abs(leafTemperature - cachedLeafTemperature) > 0.01){
		cachedLeafTemperature = leafTemperature;
		double refTemp = 298.15;
		double universalGasConstant = 8.3144598; // J/(mol*K)
		temperatureScalingFactor = exp((leafTemperature - refTemp)*pepCarboxylationActivationEnergy/(refTemp*universalGasConstant*leafTemperature))*(1 + exp((pepCarboxylationEntropyTerm - pepCarboxylationDeactivationEnergy/refTemp)/universalGasConstant))/(1 + exp((pepCarboxylationEntropyTerm - pepCarboxylationDeactivationEnergy/leafTemperature)/universalGasConstant));
		michaelisPEP = michaelisPEPAt25C*exp((leafTemperature - refTemp)*michaelisPEPActivationEnergy/(refTemp*universalGasConstant*leafTemperature));
	}
	double maxPEPCarboxylation = referenceMaxPEPCarboxylation*temperatureScalingFactor;
	pepCarboxylationRate = std::min(mesophyllC*maxPEPCarboxylation/(mesophyllC + michaelisPEP), pepRegeneration);
	// Cap PEP carboxylation rate to maximum possible flow from the atmosphere into the mesophyll cells
	pepCarboxylationRate = std::min(pepCarboxylationRate, stomatalConductance*atmosphericCO2Concentration/1.6);
	pepCarboxylationRate = std::max(pepCarboxylationRate, 0.);
	if(std::isnan(pepCarboxylationRate)) msg::error("pepCarboxylationRate: pepCarboxylationRate is NaN ");
}

std::string PEPCarboxylationRate::getName()const{
	return "PEPCarboxylationRate";
}

DerivativeBase * newInstantiationPEPCarboxylationRate(SimulaDynamic* const pSD){
   return new PEPCarboxylationRate(pSD);
}

BundleSheathCO2Concentration::BundleSheathCO2Concentration(SimulaDynamic* pSD):DerivativeBase(pSD), cachedLeafTemperature(-99999), cachedTime(-10){
	std::string name = pSD->getName().substr(0, 6); // name = sunlit or shaded
	pIrradiation = ORIGIN->getPath("/environment/atmosphere/irradiation");
	SimulaBase *probe = ORIGIN->getPath("/environment/atmosphere/CO2Concentration", "umol/mol");
	probe->get(atmosphericCO2Concentration);
	pPhotosynthesisRate = pSD->existingSibling(name + "PhotosynthesisRate", "umol/m2/s");
	if (!pPhotosynthesisRate) pPhotosynthesis = pSD->getSibling(name + "Photosynthesis", "g");
	pLeafArea = pSD->getSibling(name + "LeafArea", "cm2");
	pLeafRespirationRate = pSD->existingSibling(name + "LeafRespirationRate", "g/day");
	if (!pLeafRespirationRate) pLeafRespiration = pSD->getSibling(name + "LeafRespiration", "g");
	pLeafTemperature = pSD->getSibling(name + "LeafTemperature", "degreesC");
	pMesophyllC = pSD->getSibling(name + "MesophyllCO2Concentration", "umol/mol");
	pPEPCarboxylation = pSD->getSibling(name + "PEPCarboxylationRate", "umol/m2/s");
	std::string plantType;
	PLANTTYPE(plantType,pSD);
	SimulaBase *shootParameters = GETSHOOTPARAMETERS(plantType);
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
}

void BundleSheathCO2Concentration::calculate(const Time &t, double &sheathC){
	if (std::abs(t - cachedTime) > TIMEERROR){
		pLeafArea->get(t, leafArea);
		leafArea = leafArea/10000; // convert from cm2 to m2
		cachedTime = t;
	}
	if (leafArea < 1e-4){
		sheathC = atmosphericCO2Concentration;
		return;
	}
	double photoRate, leafRespiration, leafTemperature, mesophyllC, pepCarboxylation;
	if (!pPhotosynthesisRate){
		pPhotosynthesis->getRate(t, photoRate);
		photoRate = photoRate*1000000/(12.0111*60*60*24*leafArea); // convert from g/day to umol/(m2s)
	}
	if (!pLeafRespirationRate) pLeafRespiration->getRate(t, leafRespiration);
	pLeafTemperature->get(t, leafTemperature);
	leafTemperature = leafTemperature + 273.15; // Need leaf temperature in Kelvin
	if (pLeafRespirationRate) pLeafRespirationRate->get(t, leafRespiration);
	leafRespiration = leafRespiration*1000000/(12.0111*60*60*24*leafArea); // convert from g/day to umol/(m2s)
	pMesophyllC->get(t, mesophyllC);
	pPEPCarboxylation->get(t, pepCarboxylation);
	if (pPhotosynthesisRate) pPhotosynthesisRate->get(t, photoRate);
	double sheathRespiration = (1 - dayRespirationMesophyllFraction)*leafRespiration;
	if (std::abs(leafTemperature - cachedLeafTemperature) > 0.01){
		cachedLeafTemperature = leafTemperature;
		double refTemp = 298.15;
		double universalGasConstant = 8.3144598; // J/(mol*K)
		sheathConductance = sheathConductanceAt25C*exp((leafTemperature - refTemp)*sheathConductanceActivationEnergy/(refTemp*universalGasConstant*leafTemperature))*(1 + exp((sheathConductanceEntropyTerm - sheathConductanceDeactivationEnergy/refTemp)/universalGasConstant))/(1 + exp((sheathConductanceEntropyTerm - sheathConductanceDeactivationEnergy/leafTemperature)/universalGasConstant));
		if (sheathConductance == 0.) sheathConductance = 1e-10;
	}
	sheathC = (pepCarboxylation - photoRate + sheathRespiration)/sheathConductance + mesophyllC;
	if(std::isnan(sheathC)) msg::error("sheathC: sheathC is NaN ");
	sheathC = std::max(sheathC, 0.);
}

void BundleSheathCO2Concentration::getDefaultValue(const Time &t, double &var){
	var = atmosphericCO2Concentration;
}

std::string BundleSheathCO2Concentration::getName()const{
	return "bundleSheathCO2Concentration";
}

DerivativeBase * newInstantiationBundleSheathCO2Concentration(SimulaDynamic* const pSD){
   return new BundleSheathCO2Concentration(pSD);
}

CO2Leakage::CO2Leakage(SimulaDynamic* pSD):DerivativeBase(pSD){
	std::string name = pSD->getName().substr(0, 6); // name = sunlit or shaded
	pLeafTemperature = pSD->getSibling(name + "LeafTemperature", "degreesC");
	pMesophyllC = pSD->getSibling(name + "MesophyllCO2Concentration", "umol/mol");
	pSheathC = pSD->getSibling(name + "BundleSheathCO2Concentration", "umol/mol");
	std::string plantType;
	PLANTTYPE(plantType,pSD);
	SimulaBase *shootParameters = GETSHOOTPARAMETERS(plantType);
	SimulaBase *probe = shootParameters->getChild("bundleSheathConductanceAt25C", "mol/m2/s");
	probe->get(sheathConductanceAt25C);
	probe = shootParameters->getChild("bundleSheathConductanceActivationEnergy", "J/mol");
	probe->get(sheathConductanceActivationEnergy);
	probe = shootParameters->getChild("bundleSheathConductanceDeactivationEnergy", "J/mol");
	probe->get(sheathConductanceDeactivationEnergy);
	probe = shootParameters->getChild("bundleSheathConductanceEntropyTerm", "J/K/mol");
	probe->get(sheathConductanceEntropyTerm);
}

void CO2Leakage::calculate(const Time &t, double &leakage){
	double leafTemperature, mesophyllC, sheathC;
	pLeafTemperature->get(t, leafTemperature);
	pMesophyllC->get(t, mesophyllC);
	pSheathC->get(t, sheathC);
	leafTemperature = leafTemperature + 273.15; // Need leaf temperature in Kelvin
	double universalGasConstant = 8.3144598; // J/(mol*K)
	double refTemp = 298.15;
	double sheathConductance = sheathConductanceAt25C*exp((leafTemperature - refTemp)*sheathConductanceActivationEnergy/(refTemp*universalGasConstant*leafTemperature))*(1 + exp((sheathConductanceEntropyTerm - sheathConductanceDeactivationEnergy/refTemp)/universalGasConstant))/(1 + exp((sheathConductanceEntropyTerm - sheathConductanceDeactivationEnergy/leafTemperature)/universalGasConstant));
	leakage = sheathConductance*(sheathC - mesophyllC);
}

std::string CO2Leakage::getName()const{
	return "CO2Leakage";
}

DerivativeBase * newInstantiationCO2Leakage(SimulaDynamic* const pSD){
   return new CO2Leakage(pSD);
}

MesophyllO2Concentration::MesophyllO2Concentration(SimulaDynamic* pSD):DerivativeBase(pSD), C4Photosynthesis(false), cachedLeafTemperature(-99999), cachedTime(-10){
	std::string name = pSD->getName().substr(0, 6); // name = sunlit or shaded
	SimulaBase *probe = ORIGIN->getPath("/environment/atmosphere/O2Concentration", "mmol/mol");
	probe->get(atmosphericO2Concentration);
	std::string plantType;
	PLANTTYPE(plantType,pSD);
	SimulaBase *shootParameters = GETSHOOTPARAMETERS(plantType);
	probe = shootParameters->existingChild("C4Photosynthesis");
	if (probe) probe->get(C4Photosynthesis);
	if (C4Photosynthesis){
		pLeafTemperature = pSD->getSibling(name + "LeafTemperature", "degreesC");
		pSheathO = pSD->getSibling(name + "BundleSheathO2Concentration", "mmol/mol");
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
	} else{
		pPhotosynthesisRate = pSD->existingSibling(name + "PhotosynthesisRate", "umol/m2/s");
		if (!pPhotosynthesisRate) pPhotosynthesis = pSD->getSibling(name + "Photosynthesis", "g");
	}
	pStomatalConductance = pSD->getSibling(name + "StomatalConductance", "mol/m2/s");
	pLeafArea = pSD->getSibling(name + "LeafArea", "cm2");
	pLeafRespirationRate = pSD->existingSibling(name + "LeafRespirationRate", "g/day");
	if (!pLeafRespirationRate) pLeafRespiration = pSD->getSibling(name + "LeafRespiration", "g");
}

void MesophyllO2Concentration::calculate(const Time &t, double &mesophyllO){
	if (std::abs(t - cachedTime) > TIMEERROR){
		pLeafArea->get(t, leafArea);
		leafArea = leafArea/10000; // convert from cm2 to m2
		cachedTime = t;
	}
	if (leafArea < 1e-4){
		mesophyllO = atmosphericO2Concentration;
		return;
	}
	double conductanceOxygenWaterRatio = 0.8; // Haynes - CRC Handbook of Chemistry and Physics
	if (C4Photosynthesis){
		double leafRespiration, leafTemperature, conductance, sheathO;
		if (!pLeafRespirationRate) pLeafRespiration->getRate(t, leafRespiration);
		pLeafTemperature->get(t, leafTemperature);
		leafTemperature = leafTemperature + 273.15; // Need leaf temperature in Kelvin
		if (pLeafRespirationRate) pLeafRespirationRate->get(t, leafRespiration);
		leafRespiration = leafRespiration*1000/(12.0111*60*60*24*leafArea); // convert from g/day to mmol/(m2s)
		pStomatalConductance->get(t, conductance);
		pSheathO->get(t, sheathO);
		double mesophyllRespiration = dayRespirationMesophyllFraction*leafRespiration;
		if (std::abs(leafTemperature - cachedLeafTemperature) > 0.01){
			cachedLeafTemperature = leafTemperature;
			double universalGasConstant = 8.3144598; // J/(mol*K)
			double refTemp = 298.15;
			sheathConductance = sheathConductanceAt25C*exp((leafTemperature - refTemp)*sheathConductanceActivationEnergy/(refTemp*universalGasConstant*leafTemperature))*(1 + exp((sheathConductanceEntropyTerm - sheathConductanceDeactivationEnergy/refTemp)/universalGasConstant))/(1 + exp((sheathConductanceEntropyTerm - sheathConductanceDeactivationEnergy/leafTemperature)/universalGasConstant));
			if (sheathConductance == 0.) sheathConductance = 1e-10;
		}
		mesophyllO = (conductance*conductanceOxygenWaterRatio*atmosphericO2Concentration + 0.047*sheathConductance*sheathO - mesophyllRespiration)/(conductance*conductanceOxygenWaterRatio + 0.047*sheathConductance);
	} else{
		double photoRate, leafRespiration, conductance;
		if (!pLeafRespirationRate) pLeafRespiration->getRate(t, leafRespiration);
		if (!pPhotosynthesisRate){
			pPhotosynthesis->getRate(t, photoRate);
			photoRate = photoRate*1000000/(12.0111*60*60*24*leafArea); // convert from g/day to umol/(m2s)
		}
		if (pLeafRespirationRate) pLeafRespirationRate->get(t, leafRespiration);
		leafRespiration = leafRespiration*1000/(12.0111*60*60*24*leafArea); // convert from g/day to mmol/(m2s)
		if (pPhotosynthesisRate) pPhotosynthesisRate->get(t, photoRate);
		photoRate = photoRate*0.001; // convert from umol/m2/s to mmol/m2/s
		pStomatalConductance->get(t, conductance);
		mesophyllO = (photoRate - leafRespiration)/(conductance*conductanceOxygenWaterRatio) + atmosphericO2Concentration;
	}
	if(std::isnan(mesophyllO)) msg::error("mesophyllO: mesophyllO is NaN ");
	mesophyllO = std::max(mesophyllO, atmosphericO2Concentration);
}

void MesophyllO2Concentration::getDefaultValue(const Time &t, double &var){
	var = 1.005*atmosphericO2Concentration;
}

std::string MesophyllO2Concentration::getName()const{
	return "mesophyllO2Concentration";
}

DerivativeBase * newInstantiationMesophyllO2Concentration(SimulaDynamic* const pSD){
   return new MesophyllO2Concentration(pSD);
}

BundleSheathO2Concentration::BundleSheathO2Concentration(SimulaDynamic* pSD):DerivativeBase(pSD), cachedLeafTemperature(-99999), cachedTime(-10){
	std::string name = pSD->getName().substr(0, 6); // name = sunlit or shaded
	SimulaBase *probe = ORIGIN->getPath("/environment/atmosphere/O2Concentration", "mmol/mol");
	probe->get(atmosphericO2Concentration);
	pPhotosynthesisRate = pSD->existingSibling(name + "PhotosynthesisRate", "umol/m2/s");
	if (!pPhotosynthesisRate) pPhotosynthesis = pSD->getSibling(name + "Photosynthesis", "g");
	pLeafArea = pSD->getSibling(name + "LeafArea", "cm2");
	pLeafRespirationRate = pSD->existingSibling(name + "LeafRespirationRate", "g/day");
	if (!pLeafRespirationRate) pLeafRespiration = pSD->getSibling(name + "LeafRespiration", "g");
	pLeafTemperature = pSD->getSibling(name + "LeafTemperature", "degreesC");
	pMesophyllO = pSD->getSibling(name + "MesophyllO2Concentration", "mmol/mol");
	std::string plantType;
	PLANTTYPE(plantType,pSD);
	SimulaBase *shootParameters = GETSHOOTPARAMETERS(plantType);
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
}

void BundleSheathO2Concentration::calculate(const Time &t, double &sheathO){
	if (std::abs(t - cachedTime) > TIMEERROR){
		pLeafArea->get(t, leafArea);
		leafArea = leafArea/10000; // convert from cm2 to m2
		cachedTime = t;
	}
	if (leafArea < 1e-4){
		sheathO = atmosphericO2Concentration;
		return;
	}
	double leafTemperature, photoRate, leafRespiration, mesophyllO;
	if (!pPhotosynthesisRate){
		pPhotosynthesis->getRate(t, photoRate);
		photoRate = photoRate*1000000/(12.0111*60*60*24*leafArea); // convert from g/day to umol/(m2s)
	}
	if (!pLeafRespirationRate) pLeafRespiration->getRate(t, leafRespiration);
	pLeafTemperature->get(t, leafTemperature);
	leafTemperature = leafTemperature + 273.15; // Need leaf temperature in Kelvin
	if (pPhotosynthesisRate) pPhotosynthesisRate->get(t, photoRate);
	photoRate = photoRate*0.001; // convert from umol/m2/s to mmol/m2/s
	if (pLeafRespirationRate) pLeafRespirationRate->get(t, leafRespiration);
	leafRespiration = leafRespiration*1000/(12.0111*60*60*24*leafArea); // convert from g/day to mmol/(m2s)
	pMesophyllO->get(t, mesophyllO);
	double sheathRespiration = (1 - dayRespirationMesophyllFraction)*leafRespiration;
	if (std::abs(leafTemperature - cachedLeafTemperature) > 0.01){
		cachedLeafTemperature = leafTemperature;
		double universalGasConstant = 8.3144598; // J/(mol*K)
		double refTemp = 298.15;
		sheathConductance = sheathConductanceAt25C*exp((leafTemperature - refTemp)*sheathConductanceActivationEnergy/(refTemp*universalGasConstant*leafTemperature))*(1 + exp((sheathConductanceEntropyTerm - sheathConductanceDeactivationEnergy/refTemp)/universalGasConstant))/(1 + exp((sheathConductanceEntropyTerm - sheathConductanceDeactivationEnergy/leafTemperature)/universalGasConstant));
		if (sheathConductance == 0.) sheathConductance = 1e-10;
	}
	sheathO = (photoRate - sheathRespiration)/(0.047*sheathConductance) + mesophyllO;
	if(std::isnan(sheathO)) msg::error("sheathO: sheathO is NaN ");
	sheathO = std::max(sheathO, atmosphericO2Concentration);
}

void BundleSheathO2Concentration::getDefaultValue(const Time &t, double &var){
	var = 1.5*atmosphericO2Concentration;
}

std::string BundleSheathO2Concentration::getName()const{
	return "bundleSheathO2Concentration";
}

DerivativeBase * newInstantiationBundleSheathO2Concentration(SimulaDynamic* const pSD){
   return new BundleSheathO2Concentration(pSD);
}

O2Leakage::O2Leakage(SimulaDynamic* pSD):DerivativeBase(pSD){
	std::string name = pSD->getName().substr(0, 6); // name = sunlit or shaded
	pLeafTemperature = pSD->getSibling(name + "LeafTemperature", "degreesC");
	pMesophyllO = pSD->getSibling(name + "MesophyllO2Concentration", "mmol/mol");
	pSheathO = pSD->getSibling(name + "BundleSheathO2Concentration", "mmol/mol");
	std::string plantType;
	PLANTTYPE(plantType,pSD);
	SimulaBase *shootParameters = GETSHOOTPARAMETERS(plantType);
	SimulaBase *probe = shootParameters->getChild("bundleSheathConductanceAt25C", "mol/m2/s");
	probe->get(sheathConductanceAt25C);
	probe = shootParameters->getChild("bundleSheathConductanceActivationEnergy", "J/mol");
	probe->get(sheathConductanceActivationEnergy);
	probe = shootParameters->getChild("bundleSheathConductanceDeactivationEnergy", "J/mol");
	probe->get(sheathConductanceDeactivationEnergy);
	probe = shootParameters->getChild("bundleSheathConductanceEntropyTerm", "J/K/mol");
	probe->get(sheathConductanceEntropyTerm);
}

void O2Leakage::calculate(const Time &t, double &leakage){
	double leafTemperature, mesophyllO, sheathO;
	pLeafTemperature->get(t, leafTemperature);
	leafTemperature = leafTemperature + 273.15; // Need leaf temperature in Kelvin
	pMesophyllO->get(t, mesophyllO);
	pSheathO->get(t, sheathO);
	double refTemp = 298.15;
	double universalGasConstant = 8.3144598; // J/(mol*K)
	double sheathConductance = sheathConductanceAt25C*exp((leafTemperature - refTemp)*sheathConductanceActivationEnergy/(refTemp*universalGasConstant*leafTemperature))*(1 + exp((sheathConductanceEntropyTerm - sheathConductanceDeactivationEnergy/refTemp)/universalGasConstant))/(1 + exp((sheathConductanceEntropyTerm - sheathConductanceDeactivationEnergy/leafTemperature)/universalGasConstant));
	leakage = 0.047*sheathConductance*(sheathO - mesophyllO);
}

std::string O2Leakage::getName()const{
	return "O2Leakage";
}

DerivativeBase * newInstantiationO2Leakage(SimulaDynamic* const pSD){
   return new O2Leakage(pSD);
}

//==================registration of the classes=================
class AutoRegisterLeafGasExchangeInstantiationFunctions {
public:
   AutoRegisterLeafGasExchangeInstantiationFunctions() {
		BaseClassesMap::getDerivativeBaseClasses()["mesophyllCO2Concentration"] = newInstantiationMesophyllCO2Concentration;
		BaseClassesMap::getDerivativeBaseClasses()["PEPCarboxylationRate"] = newInstantiationPEPCarboxylationRate;
		BaseClassesMap::getDerivativeBaseClasses()["bundleSheathCO2Concentration"] = newInstantiationBundleSheathCO2Concentration;
		BaseClassesMap::getDerivativeBaseClasses()["CO2Leakage"] = newInstantiationCO2Leakage;
		BaseClassesMap::getDerivativeBaseClasses()["mesophyllO2Concentration"] = newInstantiationMesophyllO2Concentration;
		BaseClassesMap::getDerivativeBaseClasses()["bundleSheathO2Concentration"] = newInstantiationBundleSheathO2Concentration;
		BaseClassesMap::getDerivativeBaseClasses()["O2Leakage"] = newInstantiationO2Leakage;
   };
};



// our one instance of the proxy
static AutoRegisterLeafGasExchangeInstantiationFunctions p6824362572;


