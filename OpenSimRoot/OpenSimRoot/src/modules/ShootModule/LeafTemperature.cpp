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
#include "LeafTemperature.hpp"
#include "../../cli/Messages.hpp"
#include "../../engine/Origin.hpp"
#include "../PlantType.hpp"
#include <math.h>

LeafTemperature::LeafTemperature(SimulaDynamic* pSD):DerivativeBase(pSD), absorptance(0.85), IRAbsorptance(0.95), leafCharacteristicDimension(0.1), conversionFactor(1), windSpeed(5.), cacheTime(-10), cachedTranspirationRate(-10), gotTranspiration(false){
	std::string name = pSD->getName().substr(0, 6); // name = sunlit or shaded
	std::string plantType;
	PLANTTYPE(plantType,pSD);
	SimulaBase *shootParameters = GETSHOOTPARAMETERS(plantType);
	SimulaBase *probe = shootParameters->existingChild("leafAbsorptance");
	if (probe){
		probe->get(absorptance);
	}
	probe = shootParameters->existingChild("leafInfraredAbsorptance");
	if (probe){
		probe->get(IRAbsorptance);
	}
	probe = shootParameters->existingChild("leafWidth");
	if (probe){
		probe->checkUnit("m");
		probe->get(leafCharacteristicDimension);
	}
	pIrradiation = pSD->getSibling(name + "LeafIrradiation");
//	pIrradiation=ORIGIN->getChild("environment")->getChild("atmosphere")->getChild("irradiation");
	Unit u = pIrradiation->getUnit();
	if (u == "W/m2") conversionFactor = 1.;
	else if (u == "W/cm2") conversionFactor = 10000.;
	else if (u == "MJ/m2/day") conversionFactor = 1000000./(60.*60.*24.);
	else msg::error("LeafTemperature: Unknown unit for irradiation! W/m2 are the preferred units, make sure this is total W solar radiation, so the whole spectrum!");
	pAirTemperature = ORIGIN->getPath("/environment/atmosphere/averageDailyTemperature", "degreesC");
	pWindSpeed = pSD->existingPath("/environment/atmosphere/windSpeed", "m/s");
	pTranspirationRate = pSD->existingSibling(name + "PotentialTranspirationRate", "cm3/day");
	if (!pTranspirationRate) pTranspiration = pSD->getSibling(name + "PotentialTranspiration", "cm3");
	pLeafArea = pSD->getSibling(name + "LeafArea");
	pAirTemperature->get(0, cachedLeafTemperature);
}
void LeafTemperature::calculate(const Time &t, double &leafTemperature){
	// We adapt the leaf energy balance as described in Park S. Nobel - Physiochemical and Environmental Plant Physiology, Fourth Edition, 2009
	double transpirationRate;
	// Leaf temperature depends only on leaf area, air temperature, irradiation, wind speed and transpiration. Because of how SimulaVariables work, leaf area should only depend on time, which means leaf area, air temperature, irradiation and wind speed depend only on time. So we check if the transpiration rate is the same as what is stored in the cache, if this is true, we already know the leaf temperature and can return immediately. We also save if we have gotten the transpiration rate so we don't do it twice later.
	if (std::abs(t - cacheTime) < TIMEERROR){
		if (!pTranspirationRate){
			pTranspiration->getRate(t, transpirationRate);
		} else{
			pTranspirationRate->get(t, transpirationRate);
		}
		transpirationRate = transpirationRate/(leafArea*60*60*24*18.01528); // Convert from cm3/day to mol/m2/s
		gotTranspiration = true;
		if (std::abs(transpirationRate - cachedTranspirationRate) < 1e-6){
			leafTemperature = cachedLeafTemperature;
			return;
		} else{
			cachedTranspirationRate = transpirationRate;
		}
	}else {
		gotTranspiration = false;
		pLeafArea->get(t, leafArea);
		leafArea = leafArea*0.0001; // convert from cm2 to m2
		pAirTemperature->get(t, airTemperature);
	}
	if (leafArea < 1e-4){
		leafTemperature = airTemperature;
		cachedLeafTemperature = leafTemperature;
		return;
	}
	double stefanBoltzman = 5.670374419E-8; // W/m2/K4
	if (std::abs(t - cacheTime) > TIMEERROR){
		pIrradiation->get(t, irradiation);
		if (pWindSpeed) pWindSpeed->get(t, windSpeed);
		airThermalConductivity = 0.0243 + 0.00007*airTemperature; // W/m/K
		double airKinematicViscosity = 1.415e-5 + 0.09e-5*airTemperature; // m2/s
		heatOfVaporisationOfWater = 45060 - 42.5*airTemperature; // J/mol
		airTemperature = airTemperature + 273.15; // Convert to K
		double skyTemperature = airTemperature - 40; // Used to calculate infrared radiation coming from sky
		boundaryLayer = sqrt(leafCharacteristicDimension*airKinematicViscosity/windSpeed)/0.97; // m
		absorbedRadiation = absorptance*irradiation + IRAbsorptance*stefanBoltzman*(airTemperature*airTemperature*airTemperature*airTemperature + skyTemperature*skyTemperature*skyTemperature*skyTemperature); // W/m2
		airTemperature = airTemperature - 273.15; // convert back to degrees C
		cacheTime = t;
	}
	if (!gotTranspiration){
		if (!pTranspirationRate){
			pTranspiration->getRate(t, transpirationRate);
		} else{
			pTranspirationRate->get(t, transpirationRate);
		}
		transpirationRate = transpirationRate/(leafArea*60*60*24*18.01528); // Convert from cm3/day to mol/m2/s
		cachedTranspirationRate = transpirationRate;
	}
    double vaporisationHeatLoss = heatOfVaporisationOfWater*transpirationRate; // W/m2
    airTemperature = airTemperature + 273.15; // convert to K
    leafTemperature = airTemperature; // Leaf temperature also in K
	double heatConvectionFlow = 2*airThermalConductivity*(leafTemperature - airTemperature)/boundaryLayer; // W/m2
	double emittedRadiation = 2*stefanBoltzman*leafTemperature*leafTemperature*leafTemperature*leafTemperature; // W/m2
	double energyBalance = absorbedRadiation - emittedRadiation - heatConvectionFlow - vaporisationHeatLoss; // W/m2
	int counter = 0;
	// Using Newton-Raphson method to find leaf temperature for which the energy balance is zero
	while (fabs(energyBalance) > 0.1){
		leafTemperature = leafTemperature + 1e-3;
		heatConvectionFlow = 2*airThermalConductivity*(leafTemperature - airTemperature)/boundaryLayer; // W/m2
		emittedRadiation = 2*stefanBoltzman*leafTemperature*leafTemperature*leafTemperature*leafTemperature; // W/m2
		double tempEnergyBalance = absorbedRadiation - emittedRadiation - heatConvectionFlow - vaporisationHeatLoss; // W/m2
		leafTemperature = leafTemperature - 1e-3 - energyBalance*1e-3/(tempEnergyBalance - energyBalance);
		heatConvectionFlow = 2*airThermalConductivity*(leafTemperature - airTemperature)/boundaryLayer; // W/m2
		emittedRadiation = 2*stefanBoltzman*leafTemperature*leafTemperature*leafTemperature*leafTemperature; // W/m2
		energyBalance = absorbedRadiation - emittedRadiation - heatConvectionFlow - vaporisationHeatLoss; // W/m2
		counter = counter + 1;
		if (counter > 100){
			airTemperature = airTemperature - 273.15; // convert airTemperature back to degrees C
			leafTemperature = airTemperature; // Save leaf temperature in degrees C
			cachedLeafTemperature = leafTemperature;
			msg::warning("LeafTemperature: Not converging after 100 iterations, setting leaf temperature equal to air temperature.");
			return;
		}
	}
	if(std::isnan(leafTemperature)) msg::error("LeafTemperature: leaf temperature is NaN ");
	airTemperature = airTemperature - 273.15; // convert back to degrees C
	leafTemperature = leafTemperature - 273.15; // convert back to degrees C
	leafTemperature = std::max(leafTemperature, 0.);
	leafTemperature = std::min(leafTemperature, 100.); // Leaf can not be frozen or boiling
	cachedLeafTemperature = leafTemperature;
}

void LeafTemperature::getDefaultValue(const Time & t, double &var){
	pAirTemperature->get(t, var);
}

std::string LeafTemperature::getName()const{
	return "leafTemperature";
}

DerivativeBase * newInstantiationLeafTemperature(SimulaDynamic* const pSD){
   return new LeafTemperature(pSD);
}

//==================registration of the classes=================
class AutoRegisterLeafTemperatureInstantiationFunctions {
public:
   AutoRegisterLeafTemperatureInstantiationFunctions() {
		BaseClassesMap::getDerivativeBaseClasses()["leafTemperature"] = newInstantiationLeafTemperature;
   };
};



// our one instance of the proxy
static AutoRegisterLeafTemperatureInstantiationFunctions p3463573732;


