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
#ifndef PHOTOSYNTHESIS_HPP_
#define PHOTOSYNTHESIS_HPP_

#include "../../engine/BaseClasses.hpp"

class PhotosynthesisLintul:public DerivativeBase {
public:
	PhotosynthesisLintul(SimulaDynamic* const pSD);
	std::string getName()const;
protected:
	void calculate(const Time &t, double &photosynthesis);
	SimulaBase *lightInterceptionSimulator, *lightUseEfficiencySimulator, *areaSimulator, *stress, *adjust, *rca;
	Time plantingTime;
};

class PhotosynthesisLintulV2:public DerivativeBase {
public:
	PhotosynthesisLintulV2(SimulaDynamic* const pSD);
	std::string getName()const;
protected:
	void calculate(const Time &t, double &photosynthesis);
	SimulaBase *lightInterceptionSimulator, *lightUseEfficiencySimulator, *areaSimulator, *stress;
	Time plantingTime;
	double conversionFactor;
};

class CarbonLimitedPhotosynthesisRate:public DerivativeBase {
public:
	CarbonLimitedPhotosynthesisRate(SimulaDynamic* const pSD);
	std::string getName()const;
protected:
	void calculate(const Time &t, double &photosynthesis);
	SimulaBase *pLeafTemperature, *pInternalCO2Concentration, *pInternalO2Concentration, *pReferenceMaxCarboxylationEfficiency, *pLeafNitrogenConcentration;
	bool C4Photosynthesis;
	double referenceMichaelisCO2, referenceMichaelisO2, activationEnergyCO2, activationEnergyO2, carboxylationActivationEnergy, carboxylationDeactivationEnergy, carboxylationEntropyTerm;
	double CO2CompensationPointNoDayRespirationref, activationEnergyCO2CompensationPointNoDayRespiration;
	double referenceRubiscoSpecificity, rubiscoSpecificityActivationEnergy;
	double cachedTime, cachedLeafTemperature, michaelisCO2, michaelisO2, referenceMaxCarboxylationEfficiency, temperatureScalingFactor, rubiscoSpecificityReciprocal, CO2CompensationPointNoDayRespiration;
	double maxCarboxylationNitrogenProportionalityConstant, nitrogenLimit;
};

class LightLimitedPhotosynthesisRate:public DerivativeBase {
public:
	LightLimitedPhotosynthesisRate(SimulaDynamic* const pSD);
	std::string getName()const;
protected:
	void calculate(const Time &t, double &photosynthesis);
	SimulaBase *pLeafTemperature, *pInternalCO2Concentration, *pInternalO2Concentration, *pIrradiation, *pPhotoPeriod, *pReferenceJmax, *pLeafNitrogenConcentration;
	double CO2CompensationPointNoDayRespirationref, activationEnergyCO2CompensationPointNoDayRespiration;
	double absorptance, spectralQuality, JmaxActivationEnergy, JmaxDeactivationEnergy, JmaxEntropyTerm, irradianceCurvatureFactor;
	double conversionFactor;
	bool C4Photosynthesis;
	double referenceRubiscoSpecificity, rubiscoSpecificityActivationEnergy, electronTransportPartitioningFactor;
	double cachedTime, cachedLeafTemperature, referenceJmax, temperatureScalingFactor, rubiscoSpecificityReciprocal, CO2CompensationPointNoDayRespiration;
	double JmaxNitrogenProportionalityConstant, nitrogenLimit;
};

class PhosphorusLimitedPhotosynthesisRate:public DerivativeBase {
public:
	PhosphorusLimitedPhotosynthesisRate(SimulaDynamic* const pSD);
	std::string getName()const;
protected:
	void calculate(const Time &t, double &photosynthesis);
};

class PhotosynthesisFarquhar:public DerivativeBase {
public:
	PhotosynthesisFarquhar(SimulaDynamic* const pSD);
	std::string getName()const;
protected:
	void calculate(const Time &t, double &photosynthesis);
	SimulaBase *pPhotosynthesisC, *pPhotosynthesisJ, *pPhotosynthesisP, *pLeafArea, *pPhotoPeriod;
};

class PhotosynthesisRateFarquhar:public DerivativeBase {
public:
	PhotosynthesisRateFarquhar(SimulaDynamic* const pSD);
	std::string getName()const;
protected:
	void calculate(const Time &t, double &photosynthesis);
	void getDefaultValue(const Time &t, double &var);
	bool getLoopStartingAbility();
	SimulaBase *pPhotosynthesisC, *pPhotosynthesisJ, *pPhotosynthesisP;
};

class IntegratePhotosynthesisRate:public DerivativeBase {
public:
	IntegratePhotosynthesisRate(SimulaDynamic* const pSD);
	std::string getName()const;
protected:
	void calculate(const Time &t, double &photosynthesis);
	SimulaBase *pPhotosynthesis, *pLeafArea, *pPhotoPeriod;
};

class LightInterception:public DerivativeBase {
public:
	LightInterception(SimulaDynamic* const pSD);
	std::string getName()const;
protected:
	void calculate(const Time &t, double &PARINT);
	SimulaBase *irradiationSimulator, *leafAreaIndexSimulator, *KDF;
	double RDDPAR;
};

class LeafIrradiation:public DerivativeBase {
public:
	LeafIrradiation(SimulaDynamic* const pSD);
	std::string getName()const;
protected:
	void calculate(const Time &t, double &PARINT);
	SimulaBase *pBeamIrradiation, *pDiffuseIrradiation, *pSolarElevationAngle, *pLeafAreaIndex;
	double leafAbsorptanceTerm, horizontalReflection, diffuseReflection, cachedTime, cachedIrradiation;
};

class SunlitLeafIrradiation:public DerivativeBase {
public:
	SunlitLeafIrradiation(SimulaDynamic* const pSD);
	std::string getName()const;
protected:
	void calculate(const Time &t, double &PARINT);
	SimulaBase *pBeamIrradiation, *pDiffuseIrradiation, *pSolarElevationAngle, *pSunlitLeafAreaIndex, *pLeafAreaIndex, *pTotalLeafIrradiation;
	double leafAbsorptance, leafAbsorptanceTerm, horizontalReflection, diffuseReflection;
	double cachedTime, cachedSunlitInterception;
};

class ShadedLeafIrradiation:public DerivativeBase {
public:
	ShadedLeafIrradiation(SimulaDynamic* const pSD);
	std::string getName()const;
protected:
	void calculate(const Time &t, double &PARINT);
	SimulaBase *pTotalLeafIrradiation, *pSunlitLeafIrradiation, *pLeafAreaIndex, *pSunlitLeafAreaIndex, *pShadedLeafAreaIndex;
	double cachedTime, cachedShadedLightInterception;
};

class MeanLightInterception:public DerivativeBase {
public:
	MeanLightInterception(SimulaDynamic* const pSD);
	std::string getName()const;
protected:
	void calculate(const Time &t, double &PARINT);
	std::vector<SimulaBase*> lightInterceptions;
	std::vector<SimulaBase*> leafAreas;
};

#endif /*PHOTOSYNTHESIS_HPP_*/
