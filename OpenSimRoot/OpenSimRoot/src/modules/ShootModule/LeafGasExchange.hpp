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
#ifndef LEAFGASEXCHANGE_HPP_
#define LEAFGASEXCHANGE_HPP_

#include "../../engine/BaseClasses.hpp"

class MesophyllCO2Concentration:public DerivativeBase {
public:
	MesophyllCO2Concentration(SimulaDynamic* const pSD);
	std::string getName()const;
protected:
	void getDefaultValue(const Time &t, double &var);
	void calculate(const Time &t, double &concentration);
	SimulaBase *pPhotosynthesis, *pPhotosynthesisRate, *pStomatalConductance, *pLeafArea, *pLeafRespiration, *pLeafRespirationRate, *pLeafTemperature, *pSheathC, *pPEPCarboxylation;
	double atmosphericCO2Concentration, sheathConductanceAt25C, sheathConductanceActivationEnergy, sheathConductanceDeactivationEnergy, sheathConductanceEntropyTerm, dayRespirationMesophyllFraction;
	bool C4Photosynthesis;
	double cachedLeafTemperature, sheathConductance, cachedTime, leafArea;
};

class PEPCarboxylationRate:public DerivativeBase {
public:
	PEPCarboxylationRate(SimulaDynamic* const pSD);
	std::string getName()const;
protected:
	void calculate(const Time &t, double &pepCarboxylationRate);
	SimulaBase *pLeafTemperature, *pMesophyllC, *pStomatalConductance, *pMaxPEPCarboxylationAt25C;
	double pepCarboxylationActivationEnergy, pepCarboxylationDeactivationEnergy, pepCarboxylationEntropyTerm, pepRegeneration, michaelisPEPAt25C, michaelisPEPActivationEnergy, atmosphericCO2Concentration;
	double cachedTime, cachedLeafTemperature, referenceMaxPEPCarboxylation, temperatureScalingFactor, michaelisPEP;
};

class BundleSheathCO2Concentration:public DerivativeBase {
public:
	BundleSheathCO2Concentration(SimulaDynamic* const pSD);
	std::string getName()const;
protected:
	void getDefaultValue(const Time &t, double &var);
	void calculate(const Time &t, double &concentration);
	SimulaBase *pIrradiation, *pPhotosynthesis, *pPhotosynthesisRate, *pStomatalConductance, *pLeafArea, *pLeafRespiration, *pLeafRespirationRate, *pLeafTemperature, *pMesophyllC, *pPEPCarboxylation;
	double atmosphericCO2Concentration, sheathConductanceAt25C, sheathConductanceActivationEnergy, sheathConductanceDeactivationEnergy, sheathConductanceEntropyTerm, dayRespirationMesophyllFraction;
	double cachedLeafTemperature, sheathConductance, cachedTime, leafArea;
};

class CO2Leakage:public DerivativeBase {
public:
	CO2Leakage(SimulaDynamic* const pSD);
	std::string getName()const;
protected:
	void calculate(const Time &t, double &concentration);
	SimulaBase *pLeafTemperature, *pMesophyllC, *pSheathC;
	double sheathConductanceAt25C, sheathConductanceActivationEnergy, sheathConductanceDeactivationEnergy, sheathConductanceEntropyTerm;
};

class MesophyllO2Concentration:public DerivativeBase {
public:
	MesophyllO2Concentration(SimulaDynamic* const pSD);
	std::string getName()const;
protected:
	void getDefaultValue(const Time &t, double &var);
	void calculate(const Time &t, double &concentration);
	SimulaBase *pPhotosynthesis, *pPhotosynthesisRate, *pStomatalConductance, *pLeafArea, *pLeafRespiration, *pLeafRespirationRate, *pSheathO, *pLeafTemperature;
	double atmosphericO2Concentration, sheathConductanceAt25C, sheathConductanceActivationEnergy, sheathConductanceDeactivationEnergy, sheathConductanceEntropyTerm, dayRespirationMesophyllFraction;
	bool C4Photosynthesis;
	double cachedLeafTemperature, sheathConductance, cachedTime, leafArea;
};

class BundleSheathO2Concentration:public DerivativeBase {
public:
	BundleSheathO2Concentration(SimulaDynamic* const pSD);
	std::string getName()const;
protected:
	void getDefaultValue(const Time &t, double &var);
	void calculate(const Time &t, double &concentration);
	SimulaBase *pPhotosynthesis, *pPhotosynthesisRate, *pLeafArea, *pLeafRespiration, *pLeafRespirationRate, *pMesophyllO, *pLeafTemperature;
	double atmosphericO2Concentration, sheathConductanceAt25C, sheathConductanceActivationEnergy, sheathConductanceDeactivationEnergy, sheathConductanceEntropyTerm, dayRespirationMesophyllFraction;
	double cachedLeafTemperature, sheathConductance, cachedTime, leafArea;
};

class O2Leakage:public DerivativeBase {
public:
	O2Leakage(SimulaDynamic* const pSD);
	std::string getName()const;
protected:
	void calculate(const Time &t, double &concentration);
	SimulaBase *pLeafTemperature, *pMesophyllO, *pSheathO;
	double sheathConductanceAt25C, sheathConductanceActivationEnergy, sheathConductanceDeactivationEnergy, sheathConductanceEntropyTerm;
};

#endif /*LEAFGASEXCHANGE_HPP_*/
