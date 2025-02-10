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
#ifndef RATEFUNCTIONLIBRARY_HPP_
#define RATEFUNCTIONLIBRARY_HPP_

#include "../../engine/BaseClasses.hpp"

class LeafArea:public DerivativeBase {
public:
	LeafArea(SimulaDynamic* const pSV);
	std::string getName()const;
protected:
	void calculate(const Time &t, double&);
	SimulaBase *SLASimulator, *c2lSimulator, *CinDryWeight;
	Time plantingTime;
};
class SunlitLeafArea:public DerivativeBase {
public:
	SunlitLeafArea(SimulaDynamic* const pSV);
	std::string getName()const;
protected:
	void calculate(const Time &t, double&);
	SimulaBase *pSunlitLeafAreaIndex;
	double areaPerPlant, cachedTime, cachedSLAI;
};
class PotentialLeafArea:public DerivativeBase {
public:
	PotentialLeafArea(SimulaDynamic* const pSV);
	std::string getName()const;
protected:
	void calculate(const Time &t, double&);
	SimulaBase *LASimulator;
	Time plantingTime;
};

class StressAdjustedPotentialLeafArea:public DerivativeBase {
public:
	StressAdjustedPotentialLeafArea(SimulaDynamic* const pSV);
	std::string getName()const;
protected:
	void calculate(const Time &t, double&);
	SimulaBase *rgrCoefficient, *potential, *stress;
};
class LeafAreaIndex:public DerivativeBase {
public:
	LeafAreaIndex(SimulaDynamic* const pSD);
	std::string getName()const;
protected:
	void calculate(const Time &t, double &LAI);
	SimulaBase *leafAreaSimulator, *senescedLeafArea, *meanLeafAreaSimulator;
	Time plantingTime;
	double areaPerPlant;
};
class SunlitLeafAreaIndex:public DerivativeBase {
public:
	SunlitLeafAreaIndex(SimulaDynamic* const pSD);
	std::string getName()const;
protected:
	void calculate(const Time &t, double &LAI);
	SimulaBase *leafAreaIndex, *solarElevationAngle;
	double cachedTime, cachedSunlitLAI;
};
class ShadedLeafAreaIndex:public DerivativeBase {
public:
	ShadedLeafAreaIndex(SimulaDynamic* const pSD);
	std::string getName()const;
protected:
	void calculate(const Time &t, double &LAI);
	SimulaBase *leafAreaIndex, *sunlitLeafAreaIndex;
	double cachedTime, cachedShadedLAI;
};
class MeanLeafAreaIndex:public DerivativeBase {
public:
	MeanLeafAreaIndex(SimulaDynamic* const pSD);
	std::string getName()const;
protected:
	void calculate(const Time &t, double &LAI);
	double area, pa;
	SimulaBase* leafArea, *senescedLeafArea, *plantArea;
	bool plantsPlantedAtDifferentTimes, splitBySunStatus;
};
/// Simulates the ratio between actual and potential leaf area
class LeafAreaReductionCoefficient:public DerivativeBase {
public:
	LeafAreaReductionCoefficient(SimulaDynamic* const pSD);
	std::string getName()const;
protected:
	void calculate(const Time &t, double &LAI);
	SimulaBase *actual, *potential;
	double recoveryRate;
};

class CropHeight:public DerivativeBase {
public:
	CropHeight(SimulaDynamic* const pSD);
	std::string getName()const;
protected:
	void calculate(const Time &t, double &height);
	SimulaBase *pGrowthSpeed, *pStressMultiplier;
	bool pot;
};

class MaximumCanopyHeight:public DerivativeBase {
public:
	MaximumCanopyHeight(SimulaDynamic* const pSD);
	std::string getName()const;
protected:
	void calculate(const Time &t, double &height);
	std::vector <SimulaBase*> cropHeights;
	std::vector <double> plantingTimes;
};



#endif /*RATEFUNCTIONLIBRARY_HPP_*/
