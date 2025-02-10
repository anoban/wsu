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

#include "GrowthImpedance.hpp"
#include <cmath>
#include "../PlantType.hpp"
#ifndef M_PI
#define M_PI (3.14159265358979323846)
#endif


RootImpedanceFromBulkDensity::RootImpedanceFromBulkDensity(SimulaDynamic* pSD):DerivativeBase(pSD), inGrowthpoint(false){
	// Take water content first from any existing child, to allow user overrides
	// Otherwise read from SWMS, or failing that assume a constant
	pSoilWaterContent = pSD->existingChild("soilWaterContentForImpedance", "cm3/cm3");
	if(!pSoilWaterContent){
		pSoilWaterContent = pSD->getSibling("volumetricWaterContentAtTheRootSurface", "cm3/cm3");
	}
	if (!pBulkDensity) pBulkDensity = pSD->getPath("/environment/soil/bulkDensity", "g/cm3");
	if (pSD->getParent()->getName()=="growthpoint"){
		inGrowthpoint = true;
	} else{
		Coordinate rootNodePosition;
		pSD->getAbsolute(pSD->getStartTime(), rootNodePosition);
		pBulkDensity->get(rootNodePosition.y, bulkDensity);
	}
}

void RootImpedanceFromBulkDensity::calculate(const Time &t, double &imped){
	double water_content;
	pSoilWaterContent->get(t, water_content);
	if (inGrowthpoint){
		Coordinate pos;
		pSD->getAbsolute(t, pos);
		pBulkDensity->get(pos.y, bulkDensity);
	}
	// TODO make these into inputs
	// Fixed values from Vaz et al. 2011, 10.1016/j.geoderma.2011.07.016
	// Eqn form is "eqn 1" of their Table 2, formulated by Jakobsen and Dexter 1987
	// coefs for soil "LVAd", a sandy clay loam, from their Table 4.
	double intercept = 0.89;
	double bd_mult = 3.00;
	double water_mult = -15.98;

	bulkDensity = exp(intercept + bd_mult*bulkDensity + water_mult*water_content);
	imped = 0.1772*bulkDensity + 0.0134*bulkDensity*bulkDensity; // Fig 1 Pabin et al 1998, 10.1016/S0167-1987(98)00098-1

	if(imped < 0){
		imped = 0;
	}else if(imped > 1){
		imped = 1;
	}
}

std::string RootImpedanceFromBulkDensity::getName()const{
	return "rootImpedanceFromBulkDensity";
}

DerivativeBase * newInstantiationRootImpedanceFromBulkDensity(SimulaDynamic* const pSD){
   return new RootImpedanceFromBulkDensity(pSD);
}

SimulaBase* RootImpedanceFromBulkDensity::pBulkDensity = nullptr;

RootImpedanceGao::RootImpedanceGao(SimulaDynamic* pSD):DerivativeBase(pSD), inGrowthpoint(false)
{
	if (!pBulkDensity) pBulkDensity = pSD->getPath("/environment/soil/bulkDensity", "g/cm3");
	if (!pResidualWaterContent) pResidualWaterContent = pSD->getPath("/environment/soil/water/residualWaterContent", "100%");
	if (!pSaturatedWaterContent) pSaturatedWaterContent = pSD->getPath("/environment/soil/water/saturatedWaterContent", "100%");
	if (!pVoidRatio) pVoidRatio = pSD->existingPath("/environment/soil/voidRatio", "100%");

	// Precompute net stress and bulk density
	Coordinate soilbottom, soiltop;
	pSD->getPath("/environment/dimensions/minCorner")->get(soilbottom);
	pSD->getPath("/environment/dimensions/maxCorner")->get(soiltop);
	if (!precalculationsDone){
		precalculate_net_stress(cachedBulkDensity, cumulativeStress, soilbottom.y, soiltop.y);
	}
	if(cumulativeStress.empty() || cachedBulkDensity.empty()) {
		msg::error("RootImpedanceGao: Precalculation of soil weight cache failed");
	}
	if (pSD->getParent()->getName()=="growthpoint"){
		inGrowthpoint = true;
	} else{
		// If not in growthpoint the root node position will not change and we get soil variables just once. This assumes that bulk density, soil hydraulic properties etc do not change over time.
		Coordinate rootNodePosition;
		pSD->getAbsolute(pSD->getStartTime(), rootNodePosition);
		double posY = rootNodePosition.y;
		pResidualWaterContent->get(posY, wc_res);
		pSaturatedWaterContent->get(posY, wc_sat);
		if (posY > cachedBulkDensity.rbegin()->first || posY < cachedBulkDensity.begin()->first) {
			// outside soil grid => no density => no impedance
			bulkDensity = -1;
		} else{
			bulkDensity = cachedBulkDensity.lower_bound(posY)->second;
			cum_stressPower = cumulativeStress.lower_bound(posY)->second;
		}
		// if not set, will calculate at run time assuming particle density = 2.65.
		if (pVoidRatio){
			pVoidRatio->get(posY, voidRatio);
		}else{
			// back calculate from assumed particle density of 2.65 (= quartz)
			// TODO compute from sand/silt/clay instead?
			double porosity = 1 - (bulkDensity / 2.65);
			if (std::abs(porosity - wc_sat) > 0.1){
				msg::warning("ImpedanceGao: Soil physical and hydraulic parameters are inconsistent. Water content at saturation differs more than 10% from porosity implied by bulk density");
			}
			voidRatio = porosity/(1-porosity);
		}
		// Adjust bulk density to include effect of gravity:
		// (BULK g / cm3) * (1e6 cm3 / m3) * (1 kg / 1e3 g) * (GRAVITY N / kg)
		// = BULK*GRAVITY * 1e3 N / m3
		// = result is in kN / m3
		double gravity_newtons_per_kg = 9.81;
		bulkDensity = bulkDensity*gravity_newtons_per_kg;
	}
	// Used to override SWMS values. You probably want to leave this unset most of the time,
	// in which case calculate() will look up water content from the soil
	pSoilWaterContent = pSD->existingChild("soilWaterContentForImpedance", "cm3/cm3");
	if (!pSoilWaterContent){
		pSoilWaterContent = pSD->getSibling("volumetricWaterContentAtTheRootSurface", "cm3/cm3");
		pSoilHydraulicHead = pSD->getSibling("hydraulicHeadAtRootSurface", "cm"); // cm of water pressure = hPa
	} else{
		pSoilHydraulicHead = pSD->existingChild("soilHydraulicHeadForImpedance", "cm");
	}
}

void RootImpedanceGao::calculate(const Time &t, double &imped){
	if (inGrowthpoint){
		Coordinate pos;
		pSD->getAbsolute(t, pos);
		double posY = pos.y;
		pResidualWaterContent->get(posY, wc_res);
		pSaturatedWaterContent->get(posY, wc_sat);
		if (posY > cachedBulkDensity.rbegin()->first || posY < cachedBulkDensity.begin()->first) {
			// outside soil grid => no density => no impedance
			imped = 0;
			return;
		} else{
			bulkDensity = cachedBulkDensity.lower_bound(posY)->second;
			cum_stressPower = cumulativeStress.lower_bound(posY)->second;
		}
		// if not set, will calculate at run time assuming particle density = 2.65.
		if (pVoidRatio){
			pVoidRatio->get(posY, voidRatio);
		}else{
			// back calculate from assumed particle density of 2.65 (= quartz)
			// TODO compute from sand/silt/clay instead?
			double porosity = 1 - (bulkDensity / 2.65);
			if (std::abs(porosity - wc_sat) > 0.1){
				msg::warning("ImpedanceGao: Soil physical and hydraulic parameters are inconsistent. Water content at saturation differs more than 10% from porosity implied by bulk density");
			}
			voidRatio = porosity/(1-porosity);
		}
		// Adjust bulk density to include effect of gravity:
		// (BULK g / cm3) * (1e6 cm3 / m3) * (1 kg / 1e3 g) * (GRAVITY N / kg)
		// = BULK*GRAVITY * 1e3 N / m3
		// = result is in kN / m3
		double gravity_newtons_per_kg = 9.81;
		bulkDensity = bulkDensity*gravity_newtons_per_kg;
	}
	if (bulkDensity <= 0) {
		imped = 0;
		return;
	}
	// fitted empirical params from Gao et al 2016 table 3
	constexpr double F(3.560), f(0.1846);
	double wc; // water content in cm3/cm3
	pSoilWaterContent->get(t, wc);
	double Sstar = std::max(0.5, (wc-wc_res)/(wc_sat-wc_res));
	double psi;
	pSoilHydraulicHead->get(t, psi);
	//Gao et al 2016 eqn 4
	double temp = (F - voidRatio)*(F - voidRatio)/(1 + voidRatio)*std::pow(cum_stressPower - (psi)*Sstar, f);
	imped = bulkDensity*temp*temp;
	if(!std::isnormal(imped)){
		msg::error("Numerical problem: imped = " + std::to_string(imped) + " in " + pSD->getPath() + " bulkDensity = " + std::to_string(bulkDensity) + " temp = " + std::to_string(temp) + " voidRatio = " + std::to_string(voidRatio) + " cum_stressPower = " + std::to_string(cum_stressPower) + " psi = " + std::to_string(psi) + " Sstar = " + std::to_string(Sstar));
	}
}

// Compute local caches of bulk density and net stress at desired vertical resolution.
// N.B. assumes bulkDensity at a given depth is constant through the whole simulation,
// so that computing it once at ImpedanceGao instantiation time saves many repeated calculations.
// But if bulkDensity *can* change over time, do not use this function.
void RootImpedanceGao::precalculate_net_stress(std::map<double, double> &bulkd_cache, std::map<double, double> &stress_cache, const double bottom_depth, const double top_depth) {
	double layer_stress(0.0), cum_stress(0.0), stepSize(0.1);
	// fitted empirical param from Gao et al 2016 table 3
	double p(2.1931);
	for(double d = top_depth; d > bottom_depth; d -= stepSize) {
		pBulkDensity->get(d, layer_stress);
		bulkd_cache.insert({d, layer_stress});
		cum_stress += (layer_stress * stepSize * 9.81 / 100);
		double cum_stressPower = std::pow(cum_stress, p);
		stress_cache.insert({d, cum_stressPower});
	}
	precalculationsDone = true;
}

// Static member initialization
SimulaBase* RootImpedanceGao::pBulkDensity = nullptr;
SimulaBase* RootImpedanceGao::pResidualWaterContent = nullptr;
SimulaBase* RootImpedanceGao::pSaturatedWaterContent = nullptr;
SimulaBase* RootImpedanceGao::pVoidRatio = nullptr;
bool RootImpedanceGao::precalculationsDone = false;
std::map<double, double> RootImpedanceGao::cumulativeStress = {};
std::map<double, double> RootImpedanceGao::cachedBulkDensity = {};


std::string RootImpedanceGao::getName()const{
	return "rootImpedanceGao";
}

DerivativeBase * newInstantiationRootImpedanceGao(SimulaDynamic* const pSD){
	return new RootImpedanceGao(pSD);
}

RootImpedanceWhalley::RootImpedanceWhalley(SimulaDynamic* pSD):DerivativeBase(pSD), inGrowthpoint(false)
{
	if (!pBulkDensity) pBulkDensity = pSD->getPath("/environment/soil/bulkDensity", "g/cm3");
	if (!pResidualWaterContent) pResidualWaterContent = pSD->getPath("/environment/soil/water/residualWaterContent", "100%");
	if (!pSaturatedWaterContent) pSaturatedWaterContent = pSD->getPath("/environment/soil/water/saturatedWaterContent", "100%");
	if (pSD->getParent()->getName()=="growthpoint"){
		inGrowthpoint = true;
	} else{
		Coordinate rootNodePosition;
		pSD->getAbsolute(pSD->getStartTime(), rootNodePosition);
		double bulkDensity;
		pBulkDensity->get(rootNodePosition.y, bulkDensity);
		bulkDensityFactor = std::pow(10, 0.93*bulkDensity + 1.26);
		pResidualWaterContent->get(rootNodePosition.y, wc_res);
		pSaturatedWaterContent->get(rootNodePosition.y, wc_sat);
	}
	// Take water content first from any existing child, to allow user overrides
	// Otherwise read from SWMS, or failing that assume a constant
	pSoilWaterContent = pSD->existingChild("soilWaterContentForImpedance", "cm3/cm3");
	if(!pSoilWaterContent){
		pSoilWaterContent = pSD->getSibling("volumetricWaterContentAtTheRootSurface", "cm3/cm3");
		pSoilHydraulicHead = pSD->getSibling("hydraulicHeadAtRootSurface", "cm");
	} else{
		pSoilHydraulicHead = pSD->existingChild("soilHydraulicHeadForImpedance", "cm");
	}
}

void RootImpedanceWhalley::calculate(const Time &t, double &imped){
	if (inGrowthpoint){
		Coordinate pos;
		pSD->getAbsolute(t, pos);
		double posY = pos.y;
		double bulkDensity;
		pBulkDensity->get(posY, bulkDensity);
		bulkDensityFactor = std::pow(10, 0.93*bulkDensity + 1.26);
		pResidualWaterContent->get(posY, wc_res);
		pSaturatedWaterContent->get(posY, wc_sat);
	}
	double wc, psi;
	pSoilWaterContent->get(t, wc);
	double Se = (wc-wc_res)/(wc_sat-wc_res);
	pSoilHydraulicHead->get(t, psi);
	// Whalley 2007 eqn 8
	// log10 Q = 0.35 * log10(|psi| * Se) + 0.93 * rho + 1.26
	// => Q = 10 ^ (0.35 * log10(|psi| * Se) + 0.93 * rho + 1.26)
	// => Q = 10^(0.35 * log10(|psi| * Se)) * 10^(0.93 * rho + 1.26)
	// => Q = (|psi| * Se)^0.35 * 10^(0.93 * rho + 1.26)
	imped = std::pow(std::fabs(psi)*Se, 0.35)*bulkDensityFactor;

	// Empirical correction for back-transformation bias (Whalley eqn 15)
	// they derive this as the mean of antilogs of the residuals, 1/n*sum(10^(Q-Qhat))
	// but say 1.097 should be "widely applicable to soils similar to the ones studied here"
	imped *= 1.097;
}

std::string RootImpedanceWhalley::getName()const{
	return "rootImpedanceWhalley";
}

DerivativeBase * newInstantiationRootImpedanceWhalley(SimulaDynamic* const pSD){
	return new RootImpedanceWhalley(pSD);
}

// Static member initialization
SimulaBase* RootImpedanceWhalley::pBulkDensity = nullptr;
SimulaBase* RootImpedanceWhalley::pResidualWaterContent = nullptr;
SimulaBase* RootImpedanceWhalley::pSaturatedWaterContent = nullptr;

// The one static instance of impedanceCalculator.
// Will be instantiated on first use (but deallocated never, yeah?)
//
// TODO separate pointer for RootGrowthDirectionImpedance is basically just to
// avoid writing out `RootGrowthImpedanceRateMultiplier::impedanceCalculator`
// every time I reference it inside RootGrowthDirectionImpedance.
// This is silly and should be fixed by giving these classes an inheritance
// relationship instead of just friendship.
SimulaBase* RootGrowthImpedanceRateMultiplier::impedanceCalculator = nullptr;
SimulaBase* RootGrowthDirectionImpedance::impedanceCalculator = nullptr;

RootGrowthImpedanceRateMultiplier::RootGrowthImpedanceRateMultiplier(SimulaDynamic* pSD):DerivativeBase(pSD){
	fastImpedanceCalculator = pSD->existingSibling("soilPenetrationResistance", "kPa");
	if(!fastImpedanceCalculator && !impedanceCalculator){
		impedanceCalculator = ORIGIN->getPath("/environment/soil/soilPenetrationResistance", "kPa");
	}

	std::string plantType;
	PLANTTYPE(plantType, pSD);
	std::string rootType;
	int pos = 3;
	if (pSD->getParent()->getName()=="growthpoint") pos = 2;
	pSD->getParent(pos)->getChild("rootType")->get(rootType);
	SimulaBase* p(GETROOTPARAMETERS(plantType, rootType));
	halfGrowthImpedance = p->existingChild("soilImpedanceFor50PercentGrowthSlowdown", "kPa");
	if (!halfGrowthImpedance) {
		halfGrowthImpedance = pSD->getChild("soilImpedanceFor50PercentGrowthSlowdown", "kPa");
	}
	bioPore = pSD->existingSibling("inBiopore");
}

void RootGrowthImpedanceRateMultiplier::calculate(const Time &t, double &var){
	var = 1;
	if (bioPore){
		double inPore;
		bioPore->get(t, inPore);
		if (inPore == 1){
			return;
		}
	}

	// This is a Michaelis-Menten curve whose Km is interpretable as
	// "impedance where growth is reduced to 50% of unimpeded growth"
	// Default Km of 2000 kPa is a guessed average from the few studies
	// I could find, but I did no formal metaanalysis. Don't trust it too much.
	if (fastImpedanceCalculator){
		fastImpedanceCalculator->get(t, var);
	} else{
		impedanceCalculator->get(pSD, t, var);
	}
	double km = 2000;
	halfGrowthImpedance->get(t, km);
	var = 1.0 - (var/(km+var));
	if (var < 0 || var > 1 || !std::isnormal(var)) {
		msg::error("Numerical problem in RootGrowthImpedanceRateMultiplier: result is "
			+ std::to_string(var) + "but should in [0, 1]");
	}
}

std::string RootGrowthImpedanceRateMultiplier::getName() const{
	return "rootGrowthImpedanceRateMultiplier";
}

DerivativeBase * newInstantiationRootGrowthImpedanceRateMultiplier(SimulaDynamic* const pSD){
   return new RootGrowthImpedanceRateMultiplier(pSD);
}



RootDiameterImpedance::RootDiameterImpedance(SimulaDynamic* pSD):DerivativeBase(pSD){

	lengthImpedance = pSD->getSibling("rootGrowthImpedance");

	std::string plantType;
	PLANTTYPE(plantType, pSD);
	std::string rootType;
	int pos = 3;
	if (pSD->getParent()->getName()=="growthpoint") { pos = 2; }
	pSD->getParent(pos)->getChild("rootType")->get(rootType);
	SimulaBase* p(GETROOTPARAMETERS(plantType, rootType));
	diameterScalingExponent = p->existingChild("scalingExponentForRootDiameterIncreaseFromImpedance");
	if (!diameterScalingExponent) {
		diameterScalingExponent = pSD->getChild("scalingExponentForRootDiameterIncreaseFromImpedance");
	}
}

void RootDiameterImpedance::calculate(const Time &t, double &var){
	double exp = 0.0;
	lengthImpedance->get(t, var);
	diameterScalingExponent->get(t, exp);

	// Scales diameter relative to length impedance (0-1),
	// not directly relative to soil penetration resistance.
	// Two values of exp have special properties:
	// (1) exp = 0.5 => diameter increase offsets length decrease
	// 		=> no change in root volume
	//		=> architecture changes but C sink stays constant
	// (2) exp = 0 => scale = 1
	//		=> diameter unaffected by impedance
	var = 1.0 / pow(var, exp);
}

std::string RootDiameterImpedance::getName() const{
	return "rootDiameterImpedanceMultiplier";
}

DerivativeBase * newInstantiationRootDiameterImpedance(SimulaDynamic* const pSD){
   return new RootDiameterImpedance(pSD);
}



RootGrowthDirectionImpedance::RootGrowthDirectionImpedance(SimulaDynamic* pSD):DerivativeBase(pSD){
	// TO DO: Figure out a way to do this while using a faster soil penetration resistance algorithm. Can do it by slightly adjusting the fast methods in this file but that would mean loss of a lot of the speed advantages they have over the old method.
	if(!impedanceCalculator){
		impedanceCalculator = ORIGIN->getPath("/environment/soil/soilPenetrationResistance", "kPa");
	}
	growthPoint = dynamic_cast<SimulaPoint*>(pSD->getParent());
	bioPore = pSD->existingSibling("inBiopore");
}


void RootGrowthDirectionImpedance::calculate(const Time &t, Coordinate &vec){
	vec.x = 0;
	vec.y = 0;
	vec.z = 0;
	if (bioPore){
		double inPore;
		bioPore->get(t, inPore);
		if (inPore == 1){
			return;
		}
	}
	pSD->getParent()->getAbsolute(t, position);
	if (position.y >= 0){
		return;
	}
	Coordinate lastDirection;
	const SimulaPoint::Table *table = growthPoint->getTable();
	bool found(false);
	for (SimulaPoint::Table::const_reverse_iterator it(table->rbegin());
			it != table->rend(); ++it) {
		lastDirection = it->second.rate;
		if (vectorlength(lastDirection) > 1E-8) {
			found = true;
			break;
		}
	}
	if (!found) return;
	Coordinate testPosition;
	Coordinate tempvec;
	tempvec.x = 0.0;
	tempvec.y = 0.0;
	tempvec.z = 1.0;
	Coordinate perp = perpendicular(tempvec, lastDirection);
	normalizeVector(perp);
	Coordinate perp2 = perpendicular(perp, lastDirection);
	testPosition = position + lastDirection;
	double temp;
	impedanceCalculator->get(t, testPosition, temp);
	vec = vec + lastDirection*ImpedanceFactor(t, temp);
	for (double axialAngle = 10.0; axialAngle < 85.0; axialAngle += 10.0){
		for (double radialAngle = 0.0; radialAngle < 355.0; radialAngle += 10.0){
			testPosition = position + lastDirection*cos(axialAngle*M_PI/180.0) + perp*sin(axialAngle*M_PI/180.0)*cos(radialAngle*M_PI/180.0) + perp2*sin(axialAngle*M_PI/180.0)*sin(radialAngle*M_PI/180.0);
			if (testPosition.y > 0){
				vec.x = 0;
				vec.y = 0;
				vec.z = 0;
				return;
			}
			double temp;
			impedanceCalculator->get(t, testPosition, temp);
			vec = vec + (lastDirection*cos(axialAngle*M_PI/180.0) + perp*sin(axialAngle*M_PI/180.0)*cos(radialAngle*M_PI/180.0) + perp2*sin(axialAngle*M_PI/180.0)*sin(radialAngle*M_PI/180.0))*ImpedanceFactor(t, temp);
		}
	}
	vec = vec - lastDirection*dotProduct(lastDirection, vec)/dotProduct(lastDirection, lastDirection);
/// TODO The right scaling factor has to be found here, once we've settled on workable units
/// soiImpedance should handle most variable considerations, just need to ensure output is within range
// This scaling factor seems to work pretty okay
	vec = vec*0.01;
//	normalizeVector(vec);
//	bulkDensity->get(t, position, temp);
//	vec = vec*ImpedanceFactor(t, temp);
}

// This should depend on the root class, diameter etc. The shape of the function should also be different.
double RootGrowthDirectionImpedance::ImpedanceFactor(const Time &t, const double &imp){
	if (imp > maxBulkDensity) return 0;
	if (imp < minBulkDensity) return 1;
	return (maxBulkDensity - imp)/(maxBulkDensity - minBulkDensity);
}

std::string RootGrowthDirectionImpedance::getName() const{
	return "rootGrowthDirectionImpedance";
}

DerivativeBase * newInstantiationRootGrowthDirectionImpedance(SimulaDynamic* const pSD){
   return new RootGrowthDirectionImpedance(pSD);
}

BioporeController::BioporeController(SimulaDynamic* pSD):DerivativeBase(pSD), previousChecked(0.5), poreStart(-1), currentPoreLength(0), pore(false), poreSet(false), copyFrom(nullptr){
	if (pSD->getParent()->getName() != "growthpoint"){
		copyFrom = pSD->getParent(2)->getSibling("growthpoint")->getChild("inBiopore");
		return;
	}
	poreProbability = ORIGIN->getPath("/environment/soil/bioporeProbability", "1/cm");
	poreLengthMin = ORIGIN->getPath("/environment/soil/bioporeMinLength", "cm");
	poreLengthMax = ORIGIN->getPath("/environment/soil/bioporeMaxLength", "cm");
	rootLength = pSD->getParent()->getSibling("rootLength", "cm");
}


void BioporeController::calculate(const Time &t, double &inPore){
	if (copyFrom){
		if (poreSet){
			inPore = (pore) ? 1: 0;
		} else{
			copyFrom->get(t, inPore);
			pore = (inPore == 1) ? true: false;
			poreSet = true;
		}
		return;
	}
	double length;
	rootLength->get(t, length);
	if (pore){
		if (length - poreStart > currentPoreLength){
			previousChecked = length;
			pore = false;
			inPore = 0;
		} else{
			inPore = 1;
		}
	} else{
		if (length > previousChecked){
			Coordinate position;
			pSD->getParent()->getAbsolute(t, position);
			double prob;
			poreProbability->get(position.y, prob);
			prob = pow(1 - prob, length - previousChecked);
			if ((double)rand()/(double) RAND_MAX > prob){
				double minLength, maxLength;
				poreLengthMin->get(position.y, minLength);
				poreLengthMax->get(position.y, maxLength);
				currentPoreLength = minLength + ((double) rand()/(double) RAND_MAX)*(maxLength - minLength);
				pore = true;
				poreStart = length;
				inPore = 1;
			} else{
				inPore = 0;
			}
			previousChecked = length;
		} else{
			inPore = 0;
		}
	}
}

void BioporeController::getDefaultValue(const Time &t, double &var){
	var = 0;
}

std::string BioporeController::getName() const{
	return "bioporeController";
}

DerivativeBase * newInstantiationBioporeController(SimulaDynamic* const pSD){
   return new BioporeController(pSD);
}


//registration of classes
class AutoRegisterGrowthImpedanceInstantiationFunctions {
public:
   AutoRegisterGrowthImpedanceInstantiationFunctions() {
		BaseClassesMap::getDerivativeBaseClasses()["rootImpedanceFromBulkDensity"] = newInstantiationRootImpedanceFromBulkDensity;
		BaseClassesMap::getDerivativeBaseClasses()["rootImpedanceGao"] = newInstantiationRootImpedanceGao;
		BaseClassesMap::getDerivativeBaseClasses()["rootImpedanceWhalley"] = newInstantiationRootImpedanceWhalley;
		BaseClassesMap::getDerivativeBaseClasses()["rootGrowthImpedanceRateMultiplier"] = newInstantiationRootGrowthImpedanceRateMultiplier;
		BaseClassesMap::getDerivativeBaseClasses()["rootDiameterImpedanceMultiplier"] = newInstantiationRootDiameterImpedance;
		BaseClassesMap::getDerivativeBaseClasses()["rootGrowthDirectionImpedance"] = newInstantiationRootGrowthDirectionImpedance;
		BaseClassesMap::getDerivativeBaseClasses()["bioporeController"] = newInstantiationBioporeController;
   }
};

static AutoRegisterGrowthImpedanceInstantiationFunctions pgi;
