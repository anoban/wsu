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

#include "Impedance.hpp"
#include "../../cli/Messages.hpp"
#include "../../engine/Origin.hpp"
#include "../../engine/SimulaConstant.hpp"
#include <algorithm>
#include <cmath>


SoilImpedance::SoilImpedance(SimulaDynamic* pSD):DerivativeBase(pSD) {
}

void SoilImpedance::calculate(const Time &t, const Coordinate &pos, double &imped){
	Coordinate imp(0,0,0);
	this->calculate(t, pos, imp);
	imped = imp.length();
}
void SoilImpedance::calculate(SimulaBase* pCaller, const Time &t, double &imped){
	Coordinate imp(0,0,0);
	this->calculate(pCaller, t, imp);
	imped = imp.length();
}
void SoilImpedance::calculate(const Time &t, const Coordinate &pos, Coordinate &imped){
	msg::error("SoilImpedance base method not implemented yet.");
}
void SoilImpedance::calculate(SimulaBase* pCaller, const Time &t, Coordinate &imped){
	msg::error("SoilImpedance base method not implemented yet.");
}

/// Convert water content to matric potential using van Genuchten water curve parameters
/// Gao et al 2016, eqn 6, with added checks for range overflow.
double SoilImpedance::theta2psi(const double theta, const double theta_sat, const double theta_resid, const double alpha, const double n){
	if(theta > theta_sat){
		msg::warning("SoilImpedance::theta2psi: reported soil water content is greater than saturatedWaterContent. Using saturatedWaterContent; please check your parameters.");
	}
	if(theta < theta_resid){
		msg::warning("SoilImpedance::theta2psi: reported soil water content is less than residualWaterContent. Using residualWaterContent; please check your parameters.");
	}
	double th = fmin(theta, theta_sat);
	th = fmax(th, theta_resid);

	double relative_saturation = (th-theta_resid)/(theta_sat-theta_resid);

	// this is probably an oversimplification,
	// but avoids numerical problems in the exponents below
	if (relative_saturation <= 0.0 ) { return 1.0e4; }

	double pow1 = std::pow(relative_saturation, -1.0/(1.0 - 1.0/n));
	if (pow1 <= 1.0) { return 0.0; }
	return std::pow(pow1 - 1.0, 1.0/n) / alpha;
}

std::string SoilImpedance::getName()const{
	return "soilImpedance";
}



ImpedanceFromBulkDensity::ImpedanceFromBulkDensity(SimulaDynamic* pSD):SoilImpedance(pSD)
{
	bulkDensity = pSD->getPath("/environment/soil/bulkDensity");
	bulkDensity->checkUnit("g/cm3");

	// Take water content first from any existing child, to allow user overrides
	// Otherwise read from SWMS, or failing that assume a constant
	localWaterContent = pSD->existingChild("soilWaterContentForImpedance", "cm3/cm3");
	if(!localWaterContent){
		localWaterContent = pSD->getParent(3)->getChild("dataPoints")->getLastChild()->existingChild("volumetricWaterContentAtTheRootSurface", "cm3/cm3");
	}
	if(!localWaterContent){
		msg::warning("ImpedanceFromBulkDensity: neither volumetricWaterContentAtTheRootSurface nor soilWaterContentForImpedance found. Impedance calculations will assume 0.3 cm3/cm3");
		localWaterContent = new SimulaConstant<double>("soilWaterContentForImpedance", pSD, 0.3, "cm3/cm3", pSD->getStartTime());
	}
}

void ImpedanceFromBulkDensity::calculate(const Time &t, const Coordinate &pos, Coordinate &imped){
	double bulk;
	double water_content;

	bulkDensity->get(t, pos, bulk);
	localWaterContent->get(t, pos, water_content);

	// TODO make these into inputs
	// Fixed values from Vaz et al. 2011, 10.1016/j.geoderma.2011.07.016
	// Eqn form is "eqn 1" of their Table 2, formulated by Jakobsen and Dexter 1987
	// coefs for soil "LVAd", a sandy clay loam, from their Table 4.
	double intercept = 0.89;
	double bd_mult = 3.00;
	double water_mult = -15.98;

	bulk = exp(intercept + bd_mult*bulk + water_mult*water_content);
	bulk = 0.1772*bulk + 0.0134*pow(bulk, 2); // Fig 1 Pabin et al 1998, 10.1016/S0167-1987(98)00098-1

	if(bulk < 0){
		bulk = 0;
	}else if(bulk > 1){
		bulk = 1;
	}

	// HACK: Setting length of a zero-length coordinate has no effect
	// (gitlab #18), because zero-length vectors have undefined direction.
	// But since we're abusing Coordinate to represent the resistance vector
	// surrounding a point rather than the movement of the growthpoint itself,
	// (0,0,0) just means "resistance is 0 in all directions" and can be
	// safely scaled to any other magnitude that is equal in all directions.
	if(imped.length() == 0){
		imped.x = imped.y = imped.z = bulk;
	}
	imped.setLength(bulk);
}

std::string ImpedanceFromBulkDensity::getName()const{
	return "impedanceFromBulkDensity";
}



ImpedanceGao::ImpedanceGao(SimulaDynamic* pSD):SoilImpedance(pSD)
{
	//TODO this should probably move to impedanceBase; it's used by every subclass
	if (!bulkDensity) { bulkDensity = pSD->getPath("/environment/soil/bulkDensity", "g/cm3"); }

	// Precompute net stress and bulk density
	// Remove this if there is any chance bulkDensity can change during the simulation!
	Coordinate soilbottom, soiltop;
	pSD->getPath("/environment/dimensions/minCorner")->get(soilbottom);
	pSD->getPath("/environment/dimensions/maxCorner")->get(soiltop);
	precalculate_net_stress(cachedBulkDensity, cumulativeStress, bulkDensity, soilbottom.y, soiltop.y);
	if(cumulativeStress.empty() || cachedBulkDensity.empty()) {
		msg::error("ImpedanceGao: Precalculation of soil weight cache failed");
	}

	// if not set, will calculate at run time assuming particle density = 2.65.
	if (!voidRatio) { voidRatio = pSD->existingPath("/environment/soil/voidRatio", "100%"); }
	
	// TODO this should probably move to impedanceBase too
	// Used to override SWMS values. You probably want to leave this unset most of the time,
	// in which case calculate() will look up water from the calling growthpoint
	localWaterContent = pSD->existingChild("soilWaterContentForImpedance", "cm3/cm3");

	if (!saturatedWaterContent) { saturatedWaterContent = pSD->getPath("/environment/soil/water/saturatedWaterContent", "100%"); }
	if (!residualWaterContent) { residualWaterContent = pSD->getPath("/environment/soil/water/residualWaterContent", "100%"); }
	if (!vanGenuchtenAlpha) { vanGenuchtenAlpha = pSD->getPath("/environment/soil/water/vanGenuchten:alpha", "1/cm"); }
	if(!vanGenuchtenN) { vanGenuchtenN = pSD->getPath("/environment/soil/water/vanGenuchten:n", "noUnit"); }

	// When enabled, writes out calculated intermediate parameters (net stress,
	// suction stress, psi, etc) as warnings from every calculate() call.
	// Use it for diagnostics on very small models with probably no more than
	// 10 impedance callers.
	// Seriously, don't turn this on if you're simulating a full root system.
	debug = false;
	SimulaBase* dbg = pSD->existingChild("writeDebugOutputToWarnings");
	if (dbg) dbg->get(debug);
}

// Hack: Returns a dummy value and exists only to be called by Table::run.
// All other callers should use get(pCaller, t, coord) instead.
//
// We need this because Table::run tries to report the state of all
// objects that simulate double, but does it by calling
//		obj->getAbsolute(t, pos);
// 		obj->get(Time, Coord, double);
// without checking whether they need other information.
// For impedanceGao, these calls will probably have pos set to the origin,
// so a bogus value is ~expected anyway.
//
// To suppress the warning, list your impedance calculator object in
// /simulationControls/outputParameters/table/skipTheseVariables
// so that this function is never called.
void ImpedanceGao::calculate(const Time &t, const Coordinate &pos, Coordinate &imped) {
	msg::warning("ImpedanceGao: no call pointer passed so can't look up water content. Returning 0");
	imped = Coordinate(0, 0, 0);
}

void ImpedanceGao::calculate(SimulaBase* pCaller, const Time &t, Coordinate &imped){
	// fitted empirical params from Gao et al 2016 table 3
	constexpr double
		gravity_newtons_per_kg = 9.81, // OK, this one's not fitted :)
		F(3.560),
		Astar(1.000),
		f(0.1846),
		p(2.1931);

	double
		rho(0.0), // dry bulk density in g/cm3
		wc, // water content in cm3/cm3
		wc_sat, // water content at saturation in cm3/cm3
				// will probably be from fitted Van Genuchten water retention curve
		wc_resid, // residual water content (NB not the same thing as wilting point), in cm3/cm3
				  // Will probably be from fitted Van Genuchten water retention curve
		void_ratio, // = 1/(1-porosity)
		vg_alpha, // alpha parameter from fitted Van Genuchten water retention curve
		vg_n; // n parameter from fitted Van Genuchten water retention curve

	Coordinate pos;

	pCaller->getAbsolute(t, pos);
	if (pos.y > cachedBulkDensity.rbegin()->first || pos.y < cachedBulkDensity.begin()->first) {
		// outside soil grid => no density => no impedance
		imped.setLength(0);
		return;
	}
	rho = cachedBulkDensity.lower_bound(pos.y)->second;
	if (rho <= 0) {
		imped.setLength(0);
		return;
	}
	if (localWaterContent) {
		localWaterContent->get(t, wc);
	} else {
		pCaller->getSibling("volumetricWaterContentAtTheRootSurface")->get(t, wc);
	}
	saturatedWaterContent->get(t, pos, wc_sat);
	residualWaterContent->get(t, pos, wc_resid);
	vanGenuchtenAlpha->get(t, pos, vg_alpha);
	vanGenuchtenN->get(t, pos, vg_n);

	if(voidRatio){
		voidRatio->get(t, pos, void_ratio);
	}else{
		// back calculate from assumed particle density of 2.65 (= quartz)
		// TODO compute from sand/silt/clay instead?
		double porosity = 1 - (rho / 2.65);
		if (std::abs(porosity - wc_sat) > 0.1){
			msg::warning("ImpedanceGao: Soil physical and hydraulic parameters are inconsistent. Water content at saturation differs more than 10% from porosity implied by bulk density");
		}
		void_ratio = porosity/(1-porosity);
	}

	// std::cout << t << rho << "\n";

	double Sstar = std::max(0.5, (wc-wc_resid)/(wc_sat-wc_resid));

	// Adjust bulk density to include effect of gravity:
	// (BULK g / cm3) * (1e6 cm3 / m3) * (1 kg / 1e3 g) * (GRAVITY N / kg)
	// = BULK*GRAVITY * 1e3 N / m3
	// = result is in kN / m3
	rho = rho * gravity_newtons_per_kg;


	// convert water content to matric potential:
	// result is (((wc-wc_resid)/(wc_sat-wc_resid))^(-1/(1-1/vg_n)) - 1)^(1/vg_n) / vg_alpha,
	// but avoiding NaNs when terms go negative.
	double psi = SoilImpedance::theta2psi(wc, wc_sat, wc_resid, vg_alpha, vg_n);

	// Look up cached net stress from weight of overburdening soil,
	// NB this used to be recalculated live  as `sigma_s = net_stress(bulkDensity, t, pos);`
	// which loops over all soil layers and is FAR too slow to use every impedance call.
	// If bulk density can change through time, need to revisit this.
	double sigma_s(0.0);
	if (pos.y > cumulativeStress.rbegin()->first || pos.y < cumulativeStress.begin()->first) {
		imped.setLength(0.0);
		return;
	}
	sigma_s = cumulativeStress.lower_bound(pos.y)->second;

	//Gao et al 2016 eqn 4
	double impedance =
		rho * std::pow(
			Astar
				* std::pow(F - void_ratio, 2) / (1 + void_ratio)
				* std::pow(std::pow(sigma_s, p) - (-psi)*Sstar, f),
			2);
	if(!std::isnormal(impedance)){
		msg::error("Numerical problem: impedance = " + std::to_string(impedance) + " in " + pSD->getPath());
	}

	// HACK: Setting length of a zero-length coordinate has no effect
	// (gitlab #18), because zero-length vectors have undefined direction.
	// But since we're abusing Coordinate to represent the resistance vector
	// surrounding a point rather than the movement of the growthpoint itself,
	// (0,0,0) just means "resistance is 0 in all directions" and can be
	// safely scaled to any other magnitude that is equal in all directions.
	if(imped.length() == 0){
		imped.x = imped.y = imped.z = impedance;
	}
	imped.setLength(impedance);

	if(debug){
		msg::warning(
			"ImpedanceGao: path=" + pCaller->getPath()
			+ " time=" + std::to_string(t)
			+ " position=("
				+ std::to_string(pos.x) + ","
				+ std::to_string(pos.y) + ","
				+ std::to_string(pos.z)
			+ ") impedance=" + std::to_string(impedance)
			+ " void_ratio=" + std::to_string(void_ratio)
			+ " net_stress=" + std::to_string(sigma_s)
			+ " bulk_density=" + std::to_string(rho / gravity_newtons_per_kg)
			+ " theta=" + std::to_string(wc)
			+ " psi=" + std::to_string(psi)
			+ " suction_stress=" + std::to_string(Sstar * psi));
	}
}


// Calculate net stress from weight of overburdening soil
// Idea is from Gao et al 2016, implementation by CKB
// Uses odd units: (density g/cm3) * (1e6 cm3/m3) * (kg/1e3 g) * (gravity 9.81 m/s2) * (Pa/(kg/(m s2)) * (kPa/1e3 Pa) * (m/100 cm)
// 	= density * 1e6 / 1e3 * 9.81 / 1e3 / 100 kPa/cm
// 	= density * 9.81 / 100 kPa/cm
// To get total pressure at a given depth, sum over all depths above
// BEWARE: Very slow. If you can, avoid using it inside frequently-called calculations like ImpedanceGao::calculate.
double ImpedanceGao::net_stress(const SimulaBase* density, const Time &t, const Coordinate &pos, double step) {
	Coordinate over_layer(pos);
	double over_layer_weight;
	double stress = 0;
	for (over_layer.y = 0; over_layer.y > pos.y; over_layer.y -= step){
		bulkDensity->get(t, over_layer, over_layer_weight);
		stress += (over_layer_weight * step * 9.81/100);
	}
	return stress;
}

// Compute local caches of bulk density and net stress at desired vertical resolution.
// N.B. assumes bulkDensity at a given depth is constant through the whole simulation,
// so that computing it once at ImpedanceGao instantiation time saves many repeated calculations.
// But if bulkDensity *can* change over time, do not use this function.
void ImpedanceGao::precalculate_net_stress(
		std::map<double, double> &bulkd_cache,
		std::map<double, double> &stress_cache,
		const SimulaBase* density,
		const double bottom_depth,
		const double top_depth,
		const double stepsize) {
	Coordinate layer_coord;
	double layer_stress{0.0}, cum_stress{0.0};
	for(double d = top_depth; d > bottom_depth; d -= stepsize) {
		layer_coord.y = d;
		bulkDensity->get(0.0, layer_coord, layer_stress);
		bulkd_cache.insert({d, layer_stress});
		cum_stress += (layer_stress * stepsize * 9.81 / 100);
		stress_cache.insert({d, cum_stress});
	}
}

// Static member initialization; all are assigned real addresses on first ctor call
SimulaBase* ImpedanceGao::bulkDensity = nullptr;
SimulaBase* ImpedanceGao::saturatedWaterContent = nullptr;
SimulaBase* ImpedanceGao::residualWaterContent = nullptr;
SimulaBase* ImpedanceGao::voidRatio = nullptr;
SimulaBase* ImpedanceGao::vanGenuchtenAlpha = nullptr;
SimulaBase* ImpedanceGao::vanGenuchtenN = nullptr;
std::map<double, double> ImpedanceGao::cumulativeStress = {};
std::map<double, double> ImpedanceGao::cachedBulkDensity = {};


std::string ImpedanceGao::getName()const{
	return "impedanceGao";
}



ImpedanceWhalley::ImpedanceWhalley(SimulaDynamic* pSD):SoilImpedance(pSD)
{
	bulkDensity = pSD->getPath("/environment/soil/bulkDensity", "g/cm3");
	// Take water content first from any existing child, to allow user overrides
	// Otherwise read from SWMS, or failing that assume a constant
	localWaterContent = pSD->existingChild("soilWaterContentForImpedance", "cm3/cm3");
	if(!localWaterContent){
		// TODO segfaults if called from outside a growthpoint
		localWaterContent = pSD->getParent(3)->getChild("dataPoints")->getLastChild()->existingChild("volumetricWaterContentAtTheRootSurface", "cm3/cm3");
	}
	if(!localWaterContent){
		msg::warning("ImpedanceWhalley: neither volumetricWaterContentAtTheRootSurface nor soilWaterContentForImpedance found. Impedance calculations will assume 0.3 cm3/cm3");
		localWaterContent = new SimulaConstant<double>("soilWaterContentForImpedance", pSD, 0.3, "cm3/cm3", pSD->getStartTime());
	}

	saturatedWaterContent = pSD->getPath("/environment/soil/water/saturatedWaterContent", "100%");
	residualWaterContent = pSD->getPath("/environment/soil/water/residualWaterContent", "100%");
	vanGenuchtenAlpha = pSD->getPath("/environment/soil/water/vanGenuchten:alpha", "1/cm");
	vanGenuchtenN = pSD->getPath("/environment/soil/water/vanGenuchten:n", "noUnit");
}

void ImpedanceWhalley::calculate(const Time &t, const Coordinate &pos, Coordinate &imped){
	double rho, wc, wc_sat, wc_resid, vg_alpha, vg_n;
	bulkDensity->get(t, pos, rho);
	localWaterContent->get(t, pos, wc);
	saturatedWaterContent->get(t, pos, wc_sat);
	residualWaterContent->get(t, pos, wc_resid);
	vanGenuchtenAlpha->get(t, pos, vg_alpha);
	vanGenuchtenN->get(t, pos, vg_n);

	double Se = (wc-wc_resid)/(wc_sat-wc_resid);
	double psi = SoilImpedance::theta2psi(wc, wc_sat, wc_resid, vg_alpha, vg_n);


	// Whalley 2007 eqn 8
	// log10 Q = 0.35 * log10(|psi| * Se) + 0.93 * rho + 1.26
	// => Q = 10 ^ (0.35 * log10(|psi| * Se) + 0.93 * rho + 1.26)
	// => Q = 10^(0.35 * log10(|psi| * Se)) * 10^(0.93 * rho + 1.26)
	// => Q = (|psi| * Se)^0.35 * 10^(0.93 * rho + 1.26)
	double impedance =
		std::pow(std::fabs(psi) * Se, 0.35)
		* std::pow(10, 0.93 * rho + 1.26);

	// Empirical correction for back-transformation bias (Whalley eqn 15)
	// they derive this as the mean of antilogs of the residuals, 1/n*sum(10^(Q-Qhat))
	// but say 1.097 should be "widely applicable to soils similar to the ones studied here"
	impedance *= 1.097;

	// HACK: Setting length of a zero-length coordinate has no effect
	// (gitlab #18), because zero-length vectors have undefined direction.
	// But since we're abusing Coordinate to represent the resistance vector
	// surrounding a point rather than the movement of the growthpoint itself,
	// (0,0,0) just means "resistance is 0 in all directions" and can be
	// safely scaled to any other magnitude that is equal in all directions.
	if(imped.length() == 0){
		imped.x = imped.y = imped.z = impedance;
	}
	imped.setLength(impedance);
}

std::string ImpedanceWhalley::getName()const{
	return "impedanceWhalley";
}



DerivativeBase * newInstantiationSoilImpedance(SimulaDynamic* const pSD){
   return new SoilImpedance(pSD);
}
DerivativeBase * newInstantiationImpedanceFromBulkDensity(SimulaDynamic* const pSD){
   return new ImpedanceFromBulkDensity(pSD);
}
DerivativeBase * newInstantiationImpedanceGao(SimulaDynamic* const pSD){
	return new ImpedanceGao(pSD);
}
DerivativeBase * newInstantiationImpedanceWhalley(SimulaDynamic* const pSD){
	return new ImpedanceWhalley(pSD);
}

class AutoRegisterImpedanceInstantiationFunctions {
public:
	AutoRegisterImpedanceInstantiationFunctions() {
		BaseClassesMap::getDerivativeBaseClasses()["soilImpedance"] =
			newInstantiationSoilImpedance;
		BaseClassesMap::getDerivativeBaseClasses()["impedanceFromBulkDensity"] =
			newInstantiationImpedanceFromBulkDensity;
		BaseClassesMap::getDerivativeBaseClasses()["impedanceGao"] =
			newInstantiationImpedanceGao;
		BaseClassesMap::getDerivativeBaseClasses()["impedanceWhalley"] =
			newInstantiationImpedanceWhalley;
	};
};

static AutoRegisterImpedanceInstantiationFunctions p;
