/*
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
#include "DiffusionResistance.hpp"
#include "../../cli/Messages.hpp"
#include <math.h>


/********************stomatal resistance****************************/
StomatalResistance::StomatalResistance(SimulaDynamic* const pSV) :
		DerivativeBase(pSV), cachedTime(-10), temperature(25), pressure(101325) {
	std::string name = pSD->getName().substr(0, 6); // name = sunlit or shaded
	bool splitBySunStatus(false);
	if (name == "sunlit" || name == "shaded"){
		splitBySunStatus = true;
		name.at(0) = std::toupper(name.at(0));
	}
	//check if unit given in input file agrees with this function
	pSD->checkUnit("s/m");
	if (splitBySunStatus) stomatalConductance_ = pSD->existingSibling("mean" + name + "StomatalConductance", "mol/m2/s");
	else stomatalConductance_ = pSD->existingSibling("meanStomatalConductance","mol/m2/s");
	if (stomatalConductance_){
		airTemperature_ = pSD->existingPath("/environment/atmosphere/averageDailyTemperature", "degreesC");
		pressure_ = pSD->existingPath("/atmosphere/airPressure", "Pa");
		if (!airTemperature_) msg::warning("StomatalResistance: Air temperature not found, defaulting to 25 degrees C");
		if (!pressure_) msg::warning("StomatalResistance: Air pressure not found, defaulting to 101325 Pa");
	}
	dailyBulkStomatalResistance_ = pSD->existingSibling("dailyBulkStomatalResistance");
	leafAreaIndex_ 				 = pSD->getSibling("meanLeafAreaIndex"); //, "m/s"
}

std::string StomatalResistance::getName() const {
	return "stomatalResistance";
}

DerivativeBase * newInstantiationStomatalResistance(
		SimulaDynamic* const pSD) {
	return new StomatalResistance(pSD);
}


/***************************************************************************/
void StomatalResistance::calculate(const Time &t, double& r_s) {
/****************************************************************************/
	if (stomatalConductance_) {
		double g_sV;
		// TODO what gives the LICOR?
		stomatalConductance_->get(t, g_sV); //mol/m2/s
		double temperature = 25;
		double pressure = 101325;
		if (airTemperature_) airTemperature_->get(t, temperature);
		if (pressure_) pressure_->get(t, pressure);
		r_s = pressure/(g_sV*8.314*(temperature + 273.15)); // Converting to s/m using the ideal gas law
	} else {
//		Parameter rl is the inverse of the stomatal conductance per unit leaf area.
//		Several earlier studies have fixed the value of rl at 100 s m−1 for well-watered
//		agricultural crops ( Monteith, 1965, Szeicz and Long, 1969 and Allen et al., 1989) when calculations were made on a 24-h basis.
//		However, it is recognized that rl varies during the course of a day with levels of solar radiation,
//		leaf temperature, and vapor pressure gradient ( Jarvis, 1976, Stewart, 1989 and Price and Black, 1989)
//		and rl increases with environmental stresses such as soil moisture deficit
//		( Stewart, 1988, Stewart and Verma, 1992 and Hatfield and Allen, 1996).
// TODO coupling
		double r_l(100.); // default value, (m/s) is the daily bulk stomatal resistance of the well-illuminated (single) leaf
		// The bulk stomatal resistance, rl, is the average resistance of an individual leaf,
		// and has a value of about 100 s/m under well-watered conditions.
// TODO could be fitted to a crop average?
		if (dailyBulkStomatalResistance_){
			dailyBulkStomatalResistance_->get(t, r_l);
		}else{
			msg::warning("StomatalResistance: dailyBulkStomatalResistance not found, using default value of 100 m/s");
		}
		double LAI = 3;
		leafAreaIndex_->get(t,LAI);
		double lai_active;
		if(LAI > 0.8/0.3)
			lai_active = 0.5 * LAI;
		else if(LAI >= 1){
			lai_active = LAI / (0.3 * LAI+ 1.2);
		}else{
			lai_active=1./1.5;
		}
		// the active (sunlit) leaf area index, which takes into consideration
		/// the fact that generally only the upper half of dense clipped grass
		/// is actively contributing to the surface heat and vapour transfer.
		/// Allen et al. (1989) proposed equations to estimate bulk surface resistance to water flux based on
		/// the crop height of grass or alfalfa in terms of the estimate crop LAI.
		///
		/// --- stomata/ canopy diffusion resistance ---
		///
		/// If the crop is not transpiring at a potential rate, the resistance depends also on the water status of the vegetation.
		/// An acceptable approximation to a much more complex relation of the surface resistance of dense full cover vegetation is:
		//
		r_s = r_l / lai_active;
		//
		// This procedure also presents difficulties. Firstly, evaporation losses from soil must be
		// negligible. If not, r_s will not be representative of total evaporation losses.
		// Secondly, spatial and temporal sampling problems are encountered in obtaining
		// representative values of r_AD and r_AB for a plant community. Thirdly, lai is tedious
		// and often difficult to measure, particularly in developing crops. Despite these
		// limitations, this approach provides the only available method for obtaining resistance
		// values without prior knowledge of evaporation.
		// Nevertheless, in a paper Paw U and Meyers (1989), using a higher-
		// order canopy turbulence model, show clearly that the parallel resistance
		// weighted by leaf area index is problematic, even when the soil is dry, and can
		// generate serious errors when used to estimate the bulk canopy resistance.
		//
		// By assuming a crop height of 0.12 m, the surface resistance for the grass reference surface approximates to 70 s/m.
		// It approximates to 45 s/m for a 0.50-m crop height.
		// The hourly r_c (=r_s) values of 50 (clipped grass) or 30 (alfalfa) and 200 s/m (both crops), for day and nighttime respectively,
		// were concluded to be fairly accurate in matching reference evapotranspiration calculated with daily data (Walter et al., 2002).

	}
}


/********************aerodynamic resistance****************************/
AerodynamicResistance::AerodynamicResistance(SimulaDynamic* const pSV) :
		DerivativeBase(pSV) {
	//check if unit given in input file agrees with this function
	pSD->checkUnit("s/m");
	cropHeight_ = pSD->existingPath("/plants/canopyHeight", "cm");
	if (!cropHeight_) msg::warning("AerodynamicResistance: Crop height not found. Assuming 0.5 m");
	windSpeed_ 		 = pSD->existingPath("/environment/atmosphere/windSpeed"); //, "m/s"
}

std::string AerodynamicResistance::getName() const {
	return "aerodynamicResistance";
}

DerivativeBase * newInstantiationAerodynamicResistance(
		SimulaDynamic* const pSD) {
	return new AerodynamicResistance(pSD);
}


/***************************************************************************/
void AerodynamicResistance::calculate(const Time &t, double& r_a) {
/****************************************************************************/
	/// http://agsys.cra-cin.it/tools/evapotranspiration/help/Aerodynamic_resistance.html
	///
	/// Where no wind data are available within the region,
	/// a value of 2 m/s can be used as a temporary estimate.
	/// This value is the average over 2000 weather stations around the globe.
	/// In general, wind speed at 2 m, u2, should be limited to about u2 >= 0.5 m/s when used in the ETo equation.
	/// This is necessary to account for the effects of boundary layer instability and buoyancy of air in promoting
	/// exchange of vapor at the surface when air is calm. This effect occurs when the wind speed is small and
	/// buoyancy of warm air induces air exchange at the surface. Limiting u2 >= 0.5 m/s in the ETo equation improves
	/// the estimation accuracy under the conditions of very low wind speed.
	double windSpeed; // Windspeed at measurement height, which we assume to be 2 meters.
	if (windSpeed_) {
		/// --- Wind speed ---  /// --- Weibull distributed ---
		windSpeed_->get(t, windSpeed); ///< U2 being the wind speed at 2-m height in m/s

    // The aerodynamic resistance equation for open air estimations reduces to r_a = 208./U2;
	// An alternative, semi-empirical, equation of ra (Thom and Oliver, 1977)
	// is used for greenhouse conditions (Stanghellini equation), characterized by low wind speed (<1 m s-1)
	} else {
		windSpeed = 2.;
		msg::warning("AerodynamicResistance: No wind speed is set by user. Assuming a wind speed of 2 m/s.");
	}
	// There is always a little bit of wind or convection
	if (windSpeed < 0.5) windSpeed = 0.5;
	// Allen, Richard G., et al. "Crop evapotranspiration-Guidelines for computing crop water requirements-FAO Irrigation and drainage paper 56." Fao, Rome 300.9 (1998): D05109.
	// See also http://www.fao.org/3/X0490E/x0490e06.htm
	double vonKarmanConstant = 0.41;
	double measurementHeight = 2;
	double zeroPlaneDisplacementHeight = 50;
	if (cropHeight_) cropHeight_->get(t, zeroPlaneDisplacementHeight);
	zeroPlaneDisplacementHeight = zeroPlaneDisplacementHeight/100.; // Convert from cm to m
	double roughnessLengthMomentumTransfer = 0.123*zeroPlaneDisplacementHeight;
	double roughnessLengthHeatAndVapour = 0.1*0.123*zeroPlaneDisplacementHeight;
	zeroPlaneDisplacementHeight *= 2./3.;
	r_a = log((measurementHeight - zeroPlaneDisplacementHeight)/roughnessLengthMomentumTransfer)*log((measurementHeight - zeroPlaneDisplacementHeight)/roughnessLengthHeatAndVapour)/(vonKarmanConstant*vonKarmanConstant*windSpeed);
	if (std::isnan(r_a)) msg::error("AerodynamicResistance: NaN value");
	if (std::isinf(r_a)) msg::error("AerodynamicResistance: inf value");
	if (r_a <= 0) msg::error("AerodynamicResistance: Aerodynamic resistance smaller than or equal to 0. Check if crop height is less than 2, if not, adjust measurement height in the code.");
}



//==================registration of the classes=================
class AutoRegisterDiffusionResistanceClassInstantiationFunctions {
public:
	AutoRegisterDiffusionResistanceClassInstantiationFunctions() {
		BaseClassesMap::getDerivativeBaseClasses()["stomatalResistance"] = newInstantiationStomatalResistance;
		BaseClassesMap::getDerivativeBaseClasses()["aerodynamicResistance"] = newInstantiationAerodynamicResistance;
	}
};

// our one instance of the proxy
static AutoRegisterDiffusionResistanceClassInstantiationFunctions p4510656771745465435;
