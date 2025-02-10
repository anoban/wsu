
#include "SaxtonRawls.hpp"
#include "../../cli/Messages.hpp"
#include <cmath>


Saxton::Saxton(SimulaDynamic *pSD):DerivativeBase(pSD){
	sand = pSD->getPath("/soil/sand", "100%");
	clay = pSD->getPath("/soil/clay", "100%");
	gravel = pSD->getPath("/soil/gravel", "100%");
	organic = pSD->getPath("/soil/organic_matter", "100%");
	waterContent = pSD->getPath("/soil/waterContentForSaxton", "100%");
	densityFactor = pSD->getPath("/soil/densityFactorForSaxton", "noUnit");
}

Saxton::Saxton(SimulaDynamic *pSD, std::string parm):Saxton(pSD){
	returnParm = parm;
}

void Saxton::calculate(const Time &t, const Coordinate &loc, double &x){
	const double particle_density = 2.65;

	sand->get(t, loc, S);
	clay->get(t, loc, C);

	gravel->get(t, loc, Rv);
	
	// Saxton and Rawls paper confuses percent and proportion!
	// It claims all inputs are in weight percent,
	// but figure axes, table 3, and calculator screenshot in Fig 11 all
	// agree ONLY if sand/clay/gravel are proportions while OM is percent.
	// Correcting here so that user inputs stay uniform.
	organic->get(t, loc, OM);
	OM *= 100;
	
	densityFactor->get(t, loc, DF);
	if(DF < 0.9 || DF > 1.3) {
		msg::error("Invalid densityFactorForSaxton: " + std::to_string(DF) + " given, valid range 0.9 - 1.3");
	}


	// eqn 1: water content at wilting point
	par["thWP"] = -0.024*S + 0.487*C + 0.006*OM
		+ 0.005*(S*OM) - 0.013*(C*OM)
		+ 0.068*(S*C) + 0.031;
	par["thWP"] = par["thWP"] + (0.14*par["thWP"] - 0.02);

	// eqn 2: water content at field capacity
	par["thFC"] = -0.251*S + 0.195*C + 0.011*OM
		+ 0.006*(S*OM) - 0.027*(C*OM)
		+ 0.452*(S*C) + 0.299;
	par["thFC"] = par["thFC"] + (1.283*std::pow(par["thFC"], 2) - 0.374*par["thFC"] - 0.015);

	// eqn 3: water content between saturation and 33 kPa
	par["thSFC"] = 0.278*S + 0.034*C + 0.022*OM
		- 0.018*(S*OM) - 0.027*(C*OM)
		- 0.584*(S*C) + 0.078;
	par["thSFC"] = par["thSFC"] + (0.636*par["thSFC"] - 0.107);

	// eqn 4: Air entry pressure
	par["psie"] = -21.67*S - 27.93*C - 81.97*par["thSFC"]
		+ 71.12*(S*par["thSFC"]) + 8.29*(C*par["thSFC"])
		+ 14.05*(S*C) + 27.16;
	par["psie"] = par["psie"] + (0.02*pow(par["psie"], 2) - 0.113*par["psie"] - 0.70);


	// Check that psie is between -33 and zero (eqns 11-15 assume this)
	if (par["psie"] > 33) {
		// msg::error("calculated air entry pressure (" + std::to_string(par["psie"]) + ") is less than field capacity. That seems wrong.");
		msg::warning("calculated air entry pressure (" + std::to_string(par["psie"]) + ") is less than field capacity. That seems wrong.");
	} else if (par["psie"] < 0) {
		// msg::error("calculated air entry pressure (" + std::to_string(par["psie"]) + ") is above saturation. That seems wrong.");
		msg::warning("calculated air entry pressure (" + std::to_string(par["psie"]) + ") is above saturation. That seems wrong.");
	}

	// eqn 5: water content at saturation
	par["thS"] = par["thFC"] + par["thSFC"] - 0.097*S + 0.043;

	// eqn 6: normal density
	par["dens"] = (1 - par["thS"])*particle_density;

	// eqn 7-10: density adjustments
	par["dens_DF"] = par["dens"] * DF; // eqn 7
	// eqn 8: density-adjusted water content at saturation
	par["thS_DF"] = 1-(par["dens_DF"]/particle_density);
	// eqn 9: density-adjusted water content at field capacity
	par["thFC_DF"] = par["thFC"] - 0.2*(par["thS"] - par["thS_DF"]);
	// eqn 10: density-adjusted water content at (saturation minus 33 kPa)
	// "A large adjustment of density could cause Eq. [10] to become negative, thus a minimum difference of 0.5%v was set to limit the DF value in these cases."
	par["thSFC_DF"] = par["thS_DF"] - par["thFC_DF"];

	// eqn 11-15, piecewise moisture-tension function
	// TODO should this use the density-corrected thetas?
	// Saxton table 1 doesn't include "DF" subscripts in this section,
	// but "sequential application of the equations" implies it,
	// and theta_S is reportedly strongly affected by the correction
	double theta;
	waterContent->get(t, loc, theta);

	// eqns 14, 15: fitting coefficients for moisture-tension
	par["B"] = (std::log(1500) - std::log(33))
			/ (std::log(par["thFC"]) - std::log(par["thWP"])); // eqn 15
	par["A"] = exp(std::log(33) + par["B"]*std::log(par["thFC"])); // eqn 14

	if (theta < par["thWP"]) {
		par["psith"] = 1500;
		msg::error("Can't calculate water potential below wilting point");
	} else if (theta < par["thFC"]) {
		// eqn 11: tension between wilting point and field capacity
		par["psith"] = par["A"]*std::pow(theta, -par["B"]);
	} else {
		// eqn 12: tension between field capacity and air entry
		par["psith"] = 33.0 - ((theta - par["thFC"]) * (33.0 - par["psie"]) / (par["thS"] - par["thFC"]));
	} 
	if (par["psith"] < par["psie"]) {
		// eqn 13: if pressure less than air entry, assume saturation
		par["psith"] = par["psie"];
		theta = par["thS"];
	}


	// eqns 16-18: moisture-conductivity
	par["lambda"] = 1/par["B"]; // eqn 18
	par["Ks"] = 1930*std::pow(par["thS"] - par["thFC"], 3 - par["lambda"]); // eqn 16
	par["Kth"] = par["Ks"]*std::pow(theta/par["thS"], 3+(2/par["lambda"])); // eqn 17


	// // eqn 19-22: Gravel effects
	// Rv = (alpha*Rw)/(1 - Rw*(1-alpha)); // eq 19
	// rho_B = rho_N*(1-Rv) + (Rv * particle_density); // eq 20
	// PAW_B = PAW * (1 - Rv); // eq 21
	// K_b/K_S = (1 - Rw) / (1 - Rw*(1 - 3*alpha/2)); // eq 22

	// eqn 23-24: Salinity effects
	// Psi_O = 36*EC;
	// Psi_Otheta = (theta_S / theta) * (36 * EC);

	x = par[returnParm];
}

DerivativeBase * newInstantiationSaxton(SimulaDynamic* const pSD){
   return new Saxton(pSD);
}
DerivativeBase * newInstantiationSaxtonBulk(SimulaDynamic* const pSD){
   return new Saxton(pSD, "dens");
}
DerivativeBase * newInstantiationSaxtonWP(SimulaDynamic* const pSD){
   return new Saxton(pSD, "thWP");
}
DerivativeBase * newInstantiationSaxtonFC(SimulaDynamic* const pSD){
   return new Saxton(pSD, "thFC");
}
DerivativeBase * newInstantiationSaxtonSat(SimulaDynamic* const pSD){
   return new Saxton(pSD, "thS");
}
DerivativeBase * newInstantiationSaxtonKsat(SimulaDynamic* const pSD){
   return new Saxton(pSD, "Ks");
}
DerivativeBase * newInstantiationSaxtonKtheta(SimulaDynamic* const pSD){
   return new Saxton(pSD, "Kth");
}// ...TODO expose others when you need them


class AutoRegisterSaxtonInstantiationFunctions {
public:
	AutoRegisterSaxtonInstantiationFunctions() {
		BaseClassesMap::getDerivativeBaseClasses()["saxtonRawls"] = 
			newInstantiationSaxton;
		BaseClassesMap::getDerivativeBaseClasses()["saxtonRawlsBulkDensity"] = 
			newInstantiationSaxtonBulk;
		BaseClassesMap::getDerivativeBaseClasses()["saxtonRawlsWiltPoint"] = 
			newInstantiationSaxtonWP;
		BaseClassesMap::getDerivativeBaseClasses()["saxtonRawlsFieldCapacity"] = 
			newInstantiationSaxtonFC;
		BaseClassesMap::getDerivativeBaseClasses()["saxtonRawlsSaturatedWaterContent"] = 
			newInstantiationSaxtonSat;
		BaseClassesMap::getDerivativeBaseClasses()["saxtonRawlsKsat"] = 
			newInstantiationSaxtonKsat;
		BaseClassesMap::getDerivativeBaseClasses()["saxtonRawlsKth"] = 
			newInstantiationSaxtonKsat;
	};
};

static AutoRegisterSaxtonInstantiationFunctions p;
