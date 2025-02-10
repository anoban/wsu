
#ifndef SAXTONRAWLS_HPP_
#define SAXTONRAWLS_HPP_

#include "../../engine/BaseClasses.hpp"
#include <unordered_map>

/// soil texture -> estimated hydraulic properties
///
/// inputs: weight fractions of sand, clay, organic matter; volume fraction of gravel, current water content.
/// Note that Saxton and Rawls paper refers to all of these as percents,
/// however I can only match their figures by treating sand/clay/water as fraction but OM as percent.
/// This implementation takes all as fractions, for consistency, and converts the OM to percent for you.
/// Not yet implemented: Salinity corrections from EC measurements
///
/// Calculates:
///  wilting point, field capacity, saturation,
///  pressure at air entry, pressure at current water content,
///  bulk density with or without gravel, Ksat with or without gravel, K at current water content,
///  plus several fitting parameters.
/// All are stored in an unordered map named "par" and can be accessed by name.
/// additionally, since the rest of OpenSimRoot assumes that one object computes one number, the `returnParm` property controls which parameter will be returned by a plain `get()` (should be equivalent to `par[returnParm]`).
///
class Saxton : public DerivativeBase {
public:
	Saxton(SimulaDynamic* const pSD);
	Saxton(SimulaDynamic* const pSD, std::string parm);
	std::string getName()const{return "saxtonRawls";};
	std::string returnParm = "dens"; // get() returns par[returnParm]
protected:
	void calculate(const Time &t, const Coordinate &loc, double &x);
	SimulaBase
		*sand, *clay,
		*gravel, *organic, *waterContent,
		*densityFactor;
	
	// Saxton and Rawls 2006, 10.2136/sssaj2005.0117
	// Table 2. Equation symbol definitions.
	// I've collapsed their "<name> / <name>t" pairs
	// (initial value vs second calculation)
	// into single variables sequentially assigned,
	// Also using "FC" for theta at 33 kPa = field capacity
	// and "WP" for theta at 1500 kPa = wilting point
	double 
		S, // Sand, %w
		C, // Clay, %w
		DF, // Density adjustment Factor (0.9–1.3) 
		OM, // Organic Matter, %w
		// EC, // Electrical conductance of a saturated soil extract,  dS m^-1 (dS/m = mili-mho cm^-1)
		Rv; // Volume fraction of gravel (decimal), g cm^-3
		
	// computed values, stored as an unordered map so we can look them up by name
	std::unordered_map<std::string, double> par = {
	 	{"A", 0},  {"B", 0}, // Coefficients of moisture-tension, Eq. [11]	
		{"thFC", 0}, // Field Capacity moisture (33 kPa), %v
		// PAW, // Plant Avail. moisture (33–1500 kPa, matric soil), %v
		// PAW_B, // Plant Avail. moisture (33–1500 kPa, bulk soil),%v
		{"thWP", 0}, // Wilting point moisture (1500 kPa), %v
		// theta_psi, Moisture at tension psi, %v
		{"thFC_DF", 0}, // 33 kPa moisture, adjusted density, %v
		{"thSFC", 0}, // SAT-33 kPa moisture, normal density %v
		{"thSFC_DF", 0}, // SAT-33 kPa moisture, adjusted density, %v
		{"thS", 0} , // Saturated moisture (0 kPa), normal density, %v
		{"thS_DF", 0}, // Saturated moisture (0 kPa), adjusted density, %v
		{"psith", 0}, // Tension at moisture u, kPa
		// psi_et, // Tension at air entry, first solution, kPa
		{"psie", 0}, // Tension at air entry (bubbling pressure), kPa
		{"Ks", 0}, // Saturated conductivity (matric soil), mm h^-1
		// K_b, // Saturated conductivity (bulk soil), mm h^-1
		{"Kth", 0}, // Unsaturated conductivity at moisture theta, mm h^-1
		{"dens", 0}, // Normal density, g cm^-3
		// densB, // Bulk soil density (matric plus gravel), g cm^-3
		{"dens_DF", 0}, // Adjusted density, g cm^-3
		{"lambda", 0}, // Slope of logarithmic tension-moisture curve
		// alpha, // Matric soil density/gravel density (2.65) = rho/2.65
		// Rw, // Weight fraction of gravel (decimal), g g^-1
		// Psi_O, // Osmotic potential at  theta = theta_S, kPa
		// Psi_Otheta; // Osmotic potential at theta < theta_S, kPa;
	};
};


class SaxtonBulk : public Saxton {
public:
	SaxtonBulk(SimulaDynamic* const pSD);
	std::string getName()const{return "saxtonRawlsBulkDensity";};
	void get(const Time &t, const Coordinate &loc, double &x){calculate(t, loc, x);};
protected:
	// void calculate(const Time &t, double &x);
	void calculate(const Time &t, const Coordinate &loc, double &x);
};

#endif /*SAXTONRAWLS_HPP_*/
