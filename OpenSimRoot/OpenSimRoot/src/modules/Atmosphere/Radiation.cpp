/*
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
#if _WIN32 || _WIN64
#define _USE_MATH_DEFINES
#endif
#ifndef M_PI
#define M_PI (3.14159265358979323846)
#endif

#include "Radiation.hpp"
#include "../../cli/Messages.hpp"
#include <math.h>
#include "../../tools/Time.hpp"
#include "../../engine/Origin.hpp"



/***************************Radiation************************************/

Radiation::Radiation(SimulaDynamic* const pSV) :
		DerivativeBase(pSV) {
	//check if unit given in input file agrees with this function
	pSD->checkUnit("MJ/m2/day");

	dailyTemperature_   	= pSD->getPath("/environment/atmosphere/averageDailyTemperature", "degreesC");
	latitude_ 				= pSD->getPath("/environment/latitude", UnitRegistry::noUnit());
	actualDurationofSunshine_ = pSD->existingPath("/environment/atmosphere/actualDurationofSunshine");// , "hours"
	netSolarRadiation_ 		= pSD->existingPath("/environment/atmosphere/netSolarRadiation");

	minTemperature_ 		= pSD->existingPath("/environment/atmosphere/minTemperature", "degreesC");
	maxTemperature_ 		= pSD->existingPath("/environment/atmosphere/maxTemperature", "degreesC");
	actualVaporPressure_	= pSD->getSibling("actualVaporPressure"); //given in hPa

	std::string name=pSD->getName();
	if(name.find("Soil")!=std::string::npos){
		albedo_ = pSD->getPath("/environment/atmosphere/albedoSoil");
	}else{
		albedo_ = pSD->getPath("/environment/atmosphere/albedoCrop");
	}

	angstrom_as_  = pSD->existingPath("/environment/atmosphere/angstrom_as");
	angstrom_bs_  = pSD->existingPath("/environment/atmosphere/angstrom_bs");
}





std::string Radiation::getName() const {
	return "Radiation";
}

DerivativeBase * newInstantiationRadiation(
		SimulaDynamic* const pSD) {
	return new Radiation(pSD);
}

/*****************************************************************************/
void Radiation::calculate(const Time &t, double& netRadiation) {
/*****************************************************************************/
	/// Input:  Time, latitude, date
	///atmosphere
	/// Output: radiation rate, Ra (MJ m-2 day-1).
	//

	/** SYMBOLS
	 *
	 * Gsc 		= solar constant = 0.082 Mj/m2/min
	 * Ra		= extraterestrial (solar) radiation  MJ/m2/d
	 * Rs		= shortwave radiation / global radiation MJ/m2/d
	 * Rso		= clear sky solar radiation MJ/m2/d
	 * Rn 		= net radiatio MJ/m2/d
	 * Rnl		= net longwave radiation MJ/m2/d
	 * Rns		= net solar radiation MJ/m2/d
	 * albedo	= reflection rate at surface
	 * actualSun	 = actual duration of sun [hours]
	 * daylightHours = max. possible daylight hours
	 * G 		= soil heat flux
	 * omega_s	= sunset hour angle [rad]
	 * phi		= latitude [rad]
	 * dr 		= inverse relative distance Earth-Sun
	 * delta	= solar decimation [rad]
	 *
	 */
	double const Gsc = 0.082; // MJ/m2/min // solar constant

	/**
	 * Calculation procedures:
	 *
	 * --- Extraterrestrial radiation for DAILY periods Ra ---
	 *
	 * The extraterrestrial radiation, Ra,
	 * for each day of the year and for different latitudes can be estimated
	 * from the solar constant, the solar declination and the time of the year
	 */
	double phi;
	latitude_->get(t,phi);
	phi = phi * M_PI/180.; // get phi in radians [rad]

	// std::cout << "phi " << phi << std::endl;

	std::size_t myDateNumber = TimeConversion::dateToNumber(t);
//	std::cout << "Day of the Year " << myDateNumber << std::endl;

	double delta 	= 0.409*sin(2.*M_PI/365 *double(myDateNumber)-1.39);
	double dr 		= 1.+ 0.033 * cos(2.*M_PI/365 *double(myDateNumber));

	double tan_phi 	= tan(phi);
	double tan_del 	= tan(delta);
	double omega_s  = acos(-tan_phi * tan_del);
//	std::cout << "omega_s " << omega_s << std::endl;
	// OR the same with atan
//	double x = 1.- tan_phi*tan_phi * tan_del*tan_del; if(x <= 0.0) x = 0.00001;
//	double omega_s 	= M_PI*0.5 - atan(-tan_phi*tan_del/sqrt(x));

	double Ra = 24.*60. / M_PI  * Gsc* dr * (omega_s*sin(phi)*sin(delta) + cos(phi)*cos(delta)*sin(omega_s));
	// Ra is expressed in the above equation in MJ m-2 day-1. The corresponding equivalent evaporation in mm day-1 is obtained by multiplying Ra by 0.408

	// The latitude phi, expressed in radians is positive for the northern hemisphere and negative for the southern hemisphere
	// North Rhine Westphalia: 51°45'N Latitude
	// Cologne: 50°56'N
	// Juelich: 50°55'19"N --> 50.9253 decimal  // North is positive sign
	//
	// if phi in decimaldegrees, which means for juelich 50+55/60 = 50.9253
	// then phi = phi * M_PI/180 = 0.888814 (rad)

	/**
	 * Extraterrestrial radiation for HOURLY or shorter periods (Ra)
	 *
	 * t:	standard clock time at the midpoint of the period [hour]. For example for a period between 14.00 and 15.00 hours, t = 14.5,
	 * Lz:	longitude of the centre of the local time zone [degrees west of Greenwich]. For example, Lz = 75, 90, 105 and 120° for the Eastern, Central,
	 * 		Rocky Mountain and Pacific time zones (United States) and Lz = 0° for Greenwich, 330° for Cairo (Egypt), and 255° for Bangkok (Thailand),
	 * Lm: 	longitude of the measurement site [degrees west of Greenwich],
	 * Sc: 	seasonal correction for solar time [hour]
	 *
	 *
	 * double omega		= M_PI/12. * ( (t + 0.06667*(Lz - Lm)+Sc)-12. ); // solar time angle at midpoint of hourly or shorter period [rad]
	 * double omega_1 	= omega - M_PI* time1 / 24; // solar time angle at beginning of period [rad]
	 * double omega_2 	= omega + M_PI* time1 / 24; // solar time angle at end of period [rad]
	 *  // time1  length of the calculation period [hour]: i.e., 1 for hourly period or 0.5 for a 30-minute period.
	 *
	 * if(omega < -omega_s || omega > omega_s){  // sun is below horizon
	 * Ra = 0.;
	 * return;}
	 *
	 *  // The seasonal correction for solar time is:
	 * double b = 2.*M_PI*(double(myDateNumber)-81.)/364;
	 * double Sc = 0.1645*sin(2.*b)-0.1255*cos(b)-0.025*sin(b);
	 *
	 * The solar time angles at the beginning and end of the period are given by
	 * Ra = 12*60/M_PI * Gsc dr *( (omega_2-omega_1)*sin(phi)*sin(delta)+cos(phi)*cos(delta)*(sin(omega_1)-sin(omega_2) );
	 */

	/// --- Daylight hours ---
	double daylightHours = 24./M_PI * omega_s;
//	std::cout << "N " << daylightHours << std::endl;
	/// --- Solar radiation ---
	double a_s = 0.25;
	double b_s = 0.5;
	// Depending on atmospheric conditions (humidity, dust) and solar declination (latitude and month),
	// the Angstrom values as and bs will vary. Where no actual solar radiation data are available and
	// no calibration has been carried out for improved as and bs parameters,
	// the values as = 0.25 and bs = 0.50 are recommended.
	if (angstrom_as_) {
		angstrom_as_->get(t, a_s);
	}
	if (angstrom_bs_) {
		angstrom_bs_->get(t, b_s);
	}

	double Rs(0.0);
	if (netSolarRadiation_) {
		netSolarRadiation_->get(t, Rs);
	} else if (actualDurationofSunshine_) {
		double actualSunhours;
		actualDurationofSunshine_->get(t, actualSunhours);
		Rs = (a_s + b_s * actualSunhours / daylightHours) * Ra;
	} else {
		msg::error("Radiation::calculate: Provide either netSolarRadiation or actualDurationofSunshine parameter");
	}

	 /// --- Clear-sky solar radiation Rso:
	 double Rso = (a_s + b_s)*Ra;
	 // a_s+b_s fraction of extraterrestrial radiation reaching the earth on clear-sky days (actualSun = daylightHours)

	/// --- Net solar or net shortwave radiation (Rns) ---

	double albedo;
	/// albedo or canopy reflection coefficient, which is 0.23 for the hypothetical grass reference crop [dimensionless]
	/// albedo open water is 0.05
	albedo_->get(t, albedo);

	double Rns 		= (1.-albedo)*Rs;

	double const sigma 	= 0.000000004903; // Stefan-Boltzmann constant [4.903 10-9 MJ K-4 m-2 day-1],
	double stk4;

	if(minTemperature_ && maxTemperature_){
		double Tmin, Tmax;
		minTemperature_->get(t, Tmin);
		maxTemperature_->get(t, Tmax);
		double Tkmin = Tmin+273.15;
		double Tkmax = Tmax+273.15;
		double Tkmin4 = Tkmin*Tkmin*Tkmin*Tkmin;
		double Tkmax4 = Tkmax*Tkmax*Tkmax*Tkmax;
		stk4 = sigma*(Tkmin4 + Tkmax4)*0.5;
	}
	else {
		double temperature;
		dailyTemperature_->get(t, temperature);
		double Tk = temperature+273.15;
		stk4 = sigma * Tk *Tk *Tk *Tk;
	}

	double e_a;
	actualVaporPressure_->get(t,e_a);// in hPa
	e_a = e_a*0.1; // in kPa

//	std::cout << "ea  " << e_a << std::endl;
//	std::cout << "0.34 - 0.14*sqrt(e_a)  " << 0.34 - 0.14*std::sqrt(e_a) << std::endl;
//	std::cout << "1.35*Rs/Rso -0.35  " << 1.35*Rs/Rso -0.35 << std::endl;

	// 0.34 - 0.14*std::sqrt(e_a) this term is the net emissivity,
	// values 0.34 and 0.14 are correlation coefficients [Brunt, 1932; Jensen et al., 1990]
    double Rnl 	= stk4*(0.34 - 0.14*sqrt(e_a))*(1.35*Rs/Rso -0.35);
    /**
     * An average of the maximum air temperature to the fourth power and
     * the minimum air temperature to the fourth power is commonly used
     * in the Stefan-Boltzmann equation for 24-hour time steps. The term (0.34-0.14*sqrt(e_a))
     * expresses the correction for air humidity, and will be smaller if the humidity increases.
     * The effect of cloudiness is expressed by (1.35 Rs/Rso - 0.35).
     * The term becomes smaller if the cloudiness increases and hence Rs decreases.
     * The smaller the correction terms, the smaller the net outgoing flux of longwave radiation.
     * Note that the Rs/Rso term must be limited so that Rs/Rso <= 1.0.
     */

    /// --- net radiation (Rn):
    netRadiation = Rns - Rnl;

//    std::cout << "Ra " << Ra << std::endl;
//    std::cout << "Rso " << Rso << std::endl;
//    std::cout << "Rns " << Rns << std::endl;
//    std::cout << "Rnl " << Rnl << std::endl;
//    std::cout << "Rs " << Rs << std::endl;
//    std::cout << "netRadiation in MJ/m2/d " << netRadiation << std::endl;
//    std::cout << "daylightHours " << daylightHours << std::endl;
//    std::cout << "myActualVaporPressure in Radiation Method in kPa " << e_a << std::endl;

} // end calculate radiation

SineSolarElevationAngle::SineSolarElevationAngle(SimulaDynamic* const pSV):DerivativeBase(pSV), cleanUpTime(0){
	if (MAXTIMESTEP > 0.1) msg::error("SineSolarElevationAngle: Max timestep is bigger than 0.1! To avoid errors and inconsistencies it is mandatory that you set it to 0.1 or smaller when using the diurnal radiation simulator.");
	if (MINTIMESTEP > 0.1) msg::error("SineSolarElevationAngle: Min timestep is bigger than 0.1! To avoid errors and inconsistencies it is mandatory that you set it to 0.1 or smaller when using the diurnal radiation simulator.");
	std::size_t myDateNumber = TimeConversion::dateToNumber(0);
	startDay = double(myDateNumber);
	SimulaBase *probe = pSD->getPath("/environment/startYear");
	int year;
	probe->get(year);
	startYear = double(year);
	probe = pSD->getPath("/environment/latitude");
	double latitude;
	probe->get(latitude);
	latitude = latitude*M_PI/180;
	sinLatitude = sin(latitude);
	cosLatitude = cos(latitude);
	savedValues[-1] = 0;
	it1 = savedValues.begin();
}

std::string SineSolarElevationAngle::getName() const {
	return "sineSolarElevationAngle";
}

DerivativeBase * newInstantiationSineSolarElevationAngle(SimulaDynamic* const pSD){
	return new SineSolarElevationAngle(pSD);
}

/*****************************************************************************/
void SineSolarElevationAngle::calculate(const Time &t, double& sinEl){
/*****************************************************************************/
	/// Input:  Time, latitude, date, temperature, air pressure
	///atmosphere
	/// Output: radiation rate, Ra (W m-2).
	///
	/// We follow Michalsky, 1988, The astronomical almanac's algorithm for approximate solar position
	///
	/// Removed the dependence of longitude so we assume we are working in local solar time (12 noon is when the sun is at its highest)

	/** SYMBOLS
	*
	* delta = number of years passed since 1949
	* leap = number of leap years passed since 1949
	* julianDate = current Julian date
	* n = difference between current Julian date and 12:00, 1 January 2000
	* L = mean ecliptic longitude
	* g = mean ecliptic anomaly
	* l = ecliptic longitude
	* ep = obliquity of the ecliptic
	* gmst = greenwich mean standard time
	* lmst = local mean standard time
	* ra = right ascension of the sun
	* dec = declination of the sun
	* ha = hour angle of the sun
	* sinEl = sine of elevation angle of the sum
	* R = distance between earth and the sun
	* cosAOI = cosine of the angle of incidence of the sun
	*
	*/
	auto it = savedValues.find(t);
	if (it != savedValues.end()){
		sinEl = it->second;
		return;
	}

	double delta = startYear - 1949;
	double leap;
	std::modf(delta/4, &leap);
	double dayOnly;
	double hoursOnly = std::modf(t, &dayOnly);
	double julianDate = 2432916.5 + delta*365 + leap + startDay + t;
	double n = julianDate - 2451545;
	double L = 280.460 + 0.9856474*n;
	while (L >= 360){
		L = L - 360;
	}
	while (L < 0){
		L = L + 360;
	}
	double g = 357.528 + 0.9856003*n;
	while (g >= 360){
		g = g - 360;
	}
	while (g < 0){
		g = g + 360;
	}
	g = g*M_PI/180;
	double l = L + 1.915*sin(g) + 0.020*sin(2*g);
	while (l >= 360){
		l = l - 360;
	}
	while (l < 0){
		l = l + 360;
	}
	l = l*M_PI/180;
	double ep = 23.439 - 0.0000004*n;
	ep = ep*M_PI/180;
	double gmst = 6.697375 + 0.0657098242*n + 24*hoursOnly;
	double lmst = gmst;
	double ra = 0;
	double sinL = sin(l);
	double cosL = cos(l);
	if (cosL != 0) ra = atan2(cos(ep)*sinL,cosL);
	double sinDec = sin(ep)*sinL;
	double ha = lmst*M_PI/12 - ra;
	// instead of calculating the elevation angle, we calculate the sine of the elevation angle, since this is all that is needed in other functions
	sinEl = sinDec*sinLatitude + sqrt(1 - sinDec*sinDec)*cosLatitude*cos(ha);
	savedValues[t] = sinEl;
	if (t >= cleanUpTime + 5*MAXTIMESTEP){
		newCleanUpTime = t;
		it2 = std::prev(savedValues.end(), 1);
	}
	if (t >= newCleanUpTime + 5*MAXTIMESTEP){
		savedValues.erase(it1, it2);
		it1 = it2;
		cleanUpTime = newCleanUpTime;
	}
}

DiurnalRadiationSimulator::DiurnalRadiationSimulator(SimulaDynamic* const pSV):DerivativeBase(pSV), solarElevationAngle(nullptr), slope(0), saz(0), cleanUpTime(0){
	if (MAXTIMESTEP > 0.1) msg::error("DiurnalRadiationSimulator: Max timestep is bigger than 0.1! To avoid errors and inconsistencies it is mandatory that you set it to 0.1 or smaller when using the diurnal radiation simulator.");
	if (MINTIMESTEP > 0.1) msg::error("DiurnalRadiationSimulator: Min timestep is bigger than 0.1! To avoid errors and inconsistencies it is mandatory that you set it to 0.1 or smaller when using the diurnal radiation simulator.");
	std::size_t myDateNumber = TimeConversion::dateToNumber(0);
	startDay = double(myDateNumber);
	SimulaBase *probe = pSD->getPath("/environment/startYear");
	int year;
	probe->get(year);
	startYear = double(year);
	probe = pSD->getPath("/environment/latitude");
	double latitude;
	probe->get(latitude);
	latitude = latitude*M_PI/180;
	sinLatitude = sin(latitude);
	cosLatitude = cos(latitude);
	probe = pSD->existingPath("/environment/fieldSlope", "degrees");
	if (probe){
		probe->get(slope);
		slope = slope*M_PI/180;
		sinSlope = sin(slope);
		cosSlope = cos(slope);
	}
	probe = pSD->existingPath("/environment/fieldAzimuth");
	if (probe){
		probe->get(saz);
		saz = saz*M_PI/180;
	}
	pReferenceSolarRadiation = pSD->existingPath("/environment/atmosphere/solarRadiationAt1AU", "W/m2");
	pCloudCover = pSD->existingPath("/environment/atmosphere/cloudCover");
	solarElevationAngle = pSD->existingSibling("sineSolarElevationAngle");
	savedValues[-1] = 0;
	it1 = savedValues.begin();
}

std::string DiurnalRadiationSimulator::getName() const {
	return "diurnalRadiationSimulator";
}

DerivativeBase * newInstantiationDiurnalRadiationSimulator(SimulaDynamic* const pSD){
	return new DiurnalRadiationSimulator(pSD);
}

/*****************************************************************************/
void DiurnalRadiationSimulator::calculate(const Time &t, double& netRadiation){
/*****************************************************************************/
	/// Input:  Time, latitude, date, temperature, air pressure
	///atmosphere
	/// Output: radiation rate, Ra (W m-2).
	///
	/// We follow Michalsky, 1988, The astronomical almanac's algorithm for approximate solar position
	///
	/// Removed the dependence of longitude so we assume we are working in local solar time (12 noon is when the sun is at its highest)

	/** SYMBOLS
	*
	* delta = number of years passed since 1949
	* leap = number of leap years passed since 1949
	* julianDate = current Julian date
	* n = difference between current Julian date and 12:00, 1 January 2000
	* L = mean ecliptic longitude
	* g = mean ecliptic anomaly
	* l = ecliptic longitude
	* ep = obliquity of the ecliptic
	* gmst = greenwich mean standard time
	* lmst = local mean standard time
	* ra = right ascension of the sun
	* dec = declination of the sun
	* ha = hour angle of the sun
	* el = elevation angle of the sum
	* R = distance between earth and the sun
	* cosAOI = cosine of the angle of incidence of the sun
	*
	*/
	auto it = savedValues.find(t);
	if (it != savedValues.end()){
		netRadiation = it->second;
		return;
	}

	double delta = startYear - 1949;
	double leap;
	std::modf(delta/4, &leap);
	double dayOnly;
	double hoursOnly = std::modf(t, &dayOnly);
	double julianDate = 2432916.5 + delta*365 + leap + startDay + t;
	double n = julianDate - 2451545;
	double L = 280.460 + 0.9856474*n;
	while (L >= 360){
		L = L - 360;
	}
	while (L < 0){
		L = L + 360;
	}
	double g = 357.528 + 0.9856003*n;
	while (g >= 360){
		g = g - 360;
	}
	while (g < 0){
		g = g + 360;
	}
	g = g*M_PI/180;
	double sinEl, cosAOI, ra(0), ep, sinL;
	if (slope != 0 || !solarElevationAngle){
		double l = L + 1.915*sin(g) + 0.020*sin(2*g);
		while (l >= 360){
			l = l - 360;
		}
		while (l < 0){
			l = l + 360;
		}
		l = l*M_PI/180;
		ep = 23.439 - 0.0000004*n;
		ep = ep*M_PI/180;
		ra = 0;
		sinL = sin(l);
		double cosL = cos(l);
		if (cosL != 0) ra = atan2(cos(ep)*sinL, cosL);
	}
	if (solarElevationAngle){
		solarElevationAngle->get(t, sinEl);
	} else{
		double gmst = 6.697375 + 0.0657098242*n + 24*hoursOnly;
		double lmst = gmst;
		double sinDec = sin(ep)*sinL;
		double ha = lmst*M_PI/12 - ra;
		sinEl = sinDec*sinLatitude + sqrt(1 - sinDec*sinDec)*cosLatitude*cos(ha);
	}
	if (sinEl < 0){
		netRadiation = 0;
		savedValues[t] = netRadiation;
		return;
	}
	if (slope == 0){
		cosAOI = sinEl;
	} else{
		cosAOI = sinEl*cosSlope + sqrt(1 - sinEl*sinEl)*sinSlope*cos(ra - saz);
	}
	double R = 1.00014 - 0.01671*cos(g) - 0.00014*cos(2*g);
	netRadiation = 1362;
	if (pReferenceSolarRadiation) pReferenceSolarRadiation->get(t, netRadiation);
	netRadiation = netRadiation*0.8; // This atmosphere correction factor value makes it roughly equal to the beam + diffuse radiation if the atmospheric transmission coefficient is set to 0.72 at air pressures around 100240 Pa.
	netRadiation = netRadiation*cosAOI/(R*R);
	double clouds = 0.25;
	if (pCloudCover) pCloudCover->get(t, clouds);
	netRadiation = netRadiation*(1. - clouds);
	savedValues[t] = netRadiation;
	if (t >= cleanUpTime + 5*MAXTIMESTEP){
		newCleanUpTime = t;
		it2 = std::prev(savedValues.end(), 1);
	}
	if (t >= newCleanUpTime + 5*MAXTIMESTEP){
		savedValues.erase(it1, it2);
		it1 = it2;
		cleanUpTime = newCleanUpTime;
	}
}

PhotoPeriod::PhotoPeriod(SimulaDynamic* const pSV):DerivativeBase(pSV){
	std::size_t myDateNumber = TimeConversion::dateToNumber(0);
	startDay = double(myDateNumber);
	SimulaBase *probe = pSD->getPath("/environment/startYear");
	int year;
	probe->get(year);
	startYear = double(year);
	probe = pSD->getPath("/environment/latitude");
	double latitude;
	probe->get(latitude);
	latitude = latitude*M_PI/180;
	sinLatitude = sin(latitude);
	cosLatitude = cos(latitude);
}

std::string PhotoPeriod::getName() const {
	return "photoPeriod";
}

DerivativeBase * newInstantiationPhotoPeriod(SimulaDynamic* const pSD){
	return new PhotoPeriod(pSD);
}

/*****************************************************************************/
void PhotoPeriod::calculate(const Time &t, double& photoPeriod){
/*****************************************************************************/
	/// Input:  Time, latitude, date, temperature, air pressure
	///atmosphere
	/// Output: radiation rate, Ra (W m-2).
	///
	/// We follow Michalsky, 1988, The astronomical almanac's algorithm for approximate solar position
	///
	/// Removed the dependence of longitude so we assume we are working in local solar time (12 noon is when the sun is at its highest)

	/** SYMBOLS
	*
	* delta = number of years passed since 1949
	* leap = number of leap years passed since 1949
	* julianDate = current Julian date
	* n = difference between current Julian date and 12:00, 1 January 2000
	* L = mean ecliptic longitude
	* g = mean ecliptic anomaly
	* l = ecliptic longitude
	* ep = obliquity of the ecliptic
	* gmst = greenwich mean standard time
	* lmst = local mean standard time
	* ra = right ascension of the sun
	* dec = declination of the sun
	* ha = hour angle of the sun
	* el = elevation angle of the sum
	* R = distance between earth and the sun
	* cosAOI = cosine of the angle of incidence of the sun
	*
	*/

	auto it = savedValues.find(trunc(t));
	if (it != savedValues.end()){
		photoPeriod = it->second;
		return;
	}

	double currentDay = trunc(t);
	bool startWithLight;

	double delta = startYear - 1949;
	double leap;
	std::modf(delta/4, &leap);
	double dayOnly;
	double hoursOnly = std::modf(currentDay, &dayOnly);
	double julianDate = 2432916.5 + delta*365 + leap + startDay + currentDay;
	double n = julianDate - 2451545;
	double L = 280.460 + 0.9856474*n;
	while (L >= 360){
		L = L - 360;
	}
	while (L < 0){
		L = L + 360;
	}
	double g = 357.528 + 0.9856003*n;
	while (g >= 360){
		g = g - 360;
	}
	while (g < 0){
		g = g + 360;
	}
	g = g*M_PI/180;
	double l = L + 1.915*sin(g) + 0.020*sin(2*g);
	while (l >= 360){
		l = l - 360;
	}
	while (l < 0){
		l = l + 360;
	}
	l = l*M_PI/180;
	double ep = 23.439 - 0.0000004*n;
	ep = ep*M_PI/180;
	double gmst = 6.697375 + 0.0657098242*n + 24*hoursOnly;
	double lmst = gmst;
	double ra = 0;
	double sinL = sin(l);
	double cosL = cos(l);
	if (cosL != 0) ra = atan2(cos(ep)*sinL, cosL);
	double sinDec = sin(ep)*sinL;
	double ha = lmst*M_PI/12 - ra;
	double sinEl = sinDec*sinLatitude + sqrt(1 - sinDec*sinDec)*cosLatitude*cos(ha);
	if (sinEl > 0 ){
		startWithLight = true;
	} else{
		startWithLight = false;
	}
	double sunriseTime = -1;
	double sunsetTime = -1;
	for (double i = 0; i < 1.01; i += 0.001){
		double testTime = currentDay + i;
		std::modf(delta/4, &leap);
		hoursOnly = std::modf(testTime, &dayOnly);
		julianDate = 2432916.5 + delta*365 + leap + startDay + testTime;
		n = julianDate - 2451545;
		L = 280.460 + 0.9856474*n;
		while (L >= 360){
			L = L - 360;
		}
		while (L < 0){
			L = L + 360;
		}
		double g = 357.528 + 0.9856003*n;
		while (g >= 360){
			g = g - 360;
		}
		while (g < 0){
			g = g + 360;
		}
		g = g*M_PI/180;
		l = L + 1.915*sin(g) + 0.020*sin(2*g);
		while (l >= 360){
			l = l - 360;
		}
		while (l < 0){
			l = l + 360;
		}
		l = l*M_PI/180;
		ep = 23.439 - 0.0000004*n;
		ep = ep*M_PI/180;
		gmst = 6.697375 + 0.0657098242*n + 24*hoursOnly;
		lmst = gmst;
		ra = 0;
		sinL = sin(l);
		cosL = cos(l);
		if (cosL != 0) ra = atan2(cos(ep)*sinL, cosL);
		sinDec = sin(ep)*sinL;
		ha = lmst*M_PI/12 - ra;
		sinEl = sinDec*sinLatitude + sqrt(1 - sinDec*sinDec)*cosLatitude*cos(ha);
		if (startWithLight && sinEl < 0 && sunsetTime == -1){
			sunsetTime = testTime;
		}else if (startWithLight && sinEl > 0 && sunsetTime != -1 && sunriseTime == -1){
			sunriseTime = testTime;
		} else if (!startWithLight && sinEl > 0 && sunriseTime == -1){
			sunriseTime = testTime;
		} else if (!startWithLight && sinEl < 0 && sunriseTime != -1 && sunsetTime == -1){
			sunsetTime = testTime;
		}
		if (sunriseTime != -1 && sunsetTime != -1){
			if (startWithLight){
				photoPeriod = 24*(sunsetTime + 1 - sunriseTime);
			}else {
				photoPeriod = 24*(sunsetTime - sunriseTime);
			}
			savedValues[trunc(t)] = photoPeriod;
			return;
		}
	}
	msg::warning("PhotoPeriod: Either 0 or 24 hours sunlight per day.");
	photoPeriod = 0;
}

TimeUntilNextSunset::TimeUntilNextSunset(SimulaDynamic* const pSV):DerivativeBase(pSV){
	std::size_t myDateNumber = TimeConversion::dateToNumber(0);
	startDay = double(myDateNumber);
	SimulaBase *probe = pSD->getPath("/environment/startYear");
	int year;
	probe->get(year);
	startYear = double(year);
	probe = pSD->getPath("/environment/latitude");
	double latitude;
	probe->get(latitude);
	latitude = latitude*M_PI/180;
	sinLatitude = sin(latitude);
	cosLatitude = cos(latitude);
}

std::string TimeUntilNextSunset::getName() const {
	return "timeUntilNextSunset";
}

DerivativeBase * newInstantiationTimeUntilNextSunset(SimulaDynamic* const pSD){
	return new TimeUntilNextSunset(pSD);
}

/*****************************************************************************/
void TimeUntilNextSunset::calculate(const Time &t, double& timeUntilSunset){
/*****************************************************************************/
	/// Input:  Time, latitude, date, temperature, air pressure
	///atmosphere
	/// Output: radiation rate, Ra (W m-2).
	///
	/// We follow Michalsky, 1988, The astronomical almanac's algorithm for approximate solar position
	///
	/// Removed the dependence of longitude so we assume we are working in local solar time (12 noon is when the sun is at its highest)

	/** SYMBOLS
	*
	* delta = number of years passed since 1949
	* leap = number of leap years passed since 1949
	* julianDate = current Julian date
	* n = difference between current Julian date and 12:00, 1 January 2000
	* L = mean ecliptic longitude
	* g = mean ecliptic anomaly
	* l = ecliptic longitude
	* ep = obliquity of the ecliptic
	* gmst = greenwich mean standard time
	* lmst = local mean standard time
	* ra = right ascension of the sun
	* dec = declination of the sun
	* ha = hour angle of the sun
	* el = elevation angle of the sum
	* R = distance between earth and the sun
	* cosAOI = cosine of the angle of incidence of the sun
	*
	*/

	bool startFromSunsetTime(false);
	auto it = savedValues.find(trunc(t));
	if (it != savedValues.end()){
		if (it->second > t){
			timeUntilSunset = 24.*(it->second - t);
			return;
		}
		if (it->second < t){
			startFromSunsetTime = true;
			storedTime = it->second;
		}
	}

	double currentDay = trunc(t);
	bool day = false;
	for (double i = 0; i < 1.2; i += 0.001){
		double testTime;
		if (startFromSunsetTime){
			testTime = storedTime + 0.001 + i;
		} else{
			testTime = currentDay + i;
		}
		double delta = startYear - 1949;
		double leap;
		std::modf(delta/4, &leap);
		double dayOnly;
		double hoursOnly = std::modf(testTime, &dayOnly);
		double julianDate = 2432916.5 + delta*365 + leap + startDay + testTime;
		double n = julianDate - 2451545;
		double L = 280.460 + 0.9856474*n;
		while (L >= 360){
			L = L - 360;
		}
		while (L < 0){
			L = L + 360;
		}
		double g = 357.528 + 0.9856003*n;
		while (g >= 360){
			g = g - 360;
		}
		while (g < 0){
			g = g + 360;
		}
		g = g*M_PI/180;
		double l = L + 1.915*sin(g) + 0.020*sin(2*g);
		while (l >= 360){
			l = l - 360;
		}
		while (l < 0){
			l = l + 360;
		}
		l = l*M_PI/180;
		double ep = 23.439 - 0.0000004*n;
		ep = ep*M_PI/180;
		double gmst = 6.697375 + 0.0657098242*n + 24*hoursOnly;
		double lmst = gmst;
		double ra = 0;
		double sinL = sin(l);
		double cosL = cos(l);
		if (cosL != 0) ra = atan2(cos(ep)*sinL, cosL);
		double sinDec = sin(ep)*sinL;
		double ha = lmst*M_PI/12 - ra;
		double sinEl = sinDec*sinLatitude + sqrt(1 - sinDec*sinDec)*cosLatitude*cos(ha);
		if (sinEl <= 0 && day == true){
			savedValues[currentDay] = testTime;
			timeUntilSunset = 24.*(testTime - t);
			return;
		}
		if (sinEl > 0) day = true;
	}
	msg::error("TimeUntilNextSunset:: Can not calculate time until next sunset at t = " + std::to_string(t));
}

SoilRadiationFactor::SoilRadiationFactor(SimulaDynamic* const pSV):DerivativeBase(pSV){
	pMeanLeafAreaIndex = pSD->getPath("/plants/meanLeafAreaIndex");
}

std::string SoilRadiationFactor::getName() const {
	return "soilRadiationFactor";
}

DerivativeBase * newInstantiationSoilRadiationFactor(SimulaDynamic* const pSD){
	return new SoilRadiationFactor(pSD);
}

/*****************************************************************************/
void SoilRadiationFactor::calculate(const Time &t, double& factor){
/*****************************************************************************/
	pMeanLeafAreaIndex->get(t, factor);
	factor = 1 - factor;
	if (factor < 0.) factor = 0.;
}

BeamRadiationSimulator::BeamRadiationSimulator(SimulaDynamic* const pSV):DerivativeBase(pSV), solarElevationAngle(nullptr), slope(0), saz(0), cleanUpTime(0){
	if (MAXTIMESTEP > 0.1) msg::error("BeamRadiationSimulator: Max timestep is bigger than 0.1! To avoid errors and inconsistencies it is mandatory that you set it to 0.1 or smaller when using the diurnal radiation simulator.");
	if (MINTIMESTEP > 0.1) msg::error("BeamRadiationSimulator: Min timestep is bigger than 0.1! To avoid errors and inconsistencies it is mandatory that you set it to 0.1 or smaller when using the diurnal radiation simulator.");
	std::size_t myDateNumber = TimeConversion::dateToNumber(0);
	startDay = double(myDateNumber);
	SimulaBase *probe = ORIGIN->getPath("/environment/startYear");
	int year;
	probe->get(year);
	startYear = double(year);
	probe = ORIGIN->getPath("/environment/latitude");
	double latitude;
	probe->get(latitude);
	latitude = latitude*M_PI/180;
	sinLatitude = sin(latitude);
	cosLatitude = cos(latitude);
	probe = ORIGIN->existingPath("/environment/fieldSlope", "degrees");
	if (probe){
		probe->get(slope);
		slope = slope*M_PI/180;
		sinSlope = sin(slope);
		cosSlope = cos(slope);
	}
	probe = ORIGIN->existingPath("/environment/fieldAzimuth");
	if (probe){
		probe->get(saz);
		saz = saz*M_PI/180;
	}
	pAtmosphericPressure = ORIGIN->getPath("/atmosphere/airPressure", "Pa");
	pAtmosphericTransmissionCoefficient = ORIGIN->getPath("/environment/atmosphere/atmosphericTransmissionCoefficient");
	pReferenceSolarRadiation = ORIGIN->existingPath("/environment/atmosphere/solarRadiationAt1AU", "W/m2");
	solarElevationAngle = pSD->existingSibling("sineSolarElevationAngle");
	pCloudCover = pSD->existingPath("/environment/atmosphere/cloudCover");
	savedValues[-1] = 0;
	it1 = savedValues.begin();
}

std::string BeamRadiationSimulator::getName() const {
	return "beamRadiationSimulator";
}

DerivativeBase * newInstantiationBeamRadiationSimulator(SimulaDynamic* const pSD){
	return new BeamRadiationSimulator(pSD);
}

/*****************************************************************************/
void BeamRadiationSimulator::calculate(const Time &t, double& beamRadiation){
/*****************************************************************************/
	/// Input:  Time, latitude, date, temperature, air pressure
	///atmosphere
	/// Output: radiation rate, Ra (W m-2).
	///
	/// We follow Michalsky, 1988, The astronomical almanac's algorithm for approximate solar position
	///
	/// Removed the dependence of longitude so we assume we are working in local solar time (12 noon is when the sun is at its highest)

	/** SYMBOLS
	*
	* delta = number of years passed since 1949
	* leap = number of leap years passed since 1949
	* julianDate = current Julian date
	* n = difference between current Julian date and 12:00, 1 January 2000
	* L = mean ecliptic longitude
	* g = mean ecliptic anomaly
	* l = ecliptic longitude
	* ep = obliquity of the ecliptic
	* gmst = greenwich mean standard time
	* lmst = local mean standard time
	* ra = right ascension of the sun
	* dec = declination of the sun
	* ha = hour angle of the sun
	* el = elevation angle of the sum
	* R = distance between earth and the sun
	* cosAOI = cosine of the angle of incidence of the sun
	*
	*/
	auto it = savedValues.find(t);
	if (it != savedValues.end()){
		beamRadiation = it->second;
		return;
	}

	double delta = startYear - 1949;
	double leap;
	std::modf(delta/4, &leap);
	double dayOnly;
	double hoursOnly = std::modf(t, &dayOnly);
	double julianDate = 2432916.5 + delta*365 + leap + startDay + t;
	double n = julianDate - 2451545;
	double L = 280.460 + 0.9856474*n;
	while (L >= 360){
		L = L - 360;
	}
	while (L < 0){
		L = L + 360;
	}
	double g = 357.528 + 0.9856003*n;
	while (g >= 360){
		g = g - 360;
	}
	while (g < 0){
		g = g + 360;
	}
	g = g*M_PI/180;
	double sinEl, cosAOI, ep, sinL, ra(0);
	if (slope != 0 || !solarElevationAngle){
		double l = L + 1.915*sin(g) + 0.020*sin(2*g);
		while (l >= 360){
			l = l - 360;
		}
		while (l < 0){
			l = l + 360;
		}
		l = l*M_PI/180;
		ep = 23.439 - 0.0000004*n;
		ep = ep*M_PI/180;
		sinL = sin(l);
		double cosL = cos(l);
		if (cosL != 0) ra = atan2(cos(ep)*sinL, cosL);
	}
	if (solarElevationAngle){
		solarElevationAngle->get(t, sinEl);
	} else{
		double gmst = 6.697375 + 0.0657098242*n + 24*hoursOnly;
		double lmst = gmst;
		double sinDec = sin(ep)*sinL;
		double ha = lmst*M_PI/12 - ra;
		sinEl = sinDec*sinLatitude + sqrt(1 - sinDec*sinDec)*cosLatitude*cos(ha);
	}
	if (sinEl <= 0){
		beamRadiation = 0;
		savedValues[t] = beamRadiation;
		return;
	}
	if (slope == 0){
		cosAOI = sinEl;
	} else{
		cosAOI = sinEl*cosSlope + sqrt(1 - sinEl*sinEl)*sinSlope*cos(ra - saz);
	}
	double R = 1.00014 - 0.01671*cos(g) - 0.00014*cos(2*g);
	beamRadiation = 1362; // Solar constant
	if (pReferenceSolarRadiation) pReferenceSolarRadiation->get(t, beamRadiation);
	beamRadiation = beamRadiation*cosAOI/(R*R);
	double pressure, transmissionCoefficient;
	pAtmosphericPressure->get(t, pressure);
	pAtmosphericTransmissionCoefficient->get(t, transmissionCoefficient);
	double opticalAirMass = pressure/(101325*sinEl);
	beamRadiation = beamRadiation*pow(transmissionCoefficient, opticalAirMass);
	double clouds = 0.25;
	if (pCloudCover) pCloudCover->get(t, clouds);
	beamRadiation = beamRadiation*(1. - clouds);
	savedValues[t] = beamRadiation;
	if (t >= cleanUpTime + 5*MAXTIMESTEP){
		newCleanUpTime = t;
		it2 = std::prev(savedValues.end(), 1);
	}
	if (t >= newCleanUpTime + 5*MAXTIMESTEP){
		savedValues.erase(it1, it2);
		it1 = it2;
		cleanUpTime = newCleanUpTime;
	}
}

DiffuseRadiationSimulator::DiffuseRadiationSimulator(SimulaDynamic* const pSV):DerivativeBase(pSV), solarElevationAngle(nullptr), slope(0), saz(0), cleanUpTime(0){
	if (MAXTIMESTEP > 0.1) msg::error("DiffuseRadiationSimulator: Max timestep is bigger than 0.1! To avoid errors and inconsistencies it is mandatory that you set it to 0.1 or smaller when using the diurnal radiation simulator.");
	if (MINTIMESTEP > 0.1) msg::error("DiffuseRadiationSimulator: Min timestep is bigger than 0.1! To avoid errors and inconsistencies it is mandatory that you set it to 0.1 or smaller when using the diurnal radiation simulator.");
	std::size_t myDateNumber = TimeConversion::dateToNumber(0);
	startDay = double(myDateNumber);
	SimulaBase *probe = ORIGIN->getPath("/environment/startYear");
	int year;
	probe->get(year);
	startYear = double(year);
	probe = ORIGIN->getPath("/environment/latitude");
	double latitude;
	probe->get(latitude);
	latitude = latitude*M_PI/180;
	sinLatitude = sin(latitude);
	cosLatitude = cos(latitude);
	probe = ORIGIN->existingPath("/environment/fieldSlope", "degrees");
	if (probe){
		probe->get(slope);
		slope = slope*M_PI/180;
		sinSlope = sin(slope);
		cosSlope = cos(slope);
	}
	probe = ORIGIN->existingPath("/environment/fieldAzimuth");
	if (probe){
		probe->get(saz);
		saz = saz*M_PI/180;
	}
	pAtmosphericPressure = ORIGIN->getPath("/atmosphere/airPressure", "Pa");
	pAtmosphericTransmissionCoefficient = ORIGIN->getPath("/environment/atmosphere/atmosphericTransmissionCoefficient");
	pReferenceSolarRadiation = ORIGIN->existingPath("/environment/atmosphere/solarRadiationAt1AU", "W/m2");
	solarElevationAngle = ORIGIN->existingSibling("sineSolarElevationAngle");
	pCloudCover = pSD->existingPath("/environment/atmosphere/cloudCover");
	savedValues[-1] = 0;
	it1 = savedValues.begin();
}

std::string DiffuseRadiationSimulator::getName() const {
	return "diffuseRadiationSimulator";
}

DerivativeBase * newInstantiationDiffuseRadiationSimulator(SimulaDynamic* const pSD){
	return new DiffuseRadiationSimulator(pSD);
}

/*****************************************************************************/
void DiffuseRadiationSimulator::calculate(const Time &t, double& diffuseRadiation){
/*****************************************************************************/
	/// Input:  Time, latitude, date, temperature, air pressure
	///atmosphere
	/// Output: radiation rate, Ra (W m-2).
	///
	/// We follow Michalsky, 1988, The astronomical almanac's algorithm for approximate solar position
	///
	/// Removed the dependence of longitude so we assume we are working in local solar time (12 noon is when the sun is at its highest)

	/** SYMBOLS
	*
	* delta = number of years passed since 1949
	* leap = number of leap years passed since 1949
	* julianDate = current Julian date
	* n = difference between current Julian date and 12:00, 1 January 2000
	* L = mean ecliptic longitude
	* g = mean ecliptic anomaly
	* l = ecliptic longitude
	* ep = obliquity of the ecliptic
	* gmst = greenwich mean standard time
	* lmst = local mean standard time
	* ra = right ascension of the sun
	* dec = declination of the sun
	* ha = hour angle of the sun
	* el = elevation angle of the sum
	* R = distance between earth and the sun
	* cosAOI = cosine of the angle of incidence of the sun
	*
	*/
	auto it = savedValues.find(t);
	if (it != savedValues.end()){
		diffuseRadiation = it->second;
		return;
	}

	double delta = startYear - 1949;
	double leap;
	std::modf(delta/4, &leap);
	double dayOnly;
	double hoursOnly = std::modf(t, &dayOnly);
	double julianDate = 2432916.5 + delta*365 + leap + startDay + t;
	double n = julianDate - 2451545;
	double L = 280.460 + 0.9856474*n;
	while (L >= 360){
		L = L - 360;
	}
	while (L < 0){
		L = L + 360;
	}
	double g = 357.528 + 0.9856003*n;
	while (g >= 360){
		g = g - 360;
	}
	while (g < 0){
		g = g + 360;
	}
	g = g*M_PI/180;
	double sinEl, cosAOI, ep, sinL, ra(0);
	if (slope != 0 || !solarElevationAngle){
		double l = L + 1.915*sin(g) + 0.020*sin(2*g);
		while (l >= 360){
			l = l - 360;
		}
		while (l < 0){
			l = l + 360;
		}
		l = l*M_PI/180;
		ep = 23.439 - 0.0000004*n;
		ep = ep*M_PI/180;
		sinL = sin(l);
		double cosL = cos(l);
		if (cosL != 0) ra = atan2(cos(ep)*sinL, cosL);
	}
	if (solarElevationAngle){
		solarElevationAngle->get(t, sinEl);
	} else{
		double gmst = 6.697375 + 0.0657098242*n + 24*hoursOnly;
		double lmst = gmst;
		double sinDec = sin(ep)*sinL;
		double ha = lmst*M_PI/12 - ra;
		sinEl = sinDec*sinLatitude + sqrt(1 - sinDec*sinDec)*cosLatitude*cos(ha);
	}
	if (sinEl <= 0){
		diffuseRadiation = 0;
		savedValues[t] = diffuseRadiation;
		return;
	}
	if (slope == 0){
		cosAOI = sinEl;
	} else{
		cosAOI = sinEl*cosSlope + sqrt(1 - sinEl*sinEl)*sinSlope*cos(ra - saz);
	}
	double R = 1.00014 - 0.01671*cos(g) - 0.00014*cos(2*g);
	diffuseRadiation = 1362; // Solar constant
	if (pReferenceSolarRadiation) pReferenceSolarRadiation->get(t, diffuseRadiation);
	diffuseRadiation = diffuseRadiation*cosAOI/(R*R);
	double pressure, transmissionCoefficient;
	pAtmosphericPressure->get(t, pressure);
	pAtmosphericTransmissionCoefficient->get(t, transmissionCoefficient);
	double opticalAirMass = pressure/(101325*sinEl);
//	Equation taken from de Pury & Farquhar 1997 - Simple scaling of photosynthesis from leaves to canopies without the errors of big-leaf models
	diffuseRadiation = diffuseRadiation*(1 - pow(transmissionCoefficient, opticalAirMass))*0.426;
	double clouds = 0.25;
	if (pCloudCover) pCloudCover->get(t, clouds);
	diffuseRadiation = diffuseRadiation*(1. - clouds);
	savedValues[t] = diffuseRadiation;
	if (t >= cleanUpTime + 5*MAXTIMESTEP){
		newCleanUpTime = t;
		it2 = std::prev(savedValues.end(), 1);
	}
	if (t >= newCleanUpTime + 5*MAXTIMESTEP){
		savedValues.erase(it1, it2);
		it1 = it2;
		cleanUpTime = newCleanUpTime;
	}
}

//==================registration of the classes=================
class AutoRegisterRadiationClassInstantiationFunctions {
public:
	AutoRegisterRadiationClassInstantiationFunctions() {
		BaseClassesMap::getDerivativeBaseClasses()["Radiation"] = newInstantiationRadiation;
		BaseClassesMap::getDerivativeBaseClasses()["sineSolarElevationAngle"] = newInstantiationSineSolarElevationAngle;
		BaseClassesMap::getDerivativeBaseClasses()["diurnalRadiationSimulator"] = newInstantiationDiurnalRadiationSimulator;
		BaseClassesMap::getDerivativeBaseClasses()["photoPeriod"] = newInstantiationPhotoPeriod;
		BaseClassesMap::getDerivativeBaseClasses()["timeUntilNextSunset"] = newInstantiationTimeUntilNextSunset;
		BaseClassesMap::getDerivativeBaseClasses()["soilRadiationFactor"] = newInstantiationSoilRadiationFactor;
		BaseClassesMap::getDerivativeBaseClasses()["beamRadiationSimulator"] = newInstantiationBeamRadiationSimulator;
		BaseClassesMap::getDerivativeBaseClasses()["diffuseRadiationSimulator"] = newInstantiationDiffuseRadiationSimulator;
	}
};

// our one instance of the proxy
static AutoRegisterRadiationClassInstantiationFunctions p4556744351000;
