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

#ifndef SOILIMPEDANCE_HPP_
#define SOILIMPEDANCE_HPP_

#include "../../engine/BaseClasses.hpp"

/// Base class for all soil impedance calculations
///
/// Intended interface: query by time and absolute position, modules know how to find all other info.
///
/// TODO:
/// * Handle units. Currently returns a Coordinate whose vector length gives proportional slowdown, should explicitly convert bulkDensity (g/cm3) -> impedance (kPa) -> root extension (cm/day)
/// * derived classes for soil taxa and water content
class SoilImpedance: public DerivativeBase {
public:
	SoilImpedance(SimulaDynamic* const pSD);
	std::string getName()const;

	/// Look up impedance at a voxel
	///@{
	/// \param[in] t days after start of simulation
	/// \param[in] pos *absolute* xyz position of interest, probably the position of the growthpoint.
	/// \param[out] imped computed result
	///
	/// `imped` is always computed as a Coordinate with vector length equal to
	/// scalar impedence; the double form is just a shortcut for
	/// `get(t, pos, coord_imped); scalar_imped = coord_imped.length()`.
	virtual void calculate(const Time &t, const Coordinate &pos, Coordinate &imped);
	virtual void calculate(const Time &t, const Coordinate &pos, double &imped);
	virtual void calculate(SimulaBase* pCaller, const Time &t, double &imped);
	virtual void calculate(SimulaBase* pCaller, const Time &t, Coordinate &imped);
	/// @}
protected:
	static double theta2psi(const double theta, const double theta_sat, const double theta_resid, const double alpha, const double n);
};


/// Compute soil impedance from soil bulk density
///
/// Given a time and absolute position in the soil,
/// gets local bulk density and converts it to an impedance.
/// Work in progress! Units and conversion algorithm still undecided.
class ImpedanceFromBulkDensity : public SoilImpedance {
public:
	ImpedanceFromBulkDensity(SimulaDynamic* const pSD);
	std::string getName()const;

protected:
	void calculate(const Time &t, const Coordinate &pos, Coordinate &imped);
	SimulaBase *bulkDensity, *localWaterContent;
};


/// Compute soil impedance as per Gao et al 2016, accounting for weight of overburden soil
///
/// http://dx.doi.org/10.1016/j.still.2015.08.004
class ImpedanceGao : public SoilImpedance {
public:
	ImpedanceGao(SimulaDynamic* const pSD);
	std::string getName()const;

protected:
	void calculate(const Time &t, const Coordinate &pos, Coordinate &imped);
	void calculate(SimulaBase* pCaller, const Time &t, Coordinate &imped);
	SimulaBase *localWaterContent;
	static SimulaBase
		*bulkDensity,
		*saturatedWaterContent,
		*residualWaterContent,
		*voidRatio,
		*vanGenuchtenAlpha,
		*vanGenuchtenN;
	bool debug;
private:
	static double net_stress(
		const SimulaBase* density,
		const Time &t,
		const Coordinate &pos,
		double step = 1.0);
	static void precalculate_net_stress(
		std::map<double, double> &bulkd_cache,
		std::map<double, double> &stress_cache,
		const SimulaBase* density,
		const double bottom_depth,
		const double top_depth = 0.0,
		const double stepsize = 0.1);
	static std::map<double, double> cumulativeStress, cachedBulkDensity;
};

/// Compute soil impedance as per Whalley et al 2007
///
/// doi:10.1016/j.geoderma.2006.08.029
class ImpedanceWhalley : public SoilImpedance {
public:
	ImpedanceWhalley(SimulaDynamic* const pSD);
	std::string getName()const;

protected:
	void calculate(const Time &t, const Coordinate &pos, Coordinate &imped);
	SimulaBase
		*bulkDensity,
		*localWaterContent,
		*saturatedWaterContent,
		*residualWaterContent,
		*vanGenuchtenAlpha,
		*vanGenuchtenN;
};

#endif /*SOILIMPEDANCE_HPP_*/
