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

#ifndef MATERIAL_HPP_
#define MATERIAL_HPP_

#include <fstream>
#include "../../cli/Messages.hpp"
#include "../../tools/StringExtensions.hpp"
#include "../../engine/DataDefinitions/Time.hpp"
#include "../../engine/Origin.hpp"
#include "Mesh.hpp"

// old version
//#define OLDFK
#define KuppeScalingFunction

class VanGenuchten {

public:
	VanGenuchten(const Mesh_base &mesh);
	~VanGenuchten(){}


	// these functions define the unsaturated
	// hydraulic properties for each soil material
	void calc_fk(const tVector &h,const tVector &waterContent, tVector & fk)const;
	tVector calc_fc(const tVector &h)const; //
	void calc_fq(const tVector &h, tVector & fq)const;//theta as a function of h
#ifndef KuppeScalingFunction
	double calc_fk(const tVector &waterContent, const int i) const;//index version for evaporation scaling
	void calc_fq(const double &h, const int i, double &theta) const ;
#endif
	//const tVector & gethSat()const   { return hSat; }
	const tVector & getthR()const   { return thr; }//check this !!!!
	const tVector & getthSat()const   { return ths; }
	const tVector & getVanGenuchten_n()const { return n;}
	const tVector & getKs() const { return ks;}


private:
	tIndex myNumNP;
	tVector hSat;
	tVector thr;
	tVector ths;
	tVector alpha;
	tVector n;
	tVector m;
	tVector lambda;
#ifdef OLDFK
	tVector kk;
#endif
	tVector ks;
	tVector thsr,anmthsr; // just to reduce computation at cost of memory.
	double minimumConductivity;
	std::valarray<int> mask;

	SimulaBase *pm_;
	void calc_fh(const double &theta, tVector &fh ); //h as a function of theta
	//double computeConductivity(const double & currentSe, const double & m, const double & n, const double & Ks)const;
};

#endif
