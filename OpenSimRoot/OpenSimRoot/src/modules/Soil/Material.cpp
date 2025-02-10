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
#include "Material.hpp"

#include "../../math/MathLibrary.hpp"

#include <math.h>

VanGenuchten::VanGenuchten(const Mesh_base &mesh) {
	myNumNP = mesh.getNumNP();
	thr.resize(myNumNP);
	ths.resize(myNumNP);
	alpha.resize(myNumNP);
	n.resize(myNumNP);

#ifdef OLDFK
	kk.resize(myNumNP);
#endif
	ks.resize(myNumNP);


	const tVector & x = mesh.getCordX();
	const tVector & y = mesh.getCordY();
	const tVector & z = mesh.getCordZ();

	SimulaBase* p = ORIGIN->getPath("environment/soil/water/residualWaterContent");
	tFloat Y;
	for (tIndex i = 0; i != myNumNP; ++i) {
		p->get(0, Coordinate(x[i], z[i], y[i]), Y);
		thr[i] = Y; // standard van genuchten
	}

	p = ORIGIN->getPath("environment/soil/water/saturatedWaterContent");
	for (tIndex i = 0; i != myNumNP; ++i) {
		p->get(0, Coordinate(x[i], z[i], y[i]), Y);
		ths[i] = Y;
	}

	p = ORIGIN->getPath(
			"environment/soil/water/vanGenuchten:alpha");
	for (tIndex i = 0; i != myNumNP; ++i) {
		p->get(0, Coordinate(x[i], z[i], y[i]), Y);
		alpha[i] = Y;
	}

	p = ORIGIN->getPath("environment/soil/water/vanGenuchten:n");
	for (tIndex i = 0; i != myNumNP; ++i) {
		p->get(0, Coordinate(x[i], z[i], y[i]), Y);
		n[i] = Y;
	}

	pm_ = ORIGIN->existingPath("environment/soil/water/vanGenuchten:m");
	m.resize(myNumNP);
	if(pm_){
		for (tIndex i = 0; i != myNumNP; ++i) {
			p->get(0, Coordinate(x[i], z[i], y[i]), Y);
			m[i] = Y;
		}
	}else{
		m = 1. - 1. / n;
	}
	pm_ = ORIGIN->existingPath("environment/soil/water/vanGenuchten:lambda");
	if(pm_){
		lambda.resize(myNumNP);
		for (tIndex i = 0; i != myNumNP; ++i) {
			p->get(0, Coordinate(x[i], z[i], y[i]), Y);
			lambda[i] = Y;
		}
	}else{
		lambda.resize(1);
		lambda[0] = 0.5;
	}

	minimumConductivity = 0;
	p = ORIGIN->existingPath("environment/soil/water/minimumConductivity");
	if (p){
		p->checkUnit("cm/day");
		p->get(minimumConductivity);
		msg::warning("vanGenuchten: Setting minimum conductivity, which can prevent the water model from crashing.");
	}
	mask = mesh.getpotmask();


	p = ORIGIN->getPath("environment/soil/water/saturatedConductivity");
	for (tIndex i = 0; i != myNumNP; ++i) {
		p->get(0, Coordinate(x[i], z[i], y[i]), Y);
		Y*=mesh.getpotmask()[i];
		ks[i] = Y; // Ks
#ifdef OLDFK
		kk[i] = Y; // measured conductivity (not used)
#endif
	}

	thsr= ths - thr ;//available water
	anmthsr = alpha * n * m * thsr;

	return;
}


#ifdef OLDFK
void VanGenuchten::calc_fk(const tVector &h,const tVector &waterContent, tVector & fk) const {
	double Kr, HMin, HH, Qees, Qeek, Hs, Hk, Qee, Qe, Qek, FFQ, FFQk;
	for (tIndex i = 0; i < myNumNP; ++i) {
		HMin = -std::pow(1.e30, (1. / n[i])) / std::fmax(alpha[i], 1.0);
		HH = std::fmax(h[i], HMin);
		Qees = std::fmin((ths[i] - tha[i]) / (thm[i] - tha[i]),
				.999999999999999);
		Qeek = std::fmin((thk[i] - tha[i]) / (thm[i] - tha[i]), Qees);
		Hs = -1. / alpha[i] * std::pow((std::pow(Qees, (-1. / m[i])) - 1.), (1. / n[i]));
		Hk = -1. / alpha[i] * std::pow((std::pow(Qeek, (-1. / m[i])) - 1.), (1. / n[i]));
		if (h[i] < Hk) {
			Qee = std::pow((1. + std::pow((-alpha[i] * HH), n[i])), (-m[i]));
			Qe = (thm[i] - tha[i]) / (ths[i] - tha[i]) * Qee;
			Qek = (thm[i] - tha[i]) / (ths[i] - tha[i]) * Qeek;
			FFQ = 1. - std::pow((1. - std::pow(Qee, (1. / m[i]))), m[i]);
			FFQk = 1. - std::pow((1. - std::pow(Qeek, (1. / m[i]))), m[i]);
			if (FFQ < 0.0) {
				FFQ = m[i] * std::pow(Qee, (1. / m[i]));
			}
			Kr = std::pow((Qe / Qek), 0.5) * std::pow((FFQ / FFQk), 2.0) * kk[i] / ks[i];
			fk[i] = std::fmax(ks[i] * Kr, 1.e-37);
		} else {
			if (h[i] < Hs) {
				Kr = (1. - kk[i] / ks[i]) / (Hs - Hk) * (h[i] - Hs) + 1.0;
				fk[i] = ks[i] * Kr;
			} else {
				fk[i] = ks[i];
			}
		}
	}
}
#else
//this computes k from theta, the old one from h.
void VanGenuchten::calc_fk(const tVector &h,const tVector &waterContent, tVector & fk) const {
	// if separate m is used
	if (pm_) {
		for (tIndex i = 0; i < myNumNP; ++i) {
			const double currentSe = (waterContent[i] - thr[i]) / thsr[i];
			const double p = m[i] + 1. / n[i];
			const double q = 1. - 1. / n[i];
			const double z = std::pow(currentSe, (1. / m[i]));
			const double myBeta = incompleteBetaFunction(p, q, z);//very heavy function with iterative loop

			double temp = ks[i] * currentSe * (myBeta * myBeta);
			if (temp < minimumConductivity*mask[i]) temp = minimumConductivity*mask[i];
			fk[i] = temp;
			//if (fk[i]> ks[i]) 	fk[i] = ks[i];
		}
	} else {
		if (lambda.size() == myNumNP) {
			for (tIndex i = 0; i < myNumNP; ++i) {
				const double currentSe = (waterContent[i] - thr[i]) / thsr[i];
				const double a = std::pow(currentSe, 1. / m[i]); //as z
				const double b = std::pow(1. - a, m[i]);
				double temp = ks[i] * std::pow(currentSe, lambda[i]) * std::pow(1. - b, 2);
				if (temp < minimumConductivity*mask[i]) temp = minimumConductivity*mask[i];
				fk[i] = temp;
				//if (fk[i]> ks[i]) 	fk[i] = ks[i];
			}
		} else {
			for (tIndex i = 0; i < myNumNP; ++i) {
				const double currentSe = (waterContent[i] - thr[i]) / thsr[i];
				const double a = std::pow(currentSe, 1. / m[i]); //as z
				const double b = std::pow(1. - a, m[i]);
				double temp = ks[i] * std::sqrt(currentSe) * std::pow(1. - b, 2);
				if (temp < minimumConductivity*mask[i]) temp = minimumConductivity*mask[i];
				fk[i] = temp;
				//if (fk[i]> ks[i]) 	fk[i] = ks[i];
			}
		}
	}
}
#endif

#ifndef KuppeScalingFunction
//non-array version for the scaling factor
double VanGenuchten::calc_fk(const tVector &waterContent, const int i) const {
	// if separate m is used
	double fk;
	if (pm_) {
			const double currentSe = (waterContent[i] - thr[i]) / thsr[i];
			const double p = m[i] + 1. / n[i];
			const double q = 1. - 1. / n[i];
			const double z = std::pow(currentSe, (1. / m[i]));
			const double myBeta = incompleteBetaFunction(p, q, z);//very heavy function with iterative loop
			fk = ks[i] * currentSe * (myBeta * myBeta);
			//if (fk[i]> ks[i]) 	fk[i] = ks[i];
	} else {
			const double currentSe = (waterContent[i] - thr[i]) / thsr[i];
			const double a = std::pow(currentSe, 1. / m[i]);//as z
			const double b = std::pow(1. - a, m[i]);
			fk = ks[i] * std::sqrt(currentSe) * std::pow(1. - b, 2);
			//if (fk[i]> ks[i]) 	fk[i] = ks[i];
	}
	return fk;
}
#endif

// =============================================================================

/*double VanGenuchten::computeConductivity(const double &currentSe, const double &m, const double &n, const double &Ks) const {
	const double p = m + 1. / n;
	const double q = 1. - 1. / n;
	const double z = std::pow(currentSe, (1. / m));
	const double myBeta = incompleteBetaFunction(p, q, z);
	return (Ks * currentSe * (myBeta * myBeta));
}*/

#ifdef OLDFK
void VanGenuchten::calc_fc(const tVector &h, tVector & fc) const {
	double HMin, HH, Qees, Hs, C1, C2;
	for (tIndex i = 0; i < myNumNP; ++i) {
		//a during runtime very negative number, stays constant
		HMin = -std::pow(1.e30, (1. / n[i])) / std::fmax(alpha[i], 1.);
		//just checks that h is not extremely negative
		HH = std::fmax(h[i], HMin);
		//todo qees never changes, so why is it computed every time??
		//tha in our case is same as thr and thm is same as ths so Qees is simply 0.999999 in our formulation
		Qees = std::fmin((ths[i] - tha[i]) / (thm[i] - tha[i]), 0.999999999999999);//must be <1.
		//Hs is extremely small, close to 0.
		Hs = -1. / alpha[i]
				* std::pow((std::pow(Qees, (-1. / m[i]) ) - 1.), (1. / n[i]));
		if (h[i] < Hs) {
			C1 = std::pow((1. + std::pow((-alpha[i] * HH), n[i])), (-m[i] - 1.));
			C2 = (thm[i] - tha[i]) * m[i] * n[i] * (std::pow(alpha[i],n[i])) * std::pow((-HH), (n[i] - 1.)) * C1;
			fc[i] = std::fmax(C2, 1.e-37);
		} else {
			fc[i] = 0.0;
		}
	}
}
#else

// water capacity
tVector VanGenuchten::calc_fc(const tVector &h) const {
	tVector fc(0., h.size());
	for (tIndex i = 0; i < myNumNP; ++i) {
		const double ah = alpha[i] * std::fabs(h[i]);
		fc[i] = anmthsr[i]  * std::pow(ah, n[i] - 1.) / std::pow(std::pow(ah, n[i]) + 1., m[i] + 1.);
	}
	return fc;
}

#endif

// ***********************************************************************
//  FQ is for theta
#ifdef OLDFK
void VanGenuchten::calc_fq(const tVector &h, tVector &fq) const {
	double HMin, HH, Qees, Hs, Qee;
	for (tIndex i = 0; i < myNumNP; ++i) {
		HMin = -std::pow(1.e30, (1. / n[i])) / std::fmax(alpha[i], 1.);
		HH = std::fmax(h[i], HMin);
		Qees = std::fmin((ths[i] - tha[i]) / (thm[i] - tha[i]), .999999999999999);
		Hs = -1. / alpha[i] * std::pow((std::pow(Qees, (-1. / m[i])) - 1.), (1. / n[i]));
		if (h[i] < Hs) {
			Qee = std::pow((1. + std::pow((-alpha[i] * HH), n[i])), (-m[i]));
			fq[i] = std::fmax(tha[i] + (thm[i] - tha[i]) * Qee, 1.e-37);
		} else {
			fq[i] = ths[i];
		}
	}
}
#else
// theta = f(h)
void VanGenuchten::calc_fq(const tVector &h, tVector &theta) const {
	//sp.theta_R + (sp.theta_S-sp.theta_R)/pow(1. + pow(sp.alpha*abs(h),sp.n),sp.m)
	for (tIndex i = 0; i < myNumNP; ++i) {
		if (h[i] < 0.) {
			theta[i] = thr[i] + thsr[i]/std::pow(1. + std::pow(alpha[i]*std::fabs(h[i]),n[i]),m[i]);
		}else{
			theta[i] = ths[i];
		}
	}
}
#endif

#ifndef KuppeScalingFunction
//non vector version
void VanGenuchten::calc_fq(const double &h, const int i, double &theta) const {
	//sp.theta_R + (sp.theta_S-sp.theta_R)/pow(1. + pow(sp.alpha*abs(h),sp.n),sp.m)
		if (h < 0.) {
			theta = thr[i] + thsr[i]/std::pow(1. + std::pow(alpha[i]*std::fabs(h),n[i]),m[i]);
		}else{
			theta = ths[i];
		}
}
#endif

// h=f(theta) used to calculate what h is when saturated
void VanGenuchten::calc_fh(const double &theta, tVector &fh){
	for (tIndex i = 0; i < myNumNP; ++i) {
		const double hh  = thsr[i]/(theta-thr[i]);
		const double hhh = std::pow(hh,(1./m[i]));
		fh[i] = - ( std::pow((hhh - 1.),(1./n[i])) / alpha[i]);
	}
}

