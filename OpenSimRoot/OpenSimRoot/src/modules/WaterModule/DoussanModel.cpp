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
#include "DoussanModel.hpp"

#include "../PlantType.hpp"

#include "../../math/pcgSolve.hpp"
#ifndef M_PI
#define M_PI (3.14159265358979323846)
#endif

LateralHydraulicConductivity::LateralHydraulicConductivity(SimulaDynamic* pSD):
		DerivativeBase(pSD), soilK(nullptr),pNumberOfXylemVessels(nullptr), pDiameterOfXylemVessels(nullptr), inverse(false), rootSegmentStartTime(pSD->getStartTime()),soilKconversionFactor(4.){
	//get name
	std::string name(pSD->getName());
	name.erase(0,11);//erase the segment word
	name[0]=tolower(name[0]);
	//get the root type parameters
	std::string plantType, rootType;
	PLANTTYPE(plantType, pSD);
	pSD->getParent(3)->getChild("rootType")->get(rootType);
	SimulaBase *parameters(GETROOTPARAMETERS(plantType, rootType));

	cond = pSD->existingSibling(name);//note do not take from parameter section straight

	//inverse
	if(name=="lateralHydraulicConductivity"){
		inverse=true;//serial system
		pNumberOfXylemVessels = parameters->existingChild("numberOfXylemVessels", "#");
		pDiameterOfXylemVessels = parameters->existingChild("diameterOfXylemVessels", "cm");
	}else if(name=="radialHydraulicConductivity"){
		inverse=false;//parallel system
	}else{
		msg::error("HydraulicConductivity: unkown type "+name);
	}
	Unit u;
	if (cond) {
		u=cond->getUnit();
	} else if (pNumberOfXylemVessels && pDiameterOfXylemVessels) {
		u = "cm4/day/hPa";
	}else{
		msg::error("HydraulicConductivity: Neither hydraulic conductivity or xylem vessel parameters have been found. Unable to determine hydraulic conductivity.");
	}
	if(u=="cm/day/hPa"){
		//surface area based
		size=pSD->getSibling("rootSegmentSurfaceArea");
		if (inverse) msg::warning("LateralHydraulicConductivity: You are using a surface based lateral hydraulic conductivity, this does not seem correct.");
	}else if(u=="cm4/day/hPa"){
		//length based
		size=pSD->getSibling("rootSegmentLength");
		if (!inverse) msg::warning("LateralHydraulicConductivity: You are using a length based radial hydraulic conductivity, this does not seem correct.");
	}else{
		size = nullptr;
		msg::warning("LateralHydraulicConductivity: unknown unit for "+cond->getName()+". Expected cm4/day/hpa or cm/day/hpa, not scaling the values to root geometry.");
	}

	//lateral conductivity according to Hagen–Poiseuille equation
	//is (pi r^4)/(8u l) where r is radius xylem vessel, l the length and u the dynamic viscosity of water (8.90 × 10−4 Pa·s)
	//units cm3/Pa/s
	if (!inverse) {
		RCSeffect =
				parameters->existingChild(
						"reductionInRadialHydraulicConductivityDueToCorticalSenescence");
		if (RCSeffect) {
			RCSstage = pSD->existingSibling("rootCorticalSenescenceStage");
			if (RCSstage) {
				msg::warning(
						"HydraulicConductivity: simulating effects of RCS on radial hydraulic conductivity");
			} else {
				msg::warning(
						"HydraulicConductivity: NOT simulating effects of RCS on radial hydraulic conductivity");
				RCSeffect = nullptr;
			}
		}

		//rhizosphere effect
		soilK =
				pSD->existingSibling(
						"soilHydraulicConductivityAtTheRootSurface");
		if(soilK) {
			msg::warning("HydraulicConductivity:: Using rhizosphere Ksoil for reducing root conductance. This is experimental.");
			//soilKconversionFactor
			// unit of Ksoil is cm/day, or really cm2/hPa/day, and multilied with root surface (cm2) to get cm4/hpa/day
			//unit of Kradial is cm4/day/hPa
			//e.g. the soilKconversionFactor is unitless, weights the resistance
			// the circumference of a root increases 4 times, if the diameter increases 4 times. note that this is all not very mechanistic and in the future is better replaced.
			SimulaBase* kc=parameters->existingChild(	"weightForRhizosphereConductanceInRadialConductance");
			if(kc) kc->get(soilKconversionFactor);
		}

	} else {
		RCSeffect = nullptr;
	}

}
std::string LateralHydraulicConductivity::getName() const {
	return "hydraulicConductivity";
}
void LateralHydraulicConductivity::calculate(const Time &t, double &c) {
	double s = 1;
	if (size) size->get(t, s);
	c = 0;
	if (cond) cond->get(t, c);
	if(inverse){
		if (pNumberOfXylemVessels){
			if (!pDiameterOfXylemVessels) msg::error("LateralHydraulicConductivity: diameterOfXylemVessels not found while numberOfXylemVessels is. Both or none are needed.");
			double numberOfXylemVessels, diameterOfXylemVessels;
			pNumberOfXylemVessels->get(t - rootSegmentStartTime, numberOfXylemVessels);
			pDiameterOfXylemVessels->get(t - rootSegmentStartTime, diameterOfXylemVessels);
			c = (M_PI*60*60*24*numberOfXylemVessels*pow(diameterOfXylemVessels/2, 4))/(8*8.9*pow(10, -6));
		}
		//by length, the longer the root the lower the lateral con
		if (s > 1e-9) {
			c /= s;
		} else {
			c /= 1e-9;
		}
	} else {
		c *= s;
	}
	if(RCSeffect){
		double e;
		RCSstage->get(t, s);
		RCSeffect->get(s, e);
		c *= (1 - e);
	}

	// todo this is very coarse and I guess better models have been produced.
	if(soilK){
		double sk;
		soilK->get(t,sk);
		sk *= s; // to go to root surface area. This is still for a 1 cm path length, without radial correction or what so ever.
		const double nc = 1/(1/c+soilKconversionFactor/sk); //sum of root and rhizosphere
		if(c>10*nc) msg::warning("Rhizosphere drying reduces radial conductance to roots by more than ten fold.");
		//std::cout<<std::endl<<"c="<<c<<" sk="<<sk<<" nc="<<nc;
		c=nc;
	}
}

DerivativeBase * newInstantiationLateralHydraulicConductivity(
		SimulaDynamic* const pSD) {
	return new LateralHydraulicConductivity(pSD);
}

WaterUptakeDoussanModel::WaterUptakeDoussanModel(SimulaDynamic* pSD) :
		DerivativeBase(pSD),psiCrit_(nullptr), psiRed_(nullptr), exponent_(nullptr), pRootWaterOutflowRate(nullptr), wuList(),
		KhList(), LrList(), psi_sList(), ni(), neighborMap(), psi_s(), Lr(), b(), psi_x(), C(), psiCollar(-100),
		collarnodeIndex(-1), largeKh(15), nodeCounter(0), collarKh(0.), useGravity(false) {

	pSD->avoidPredictorCorrectedLoops(); //speed up, should be ok
	SimulaBase::signalMeAboutNewObjects(pSD);
	pSD->getSibling("plantType")->get(plantType_);
	auto *ps=pSD->getSibling("plantPosition")->existingChild("shoot");
	if(ps){
		potentialTranspirationRate = ps->existingChild("potentialTranspirationRate", "cm3/day");
		potentialTranspiration = ps->existingChild("potentialTranspiration", "cm3");
	}
	if (!potentialTranspiration && !potentialTranspirationRate) {
		msg::warning(
				"WaterUptakeDoussanModel: potentialTranspiration not found, assuming predefined collar potential");
	}

	SimulaBase::Positions l;
	SimulaBase::getAllPositions(l);
	int count = 0;
	for (auto it(l.begin()); it != l.end(); ++it) {
		addObject(it->second);
		++count;
	}

	auto p = pSD->existingSibling("waterPotentialAtTheCollar", "hPa");
	if (p) {
		waterPotentialAtTheCollar = dynamic_cast<SimulaTable<double>*>(p);
		if (waterPotentialAtTheCollar && (potentialTranspiration || potentialTranspirationRate))
			waterPotentialAtTheCollar->set(pSD->getStartTime(), 0);
	} else {
		if (!potentialTranspiration && !potentialTranspirationRate)
			msg::error(
					"WaterUptakeDoussan: neither potentialTranspiration nor waterPotentialAtTheCollar given.");
	}
	
	//TODO check this	
	auto temp = pSD->existingSibling("rootWaterOutflowRate", "cm3/day");
	if (temp){
		pRootWaterOutflowRate = dynamic_cast<SimulaTable<double>*>(temp);
	}
	const auto plantType=plantType_;
	ps=ORIGIN->getChild("rootTypeParameters")->getChild(plantType)->existingChild("shoot");
	if(ps){
		psiCrit_       = ps->existingChild("minimumWaterPotentialAtTheCollar", "hPa");
		psiRed_      = ps->existingChild("waterPotentialAtTheCollarAtOnsetOfStomatalClosing", "hPa");
		exponent_= ps->existingChild("exponentForStomatalClosure");
	}
	if(!psiCrit_) msg::warning("Doussan: minimumWaterPotentialAtTheCollar not found, defaulting to -15000hPa.");
	if(!psiRed_) msg::warning("Doussan: waterPotentialAtTheCollarAtOnsetOfStomatalClosing not found, defaulting to -4000hPa.");
	if(!exponent_) msg::warning("Doussan: exponentForStomatalClosure not found, defaulting to 2");
}

std::string WaterUptakeDoussanModel::getName() const {
	return "waterUptakeDoussanModel";
}
void WaterUptakeDoussanModel::addObject(SimulaBase *rootNode) {
	//this is called each time a new root node is inserted.
	//check that this is a root node, not growth point or something else.
	std::string name = rootNode->getName();
	if (name.substr(0, 9) != "dataPoint") {
		msg::warning("WaterUptakeDoussanModel: Ignoring node with name " + name,
				3);
		return;
	}

	//check if the root node belongs to the same plant
	std::string plantType;
	PLANTTYPE(plantType, rootNode);
	if (plantType != plantType_)
		return;

	neighborMap[rootNode] = wuList.size();

	//store the node info in a structure
	SimulaTable<double>* pwu = dynamic_cast<SimulaTable<double>*>(rootNode->getChild("rootSegmentWaterUptakeRate"));

	if (!pwu)
		msg::error(
				"WaterUptakeDoussanModel: failed to find table with name rootSegmentWaterUptakeRate for storing data");
	wuList.push_back(pwu);

	//set initial value
	pwu->set(pwu->getStartTime(),0.);

}

void WaterUptakeDoussanModel::findConnections() {

	auto mit(wuList.begin()); //note that the neighbormap is sorted, and so changes in order all the time.
	if (nodeCounter)
		std::advance(mit, nodeCounter);

	for (; mit != wuList.end(); ++mit) {

		//Previous node in the graph
		SimulaBase * rootNode = (*mit)->getParent();
		SimulaBase * rootAnker = rootNode->getParent(2);
		SimulaBase * prevRootNode;
		if (rootAnker->getName() == "hypocotyl") {
			//note this returns nullptr at the collar node;
			prevRootNode = rootNode->getNextSibling(); //getParent()->getChildren()->getNext(rootNode->getName());
		} else {
			prevRootNode = rootNode->getPreviousSibling(); //getParent()->getChildren()->getPrevious(rootNode->getName());
			if (!prevRootNode) {
				//this is first node of the root, hook it up to the parent root
				if (rootAnker->getName() == "primaryRoot") {
					//this is a primary root -> connect it to the hypocotyl.
					prevRootNode = rootAnker->getSibling("hypocotyl")->getChild(
							"dataPoints")->getFirstChild();
				} else {
					//this is a branch -> find the nearest datapoint in the parent root.
					SimulaBase::List l;
					rootAnker->getParent(3)->getChild("dataPoints")->getAllChildren(
							l);
					if (l.empty())
						msg::error("WaterUptakeDoussan: can't connect branch");
					double d = 1e10;
					Coordinate pos, npos;
					rootAnker->get(pos); //note that rootNode has position 0 0 0 relative to the root anker
					for (auto it = l.begin(); it != l.end(); ++it) {
						(*it)->get(npos);
						npos -= pos;
						double ld = npos.squareSum();
						if (ld < d) {
							d = ld;
							prevRootNode = (*it);
						}
					}
				}
			}
		}

		//neigbor map
		int neighborIndex = -1; //negative to indicate collar node.
		if (prevRootNode) {
			auto it = neighborMap.find(prevRootNode);
			if (it != neighborMap.end()) {
				neighborIndex = it->second;
			} else {
				msg::error(
						"WaterUptakeDoussanModel::addObject: program bug detected: previous root node not found in neighbor map.");
			}
		} else {
			//collar node has negative number, but should point to this root node and this root node becomes the new collar node.
			for (std::size_t i = 0; i < ni.size(); ++i) {
				if (ni[i] == -1) {
					ni[i] = (int) ni.size();
					break; //found the collar node.
				}
			}
		}
		ni.push_back(neighborIndex);

		//collect node info
		SimulaBase * pLateralConductivity, *pRadialConductivity,
				*pRootSurfaceWaterPotential;
		//pRootSurface = rootNode->getChild("rootSegmentSurfaceArea", "cm2");
		//pRootLength = rootNode->getChild("rootSegmentLength","cm");
		pLateralConductivity = rootNode->getChild(
				"rootSegmentLateralHydraulicConductivity", "cm3/day/hPa");
		pRadialConductivity = rootNode->getChild(
				"rootSegmentRadialHydraulicConductivity", "cm3/day/hPa");
		pRootSurfaceWaterPotential = rootNode->getChild(
				"hydraulicHeadAtRootSurface", "cm");//cm water pressure, same as hPa

		//store the node info in a structure
		SimulaBase*ppxp = rootNode->existingChild("xylemWaterPotential");
		SimulaTable<double>* pxp = nullptr;
		if (ppxp)
			pxp = dynamic_cast<SimulaTable<double>*>(ppxp);

		xpList.push_back(pxp);
		KhList.push_back(pLateralConductivity);
		LrList.push_back(pRadialConductivity);
		psi_sList.push_back(pRootSurfaceWaterPotential);

		//lengthList.push_back(pRootLength);

		//msg::warning("WaterUptakeDoussanModel: Added root node for plant "+plantType_);
	}
	if (xpList.size() != neighborMap.size())
		msg::error("WaterUptakeDoussanModel:: programming error detected");
	nodeCounter = neighborMap.size();

	//write connection tree.
	/*std::cout<<std::endl<<"connections table";
	 for (unsigned int i=0 ; i<nodeCounter ; ++i){
	 auto j=ni[i];
	 if(j>=0){
	 std::cout<<std::endl<<" connecting "<<wuList[i]->getParent(3)->getName()<<"/"<<wuList[i]->getParent()->getName()<<" to "<<wuList[j]->getParent(3)->getName()<<"/"<<wuList[j]->getParent()->getName();
	 double v(0.);
	 //LrList[i]->get(0.,v);
	 //std::cout<<std::endl<<" lr of "<<LrList[i]->getParent(3)->getName()<<"/"<<LrList[i]->getParent()->getName()<<" is "<<v;
	 }else{
	 std::cout<<std::endl<<" connecting "<<wuList[i]->getParent(3)->getName()<<"/"<<wuList[i]->getParent()->getName()<<" to collarnode";
	 }

	 }*/

}
void WaterUptakeDoussanModel::build(const Time &t, const double &psiCrit, const double &psiRed, const double &exponent) {
	//delayed connecting of floating nodes.
	if (nodeCounter < neighborMap.size()) {
		findConnections();
	}

	//construct sparse matrix C and right hand side b
	std::size_t n(wuList.size());
	b.resize(n);
	C.resetToZero();//jp not sure if this is needed
	C.resize(n);
	Lr.resize(n);
	psi_s.resize(n);
	gravity.resize(n);
	if (psi_x.size() != n) {
		auto cp(psi_x);
		psi_x.resize(n, 0);
		for (std::size_t i = 0; i < cp.size() && i < n; ++i) {
			psi_x[i] = cp[i];
		}
	} else {
		msg::warning("WaterUptakeDoussanModel::build: using same matrix", 3);
	}

	//C and the diagonal for Lr
	gravity=0.;
	for (std::size_t i = 0; i < n; ++i) {
		double lr(0.), p(0.);
		SimulaBase *obj = LrList[i];
		if (obj && obj->evaluateTime(t)) {
			psi_sList[i]->get(t, p);
			obj->get(t, lr);

			//correct for gravity
			Coordinate pos,npos;
			obj->getAbsolute(t,pos);
			obj->getSibling("rootDiameter")->followChain(t)->getAbsolute(t,npos);
			if(useGravity) gravity[i]=(pos.y+npos.y)/2;
			p+=gravity[i];

			//scale down lr based on p
			double scaling=getScalingFactorRadialConductivity(p, psiCrit, psiRed, exponent);
			lr*=scaling;
			//if(lr<1.0e-20) p=-100.0; //just disable this
			//if(p<-1e6) p=-1e6; //avoid to negative values in grid
		}
		C.insert(i, i, lr);	//sets the diagonal , so that in next loop the addValueUnsafely can be called on it.
		Lr[i] = lr;
		psi_s[i] = p;
		b[i] = p * lr;
	}

	//C and boundary condition for collar node on the right hand side
	int count = 0;
	std::valarray<double> y(0., n);
	for (std::size_t i = 0; i < n; ++i) {
		auto j = ni[i];

		if (j < 0) {
			SimulaBase *obj = KhList[i];
			double v;
			if (obj && obj->evaluateTime(t)) {
				obj->get(t, v);
				if (v < 1.e-6)
					v = 1.e-6;
				if (v > largeKh)
					v = largeKh;
				Coordinate pos;		//,npos;
				obj->getAbsolute(t, pos);
				y[i] = pos.y;
			} else {
				v = largeKh;
			}
			//v=2;//for debuggin

			//collar node
			collarnodeIndex = i;
			collarKh = v;
			b[i] += v * psiCollar; // b(i) = b(i) + K_c * psi_c
			C.addValueUnsafely(i, i, v);
			++count;
			if (count > 1)
				msg::error(
						"WaterUptakeDoussanModel: Two collar nodes included, check code");
		} else {
			SimulaBase *obj = KhList[i];
			double v;
			if (obj && obj->evaluateTime(t)) {
				obj->get(t, v);
				if (v < 1.e-6)
					v = 1.e-6;
				if (v > largeKh)
					v = largeKh;
			} else {
				v = largeKh;
			}
			//v=2;//for debuggin
			C.addValueUnsafely(i, i, v);
			C.insert(i, j, -v); //this is twice, given symmetry
			C.addValueUnsafely(j, j, v);
		}
	}
}

void WaterUptakeDoussanModel::writeValues(const Time &t) {
	double outflowRate = 0;
	//set the uptake and xylem water potentials
	for (std::size_t i = 0; i < Lr.size(); ++i) {
		double jf = Lr[i] * (psi_s[i] - psi_x[i]);
		if(std::isnan(jf)) msg::error("WaterUptakeDoussanModel: Uptake is NaN. Lr="+std::to_string(Lr[i])+" PSIsoil="+std::to_string(psi_s[i])+" PSIxylem="+std::to_string(psi_x[i]));
		if (pRootWaterOutflowRate && jf < 0){
			outflowRate += jf;
		}
		if (wuList[i]->evaluateTime(t)) {
			wuList[i]->set(t, jf);
			if (xpList[i])
				xpList[i]->set(t, psi_x[i]-gravity[i]);
		} else {
			if (fabs(jf) > 1e-10)
				msg::warning(
						"WaterUptakeDoussanModel: uptake from a root segment outside it's lifetime");
		}
	}

	if (waterPotentialAtTheCollar && (potentialTranspiration || potentialTranspirationRate))
		waterPotentialAtTheCollar->set(t, psiCollar);
	if (pRootWaterOutflowRate) pRootWaterOutflowRate->set(t, outflowRate);
}

double WaterUptakeDoussanModel::getScalingFactorTrans(const double psiCollar_,const double & psiCrit, const double & psiRed, const double & exponent){
	double scaling;
	if(psiCollar_>psiRed){
		scaling=1.;
	}else if (psiCollar_<psiCrit){
		scaling=0.;
	}else{
		const double rr      =   (psiCollar_ - psiCrit) / (psiRed - psiCrit);
		scaling =  pow((0.5 - 0.5 * cos(M_PI * rr)), exponent);
	}
	return (scaling);
}
//todo now the same as trans, but needs theoretic basis. Possibly related to the mualem curve, not sure.
double WaterUptakeDoussanModel::getScalingFactorRadialConductivity(const double & h, const double & hCrit, const double & hRed, const double & exponent){
	double scaling;
	if(h>hRed){
		scaling=1.;
	}else if (h<hCrit){
		scaling=0.;
	}else{
		const double rr      =   (h - hCrit) / (hRed - hCrit);
		scaling   =  pow((0.5 - 0.5 * cos(M_PI * rr)), exponent);
	}
	return (scaling);
}



double WaterUptakeDoussanModel::getUptake(double& psiCollarMax, double& psiCollarMin, const double& potTrans, const double & psiCrit, const double & psiRed, const double & exponent, double & jsum, double & trans){
	//set transpiration
	if (psiCollar < psiCrit){
		trans = 0.;
	}else if (psiCollar < psiRed){
		trans = potTrans*getScalingFactorTrans(psiCollar, psiCrit,psiRed,exponent);
	}else{
		trans = potTrans;
	}
	//update boundary condition at the collar
	b[collarnodeIndex] = Lr[collarnodeIndex]*psi_s[collarnodeIndex] + collarKh*psiCollar;
	//Solve (result is psi_x).
	double pcgprecision=fabs(trans*1e-12+1e-15);
	Pcg::solve(C, b, psi_x, pcgprecision, 150000, SYM_GAUSS_SEIDEL);//note double precision only give us 14 digits
	//Pcg::solve(C, b, psi_x, pcgprecision, 150000, JACOBI);//note double precision only give us 14 digits
	//check result
//	std::valarray<double> b2(0.,b.size());
//	C.vectorMultiply(psi_x, b2);
//	b2-=b;
//	if(b2.max()>1e-5 || b2.min()<-1e-5){
//		std::cout<<"WaterUptakeDoussanModel::getUptake: PCG solver inaccurate solution";
//	}

	//sum of all the fluxes
	jsum = (Lr * (psi_s - psi_x)).sum();
	if(std::isnan(jsum)) {
		std::cout<<std::setprecision(8)<<std::endl;
										std::cout<<std::endl<<"matrix C:";
										std::cout<<std::setprecision(8)<<std::endl;

									    C.print_sparse();
									    std::cout<<std::endl<<"rhs : sol";
									    auto j = (Lr * (psi_s - psi_x));
									    for(unsigned int i=0;i<b.size();++i)
									    	std::cout<<std::setprecision(8)<<std::endl<<i<<" ni="<<ni[i]<<" b="<<b[i]<<" Lr="<<Lr[i]<<" psi_x="<<psi_x[i]<<" psi_s="<<psi_s[i]<<" j="<<j[i];
									    std::cout<<std::setprecision(8)<<std::endl<<"trans="<<trans<<" jsum="<<jsum<<" psiCollar="<<psiCollar;
									    std::cout<<std::setprecision(8)<<std::endl;
									    std::cout<<std::setprecision(8)<<std::endl;
		msg::error("WaterUptakeDoussanModel: jsum of the solution is NaN");
	}
	// range limits.
	if(trans>jsum) psiCollarMax=psiCollar; //Psicollar needs to be more negative
	if(trans<jsum) psiCollarMin=psiCollar; //Psicollar needs to be less negative

    return (jsum-trans);
}

void WaterUptakeDoussanModel::calculate(const Time &t, double &jsum) {
	//get the target transpiration rate. (note this must happen before we run find connections, as it could trigger a update
	double trans = 0.;
	double psiCrit = -15000; //15000 hPa or 1.5 MPa
	double psiRed = -4000; //  cm,  -4000 equal to -0.4 MPa
	double exponent = 2.;
	//todo, so far these are constants, maybe assume so and move this to the constructor?
	if(psiCrit_)   psiCrit_->get(t-pSD->getStartTime(),psiCrit);
	if (psiRed_) psiRed_->get(t-pSD->getStartTime(), psiRed);
	if (exponent_)	exponent_->get(t-pSD->getStartTime(), exponent);


	// build the matrix and righthand side
	// typically the conductance should go down faster than the closure of stomata which in our case happens with a lower exponent.
	int count=0;
	unsigned int n=wuList.size();
	build(t, psiCrit, psiRed, exponent/2.);
	//take car of callback loops that might have resulted from the get calls. These could have updated the wuList, and as a consequence, the matrix building is inconsistent.
	while(n!=wuList.size()) {
		n=wuList.size();
//		std::cout<<"WaterUptakeDoussanModel: Dependency loop at t="<<t<<" count="<<count;
		build(t, psiCrit, psiRed, exponent/2.);
		++count;
		if(count>10)  msg::error("WaterUptakeDoussanModel: wuList grew during building matrix. Retried 10 times without success ");
	}


	double psiCollarMin=-1e6;//needs some bounds otherwise jsum will be nan.
	double psiCollarMax=1e5;//needs som bounds otherwise jsum will be nan.
	if (!potentialTranspiration && !potentialTranspirationRate) {
		waterPotentialAtTheCollar->get(t, psiCollar);
		getUptake(psiCollarMax, psiCollarMin,trans,psiCrit,psiRed,exponent, jsum, trans);//works as long as collar is not touched we do not care about the error.
	} else {
		//const double delta=500;

		if (potentialTranspirationRate){
			potentialTranspirationRate->get(t, trans);
		} else {
			potentialTranspiration->getRate(t, trans);
	    }

	    const double potTrans=trans;
		double smalltrans = 1.e-5 +  0.0001 * fabs(trans); //minimum precision

	//minimize fabs(trans-jsum) by changing collarpotential
		// last times data

		double error = 1.0e30;//getUptake(psiCollarMax, psiCollarMin, potTrans,psiCrit,psiRed,exponent, jsum, trans);
		double psiCollarOld = psiCollar;
		double jsumOld = jsum;
		double transOld = trans;

		int count = 0;
		while (fabs(error)> smalltrans && count<50){
			double dpsi = -100.;
			switch (count) {
				case 0:
					dpsi=0;
					break;
				case 1:
					if(jsum>trans){
						dpsi=50;
					}else{
						dpsi=-50;
					}

					break;
				default:
					//if(abs(jsum - jsumOld)>1e-10){
					if(abs(psiCollarOld - psiCollar)>1e-10){
						//assuming linear curves for trans(psiCollar) and jsum(psicollar) which cross
						//note that these curves are only approx linear, see trans function and jsum is only linear if the conductances are constant, which they are not

						const double at=(trans-transOld)/(psiCollar-psiCollarOld);
						const double aj=(jsum-jsumOld)/(psiCollar-psiCollarOld);
						const double bt=transOld-at*psiCollarOld;
						const double bj=jsumOld-aj*psiCollarOld;
						dpsi = (bj-bt)/(at-aj)-psiCollar;

						//just to avoid the solution walks away. 
						double mdpsi=psiCollarMax-psiCollarMin;
						if(dpsi>mdpsi ) dpsi=mdpsi/2;
						if(dpsi<-mdpsi ) dpsi=-mdpsi/2;

					}else{
						if(jsum>trans){
							dpsi=(psiCollarMax-psiCollarMin)/2;
						}else{
							dpsi=(psiCollarMax-psiCollarMin)/2;
						}
					}
					break;
			}
			psiCollarOld = psiCollar;
			jsumOld = jsum;
			transOld = trans;
			psiCollar+=dpsi;

			error = getUptake(psiCollarMax, psiCollarMin, potTrans, psiCrit,psiRed,exponent, jsum, trans);
			++count;

			//if(count>3)
			//std::cout<<std::setprecision(8)<<std::endl<<"doussan iterations "<< count<<" time="<<t<<" jsum="<<jsum<<" trans="<<trans<<" potTrans="<<potTrans<<" psiCollar"<<psiCollar;
//
//			if (count > 50  || (psi_x.max()>psi_s.max() && fabs(error)<= smalltrans) ) {
//
//								std::cout<<std::setprecision(8)<<std::endl;
//								std::cout<<std::endl<<"matrix C:";
//								std::cout<<std::setprecision(8)<<std::endl;
//
//							    C.print_sparse();
//							    std::cout<<std::endl<<"rhs : sol";
//							    auto j = (Lr * (psi_s - psi_x));
//							    for(unsigned int i=0;i<b.size();++i)
//							    	std::cout<<std::setprecision(8)<<std::endl<<i<<" ni="<<ni[i]<<" b="<<b[i]<<" Lr="<<Lr[i]<<" psi_x="<<psi_x[i]<<" psi_s="<<psi_s[i]<<" j="<<j[i];
//							    std::cout<<std::setprecision(8)<<std::endl<<" t="<<t<<"trans="<<trans<<" jsum="<<jsum<<" psiCollar="<<psiCollar;
//							    std::cout<<std::setprecision(8)<<std::endl;
//							    std::cout<<std::setprecision(8)<<std::endl;
//
//							    build(t, psiCrit, psiRed, exponent/2.);
//							    C.print_sparse();
//
//								break;
//				}
			if(count>50) msg::warning("WaterUptakeDoussanModel: Can not find accurate solution for Doussan model at time = " + std::to_string(t)+ ", continuing with an error of (ml) "+std::to_string(error));
		}
	}
	if(std::isnan(psiCollar)) msg::error("WaterUptakeDoussanModel: psiCollar is NaN");

	//set the uptake for all the root nodes.
	writeValues(t);
}

DerivativeBase * newInstantiationWaterUptakeDoussanModel(
		SimulaDynamic* const pSD) {
	return new WaterUptakeDoussanModel(pSD);
}


//Schnepf implements doussan+gravity
class WaterUptakeSchnepf:public WaterUptakeDoussanModel{
public:
	WaterUptakeSchnepf(SimulaDynamic* const pSD):WaterUptakeDoussanModel(pSD){
		useGravity=true;
	}
	virtual std::string getName()const{
		return "waterUptakeAlmDoussanSchnepfModel";
	};

};

DerivativeBase * newInstantiationWaterUptakeSchnepf(
		SimulaDynamic* const pSD) {
	return new WaterUptakeSchnepf(pSD);
}


HydraulicConductivityRootSystem::HydraulicConductivityRootSystem(
		SimulaDynamic* pSD) :
		DerivativeBase(pSD) {
	Unit u = pSD->getUnit();
	if (u == "cm3/day/g/hPa") {
		size = pSD->getSibling("rootDryWeight", "g");
	} else if (u == "cm3/day/cm2/hPa") {
		size = pSD->getSibling("rootSurfaceArea", "cm2");
	} else if (u == "cm3/day/cm3/hPa") {
		size = pSD->getSibling("rootVolume", "cm3");
	} else {
		msg::error(
				"HydraulicConductivityRootSystem: unknown unit " + u.name
						+ " for " + pSD->getName()
						+ " use cm3/day/g/hPa, cm3/day/cm3/hPa or cm3/day/cm2/hPa");
	}
	collarPotential = pSD->getSibling("waterPotentialAtTheCollar", "hPa");
	flowrate = pSD->getSibling("rootWaterUptake", "cm3");

}
std::string HydraulicConductivityRootSystem::getName() const {
	return "hydraulicConductivityRootSystem";
}
void HydraulicConductivityRootSystem::calculate(const Time &t, double &K) {
	double s, h, f;
	size->get(t, s);
	if (s > 0) {
		collarPotential->get(t, h);
		flowrate->getRate(t, f);
		K = f / s / (-h); //ml/day/g/hPa
	} else {
		K = 0;
	}
}
DerivativeBase * newInstantiationHydraulicConductivityRootSystem(
		SimulaDynamic* const pSD) {
	return new HydraulicConductivityRootSystem(pSD);
}

//Register the module
static class AutoRegisterDoussanInstantiationFunctions {
public:
	AutoRegisterDoussanInstantiationFunctions() {
		BaseClassesMap::getDerivativeBaseClasses()["waterUptakeDoussanModel"] =
				newInstantiationWaterUptakeDoussanModel;
		BaseClassesMap::getDerivativeBaseClasses()["waterUptakeAlmDoussanSchnepfModel"] =
				newInstantiationWaterUptakeSchnepf;
		BaseClassesMap::getDerivativeBaseClasses()["hydraulicConductivity"] =
				newInstantiationLateralHydraulicConductivity;
		BaseClassesMap::getDerivativeBaseClasses()["hydraulicConductivityRootSystem"] =
				newInstantiationHydraulicConductivityRootSystem;
	}
} l6k5tfottos789g97f;

