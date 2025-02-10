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

#include "Swms3d.hpp"

std::deque<SimulaBase *> Swms3d::rootNodeQueue_;
OutputSoilVTK * Swms3d::outputVTK(nullptr);

Watflow* Swms3d::primaryWater_(nullptr);

Swms3d::Swms3d(SimulaDynamic* const pSD) :
		DerivativeBase(pSD), water(nullptr), dt(fmin(0.001,pSD->preferedTimeStep())), /* start time */
		ItCum(0), /* TODO this can be deleted if we do not want information about the number of the whole iterations of watflow */
		Iter(5), /* set iter to 5 so that the timestep is not changed the first time round. */
		TLevel(1), /* Time-level (current time-step number) */
		//PLevel(0), /* Print time-level (current print-time number). */
		cumMineralisation(0.), dtOpt(dt), dtOld(dt), tOld(pSD->getStartTime()), /* initialize tOld (tOld = tInit = 0.) */
		tAtm(0), /* TODO see further comments on atm, atmosphere in this file */
		 TPrint(0), pdt(0), printVTU(false) {

}

void Swms3d::initialize_() {
	water = new Watflow(pSD);
	primaryWater_ = water;

	//backward compatibility mode
	SimulaBase* p = ORIGIN->existingPath("simulationControls/SWMS3D/nutrientToSimulate");
	std::string nutrient;
	if (p) {
		p->get(nutrient);
		solute.push_back(new Solute(*water, nutrient)); //only backward compatible instantiation of solute
		msg::warning("Swms3d:: simulating "+nutrient,1);
	}

	//just check for nutrients
	SimulaBase::List list;
	pSD->getParent()->getAllChildren(list);
	for (auto it : list) {
		SimulaBase* pp(it->existingChild("includeThisNutrientInSWMSSimulation"));
		int factor(1);
		SimulaBase* pm(it->existingChild("meshRefinementFactor"));
		if (pm)
			pm->get(factor);
// 		TODO meshdirection used defined or random or hard coded
//		int meshdirection(1);
//		SimulaBase* pd(it->existingChild("meshdirection"));
//		if(pd) pd->get(meshdirection);
		if (pp && it->getName() != nutrient) {
			bool b;
			pp->get(b);
			if (b)
				solute.push_back(new Solute(*water, it->getName(), factor));
			msg::warning("Swms3d:: simulating "+it->getName(),1);
		}
	}

	outputVTK = new OutputSoilVTK(water, solute);

//    SeepF  = false; // off in simroot (assuming there is not water table in the column)

	// tprint
	p = ORIGIN->existingPath("simulationControls/outputParameters/soilVTU/timeInterval");
	if(!p){
		p = ORIGIN->existingPath("simulationControls/outputParameters/VTU/timeInterval");
		if(!p){
			p = ORIGIN->getPath("simulationControls/outputParameters/defaults/timeInterval");
		}
	}
	p->get(pdt);

	p = ORIGIN->existingPath("simulationControls/outputParameters/soilVTU/startTime");
	if(!p){
		p = ORIGIN->existingPath("simulationControls/outputParameters/VTU/startTime");
		if(!p){
			p = ORIGIN->getPath("simulationControls/outputParameters/defaults/startTime");
		}
	}
	p->get(TPrint);
	p = ORIGIN->existingPath("simulationControls/outputParameters/soilVTU/run");
	if(!p){
		p = ORIGIN->existingPath("simulationControls/outputParameters/VTU/run");
	}
	if(p) p->get(printVTU); // 1
	if (TPrint <= tOld) {
		TPrint += pdt;
	}
	// initial step
	Time maxtimestepsolute(1e9);
	for (auto it : solute) {
		double m = it->getMaxTimeStep();
		if (maxtimestepsolute > m)
			maxtimestepsolute = m;
	}
	getTimeStepSWMS(dt, Iter, 1e9, tAtm, tOld, maxtimestepsolute);
	const double t=tOld+dt;

	// open pvd file and write header (vtd) for vtk output
	if (printVTU && TPrint < t+TIMEERROR) {
		outputVTK->outvtk_new(tOld);
	}
	while (TPrint<t+TIMEERROR) TPrint += pdt;

	//++PLevel; // so that we have Plevel = 0 initial and PLevel = 1 after the first print for calculate

}

std::string Swms3d::getName() const {
	return "Swms3d";
}

void Swms3d::addObject(SimulaBase *rootNode) {
	//match root nodes to fem nodes and add them to the list.
	std::string name = rootNode->getName();
	std::string parentname = rootNode->getParent()->getName();
	int pnl = parentname.size();
	if (name == "growthpoint" && parentname.substr(pnl - 8, pnl) != "Template") {
		//msg::warning("Simunek::addObject: Adding growthpoint");
		rootNodeQueue_.push_back(rootNode);
	} else if (name.substr(0, 9) == "dataPoint") {
		//msg::warning("Simunek::addObject: Adding node");
		rootNodeQueue_.push_back(rootNode);
	} else {
		//msg::warning("Simunek::addObject: Ignoring node with name "+name);
		return;
	}
}

void Swms3d::calculate(const Time &time, double &dummy) {
	if (!water)
		this->initialize_(); //finish what could not be done in the constructor.
	dummy = 0;

//	if(rootNodeQueue_.empty()) {
//		msg::error("Bad addObjects: rootNodeQueue_ empty");
//	}

	SimulaBase::updateAll(time);
	if (!rootNodeQueue_.empty()) {
		auto t=std::max(time,tOld);
		water->updateLists(t, rootNodeQueue_);

		for (auto it : solute) {
			it->updateLists(t, rootNodeQueue_);
		}
		rootNodeQueue_.clear();
	}

	// time loop -------------------------------------------

	Time oldWallTime=ObjectGenerator::setWallTime(time);//hack as this is not owned by SimulaTimeDriven
	while (tOld  < time) {
		// Time governing
		Time maxtimestepsolute(1e9);
		for (auto it : solute) {
			double m = it->getMaxTimeStep();
			if (maxtimestepsolute > m)
				maxtimestepsolute = m;
		}
		getTimeStepSWMS(dt, Iter, TPrint, tAtm, tOld, maxtimestepsolute);
		ObjectGenerator::setWallTime(tOld+dt);//hack as this is not owned by SimulaTimeDriven

		// Calculate water flow
		//note it can change dt
		double dtori(dt);
		water->calculate(ItCum, dt, pSD->minTimeStep(), dtOld, tOld, TLevel, Iter);
		if (dt < dtori - pSD->minTimeStep()){
			dtOpt = dt; //system got reset due to maxit was reached.
			ObjectGenerator::setWallTime(tOld+dt);//hack as this is not owned by SimulaTimeDriven
		}

		// Calculate solute transport
		for (auto it : solute) {
			it->calculate(tOld, dt);
		}

		double t = tOld + dt;
		if (printVTU && std::fabs(TPrint - t) < 1.1*TIMEERROR) {
				outputVTK->outvtk_new(t);
		}
		while (TPrint<t+TIMEERROR) TPrint += pdt;

		++TLevel;
		tOld = t;
		dtOld = dt;
		// --- end of time loop -------------------------------------------------
	}
	ObjectGenerator::resetWallTime(oldWallTime);//hack as this is not owned by SimulaTimeDriven
}

void Swms3d::getTimeStepSWMS(double & deltaT, const int& Iter, const double& tPrint, const double& tAtm, const double& lastTime, const double& dtMaxC) {
	//this should use a similar timestepping logic as int baseclasses.cpp
	//todo Should use the preferedTimeStep(), minTimeStep() etc from SimulaDynamic

	//preferred time step is dtOpt. this is increased if the number of iterations is small, and decreased if it is large
	//note that number of iterations is often not improved by decreasing the timestep.
	if (Iter < 2) {
		dtOpt *= 2.;
	} else if (Iter < 5) {
		dtOpt *= 1.5;
	} else if (Iter < 20) {
		//do nothing
	}else if (Iter < 100) {
		dtOpt /= 1.5;
	} else {
		dtOpt /= 2.;
	}
	dtOpt = std::min(dtMaxC, dtOpt);
	dtOpt = std::max(pSD->minTimeStep(), dtOpt);
	dtOpt = std::min(pSD->maxTimeStep(), dtOpt);

	if (dtMaxC < pSD->minTimeStep())
		msg::warning("Swms3d::getTimeStepSWMS: Ignoring Peclet courant criteria for timestep, as it reduces the timestep below minimum dt");

	/*!jouke: tAtm not implemented with simroot input files.
	 !todo it would be good to implement tatm so that precipitation is synchronized.
	 !everything goes alright as long a tprint and tatm fall together ( in our case on a daily basis)
	 */

	//round off timesteps to multiple of min time step, so that processes with same minTimeStep are in sync
	double newTime = lastTime + dtOpt;

	//timestep for printing
	if (tPrint > lastTime+TIMEERROR && newTime > tPrint) {
		newTime = tPrint;
	}

	//wall time
	Time tFix = SimulaTimeDriven::getWallTime(newTime);
	if (tFix > lastTime) {
		if (newTime > tFix) {
			newTime = tFix;
		}
	} else {
		msg::warning("Swms3d::getTimeStepSWMS:: wallTime before lastTime, this is bad - ignoring wall time.");
	}

	//syncing time
	Time synctime = MINTIMESTEP * floor((newTime + TIMEERROR) / MINTIMESTEP);
	if (synctime - lastTime > TIMEERROR) {
		// we can round off the timestep to a multiple of MINTIMESTEP
		//snapping the timestep to the default rhytm at which the other processes synchronize.
		deltaT = synctime - lastTime;
		if(deltaT==0.)
			msg::error("Swms3d::getTimeStepSWMS dt==0.");
	} else {
		//the TIMESTEP is smaller than MINTIMESTEP, and so we need to in small steps, such that we end up nicely at MINTIMESTEP
		//trying to balance the timestep so that the timestep snapping does not cause large oscillations in the timestep.
		//note this is possible if the process uses timesteps smaller than the default MINTIMESTEP, that is
		//MINTIMESTEP>pSD->getMinTimeStep() and we want to reduce the timestep so that multiple steps will
		//cause synchronization with MINTIMESTEP.
		double mindt = std::min(MINTIMESTEP, (tPrint - lastTime));
		int steps = 1+ (int) floor(mindt / (newTime - lastTime));
		deltaT = mindt / steps;
		if(deltaT==0.)
			msg::error("Swms3d::getTimeStepSWMS dt==0.");
	}

	newTime = lastTime + deltaT;
	//if(isnan(deltaT)) msg::error("GetTime: nan for dt");
	//std::cout<<std::endl<<"timestep swms "<<deltaT<<" at time "<<lastTime;

}

DerivativeBase * newInstantiationSwms3d(SimulaDynamic* const pSD) {
	return new Swms3d(pSD);
}

//Calls SWMS model so it will store the root surface concentration and hydraulic head in their respective tables
GetValuesFromSWMS::GetValuesFromSWMS(SimulaDynamic* pSD) :
		DerivativeBase(pSD) {
	//root diameter
	link = ORIGIN->getChild("soil")->existingChild("Swms3d");
	if (!link)
		link = ORIGIN->getChild("soil")->getChild("Simunek");
	self = dynamic_cast<SimulaTable<Time>*>(pSD);
	if (!self)
		msg::error("GetValuesFromSWMS: Failed dynamic cast to SimulaTable<Time> for " + pSD->getPath());
}
void GetValuesFromSWMS::calculate(const Time &t, double &v) {
	//update
	//todo, this needs loop protection, and then the loop protection in simulaExternal can be removed.
	link->get(t, v);

	//set initial value
	//this actually should not be necessary, but sometimes the table remains empty due to predictor corrected loops, and swms then simply returning
	//todo: now we keep tack of the growthpoint, we better use that, but would have to shift the time to the moment the gp passed the position.
	if (self->setStartTime()) {
		std::string dpn(pSD->getParent()->getName());
		std::string nn;
		///@todo since we do not simulate the conc//hh at the growth point surface area, we do not know what this value should be
		if (pSD->getName() == "nutrientConcentrationAtTheRootSurface") {
			nn = dpn;
			dpn = pSD->getParent(2)->getName();
			double v(0);
			if (dpn == "growthpoint") {
				self->set(self->getStartTime(), v);	//just set zero, as it starts inside the root
			} else {
				SimulaBase* pp(pSD->getParent(2)->getPreviousSibling());	//pSD->getParent(3)->getChildren()->getPrevious(dpn));
				if (pp) {
					pp->getChild(nn)->getChild(pSD->getName())->get(pSD->getStartTime(), v);
					self->set(self->getStartTime(), v);
				}else{
					self->set(self->getStartTime(), v);
					self->set(self->getStartTime() + MAXTIMESTEP, v);//point inside the root is not so important and we do not want the extrapolation warning message on these points
				}
			}
		} else if (pSD->getName() == "volumetricWaterContentAtTheRootSurface") {
			double v(0.2);
			if (dpn == "growthpoint") {
				self->set(self->getStartTime(), v);	//just set zero, as it starts inside the root
			} else {
				SimulaBase* pp(pSD->getParent()->getPreviousSibling());	//pSD->getParent(2)->getChildren()->getPrevious(dpn));
				if (pp) {
					pp->getChild(pSD->getName())->get(pSD->getStartTime(), v);
					self->set(self->getStartTime(), v);
				}else{
					self->set(self->getStartTime(), v);
					self->set(self->getStartTime() + MAXTIMESTEP, v);//point inside the root is not so important and we do not want the extrapolation warning message on these points
				}
			}
		} else if (pSD->getName() == "hydraulicHeadAtRootSurface") {
			double v(-10000);
			if (dpn == "growthpoint") {
				self->set(self->getStartTime(), v);	//just set zero, as it starts inside the root
			} else {
				SimulaBase* pp(pSD->getParent()->getPreviousSibling());	//pSD->getParent(2)->getChildren()->getPrevious(dpn));
				if (pp) {
					pp->getChild(pSD->getName())->get(pSD->getStartTime(), v);
					self->set(self->getStartTime(), v);
				}else{
					self->set(self->getStartTime(), v);
					self->set(self->getStartTime() + MAXTIMESTEP, v);//point inside the root is not so important and we do not want the extrapolation warning message on these points
				}
			}
		} else if (pSD->getName() == "soilHydraulicConductivityAtTheRootSurface") {
			double v(0.);
			if (dpn == "growthpoint") {
				self->set(self->getStartTime(), v);	//just set zero, as it starts inside the root
			} else {
				SimulaBase* pp(pSD->getParent()->getPreviousSibling());	//pSD->getParent(2)->getChildren()->getPrevious(dpn));
				if (pp) {
					pp->getChild(pSD->getName())->get(pSD->getStartTime(), v);
					self->set(self->getStartTime(), v);
				}else{
					self->set(self->getStartTime(), v);
					self->set(self->getStartTime() + MAXTIMESTEP, v);//point inside the root
				}
			}
		} else {
			self->set(self->getStartTime(), 0);
			msg::warning(
					"GetValuesFromSWMS: initializing with 0 for " + pSD->getPath()
							+ " as no initialization method for this data type has been implemented in the code.");
		}
	}
}
std::string GetValuesFromSWMS::getName() const {
	return "getValuesFromSWMS";
}
DerivativeBase * newInstantiationGetValuesFromSWMS(SimulaDynamic* const pSD) {
	return new GetValuesFromSWMS(pSD);
}

//==================registration of the classes=================
class AutoRegisterRateClassInstantiationSwms3d {
public:
	AutoRegisterRateClassInstantiationSwms3d() {
		BaseClassesMap::getDerivativeBaseClasses()["Swms3d"] = newInstantiationSwms3d;
		BaseClassesMap::getDerivativeBaseClasses()["Simunek"] = newInstantiationSwms3d;//just for backward compatibility
		BaseClassesMap::getDerivativeBaseClasses()["getValuesFromSWMS"] = newInstantiationGetValuesFromSWMS;
	}
};

// our one instance of the proxy
static AutoRegisterRateClassInstantiationSwms3d p4351486135415327;
