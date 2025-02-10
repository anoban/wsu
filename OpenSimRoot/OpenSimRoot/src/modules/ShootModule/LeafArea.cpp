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

#include "LeafArea.hpp"
#include "../../cli/Messages.hpp"
#include "../../engine/Origin.hpp"
#include "../PlantType.hpp"
#include <math.h>

#if _WIN32 || _WIN64
#include <algorithm>
#endif

PotentialLeafArea::PotentialLeafArea(SimulaDynamic* const pSV):
	DerivativeBase(pSV),LASimulator(nullptr)
{
	//check if unit given in input file agrees with this function
	pSD->checkUnit("cm2");
	//planting time
	plantingTime=pSD->getStartTime();
	//plant type
	std::string plantType;
	PLANTTYPE(plantType,pSD)
	//plant parameters
	if (pSD->getParent()->getName() == "mainShoot"){
		LASimulator = GETSHOOTPARAMETERS(plantType)->existingChild("leafAreaExpansionRate", "cm2/day");
		if (!LASimulator) LASimulator=GETSHOOTPARAMETERS(plantType)->getChild("leafAreaExpantionRate","cm2/day");
	}else if(pSD->getParent()->getName()!="shoot"){
		if (GETSHOOTPARAMETERS(plantType)->existingChild("tillers")){
			LASimulator=GETSHOOTPARAMETERS(plantType)->getChild("tillers")->existingChild("leafAreaExpansionRatePerTiller", "cm2/day");
			if (!LASimulator) LASimulator=GETSHOOTPARAMETERS(plantType)->getChild("tillers")->existingChild("leafAreaExpantionRatePerTiller", "cm2/day");
		}
		if (!LASimulator) LASimulator=GETSHOOTPARAMETERS(plantType)->existingChild("leafAreaExpansionRatePerTiller","cm2/day");
		if (!LASimulator) LASimulator=GETSHOOTPARAMETERS(plantType)->getChild("leafAreaExpantionRatePerTiller","cm2/day");
	}else{
		SimulaBase *p(ORIGIN->existingChild("tillerTemplate"));
		if(p && p->existingChild("potentialLeafArea")){
			//add up tillers;
			msg::warning("PotentialLeafArea::leafArea is sum of tillers");
		}else{
			LASimulator=GETSHOOTPARAMETERS(plantType)->existingChild("leafAreaExpansionRate","cm2/day");
			if (!LASimulator) LASimulator=GETSHOOTPARAMETERS(plantType)->getChild("leafAreaExpantionRate","cm2/day");
		}
	}
}
void PotentialLeafArea::calculate(const Time &t, double& r){
	//get rate from parameter set
	if(LASimulator){
		LASimulator->get(t-plantingTime,r);
	}else{
		SimulaBase::List l;
		pSD->getSibling("tillers")->getAllChildren(l,t);
		pSD->getSibling("mainShoot")->getChild("potentialLeafArea")->getRate(t,r);
		for(auto &i:l){
			double la;
			i->getChild("potentialLeafArea")->getRate(t,la);
			r+=la;
		}
	}
}
std::string PotentialLeafArea::getName()const{
	return "potentialLeafGrowthRate";
}

DerivativeBase * newInstantiationPotentialLeafArea(SimulaDynamic* const pSV){
   return new PotentialLeafArea(pSV);
}

//if we are simulating stress - switch to relative leafAreaExpantionRate as soon as stress factor has reduced growth
StressAdjustedPotentialLeafArea::StressAdjustedPotentialLeafArea(SimulaDynamic* const pSV):
	DerivativeBase(pSV)
{
	//simulators
	rgrCoefficient=pSD->existingSibling("leafAreaReductionCoefficient","cm2/cm2");
	potential=pSD->getSibling("potentialLeafArea",pSD->getUnit());
	//stress factor
	stress=pSD->getParent(3)->existingChild("stressFactor:impactOn:leafAreaExpansionRate");
	if (!stress) stress=pSD->getParent(3)->existingChild("stressFactor:impactOn:leafAreaExpantionRate");
	if(!stress) {
		stress=pSD->getParent(3)->getChild("stressFactor");
		msg::warning("StressAdjustedPotentialLeafArea:: \"stressFactor:impactOn:leafAreaExpantionRate\" or \"stressFactor:impactOn:leafAreaExpansionRate\" not found, using raw stress factor values");
	}
}
void StressAdjustedPotentialLeafArea::calculate(const Time &t, double& r){
	//get rate from parameter set
	potential->getRate(t,r);
	//relative growth rate for potential growth
	double c;
	if(rgrCoefficient){
		rgrCoefficient->get(t,c);
		r*=c;
	}
	//stress
	stress->get(t,c);
	if(c<1){
		r*=c;
	}
	if(r<0) {
		msg::warning("StressAdjustedPotentialLeafArea: fixing negative growth");
		r=0;
	}
}
std::string StressAdjustedPotentialLeafArea::getName()const{
	return "stressAdjustedPotentialLeafGrowthRate";
}


DerivativeBase * newInstantiationStressAdjustedPotentialLeafArea(SimulaDynamic* const pSV){
   return new StressAdjustedPotentialLeafArea(pSV);
}


LeafArea::LeafArea(SimulaDynamic* const pSD):
	DerivativeBase(pSD), SLASimulator(nullptr), c2lSimulator(nullptr), CinDryWeight(nullptr)
{
	//check if unit given in input file agrees with this function
	if(pSD->getUnit()!="cm2") msg::error("LeafArea: Expecting unit cm2 but found "+pSD->getUnit().name);
	//plant type
	std::string plantType;
	PLANTTYPE(plantType,pSD)
	//see if carbon model is run
	c2lSimulator=pSD->existingSibling("carbonAllocation2Leafs","g");
	//planting time
	pSD->getParent(3)->getChild("plantingTime")->get(plantingTime);
	//plant parameters
	SimulaBase *parameters(GETSHOOTPARAMETERS(plantType));
	if(c2lSimulator){
		SLASimulator=parameters->getChild("specificLeafArea","g/cm2");
	}else{
		//plant parameters
		SLASimulator=parameters->existingChild("leafAreaExpansionRate");
		if (!SLASimulator) SLASimulator=parameters->getChild("leafAreaExpantionRate");
	}
	//carbon conversion factor.
	CinDryWeight=pSD->getParent(3)->getChild("carbonToDryWeightRatio","100%");
}
void LeafArea::calculate(const Time &t, double&r){
	//note that the original version starts with an exponential growth curve instead of using photosynthesis
	//local time
	Time localTime(t - plantingTime);
	if(c2lSimulator){
		//get portion of shoot allocated carbon that is used for leafs.
		c2lSimulator->getRate(t,r);
		//get specific leaf area SLA
		double sla;
		SLASimulator->get(localTime,sla);
		if(sla<1e-10) msg::error("LeafArea: SLA is too small.");
		r/=sla;
		//C in dry weight
		double cdw;
		CinDryWeight->get(t,cdw);
		//multiply and return the result - assume that carbon to drymatter ratio is 0.54
		r/=cdw;
		if(r<0) msg::error("LeafArea:Negative leaf growth");
		if(std::isnan(r)) {
			msg::error("LeafArea: area is NaN");
		}
	}else{
		//no carbon model, use potential growth
		//code duplication of the potential leaf area growth function.
		//this is only here for backward compatibility reasons, as no we use the replace option in the inputfiles to switch from potential to actual (carbon driven).
		SLASimulator->get(localTime,r);
	}
	if(std::isnan(r)) {
		msg::error("LeafArea: area is NaN");
	}

}
std::string LeafArea::getName()const{
	return "leafArea";
}

DerivativeBase * newInstantiationLeafArea(SimulaDynamic* const pSD){
   return new LeafArea(pSD);
}

SunlitLeafArea::SunlitLeafArea(SimulaDynamic* const pSD):DerivativeBase(pSD), cachedTime(-10){
	std::string name = pSD->getName().substr(0, 6); // name = sunlit or shaded
	SimulaBase *pAreaPerPlant = pSD->getSibling("areaPerPlant", "cm2");
	pAreaPerPlant->get(areaPerPlant);
	pSunlitLeafAreaIndex = pSD->getSibling(name + "LeafAreaIndex");
}
void SunlitLeafArea::calculate(const Time &t, double&r){
	if (std::abs(t - cachedTime) < TIMEERROR){
		r = cachedSLAI;
		return;
	}
	double LAI;
	pSunlitLeafAreaIndex->get(t, LAI);
	r = areaPerPlant*LAI;
	cachedTime = t;
	cachedSLAI = r;
}
std::string SunlitLeafArea::getName()const{
	return "sunlitLeafArea";
}

DerivativeBase * newInstantiationSunlitLeafArea(SimulaDynamic* const pSD){
   return new SunlitLeafArea(pSD);
}


LeafAreaIndex::LeafAreaIndex(SimulaDynamic* pSD):DerivativeBase(pSD)
{
	//check if unit given in input file agrees with this function
	pSD->checkUnit("cm2/cm2");
	//simulators
	leafAreaSimulator=pSD->getSibling("leafArea","cm2");
	senescedLeafArea=pSD->existingSibling("senescedLeafArea","cm2");
	//plant type
	std::string plantType;
	PLANTTYPE(plantType,pSD)
	//get area per plant
	SimulaBase *areaSimulator=ORIGIN->getChild("rootTypeParameters")->getChild(plantType)->getChild("shoot")->getChild("areaPerPlant","cm2");
	areaSimulator->get(areaPerPlant);
	//planting time
	plantingTime=pSD->getStartTime();
	//meanLeafAreaSimulator
	meanLeafAreaSimulator=pSD->getParent(4)->existingChild("meanLeafAreaIndex","cm2/cm2");
	if(meanLeafAreaSimulator) {
		msg::warning("LeafAreaIndex: using mean leaf area based on box size and the leaf area of all plants.",1);
	}else{
		msg::warning("LeafAreaIndex: using mean leaf area based on areaPerPlant.",1);
	}
}
void LeafAreaIndex::calculate(const Time &t, double &LAI){
		//get leaf area
		double LA;
		leafAreaSimulator->get(t,LA);//cm2
		if(senescedLeafArea){
			double sLA;
			senescedLeafArea->get(t,sLA);//cm2
			LA-=sLA;
		}
		//calculate LAI
		LAI=LA/areaPerPlant;//cm2/cm2


		///user should be very careful with this.
		if(meanLeafAreaSimulator){
			double mLAI;
			meanLeafAreaSimulator->get(t,mLAI);
//			if(mLAI>0){
//				if(fabs((LAI-mLAI)/mLAI)>0.05){
//					msg::warning("LeafAreaIndex:: mean LAI and individual plant LAI are not the same but the model will not produce correct results for structured canopies. Check that the parameter areaPerPlant fits the box size divided by the number of plants and that plants are equally sized. using mLAI", 1); // This generates an extremely high number of false positives sometimes so turning it off
//				}
//			}
			LAI=mLAI;
		}
}
std::string LeafAreaIndex::getName()const{
	return "leafAreaIndex";
}

DerivativeBase * newInstantiationLeafAreaIndex(SimulaDynamic* const pSD){
   return new LeafAreaIndex(pSD);
}

SunlitLeafAreaIndex::SunlitLeafAreaIndex(SimulaDynamic* pSD):DerivativeBase(pSD), cachedTime(-10){
	leafAreaIndex = pSD->getSibling("leafAreaIndex");
	solarElevationAngle = ORIGIN->getPath("/environment/atmosphere/sineSolarElevationAngle");
}
void SunlitLeafAreaIndex::calculate(const Time &t, double &sunLAI){
	if (std::abs(t - cachedTime) < TIMEERROR){
		sunLAI = cachedSunlitLAI;
		return;
	}
	double sinEl;
	solarElevationAngle->get(t, sinEl);
	if (sinEl <= 0){
		sunLAI = 0;
		return;
	}
	double LAI;
	leafAreaIndex->get(t, LAI);
	double extinctionCoefficient = 0.5/sinEl;
	sunLAI = (1 - exp(-extinctionCoefficient*LAI))/extinctionCoefficient;
	if (sunLAI > LAI) sunLAI = LAI;
	if (sunLAI > 1) sunLAI = 1;
	cachedTime = t;
	cachedSunlitLAI = sunLAI;
}
std::string SunlitLeafAreaIndex::getName()const{
	return "sunlitLeafAreaIndex";
}

DerivativeBase * newInstantiationSunlitLeafAreaIndex(SimulaDynamic* const pSD){
   return new SunlitLeafAreaIndex(pSD);
}

ShadedLeafAreaIndex::ShadedLeafAreaIndex(SimulaDynamic* pSD):DerivativeBase(pSD), cachedTime(-10){
	leafAreaIndex = pSD->getSibling("leafAreaIndex");
	sunlitLeafAreaIndex = pSD->getSibling("sunlitLeafAreaIndex");
}
void ShadedLeafAreaIndex::calculate(const Time &t, double &shadedLAI){
	if (std::abs(t - cachedTime) < TIMEERROR){
		shadedLAI = cachedShadedLAI;
		return;
	}
	double LAI, sunLAI;
	leafAreaIndex->get(t, LAI);
	sunlitLeafAreaIndex->get(t, sunLAI);
	shadedLAI = LAI - sunLAI;
	cachedTime = t;
	cachedShadedLAI = shadedLAI;
}
std::string ShadedLeafAreaIndex::getName()const{
	return "shadedLeafAreaIndex";
}

DerivativeBase * newInstantiationShadedLeafAreaIndex(SimulaDynamic* const pSD){
   return new ShadedLeafAreaIndex(pSD);
}

MeanLeafAreaIndex::MeanLeafAreaIndex(SimulaDynamic* pSD):DerivativeBase(pSD), plantsPlantedAtDifferentTimes(false), splitBySunStatus(false)
{
	std::string name = pSD->getName().substr(4, 6); // name = sunlit or shaded
	if (name == "Sunlit" || name == "Shaded"){
		name.at(0) = std::tolower(name.at(0));
		splitBySunStatus = true;
	}
	//check if unit given in input file agrees with this function
	pSD->checkUnit("cm2/cm2");
	if (splitBySunStatus){
		leafArea = pSD->getSibling(name + "LeafArea", "cm2");
		senescedLeafArea = pSD->existingSibling(name + "SenescedLeafArea","cm2");
	} else{
		leafArea=pSD->getSibling("leafArea","cm2");
		senescedLeafArea=pSD->existingSibling("senescedLeafArea","cm2");
	}
	plantArea=pSD->getSibling("plantArea","cm2");
	plantArea->get(0,pa);
	msg::warning("MeanLeafAreaIndex: Assuming all plants are planted at t = 0. If this is not the case, add 'plantsPlantedAtDifferentTimes' under 'environment'.");
	//area
	Coordinate minCorner, maxCorner;
	SimulaBase * params(ORIGIN->getChild("environment")->getChild("dimensions"));
	params->getChild("minCorner")->get(minCorner);
	params->getChild("maxCorner")->get(maxCorner);
	double x=fabs(maxCorner.x-minCorner.x);
	double z=fabs(maxCorner.z-minCorner.z);
	area=x*z;
	if(area<0.1) msg::error("MeanLeafAreaIndex: very small area calculated, check values for min and max corners");
	if(fabs(pa-area)>1.) msg::warning("MeanLeafAreaIndex: plant area and area of the box are not the same. Using plant Area");
	SimulaBase *probe = params->existingChild("plantsPlantedAtDifferentTimes");
	if (probe) probe->get(plantsPlantedAtDifferentTimes);
}
void MeanLeafAreaIndex::calculate(const Time &t, double &LAI){
	//calculate LAI
	leafArea->get(t,LAI);
	if(senescedLeafArea){
		double sLA;
		senescedLeafArea->get(t,sLA);
		LAI-=sLA;
	}
	if (plantsPlantedAtDifferentTimes){
		plantArea->get(t,pa);
		if(fabs(pa-area)>1.) msg::warning("MeanLeafAreaIndex: plant area and area of the box are not the same. Using plant Area");
	}
	LAI/=pa;//cm2/cm2
}
std::string MeanLeafAreaIndex::getName()const{
	return "meanLeafAreaIndex";
}

DerivativeBase * newInstantiationMeanLeafAreaIndex(SimulaDynamic* const pSD){
   return new MeanLeafAreaIndex(pSD);
}


LeafAreaReductionCoefficient::LeafAreaReductionCoefficient(SimulaDynamic* pSD):DerivativeBase(pSD),
		actual(pSD->getSibling("leafArea","cm2")),
		potential(pSD->getSibling("potentialLeafArea","cm2")),
		recoveryRate(0)
{
	//check if unit given in input file agrees with this function
	pSD->checkUnit("cm2/cm2");
	//recovery
	std::string plantType;
	PLANTTYPE(plantType,pSD)
	SimulaBase* p(ORIGIN->getChild("rootTypeParameters")->getChild(plantType)->getChild("shoot")->existingChild("increaseInRGRDuringRecovery"));
	if(p) p->get(recoveryRate);

}
void LeafAreaReductionCoefficient::calculate(const Time &t, double &c){
	//time shift.
	/**@todo this introduces a small timeshift dependency but since otherwise
	 * leaf area depends on an estimate it will introduce small reductions
	 * which cause balance errors and often the following error message:
	 *
	 * CarbonAvailableForGrowth:  negative values: carbon production lower than compulsory cost of respiration/exudates?
	 */
	Time rt=std::max(t - actual->maxTimeStep(), pSD->getStartTime() );
	std::size_t np=pSD->getPredictorSize();

	//coefficient
	double p;
	potential->get(rt,p);
	if(p>1.e-4){
		actual->get(rt,c);
		c/=p;
		c+=recoveryRate;
		//limits
		if(c>0.9999){
			c=1.;
		}else if(c<1e-4){
			if(np < pSD->getPredictorSize()){
				c=1.;//this is important for c=0 estimated occurs when shoot starts growing, and the estimated allocation is 0 (for there was no allocation at all) but the shoot actually simply needs to grow potentially.
			}else{
				c=std::max(0.,c);
			}
		}else{
			msg::warning("LeafAreaReductionCoefficient: Reducing leaf area",1);
		}
	}else{
		c=1.;
	}
}
std::string LeafAreaReductionCoefficient::getName()const{
	return "leafAreaReductionCoefficient";
}

DerivativeBase * newInstantiationLeafAreaReductionCoefficient(SimulaDynamic* const pSD){
   return new LeafAreaReductionCoefficient(pSD);
}

CropHeight::CropHeight(SimulaDynamic* pSD):DerivativeBase(pSD), pStressMultiplier(nullptr), pot(false){
	auto found=pSD->getName().find("potential");
	if(found!=std::string::npos) pot=true;
	std::string plantType;
	PLANTTYPE(plantType,pSD);
	SimulaBase *params = GETSHOOTPARAMETERS(plantType);
	pGrowthSpeed = params->existingChild("potentialCropHeightGrowthSpeed", "cm/day");
	if (pGrowthSpeed && !pot){
		pStressMultiplier = pSD->getParent(2)->existingSibling("stressFactor:impactOn:cropHeightGrowthSpeed");
		if (!pStressMultiplier) msg::warning("CropHeight: Potential growth speed for crop height given but stress factor not found. Continuing without stress effects.");
	}
	if (!pGrowthSpeed) pGrowthSpeed = params->existingChild("cropHeightGrowthSpeed", "cm/day");
	if (!pGrowthSpeed){
		pGrowthSpeed = params->existingChild("cropHeight", "cm");
		if (pGrowthSpeed) msg::warning("CropHeight: Crop height given explicitly in input file, make sure this is a SimulaDerivative.");
		if (!pGrowthSpeed) msg::warning("CropHeight: No growth rates or values given for crop height. Setting crop height to 50 cm.");
	}
}

void CropHeight::calculate(const Time &t, double &height){
	height = 50.;
	double multiplier = 1.;
	if (pGrowthSpeed) pGrowthSpeed->get(t, height);
	if (pStressMultiplier) pStressMultiplier->get(t, multiplier);
	height *= multiplier;
}

std::string CropHeight::getName()const{
	return "cropHeight";
}

DerivativeBase * newInstantiationCropHeight(SimulaDynamic* const pSD){
   return new CropHeight(pSD);
}

MaximumCanopyHeight::MaximumCanopyHeight(SimulaDynamic* pSD):DerivativeBase(pSD){
	SimulaBase::List plants;
	ORIGIN->getChild("plants")->getAllChildren(plants);
	for(auto & it:plants){
		SimulaBase*	probe=(it)->existingChild("plantingTime");
		if(probe) {
			Time pt;
			probe->get(pt);
			plantingTimes.push_back(pt);
			probe=(it)->getChild("plantPosition")->getChild("shoot")->existingChild("cropHeight", "cm");
			cropHeights.push_back(probe);
			if (!probe){
				msg::warning("MaximumCanopyHeight: cropHeight not found for " + (it)->getName() + ". Assuming a constant height of 50 cm for this plant.");
			}
		}//else ignore, this is not a plant
	}
}

void MaximumCanopyHeight::calculate(const Time &t, double &height){
	height = 1.;
	for (unsigned int i = 0; i < plantingTimes.size(); i++){
		if (t >= plantingTimes[i]){
			double temp = 50;
			if (cropHeights[i]){
				cropHeights[i]->get(t, temp);
			}
			if (temp > height) height = temp;
		}
	}
}

std::string MaximumCanopyHeight::getName()const{
	return "maximumCanopyHeight";
}

DerivativeBase * newInstantiationMaximumCanopyHeight(SimulaDynamic* const pSD){
   return new MaximumCanopyHeight(pSD);
}



//==================registration of the classes=================
class AutoRegisterLeafAreaClassInstantiationFunctions {
public:
   AutoRegisterLeafAreaClassInstantiationFunctions() {
	  BaseClassesMap::getDerivativeBaseClasses()["leafArea"] = newInstantiationLeafArea;
	  BaseClassesMap::getDerivativeBaseClasses()["sunlitLeafArea"] = newInstantiationSunlitLeafArea;
	  BaseClassesMap::getDerivativeBaseClasses()["constantLeafGrowthRate"] = newInstantiationPotentialLeafArea;
	  BaseClassesMap::getDerivativeBaseClasses()["potentialLeafGrowthRate"] = newInstantiationPotentialLeafArea;
	  BaseClassesMap::getDerivativeBaseClasses()["stressAdjustedPotentialLeafGrowthRate"] = newInstantiationStressAdjustedPotentialLeafArea;
	  BaseClassesMap::getDerivativeBaseClasses()["potentialLeafArea"] = newInstantiationPotentialLeafArea;
 	  BaseClassesMap::getDerivativeBaseClasses()["leafAreaIndex"] = newInstantiationLeafAreaIndex;
 	  BaseClassesMap::getDerivativeBaseClasses()["sunlitLeafAreaIndex"] = newInstantiationSunlitLeafAreaIndex;
 	  BaseClassesMap::getDerivativeBaseClasses()["shadedLeafAreaIndex"] = newInstantiationShadedLeafAreaIndex;
 	  BaseClassesMap::getDerivativeBaseClasses()["meanLeafAreaIndex"] = newInstantiationMeanLeafAreaIndex;
 	  BaseClassesMap::getDerivativeBaseClasses()["leafAreaReductionCoefficient"] = newInstantiationLeafAreaReductionCoefficient;
 	  BaseClassesMap::getDerivativeBaseClasses()["cropHeight"] = newInstantiationCropHeight;
 	  BaseClassesMap::getDerivativeBaseClasses()["maximumCanopyHeight"] = newInstantiationMaximumCanopyHeight;
   };
};



// our one instance of the proxy
static AutoRegisterLeafAreaClassInstantiationFunctions p45567987ghg34;


