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
#if _WIN32 || _WIN64
#define _USE_MATH_DEFINES
#endif

#include "Watflow.hpp"

#include "../../engine/Origin.hpp"
#include "../../engine/SimulaConstant.hpp"


#define GRAVITY //comment out to have not gravity

//#include <fstream> // debug matrix - file output
#include "../../math/pcgSolve.hpp"
#ifndef M_PI
#define M_PI (3.14159265358979323846)
#endif

//*************************************************
Water::Water(SimulaDynamic* pSD){
}

void Water::calculate(const Time &t, double &theta){
}

std::string Water::getName() const {
	return "Water";
}
Watflow::Watflow(SimulaDynamic* const pSD):
waterMesh(), /* empty mesh obj*/
AtmInF(true),
FreeD(true),
zeroFluxBottomBoundary_(false),
allowOutflowFromRoots(false),
NumNP(waterMesh.getNumNP()), // copies
NumEl(waterMesh.getNumEl()),
NumBP(waterMesh.getNumBP()),
Q(0.,NumNP),   	 /* based on hOld */
SSM(nullptr),
par(waterMesh),
prec_(nullptr), interception_(nullptr), evaporation_(nullptr),

x(waterMesh.getCordX()),
y(waterMesh.getCordY()),
z(waterMesh.getCordZ()),
hNew(0.,NumNP),
hOld(0.,NumNP),
matchmode(1),
B(0.,NumNP),
RHS(0.,NumNP),
DS(0.,NumNP),
Sink(NumNP),
Con(NumNP),
ConO(NumNP), /* Nodal values of the hydraulic conductivity at the old time level used in solute  */
thOld(NumNP),
thNew(NumNP),
Adiag_boundary_nodes_(0., NumNP),
rLen(waterMesh.getrLen()),
prec(0.),
rSoil(0.),
hCritS(0.),    /// (max. allowed pressure head at the soil surface)
/// hcrtiA = Minimum allowed pressure head at the soil surface [L] (Table 8.11).
hCritA(-1000000.),
// hCritA=-abs(hCritA)
// the water pressure cannot fall below a limit value (defined by the temperature and the relative humidity of the air)
// If this value is reached, the boundary condition is switched to a Dirichlet type with h = hCritA = rel. humidity of air
// and is maintained until the potential evaporation rate decreases or rainfall occurs. (see shift subroutine in watflow3.f)
AreaOfDomain(waterMesh.getAreaOfDomain()), VolumeOfDomain(waterMesh.getVolumeOfDomain()), Nsub(waterMesh.getNsubel()),
evapoScalingWet(0.2),evapoScalingDry(0.002),evapoScalingExponent(2.),
totWaterInColumn(nullptr),totWaterChange(nullptr), sumTopBoundaryFlux_(nullptr), sumBottomBoundaryFlux_(nullptr), totSink_(nullptr),relativeWaterContentInTopLayer(nullptr)
{

	// set atm parameter
	prec_ 		  	= pSD->getPath("/environment/atmosphere/precipitation", "cm/day");
	interception_ 	= pSD->existingPath("/atmosphere/interception", "cm/day");
	SimulaBase* potentialEvaporation_ = pSD->existingPath("/atmosphere/potentialEvaporation", "cm/day");
	evaporation_ 	= pSD->existingPath("/atmosphere/evaporation", "cm/day");
	SimulaBase* evaporationEn_ 	= pSD->existingPath("/environment/atmosphere/evaporation", "cm/day");
	if (potentialEvaporation_) {
		msg::warning("Watflow:: using potentialEvaporation from atmosphere section",1);
		evaporation_=potentialEvaporation_;
	} else if (evaporation_) {
		msg::warning("Watflow:: using evaporation from atmosphere section",1);
	} else if (evaporationEn_) {
		evaporation_=evaporationEn_;
		msg::warning("Watflow:: using evaporation from environment section, i.e. given, predefined.",1);
	} else {
		msg::error("Watflow:: No evaporation defined in input file.");
	}

	//get the values for the evaporation scaling function
	//double thetaScalingWet,thetaScalingDry,thetaScalingExponent;
	SimulaBase* p = pSD->existingPath("/environment/soil/water/relativeOffsetAirDrySoil");
	if(p) p->get(evapoScalingDry);
	p = pSD->existingPath("/environment/soil/water/relativeOffsetKlimitedEvaporation");
	if(p) p->get(evapoScalingWet);
	p = pSD->existingPath("/environment/soil/water/evaporationScalingExponent");
	if(p) p->get(evapoScalingExponent);

	// initialize matrix
	SSM = new SparseSymmetricMatrix(NumNP);
	waterMesh.easy_init_matrix(*SSM);

	//top boundary condition is by default neuman with prescribed precipitation or evaporation. In case the column gets too dry, we can avoid further drying by switching to a dirichlet boundary
	SimulaBase * pc=ORIGIN->existingPath("environment/soil/water/minimalSurfaceHydraulicHead","cm");
	if(pc) {
		pc->get(hCritA);
		msg::warning("WatFlow: Setting minimalSurfaceHydraulicHead to "+convertToString(hCritA));
	}

	//bottom boundary condition is by default free drainage, but here we switch based on user input
	depthWaterTable_ = ORIGIN->existingPath("environment/soil/water/depthOfWaterTable");
	if(depthWaterTable_) {
		msg::warning("Watflow: Using Dirichlet bottom boundary based on specified depthOfWaterTable.",3);
		FreeD=false; ///todo enumerator would be better, so we can simply switch between various boundaries
	}else{
		msg::warning("Watflow: Using Free drainage bottom boundary.",3);
	}

	SimulaBase* zeroFlux = ORIGIN->existingPath("environment/soil/water/zeroFluxBottomBoundary");
	if(zeroFlux) {
		zeroFlux->get(zeroFluxBottomBoundary_);
		if(zeroFluxBottomBoundary_ && depthWaterTable_) msg::error("Watflow: Both depthOfWaterTable and zeroFluBottomBoundary set. Use one of the two.");
		msg::warning("Watflow: Using zero flux bottom boundary condition.",3);
	};
	p = ORIGIN->existingPath("environment/soil/water/allowOutflow");
	if (p) p->get(allowOutflowFromRoots);
	if (allowOutflowFromRoots) msg::warning("Watflow: Allowing water to flow out of roots so you can simulate, among other things, hydraulic lift");

	// set material!!!! set con and thold and thnew
	p = ORIGIN->getPath("environment/soil/water/initialHydraulicHead");
	double Y;
	for (tIndex i = 0; i != NumNP; ++i) {
		p->get(0, Coordinate(x[i], z[i], y[i]), Y);
		hOld[i]  = Y;
		hNew[i]  = Y; // hNew = hOld;
	}

	par.calc_fq(hOld, thOld);
	par.calc_fk(hOld, thOld, ConO);
	thNew = thOld;
	Con = ConO;

	// find out what method to use for matching roots to fem nodes
	p = ORIGIN->existingPath("environment/dimensions/rootMatchingMode");
	if (p) {
		std::string mode;
		p->get(mode);
		if (mode == "postma and lynch, 2011") {
			matchmode = 1;
		} else if (mode == "ignore root placement") {
			matchmode = 2;
			msg::warning("Watflow::Watflow: using rootMatchingMode: 'ignore root placement' as set in environment/dimensions/rootMatchingMode",1);
		} else {
			msg::error("Watflow::Watflow: unknown string for environment/dimensions/rootMatchingMode ");
		}
	}

	// Initialize atmosphere
	set_atmosphere(0); // thNew is thOld, so the first atmosphere coupling is good.
	neumann_boundary_condition(); //computes initial Q simply for the first output

	//output
	p = ORIGIN->getChild("soil")->existingChild("water");
	if(p){
		SimulaBase* d = p->existingChild("totalWaterInColumn", "cm3");
		if (d) totWaterInColumn = dynamic_cast<SimulaTable<Time>*>(d);
		d = p->existingChild("relativeWaterContentInTopLayer");
		if (d) relativeWaterContentInTopLayer = dynamic_cast<SimulaTable<Time>*>(d);
		d = p->existingChild("totalWaterInLayer1", "cm3");
		if (d){
			int i = 1;
			std::string layerName = "totalWaterInLayer" + std::to_string(i);
			d = p->existingChild(layerName, "cm3");
			while (d){
				totWaterInLayers.push_back(dynamic_cast<SimulaTable<Time>*>(d));
				double lb, ub;
				d->getChild("lowerBound", "cm")->get(lb);
				d->getChild("upperBound", "cm")->get(ub);
				lowerBounds_.push_back(lb);
				upperBounds_.push_back(ub);
				i++;
				std::string layerName = "totalWaterInLayer" + std::to_string(i);
				d = p->existingChild(layerName, "cm3");
			}
		}
		d = p->existingChild("totalWaterChangeInColumn", "cm3");
		if (d) totWaterChange = dynamic_cast<SimulaTable<Time>*>(d);
		d = p->existingChild("topBoundaryFluxRate", "cm3/day");
		if (d) {
			sumTopBoundaryFlux_ = dynamic_cast<SimulaTable<Time>*>(d);
			sumTopBoundaryFlux_->setInterpolationMethod("step");
		}
		d = p->existingChild("bottomBoundaryFluxRate", "cm3/day");
		if (d) {
			sumBottomBoundaryFlux_ = dynamic_cast<SimulaTable<Time>*>(d);
			sumBottomBoundaryFlux_->setInterpolationMethod("step");
		}
		d = p->existingChild("totalSinkRate", "cm3/day");
		if (d) {
			totSink_ = dynamic_cast<SimulaTable<Time>*>(d);
			totSink_->setInterpolationMethod("step");
		}
	}

	reportTotWaterColumn(0,0);
	this->setrootsurfacevalues(0., 0.0001);
}

//#define EXPLICIT

void Watflow::calculate(tIndex & ItCum, double & dt, const double & dtMin,  const double & dtOld,
		const double & tOld, const tIndex & TLevel, tIndex & Iter) {
	this->set_atmosphere(tOld+dt/2.); // midpoint for evaporation/precipitation

	//update hold to old hnew
	//auto hdiff=hNew-hOld;
	//fast swapping of pointers, note that hNew must be set, the others will be set based on it.
	std::swap(hOld,hNew);
	std::swap(thOld, thNew);
	std::swap(ConO, Con); // gives ConO an initial value use in solute


	//extrapolation as first guess seems to mostly cause convergence errors.
		//new estimate for hnew, I am not seeing much speed up because of this.
	    hNew=hOld;
	    /*const auto sdt= 1+ 0.8*dt / dtOld;
		hNew += hdiff * sdt; //note the swap above, hnew must be set, value can be between 1 an d2.
		for (tIndex i = 0; i < NumNP; ++i) {
			if(hNew[i]>0.)
				hNew[i]=0.;
		}

		if(hOld.max()>0.) {
    	//msg::error("Watflow:: positive pressures, set to 0. max h="+convertToString(hOld.max())+" at "+convertToString(tOld));
    	msg::warning("Watflow:: positive pressures, saturated conditions setting values to 0.");
    	for (tIndex i = 0; i < NumNP; ++i) {
    		if(hOld[i]>0.){
    			hOld[i]=0.;
    			hNew[i]=0.;
    			thOld[i]=par.getthSat()[i];
    			//thNew[i]=thOld[i]; not needed will be set based on hnew.
    		}
    	}
    }*/

	Iter   = 0; // number of iterations in the newton loop
	togglePredictionMode(true);
	double EpsTh(1e9), EpsH(1e9);
	//bool dappen=true;
	//todo would it make sense to scale this to timestep, only if we have consistent errors.
	const double TolTh(1.e-3); // convergence criteria, just ofr speed divided by thsat-thr which is more or less 0.4 for most soils
	const double TolH(1.e-3);        // convergence criteria
	const std::size_t MaxIt(800);




	for (;;) {
		// --- Beginning of newton iteration loop --------------------------------------
		par.calc_fq(hNew, thNew);
		par.calc_fk(hNew, thNew, Con);
		const tVector Cap(par.calc_fc(hNew));

		//build matrix and right hand side
//std::cout<<std::endl<<"t*1000="<<tOld*1000<<" iter="<<Iter;
		//if(Iter==0)
		//std::cout<<std::endl<<"resetting"<<Iter;
		this->unsetrootsurfacevalues(tOld+TIMEERROR);
		this->setrootsurfacevalues(tOld+dt, dtMin);
		reset_system(tOld,dt, Iter, Cap);

		// debug lines
//			std::ofstream ssm;
//			ssm.open("ssm.txt");
//			SSM->print_sparse(ssm);
//			ssm.close();
//			exit(0);
//		std::cout<<" start";

		// solve Parameterlist: A, b, x. b is rhs, x is solution.
		const auto hTemp = hNew;//container to keep the hnew of the previous iteration
		auto pcgerr=Pcg::solve(*SSM, RHS, hNew, 1.e-15, 5000, SYM_GAUSS_SEIDEL);
		//auto pcgerr=Pcg::solve(*SSM, RHS, hNew, 1.e-15, 5000, JACOBI); //try jacobi
//		std::cout<<" finished";

		++Iter;
		++ItCum;

		// restart solving with a smaller timestep
		if(!pcgerr){
			for(auto & hlk: hNew){
				if(!std::isfinite(hlk)) {
					hlk=-100000;
					pcgerr=true;
				}
			}
		}

		if(pcgerr ){
			if(dt < 1.01 * dtMin) msg::error("Watflow:: PCG solver did not converge and the time step is at minimum. Can't find a solution.");
			hNew  = hOld;//not sure if we could continue with the bogus solution of the previous try, so reset?
			const auto pcgerr2=true;
			//const auto pcgerr2=Pcg::solve(*SSM, RHS, hNew, 1.e-15, 5000, JACOBI); //try jacobi
			if(pcgerr2){
				hNew  = hOld;
				this->unsetrootsurfacevalues(tOld+TIMEERROR);//remove bogus values in the tables
				dt = std::max(dt / 10., dtMin);
				msg::warning("Watflow::calculate: PCG did not converge fast enough, reducing the time step by a factor 10 and trying again.");
				// reset
				Iter = 0;
				this->set_atmosphere(tOld+dt/2.); // midpoint for evaporation/precipitation
				// do the next iteration
				continue;
			}else{
				msg::warning("Watflow::calculate: SYM_GAUSS_SEIDEL did not converge fast enough, used JACOBI solver instead.");
			}
		}


		//  Test for convergence
//		ItCrit = true;
		double EpsThOld(EpsTh), EpsHOld(EpsH);
		EpsTh=0.; EpsH=0.;

		for (tIndex i = 0; i < NumNP; ++i) {
			double c=std::fabs(Cap[i]); //cap is still based on previous hnew
			if(c<TolTh/TolH) c=TolTh/TolH; // not sure why this would be necessary.  as if c is smaller then this EpsH does the trick anyway.
			const double hd= std::fabs(hNew[i] - hTemp[i]);
			EpsTh = std::fmax(EpsTh,c * hd); //why not simply recalc theta?
			EpsH = std::fmax(EpsH,hd);
		}
		//std::cout<<std::endl<<"time="<<tOld<<" iter="<<Iter<<" epsth="<<EpsTh/TolTh<<Iter<<" epsh="<<EpsH/TolH;

		//convergence
		double conTh=EpsTh/EpsThOld;
		double conH =EpsH /EpsHOld ;

		if (EpsTh > TolTh || EpsH > TolH  ) { // || Iter<10) {
		// we do not meet the criteria, and have to do a new iteration with a better estimate for hnew
//			ItCrit = false;
			if (Iter == 1) {
				//one off time extrapolation of hnew
				//hNew+=hNew-hOld;
			} else if (Iter < MaxIt) {
				// do the next iteration

				/*if (!dappen) {
					//we had a solution, but ran away from it again try with strong dappening
					//dappen=true;
					hNew = hTemp * 0.5 + hNew * 0.5;
				} else if (conTh > 1.5 || conH > 1.5) {
					hNew = hTemp * 0.1 + hNew * 0.9;
				} else if (conTh > 1.2 || conH > 1.2) {
					hNew = hTemp * 0.01 + hNew * 0.99;
				} else {
					hNew = hTemp * 0.001 + hNew * 0.999;
				}*/
				hNew = hTemp * 0.1 + hNew * 0.9;
				// do the next iteration but try to reduce oscillations in hnew
				//if (EpsTh > TolTh*10. || EpsH > TolH*10. )
				//}
			} else {
				if (dt < 1.01 * dtMin) {
					msg::warning(
							"Watflow::calculate: timestep is less than dtMin (=" + convertToString(dtMin)
									+ ") and convergence in time is too slow, continuing with inaccurate results");
					if (conTh > 1. || conH > 1.) {
						hNew = hTemp; //previous result was better
					}
					break; // exit timeloop with inaccurate result
				} else {
					// restart solving with a smaller timestep
					hNew = hOld;
					this->unsetrootsurfacevalues(tOld + TIMEERROR); //remove bogus values in the tables
					dt = std::max(dt / 3., dtMin);
					msg::warning("Watflow::calculate: Did not converge fast enough, reducing timestep by a factor 3 and trying again.");
					// reset
					Iter = 0;
					this->set_atmosphere(tOld + dt / 2.); // midpoint for evaporation/precipitation
					// do the next iteration
				}
			}
	//	} else if (dappen) { // ItCrit is true, so we can leave the for loop
			//do one more loop without dappen
		//	dappen = false;
		} else { // ItCrit is true, so we can leave the for loop
				 //we reached the tolerance criteria and have an accurate enough result
				 //we update the info, as thnew are still based on previous iteration hnew.
			par.calc_fq(hNew, thNew);
			par.calc_fk(hNew, thNew, Con);

			if (Iter > MaxIt / 2) {
				//report if we used many iterations
				msg::warning("Watflow::calculate: needing more than " + std::to_string(MaxIt / 2) + " iterations to solve problem", 1);
			}
			break;
		}

	} // --- End of iteration loop --------------------------------------------


	// OUTPUTS
	//   P-Level information
	computeQBackwards(dt); //to report values for Q.
	reportTotWaterColumn(tOld,dt);
	// set current Conc & hhead at root surface
	double t = tOld + dt;
	togglePredictionMode(false);
	this->setrootsurfacevalues(t, dtMin);

} // end method


void Watflow::reset_system(const double & tOld, const double & dt, const tIndex & Iter, const tVector & Cap) {

	// (Re-)Initialization
	B = 0.;
	SSM->resetToZero(); // to reset the matrix here it is essential to have a matrix with neighboring entries first!
	tIndex selcount(0);
	tIndex count(0);
	const auto & Bi(waterMesh.getBi()), Ci(waterMesh.getCi()), Di(waterMesh.getDi());
	for (auto & subit : waterMesh.getSubelementlist() ) {
			const tIndex i = subit[0],  j = subit[1], 	k = subit[2], 	l = subit[3];

			// sumConE is conductivity sum per sub-element
			const double sumConE = ( Con[i]  + Con[j]  + Con[k]  + Con[l] )/ 24.; //(1./4. * 1./6. = 1./24.)

			// see eq 4.5; Di from the determinant the linear distances in certain direction, here gravity which is unity, so it is sumConE * the determinant in the z direction
#ifdef GRAVITY
			B[i] += Di[count] * sumConE;
			B[j] += Di[count + 1] * sumConE;
			B[k] += Di[count + 2] * sumConE;
			B[l] += Di[count + 3] * sumConE;
#endif

			const double AMul = sumConE / 6. / waterMesh.getSubelemetVolumes()[selcount];
			++selcount;  // go through each subelement
			// for the 36 and 6 see equation 4.5 and 4.6 on page 18 of the manual,
			// they come from from the 6 directions x,y,z in positive and negative and all combinations of directions

			for (tIndex ii = 0; ii < 4; ++ii) { // all neighboring combinations go into the matrix
				const tIndex iG = subit[ii];
				const tIndex cii = count+ii;
				for (tIndex jj = ii; jj < 4; ++jj) {
					const tIndex jG = subit[jj];
					const tIndex cjj = count+jj;
					const double temp_value = AMul * (Bi[cii] * Bi[cjj] + Ci[cii] * Ci[cjj] + Di[cii] * Di[cjj]);//todo could be optimized as this follows a simple rhythm in a regular grid.
					SSM->addValueUnsafely(iG, jG, temp_value);
				}
			}
			count += 4; // go through each node of each subelement
	}

	//if (0 == Iter) {//commenting this out calls setwatersink new but does not necessarily force new values.
	DS = 0.;
	tIndex it_subelem = 0;
	setWaterSink(tOld, tOld+dt);
	for (auto & it : waterMesh.getSubelementlist() ) {
		const tIndex i = it[0], 	j = it[1], 	k = it[2],  	l = it[3];
		const double VE = waterMesh.getSubelemetVolumes()[it_subelem];
		++it_subelem;

		//  here we go back to nodal values //todo this smoothes the sink terms.
		DS[i] += VE * Sink[i] * 0.25;
		DS[j] += VE * Sink[j] * 0.25;
		DS[k] += VE * Sink[k] * 0.25;
		DS[l] += VE * Sink[l] * 0.25;
	}
	//}



/// Determine boundary fluxes
// this does normally nothing, but if the given boundary condition cause the soil to be too wet or too dry,
// then shift will change the boundary condition

	neumann_boundary_condition(); //computes Q

	auto &F(waterMesh.getvol());

	RHS = - B + F*( Cap*hNew - thNew + thOld )/dt + Q - DS;

	dirichlet_boundary_condition(tOld+dt/2.);


	for (tIndex i = 0; i < NumNP; ++i) {
		const double temp_value = waterMesh.getvol()[i] * Cap[i] / (dt);
		SSM->addValueUnsafely(i, i, temp_value);
	}
	return;
}

///computes Q
void Watflow::neumann_boundary_condition() {
	Q=0.;
	//set Q for the top boundary nodes.
		for (auto n : waterMesh.getTopBoundaryNodes()) {
			// set scaling
			const double scaling(getScaling(n));
		Q[n] = - waterMesh.getWidth()[n] * (rSoil * scaling - prec); // L²*L = L³, cm³
		}

	if (zeroFluxBottomBoundary_) {
		//neumann at bottom boundary
		//todo for this to work, getWidth need to have indexes numnp, and not numbp
		for (auto i : waterMesh.getBottomBoundaryNodes()) {
			Q[i] = 0.;
		}
	}else if (FreeD) {
		//neumann at bottom boundary
		//todo for this to work, getWidth need to have indexes numnp, and not numbp
		for (auto i : waterMesh.getBottomBoundaryNodes()) {
			Q[i] = -waterMesh.getWidth()[i] * Con[i]; // TODO discuss why this flux is calculated with ConO (old Con)
			if(std::isnan(Q[i])) {
				/*std::valarray<double> tcon(0., NumNP);
				par.calc_fk(hOld, tcon);
				std::cout<<std::endl<<waterMesh.getWidth()[i]<<" "<<ConO[i]<<" "<<hOld[i]<<" "<<tcon[i]<<std::endl;*/
				msg::error("WatFlow: Q[i] is NAN");
			}
		}
	}
	//msg::warning("Watflow:: Q is max "+convertToString(Q.max())+" and min "+convertToString(Q.min()));
	return;
}

void Watflow::computeQBackwards(const double dt){
	//This code computes what Q should have been when using neumann to get the same hnew result as we just computed with dirichlet.

	//the values on diagonal of A were overwritten when doing dirichlet, we restore them first
	for (auto n : waterMesh.getTopBoundaryNodes()) {
		if(fabs(RHS[n]/hNew[n]) > 10.e20) SSM->insert(n,n,Adiag_boundary_nodes_[n]);
			}
	for (auto n : waterMesh.getBottomBoundaryNodes()) {
		if(fabs(RHS[n]/hNew[n]) > 10.e20) SSM->insert(n,n,Adiag_boundary_nodes_[n]);
		}
	//auto Qcheck=Q;


	//compute back the right hand side
	tVector temp_A_times_hNew(NumNP);
	SSM->vectorMultiply(hNew, temp_A_times_hNew);

	//compute Q from the right hand side (formula 4.4)
	auto &F(waterMesh.getvol());
	//for (tIndex n = 0; n < NumNP; ++n) {
	for (auto n : waterMesh.getTopBoundaryNodes()) {
		//if (waterMesh.getKode()[n] < 1) {
		//	continue;
		//} else {
		if(fabs(RHS[n]/hNew[n]) > 10.e20){
			//Dirichlet boundary condition.
			//	RHS = - B + F*( Cap*hNew - thNew + thOld )/dt + Q - DS;
			Q[n] = temp_A_times_hNew[n] + B[n] + DS[n] - F[n] * (thNew[n] - thOld[n]) / (dt); //eq 4.4 of the manual
			if(std::isnan(Q[n])){
				if(std::isnan(temp_A_times_hNew[n])) msg::error("WatFlow: temp_A_times_hNew[n] is NAN");
				if(std::isnan(B[n])) msg::error("WatFlow: B[n] is NAN");
				if(std::isnan(DS[n])) msg::error("WatFlow: DS[n] is NAN");
				if(std::isnan(F[n])) msg::error("WatFlow: F[n] is NAN");
				if(std::isnan(thNew[n])) msg::error("WatFlow: thNew[n] is NAN");
				if(std::isnan(thOld[n])) msg::error("WatFlow: thOld[n] is NAN");
				if(dt == 0) msg::error("WatFlow: dt = 0");
				msg::warning("WatFlow: temp_A_times_hNew[n] = " + std::to_string(temp_A_times_hNew[n]) + " B[n] = " + std::to_string(B[n]) + " DS[n] = " + std::to_string(DS[n]) + " F[n] = " + std::to_string(F[n]) + " thNew[n] = " + std::to_string(thNew[n]) + " thOld[n] = " + std::to_string(thOld[n]) + " dt = " + std::to_string(dt));
				msg::error("WatFlow: Q[n] is NAN");
			}
	}
//			}
	}
	for (auto n : waterMesh.getBottomBoundaryNodes()) {
		//if (waterMesh.getKode()[n] < 1) {
		//	continue;
		//} else {
		if(fabs(RHS[n]/hNew[n]) > 10.e20){
			//Dirichlet boundary condition.
			Q[n] = temp_A_times_hNew[n] + B[n] + DS[n] - F[n] * (thNew[n] - thOld[n]) / (dt); //eq 4.4 of the manual
			if(std::isnan(Q[n])) msg::error("WatFlow: Q[n] is NAN (2)");

		}
//		}
	}

	//Qcheck-=Q;
	//std::cout<<std::endl<<Q.max()<<" "<<Q.min()<<" "<<Qcheck.max()<<" "<<Qcheck.min()<<" ";

}

void Watflow::dirichlet_boundary_condition(const Time &t) {
	//The top boundary is assumed to have a predescribed flux coming
	//from precipitation and or evaporation. This flux is already set in Q by
	//setAtmosphere.
	//here we just do not except that flux if it leads to too dry soil or too wet soil, in which case we fix hnew
	for (auto n : waterMesh.getTopBoundaryNodes()) {
		const double scaling(getScaling(n));
		if (hNew[n] <= hCritA && rSoil*scaling>prec+1e-6) { // Evaporation capacity is exceeded
			msg::warning("Watflow: top soil too dry, in order to have the evaporation. ");
			hNew[n] = hCritA;
			Adiag_boundary_nodes_[n]=(*SSM)(n,n);
			SSM->insert(n, n, 10.e30); // fill diagonal
			RHS[n] = 10.e30 * hNew[n];  // rhs
			Q[n] = 0.;  //Computed later
			//continue;
		} else if (hNew[n] >= hCritS && rSoil*scaling<prec-1e-6) { // Infiltration capacity is exceeded
			msg::warning("Watflow: topsoil too wet, and can not take up precipitation, assuming runoff.");
			hNew[n] = hCritS;
			Adiag_boundary_nodes_[n]=(*SSM)(n,n);
			SSM->insert(n, n, 10.e30); // fill diagonal
			RHS[n] = 10.e30 * hNew[n];  // rhs
			Q[n] = 0.; //Computed later
		}
	}

	/// Free Drainage
	if(depthWaterTable_){//(assumes FreeD is false)
		//dirichlet at the bottom boundary.
		//todo implement code to get values for h (in case of changing groundwatertable over time
		double hbot=-300;
		depthWaterTable_->get(t,hbot);
		//msg::warning("Watflow::dirichlet: Dirichlet boundary set active");
		for (auto i : waterMesh.getBottomBoundaryNodes()) {
			hNew[i]=hbot-z[i];
			Adiag_boundary_nodes_[i]=(*SSM)(i,i);
			SSM->insert(i, i, 10.e30); // fill diagonal
			RHS[i] = 10.e30 * hNew[i];  // rhs
		}
	}

	return;
}

// **********************************************************************************
// scaling function for getting the actual evaporation from the potential evaporation

#ifdef KuppeScalingFunction
double Watflow::getScaling(const int n) {
	const double th=thNew[n];
	double thr=par.getthR()[n];
	double ths=par.getthSat()[n];
	const double thd=ths-thr;

	thr +=  evapoScalingDry * thd;
	ths -=  evapoScalingWet * thd;

	double scaling;
	if (th <= thr) {
		scaling = 0.0;
	} else if (th <= ths) {
		const double thRel = (th - thr) / (ths - thr);
		scaling = pow((0.5 - 0.5 * cos(M_PI * thRel)), evapoScalingExponent);
	} else {
		scaling = 1.0;
	}

	// post scaling procedure
	if (prec > rSoil * scaling) {
		if (prec > rSoil) {
			scaling = 1.;
		} else {
			scaling = prec / rSoil;
		}
	}
	return scaling;
}
#else
//this is all nonsense, do not use.
double Watflow::getScaling(const int n) {

	//rsoil is evaporation in cm/day
	// k is in cm/day

	const double th=thNew[n];
	double thr=par.getthR()[n];
	double ths=par.getthSat()[n];
	const double thd=ths-thr;

	thr +=  evapoScalingDry * thd;
	par.calc_fq(-10000.,0,ths);

	double scaling;
	if (th <= thr) {
		scaling = 0.0;
	} else if (th <= ths) {
		const double thRel = (th - thr) / (ths - thr);
		scaling = pow((0.5 - 0.5 * cos(M_PI * thRel)), evapoScalingExponent);
	} else {
		scaling = 1.0;
	}


	// post scaling procedure
	if (prec > rSoil * scaling) {
		if (prec > rSoil) {
			scaling = 1.;
		} else {
			scaling = prec / rSoil;
		}
	}
	return scaling;
}
#endif

void Watflow::set_atmosphere(const Time & t) {
	// local
	double  interception(0.);

	// precipitation
	prec_->get(t, prec);
	prec = std::fabs(prec);

	// interception
	if (prec > 1.e-6 && interception_) {
			interception_->get(t, interception);
			prec = prec - std::fabs(interception);
			msg::warning("Watflow::set_atmosphere: using interception from atmosphere section",1);
		}

	// evaporation
	evaporation_->get(t, rSoil);
	rSoil = std::fabs(rSoil);

	return;
} // end set_atmosphere method

void Watflow::setWaterSink(const double & t0, const double & t1) {
	if (t1 <= t0)
		return;
	Sink = 0.;
	for (unsigned int i(0); i < femIndexes_.size(); ++i) {
		if (!waterUptake_[i]->evaluateTime(t0))
			continue;
		std::vector<int> &ind = femIndexes_[i];
		if (ind[0] == -1) {
			//!ROOTS OUTSIDE GRID no nutrients and no water
			continue;
		} else {
			double wuptake;//, wu1;
			//waterUptake_[i]->get(t1, wuptake);
			//waterUptake_[i]->get(t0, wu1);
			//wuptake -= wu1;
			//wuptake /= (t1 - t0);
			waterUptake_[i]->get(t1,wuptake);
			if (allowOutflowFromRoots){
				if (fabs(wuptake) < 1e-15) continue;
			} else{
				if (wuptake < 1e-15) continue;
			}

			std::vector<int> * pind;
			std::vector<int> indexes;
			std::vector<double> weights;
			std::vector<double> * pweights;
			double sum;
			//double * psum;
			if (ind[0] == -2) {
				//growthpoint, needs matching
				Coordinate pos;
				waterUptake_[i]->getParent(2)->get(t0, pos);
				std::vector<double> &base = femWeights_[i];
				pos.x += base[0];
				pos.y += base[1];
				pos.z += base[2];
				waterMesh.matchNodes(pos, indexes, weights, sum); //todo: somewhat ugly, depends on order of things in several other places
				pind = &indexes;
				pweights = &weights;
				//psum=&sum;
			} else {
				pind = &ind;
				pweights = &femWeights_[i];
				//psum=&femSumWeights_[i];
			}
			if (pind->size() > 1) {
				double sumweights(0);
				for (unsigned int j = 0; j < pind->size(); ++j) {
					sumweights += (*pweights)[j];
				}
				for (unsigned int j = 0; j < pind->size(); ++j) {
					int femi = pind->at(j);
					Sink[femi] += wuptake * (*pweights)[j] / sumweights;
				}
			} else {
				//push the first nodal values
				int femi = pind->at(0);
				Sink[femi] += wuptake;
			}
			// make sure we do not take up water from dry soil
			bool warn=false;
			/*for (unsigned int j = 0; j < hOld.size(); ++j) {
			    if(hOld[j] < -15000.) {
			    	double s=  -1000./(hOld[j]+14000.);
			    	if(hOld[j]  <  -20000.) s=0.;
			    	Sink[j]*=s;
			    	warn=true;
			    }
			}*/
			if(warn) msg::warning("Watflow: Trying to take up water from dry node. Setting uptake for that node to 0. This leads to water balance errors");
			//if(warn) std::cout<<std::endl<<" limiting uptake at t="<<t0;


		}

	}
	Sink /= waterMesh.getvol();
}

void Watflow::updateLists(const Time &t, const std::deque<SimulaBase *> & rootNodeQueue_) {
	unsigned int j=waterUptake_.size();
	const std::string wr="rootSegmentWaterUptakeRate";
	for (auto & it : rootNodeQueue_) {
		SimulaBase* u = it->existingChild(wr);//NOTE this
		if(!u){
			u=new SimulaConstant<double>(wr,it,"ml/day",it->getStartTime(),it->getEndTime());
		}
		waterUptake_.push_back(u);
		u->garbageCollectionOn();

		u = it->existingChild("hydraulicHeadAtRootSurface");
		SimulaTable<Time> *tbl;
		if (u) {
			tbl = dynamic_cast<SimulaTable<Time> *>(u);
			u->garbageCollectionOn();
			if (!tbl)
				msg::error("Watflow::updateLists: hydraulicheadAtRootSurface must be a time table");
			// If water can flow out of roots, table extrapolation of hydraulic heads leads to issues in drying soil so we turn it off
			//if (allowOutflowFromRoots) tbl->setExtrapolationMethod("lastValue");
			hydraulicHead_.push_back(tbl);
		} else {
			hydraulicHead_.push_back(nullptr);
			msg::warning("Watflow::updateLists: hydraulicheadAtRootSurface not found",2);
		}

		u = it->existingChild("volumetricWaterContentAtTheRootSurface");
		if (u) {
			tbl = dynamic_cast<SimulaTable<Time> *>(u);
			u->garbageCollectionOn();
			if (!tbl)
				msg::error("Watflow::updateLists: volumetricWaterContentAtRootSurface must be a time table");
			volumetricWaterContent_.push_back(tbl);
		} else {
			volumetricWaterContent_.push_back(nullptr);
			msg::warning("Watflow::updateLists: volumetricWaterContentAtRootSurface not found",2);
		}

		u = it->existingChild("soilHydraulicConductivityAtTheRootSurface");
		if (u) {
			tbl = dynamic_cast<SimulaTable<Time> *>(u);
			u->garbageCollectionOn();
			if (!tbl)
				msg::error("Watflow::updateLists: soilHydraulicConductivityAtTheRootSurface must be a time table");
			rhizosphereConductance_.push_back(tbl);
		} else {
			rhizosphereConductance_.push_back(nullptr);
			msg::warning("Watflow::updateLists: soilHydraulicConductivityAtTheRootSurface not found",2);
		}


		//u=p->getChild("Cmin");

		if (it->getName() == "growthpoint") {
			Coordinate pos;
			it->getBase(t, pos);
			std::vector<double> w(3);
			w[0] = pos.x;
			w[1] = pos.y;
			w[2] = pos.z;
			femIndexes_.push_back(std::vector<int>(1, -2));
			femWeights_.push_back(w);
			//femSumWeights_.push_back(-1);
		} else {
			std::vector<int> indexes(27);
			std::vector<double> weights(27);
			double sum;
			Coordinate pos;
			it->getAbsolute(t, pos);

			waterMesh.matchNodes(pos, indexes, weights, sum);
			femIndexes_.push_back(indexes);
			femWeights_.push_back(weights);
			//femSumWeights_.push_back(sum);
		}

	}
	setrootsurfacevalues(t, TIMEERROR, j);//set root values for added root segments.

}

void Watflow::setrootsurfacevalues(const Time & t1, double dtMin, unsigned int j) {
	const double t=(double)ceil(t1/TIMEERROR)*TIMEERROR;//rounding of t, as very small rounding off differences can cause steep gradients in the interpolation tables, resulting in bad extrapolation results.
	// make sure lists are up to date
	double hhsur, thsur, Ksur(0.);
	for (unsigned int i(j); i < femIndexes_.size(); ++i) {
		if (!waterUptake_[i]->evaluateTime(t))
			continue;
		std::vector<int> &ind = femIndexes_[i];
		if (ind[0] == -1) {
			//!ROOTS OUTSIDE GRID no water
			hhsur = -1e8;
			thsur = 0.0;
		} else {
			std::vector<int> * pind;
			std::vector<int> indexes;
			std::vector<double> weights;
			std::vector<double> * pweights;
			double sum(0.0);
			//double * psum;
			if (ind[0] == -2) {

				//growthpoint, needs matching
				Coordinate pos;
				auto p=waterUptake_[i]->getParent();
				p->get(std::max(p->getStartTime(),t-MAXTIMESTEP) , pos);//TODO hack to make sure we do not enter predictor corrected loops here. I guess precision is not that great.
				std::vector<double> &base = femWeights_[i];
				pos.x += base[0];
				pos.y += base[1];
				pos.z += base[2];
				waterMesh.matchNodes(pos, indexes, weights, sum);
				pind = &indexes;
				pweights = &weights;
				//psum=&sum;
			} else {
				pind = &ind;
				pweights = &femWeights_[i];
				//psum=&Simunek::femSumWeights_[i];
			}
			if (pind->size() > 1) {
				//determine weights based on distances
				hhsur = 0;
				thsur = 0;
				Ksur = 0.;
				//std::cout<<std::endl<<"weigts";
				double tmps = 0;
				for (unsigned int j = 0; j < pind->size(); ++j) {
					int femi = pind->at(j);
					double w(pweights->at(j));
					tmps  += w;
					hhsur += w * hNew[femi];
					thsur += w * thNew[femi];
					Ksur  +=  w * Con[femi];
				}
				hhsur /= tmps;
				thsur /= tmps;
				Ksur /= tmps;
			} else {
				//push the first nodal values
				int femi = pind->at(0);
				if (femi < 0) {
					// growthpoint matching above returned -1 => point is outside of grid
					hhsur = -1e8;
					thsur = 0.0;
				} else {
					hhsur = hNew[femi];
					thsur = thNew[femi];
					Ksur  =  Con[femi];
				}
			}

		}
		if(hhsur< -15000) hhsur=-15000;
		if(hhsur> 0) hhsur=0;

//			std::cout.precision(5);
		if (hydraulicHead_[i]){
			if(hydraulicHead_[i]->size()==0 && t>hydraulicHead_[i]->getStartTime()){//coming from nodes that are just added. Put data in based on their start time so they do not complain.
				//todo why not do this when added?
				double ts=hydraulicHead_[i]->getStartTime();
				hydraulicHead_[i]->set(ts, hhsur);
			}
			hydraulicHead_[i]->set(t, hhsur);
		}
		if (volumetricWaterContent_[i]){
			if(volumetricWaterContent_[i]->size()==0 && t>volumetricWaterContent_[i]->getStartTime()){//coming from nodes that are just added. Put data in based on their start time so they do not complain.
				//todo why not do this when added?
				double ts=volumetricWaterContent_[i]->getStartTime();
				volumetricWaterContent_[i]->set(ts, thsur);
			}
			volumetricWaterContent_[i]->set(t, thsur);
		}
		if (rhizosphereConductance_[i]){
			if(rhizosphereConductance_[i]->size()==0 && t>rhizosphereConductance_[i]->getStartTime()){//coming from nodes that are just added. Put data in based on their start time so they do not complain.
				//todo why not do this when added?
				double ts=rhizosphereConductance_[i]->getStartTime();
				rhizosphereConductance_[i]->set(ts, Ksur);
			}
			rhizosphereConductance_[i]->set(t, Ksur);
		}


	}
}

void Watflow::unsetrootsurfacevalues(const Time & t) {
	for (unsigned int i(0); i < femIndexes_.size(); ++i) {
		if (!waterUptake_[i]->evaluateTime(t))
			continue;
		if (hydraulicHead_[i]){
			hydraulicHead_[i]->removeAfter(t);
		}
		if (volumetricWaterContent_[i]){
			volumetricWaterContent_[i]->removeAfter(t);
		}
		if (waterUptake_[i]){
			waterUptake_[i]->removeAfter(t);
		}
	}
}

void Watflow::togglePredictionMode(const bool &m)const {
	if (m) {
		for (unsigned int i(0); i < femIndexes_.size(); ++i) {
			if (hydraulicHead_[i]) {
				hydraulicHead_[i]->setPredictionModeON();
			}
			if (volumetricWaterContent_[i]) {
				volumetricWaterContent_[i]->setPredictionModeON();
			}

		}

	} else {
		for (unsigned int i(0); i < femIndexes_.size(); ++i) {
			if (hydraulicHead_[i]) {
				hydraulicHead_[i]->setPredictionModeOFF();
			}
			if (volumetricWaterContent_[i]) {
				volumetricWaterContent_[i]->setPredictionModeOFF();
			}

		}
	}
}


void Watflow::reportTotWaterColumn(const Time&t,const double & dt)const{

	const double tnew=t+dt;

	double thTot=waterMesh.integrateOverMesh(thNew);
	if (totWaterInColumn)
		totWaterInColumn->set(tnew, thTot);
	if (relativeWaterContentInTopLayer){
		double relativeWaterContent = 0;
		for (auto i:waterMesh.getTopBoundaryNodes()){
			relativeWaterContent += (thNew[i] - par.getthR()[i])/(par.getthSat()[i] - par.getthR()[i]);
		}
		relativeWaterContent /= waterMesh.getTopBoundaryNodes().size();
		relativeWaterContentInTopLayer->set(tnew, relativeWaterContent);
	}
	if (totWaterInLayers.size() > 0){
		for (std::size_t i = 0; i < totWaterInLayers.size(); i++){
			double watInLayer = waterMesh.integrateOverMeshLayer(thNew, lowerBounds_[i], upperBounds_[i]);
			totWaterInLayers[i]->set(tnew, watInLayer);
		}
	}
	static double initThTot(0.);
	if(dt==0.)initThTot=thTot;
	if (totWaterInColumn)
		totWaterChange->set(tnew, thTot-initThTot);


	//double thChange(0);
	//if(dt>0) thChange = (thTot-waterMesh.integrateOverMesh(thOld))/dt;

	//double hMean = waterMesh.integrateOverMesh(hNew)/VolumeOfDomain;

	double topBoundaryFlux(0);
	for (auto i : waterMesh.getTopBoundaryNodes()) {
		topBoundaryFlux += Q[i];
	}
	if(std::isnan(topBoundaryFlux)) msg::error("WatFlow: Top boundary flux is NAN");
	if(sumTopBoundaryFlux_)sumTopBoundaryFlux_->set(t,topBoundaryFlux);

	if(dt==0. && msg::getVerboseLevel()>1){
		double surf(0.);
		for (auto i : waterMesh.getTopBoundaryNodes()) {
			surf += waterMesh.getWidth()[i];
		}
		msg::warning("Watflow::Surface area water mesh is "+convertToString(surf));
	};

	double cumBottomBoundaryFlux(0);
	for (auto i : waterMesh.getBottomBoundaryNodes()) {
		cumBottomBoundaryFlux += Q[i];
	}
	if(std::isnan(cumBottomBoundaryFlux)) msg::error("WatFlow: Bottom boundary flux is NAN");
	if(sumBottomBoundaryFlux_)sumBottomBoundaryFlux_->set(t,cumBottomBoundaryFlux);

	double vMeanR = waterMesh.integrateOverMesh(Sink);//root uptake in per day
	if(totSink_) totSink_->set(t,-vMeanR);



}


WaterMassBalanceTest::WaterMassBalanceTest(SimulaDynamic* pSD) :
		DerivativeBase(pSD), relMassBalanceError_(nullptr) {

	//todo come up with better names for these
	totWaterChange = pSD->getSibling("totalWaterChangeInColumn", "cm3");
	sumTopBoundaryFlux_ = pSD->getSibling("topBoundaryFlux", "cm3");
	sumBottomBoundaryFlux_ = pSD->getSibling("bottomBoundaryFlux", "cm3");
	totSink_ = pSD->getSibling("totalSink", "cm3");
	totalWaterUptake_ = ORIGIN->existingPath("/plants/rootWaterUptake", "cm3");
	if (!totalWaterUptake_) {
		msg::warning("WaterMassBalanceTest: not including water uptake as the table \"/plants/rootWaterUptake\" is missing");
	}

	SimulaBase *d = pSD->existingSibling("relativeMassBalanceError");
	if (d)
		relMassBalanceError_ = dynamic_cast<SimulaTable<Time>*>(d);

	pSD->getSibling("totalWaterInColumn", "cm3")->get(0,ref);
}

void WaterMassBalanceTest::calculate(const Time & t, double &error) {
	double change(0);
	totWaterChange->get(t, change);

	double tboundary(0);
	sumTopBoundaryFlux_->get(t, tboundary);

	double bboundary(0);
	sumBottomBoundaryFlux_->get(t, bboundary);

	double sink(0);
	totSink_->get(t, sink);

	double uptake(0);
	if (totalWaterUptake_) {
		totalWaterUptake_->get(t, uptake);
	}

	//todo sink contains mineralization
	double sumFluxes(tboundary + bboundary + sink);
	error = (change - sumFluxes) / ref;
	if (std::fabs(error) > 0.01)
		msg::warning("WaterMassBalanceTest: mass balance is off by more than 1% of "+std::to_string(ref)+" ml initial water in column");

	if(relMassBalanceError_) relMassBalanceError_->set(t, error);

	//abs error returned
	error = change - sumFluxes;
}

std::string WaterMassBalanceTest::getName() const {
	return "waterMassBalanceTest";
}

DerivativeBase * newInstantiationWaterMassBalanceTest(SimulaDynamic* const pSD) {
	return new WaterMassBalanceTest(pSD);
}

//Register the module
class AutoRegisterWaterFunctions {
public:
	AutoRegisterWaterFunctions() {
		BaseClassesMap::getDerivativeBaseClasses()["waterMassBalanceTest"] = newInstantiationWaterMassBalanceTest;
	};
};

// our one instance of the proxy
static AutoRegisterWaterFunctions l8r9h38hr9h9h9;



