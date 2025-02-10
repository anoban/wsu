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

#include "../export/ExportLibrary.hpp"

#include <fstream>
#include "../cli/Messages.hpp"
#include "../cli/Info.hpp"
#include "../engine/SimulaBase.hpp"
#include "../engine/Origin.hpp"
#include "../math/MathLibrary.hpp"
#include <vector>
#include <map>
#include <iomanip>
#include "../export/3dimage/rawimage.hpp"
#include "../export/ExportBaseClass.hpp"
#include "../export/General/GarbageCollection.hpp"
#include "../export/General/PrimeModel.hpp"
#include "../export/General/ProbeAllObjects.hpp"
#include "../export/Text/TabledOutput.hpp"
#include "../export/Text/Stats.hpp"
#include "../export/Text/WriteModel2File.hpp"
#include "../export/VTK/VTU.hpp"
#include "../export/RSML/RSML.hpp"
#include "../tools/StringExtensions.hpp"

#if _WIN32 || _WIN64
#include "windows.h"
#include "psapi.h"
//#include "Wincon.h"
   static double maxMemUse(0);
#else
   static double maxMemUse(0);
#endif

#ifdef __APPLE__
	#include <mach/mach.h>
#endif

ModuleList modules;///todo runExportMoudules needs to be converted to a proper class which exposes this.
static std::string inputfoldername;
std::string getCurrentInputFolderName(){//hack solang cli does not provide a standard way to retrieve this
	return inputfoldername;
}

void runExportModules(int &exitCode, const std::string &dir){
	try{
		inputfoldername=dir;
		while(!inputfoldername.empty()) {
			if(*(inputfoldername.rbegin())=='/') break;
			inputfoldername.erase(--inputfoldername.end());
		}

		std::cout << ANSI_Black << "Running modules\n" << std::flush;
		///@todo we want to loop through list of registered functions.
		//instantiation of export modules,
		///@todo muliple instantiations of rawimage under different names is now possible. Instantiations should really be based on a loop through objects in the simulacontrolparameter list, but we need to recognize the type
		//@todo when not enabled, should be destructed again?
		PrimeModel pm;
		ProbeAllObjects pao;
		VTU vtu("VTU");
		VTU vtp("vtp");
		VTU vtuwire("wireframe.vtu");
		RSML rsml("RSML");
		RawImage raw1("rasterImage");
		Table tbl;
		Stats tblstat;
		GarbageCollection grb;
		ModelDump mdp;
		if(pm.enabled()) modules.push_back(&pm);
		if(pao.enabled()) modules.push_back(&pao);
		if(tbl.enabled()) modules.push_back(&tbl);
		if(vtu.enabled()) modules.push_back(&vtu);
		if(vtp.enabled()) modules.push_back(&vtp);
		if(vtuwire.enabled()) modules.push_back(&vtuwire);
		if(rsml.enabled()) modules.push_back(&rsml);
		if(raw1.enabled()) modules.push_back(&raw1);
		/*if(raw2.enabled()) modules.push_back(&raw2);
		if(raw3.enabled()) modules.push_back(&raw3);
		if(raw4.enabled()) modules.push_back(&raw4);
		if(raw5.enabled()) modules.push_back(&raw5);*/
		//make sure this runs last?
		if(grb.enabled()) modules.push_back(&grb);
		if(mdp.enabled()) modules.push_back(&mdp);
		if(tblstat.enabled()) modules.push_back(&tblstat);

		//initiation of export modules
		for(ModuleList::iterator it(modules.begin()); it!=modules.end(); ++it){
			(*it)->initialize();
		}

		//time loop for export modules
		std::cout<<std::fixed;

		//max endtime
		Time endTime(0), intervaltime(1.e15), startTime(1.e15);
		for(ModuleList::iterator it(modules.begin()); it!=modules.end(); ++it){
			endTime=std::max(endTime,(*it)->getEndTime());
			intervaltime=std::min(intervaltime,(*it)->getIntervalTime());
			startTime=std::min(startTime,(*it)->getStartTime());
		}

		//screen ouput
		unsigned int numberOfDigitsOnScreen=0;
		if (intervaltime<1) numberOfDigitsOnScreen=1;
		if (intervaltime<0.1) numberOfDigitsOnScreen=2;
		if (intervaltime<0.01) numberOfDigitsOnScreen=3;
		if (intervaltime<0.001) numberOfDigitsOnScreen=4;
		if (intervaltime<0.0001) numberOfDigitsOnScreen=5;

		//experimental different time loop.
		while(true){
			double currentTime(1e15);
			ExportBase* nit = nullptr;
			bool match=false;
			for(ModuleList::iterator it(modules.begin()); it!=modules.end(); ++it){
				if( (*it)->enabled() && (*it)->getCurrentTime() < currentTime) {
					nit=*it;
					currentTime=nit->getCurrentTime();
					match=true;
					//std::cout<<std::endl<<" found "<<nit->getName();
				}
			}
			if(!match) break;
			//std::cout<<std::endl<<" running "<<nit->getName()<<" at time "<<nit->getCurrentTime();
			nit->run(nit->getCurrentTime());
			if(nit->getEndTime() - nit->getCurrentTime()  < 0.5*nit->getIntervalTime()+TIMEERROR) nit->enabled()=false;
			const double t=nit->getCurrentTime();
			nit->getCurrentTime() = nit->getNextOutputTime();

			const bool doneSomething=true;


#ifdef WRITEGRAPH


			if(doneSomething && t>11.9 && t<12.1){
				auto &g(ORIGIN->getDependencieGraph());
				//const std::vector<std::string> &l(ORIGIN->getDependencieGraphLabels());

				std::string filename("graph_"+std::to_string(t)+".dot");
				std::ofstream os;
				os.open( filename.c_str() );
				if ( !os.is_open() ) msg::error("generateGraph: Failed to open "+filename);

				os<<"strict digraph {\nlabel=\"OpenSimRoot\"\n   ";
				std::set<SimulaBase*> vc;
				std::set<SimulaBase*> vd;


				for (auto &i:g){
					auto tp1=i.first->getType();
					auto tp2=i.second->getType();
					auto n1=i.second->getName();
					if(n1=="rootType") continue;
					if(n1=="plantType") continue;
					if (tp1.substr(0,14)=="SimulaConstant"){
						vc.insert(i.first);//todo this should never happen
					}else{
						vd.insert(i.first);
					}
					if (tp2.substr(0,14)=="SimulaConstant"){
						vc.insert(i.second);
					}else{
						vd.insert(i.second);
					}
				}
				os<<"\n   node[shape=\"diamond\", style=\"\"]\n";
				for (auto i:vc){
					os<<" \"Const "<<i->getPrettyName()<<"\";\n";
				}
				os<<"\n   node[shape=\"box\", style=\"rounded\"]\n";
				for (auto i:vd){
					os<<" \""<<i->getPrettyName()<<"\";\n";
				}
				os<<"\n";


				for (auto &i:g){
					auto tp1=i.first->getType();
					auto tp2=i.second->getType();
					auto n1=i.second->getName();
					auto n2=i.first->getName();
					if(n1=="rootType") continue;
					if(n1=="plantType") continue;
					auto np1=i.second->getPrettyName();
					auto np2=i.first->getPrettyName();
					if (tp1.substr(0,14)=="SimulaConstant"){
						np2="Const "+np2;
					}
					if (tp2.substr(0,14)=="SimulaConstant"){
						np1="Const "+np1;
					}
					//auto p=i.second->getParent();
					//if( p && p->getName()=="dataPoints")


					//std::replace( n1.begin(), n1.end(), ':', '_');
					//std::replace( n1.begin(), n1.end(), ';', '_');
					//std::replace( n2.begin(), n2.end(), ':', '_');
					//std::replace( n2.begin(), n2.end(), ';', '_');
					os<<"\n   \""<<np1<<"\"";
					os<<" -> \""<<np2<<"\"";
					auto t2=i.second->getType();
					if (t2.substr(0,14)=="SimulaConstant"){
						double val;
						i.second->get(val);
						os<<"[label=\""<< val<<"\"]";
					}
				}
				os<<"\n}";
				os.close();
				std::string c("dot -Tpng -o "+filename+".png "+filename);
				system(c.c_str());
				ORIGIN->clearDependencieGraph();
			}

#endif

			if(doneSomething){
				auto walltime = std::time(nullptr);
				auto local_walltime = std::localtime(&walltime);
//note that on mingw this does not link, but the "linux" code does.
#if _WIN32 || _WIN64
				
				MEMORYSTATUSEX memInfo;
				memInfo.dwLength = sizeof(MEMORYSTATUSEX);
				GlobalMemoryStatusEx(&memInfo);
				//DWORDLONG totalPhysMem = memInfo.ullTotalPhys;
				PROCESS_MEMORY_COUNTERS_EX pmc;
				pmc.cb = sizeof(pmc);
				GetProcessMemoryInfo(GetCurrentProcess(), (PROCESS_MEMORY_COUNTERS*)&pmc, pmc.cb);
				size_t mem = pmc.WorkingSetSize;
				double m = mem/1000000.;
				std::cout << std::fixed << std::setprecision(1);
				std::cout
					<< std::put_time(local_walltime, "%T") << ": "
					<< t << "/" << endTime
					<< " days. Mem " << floor(m)
					<< " mB. #obj.=" << SimulaBase::numberOfInstantiations
					<< " x64b/obj.=" << (m*131072. / ((double)SimulaBase::numberOfInstantiations))
					<< "\r"
					<< std::flush;
#else
	#if __APPLE__
				mach_task_basic_info_data_t memInfo;
				mach_msg_type_number_t memInfoCount = MACH_TASK_BASIC_INFO_COUNT;
				mach_vm_size_t m(0);
				if(KERN_SUCCESS == task_info(mach_task_self(), MACH_TASK_BASIC_INFO, (task_info_t)&memInfo, &memInfoCount)){
					m = memInfo.resident_size / (1024*1024); // OS X reports in bytes not pages
				}
	#else
				/*memory usage status*/
				/*TODO maybe call something smart like a garbage collector or so to reduce memory usage when high?*/
				/*TODO discussion on internet suggests statm is not correct - status is better*/
				std::ifstream statm("/proc/self/statm");
				/*TODO this assumes a page size of 4096, for the correct page size run "getconf PAGE_SIZE" */
				double m;
				statm>>m;statm>>m;m*=4./1024.;// strictly 4096/1024/1024
				statm.close();
	#endif
				if(m>maxMemUse)maxMemUse=m;
				/*progress indicator*/
				std::cout << std::fixed << std::setprecision(numberOfDigitsOnScreen);
				//1024.*1024./8. = 131072 to get from mB to 64bit units
				std::cout
					<< std::put_time(local_walltime, "%T") << ": "
					<< t << "/" << endTime
					<< " days. Mem " << floor(m)
					<< " mB. #obj.=" << SimulaBase::numberOfInstantiations
					<< " x64b/obj.=" << (m*131072./((double)SimulaBase::numberOfInstantiations))
					<< "\r"
					<< std::flush;
#endif
			}
		}
		std::string resourceUsage="At "+convertToString(endTime)+" days mem usage was "+convertToString(floor(maxMemUse))+" mB.";
		msg::warning(resourceUsage);

#if _WIN32_WINNT_WIN7
		std::cout << std::endl;
#endif 
		ANSI_OKmessage
		std::cout<<std::endl<<"Finalizing output:";

		for(ModuleList::iterator it(modules.begin()); it!=modules.end(); ++it){
			(*it)->finalize();
		}

		ANSI_OKmessage



	}	catch (std::exception& error){
		ANSI_FAILmessage
		std::cout << ANSI_Red <<error.what()<< ANSI_Black;
		exitCode=1;
	}
}


//TODO move this to more sensible place
void terminate(){
	// Tell about possible warnings
	writeWarnings();

	//write run time
	std::cout<<getExecutionTime();

	//reset terminal
	resetTerminal();
}

std::string getExecutionTime() {
	std::size_t seconds(clock() / CLOCKS_PER_SEC);
	std::size_t minutes(seconds / 60);
	std::size_t hours(minutes / 60);
	seconds = seconds - 60 * minutes;
	minutes = minutes - 60 * hours;
	std::stringstream r;
	r << "Simulation took (hours:minutes:seconds): " << hours << ":" << minutes << ":" << seconds;
	return (r.str());
}

void resetTerminal(){
	std::cout<<ANSI_ResetTerminal<<std::endl<<std::flush;
}
