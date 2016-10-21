// Copyright (C) 2010, 2011, 2012, 2013, 2014 Steffen Rendle
// Contact:   srendle@libfm.org, http://www.libfm.org/
//
// This file is part of libFM.
//
// libFM is free software: you can redistribute it and/or modify
// it under the terms of the GNU General Public License as published by
// the Free Software Foundation, either version 3 of the License, or
// (at your option) any later version.
//
// libFM is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
// GNU General Public License for more details.
//
// You should have received a copy of the GNU General Public License
// along with libFM.  If not, see <http://www.gnu.org/licenses/>.
//
//
// fm_learn_sgd.h: Stochastic Gradient Descent based learning for
// classification and regression
//
// Based on the publication(s):
// - Steffen Rendle (2010): Factorization Machines, in Proceedings of the 10th
//   IEEE International Conference on Data Mining (ICDM 2010), Sydney,
//   Australia.

#ifndef FM_LEARN_SGD_ELEMENT_H_
#define FM_LEARN_SGD_ELEMENT_H_

#include "fm_learn_sgd.h"
#include <vector>
#include <algorithm>
#include<sstream>
using namespace std;
class fm_learn_sgd_element: public fm_learn_sgd {
	public:
		virtual void init() {
			fm_learn_sgd::init();

			if (log != NULL) {
				log->addField("rmse_train", std::numeric_limits<double>::quiet_NaN());
			}
		}
		virtual void learn(DataSubset& train, DataSubset& test) {
			fm_learn_sgd::learn(train, test);

			std::cout << "SGD: DON'T FORGET TO SHUFFLE THE ROWS IN TRAINING DATA TO GET THE BEST RESULTS." << std::endl;
			// SGD
			std::vector<uint> random(train.data->getNumRows());
			for(uint ind = 0; ind < train.data->getNumRows(); ind++)
			{
				random[ind] = ind;
			}
			random_shuffle(random.begin(),random.end());

			std::ofstream file_rmse;
            std::string file;
            std::stringstream convert; // stringstream used for the conversion

            convert << fm->k0<<fm->k1<<fm->num_factor;//add the value of Number to the characters in the stream

             std::string str = convert.str();
            file="test_rmse_" + str + "_sgd";
            file_rmse.open(file.c_str());
            file_rmse.close();

			for (int i = 0; i < num_iter; i++) {
				time_t now=time(0);
				char * temp=ctime(&now);
				std::cout<<temp<<std::endl;
				double iteration_time = getusertime();
				std::random_shuffle(random.begin(),random.end());
				//for (train.data->begin(); !train.data->end(); train.data->next()) {
				for(uint j = 0; j < train.data->getNumRows(); j++) {
					double p = fm->predict(train.data->getRow(random[j]), sum, sum_sqr);
					//std::cout<<p<<std::endl;
					double mult = 0;
					if (task == 0) {
						p = std::min(max_target, p);
						p = std::max(min_target, p);
						mult = -(train.target(random[j])-p);
					} else if (task == 1) {
						mult = -train.target(random[j])*(1.0-1.0/(1.0+exp(-train.target(random[j])*p)));
					}
					SGD(train.data->getRow(random[j]), mult, sum);
				}
				std::ofstream file_rmse;
                std::string file;
                std::stringstream convert; // stringstream used for the conversion

                convert << fm->k0<<fm->k1<<fm->num_factor;//add the value of Number to the characters in the stream

                std::string str = convert.str();
                file="test_rmse_" + str + "_sgd";
                file_rmse.open(file.c_str(), std::ios_base::app);
				
				iteration_time = (getusertime() - iteration_time);
 				//double rmse_train = evaluate(train);
				double rmse_test = evaluate(test);
				file_rmse<<rmse_test<<"\n";
				std::cout << "#Iter=" << std::setw(3) << i << "\tTest=" << rmse_test << std::endl;
				if (log != NULL) {
//					log->log("rmse_train", rmse_train);
					log->log("time_learn", iteration_time);
					log->newLine();
				}
			}
		}

};

#endif /*FM_LEARN_SGD_ELEMENT_H_*/
