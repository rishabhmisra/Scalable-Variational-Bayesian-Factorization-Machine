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

#ifndef FM_LEARN_SGD_ONLINE_H_
#define FM_LEARN_SGD_ONLINE_H_

#include "fm_learn_sgd.h"
#include <vector>
#include <algorithm>
#include<sstream>
using namespace std;
class fm_learn_sgd_online: public fm_learn_sgd {
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


			uint tmp,size_except_last;
			uint total_cases = train.num_cases;
			tmp = ceil((double)(train.num_cases)/num_batch);
			size_except_last = tmp;
			uint* shuffle = new uint[train.num_cases];
			for(uint i=0;i<train.num_cases;i++)
			{
				shuffle[i] = i+1;
			}

			std::ofstream file_rmse;
            std::string file;
            std::stringstream convert; // stringstream used for the conversion

            convert << fm->k0<<fm->k1<<fm->num_factor;//add the value of Number to the characters in the stream

             std::string str = convert.str();
            file="test_rmse_" + str + "_sgd_online";
            file_rmse.open(file.c_str());
            file_rmse.close();

			for (int i = 0; i < num_iter; i++) {
				time_t now=time(0);
				char * temp=ctime(&now);
				std::cout<<temp<<std::endl;
				double iteration_time = getusertime();
				random_shuffle(shuffle,shuffle + train.num_cases);

				ofstream batch[num_batch];
				for(uint z=0;z<num_batch;z++)
				{
					std::ostringstream sstream;
					sstream << training_file <<"batch" << z+1;
					const char* file = (sstream.str()).c_str();
					batch[z].open(file, ios::out | ios::trunc);
				}
				uint index = 0;
				ifstream train_file(training_file.c_str());
				while (index<train.num_cases) {

					std::string line;
					std::getline(train_file, line);
					tmp = shuffle[index];
					index++;
					uint group = ceil(((double)tmp/size_except_last));
					batch[group-1] << line <<"\n";
				}
				train_file.close();

				for(uint z=0;z<num_batch;z++)
				{
					batch[z].close();
				}


                for(uint k = 1; k <= num_batch; k++)
				{
                    std::ostringstream sstream;
					sstream << training_file <<"batch"<< k;
					std::string file = sstream.str();
					DataSubset train1(0,true,true); // no transpose data
					train1.load(file,fm->num_attribute);

                    for(uint j = 0; j < train1.data->getNumRows(); j++) {
                        double p = fm->predict(train1.data->getRow(j), sum, sum_sqr);
                        double mult = 0;
                        if (task == 0) {
                            p = std::min(max_target, p);
                            p =std::max(min_target, p);
                            mult = -(train1.target(j)-p);
                        } else if (task == 1) {
                            mult = -train1.target(j)*(1.0-1.0/(1.0+exp(-train1.target(j)*p)));
                        }
                        SGD(train1.data->getRow(j), mult, sum);
                    }
					delete[] ((LargeSparseMatrixMemory<float>*) train1.data)->data.value[0].data;
                    delete[] ((LargeSparseMatrixMemory<float>*) train1.data)->data.value;
                    delete[] ((LargeSparseMatrixMemory<float>*) train1.data_t)->data.value[0].data;
                    delete[] ((LargeSparseMatrixMemory<float>*) train1.data_t)->data.value;
				}

				std::ofstream file_rmse;
                std::string file;
                std::stringstream convert; // stringstream used for the conversion

                convert << fm->k0<<fm->k1<<fm->num_factor;//add the value of Number to the characters in the stream

                std::string str = convert.str();
                file="test_rmse_" + str + "_sgd_online";
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
