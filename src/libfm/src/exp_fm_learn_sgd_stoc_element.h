// Author Avijit Saha
// This file contains exponential family stochastic gradient descent
//

#ifndef EXP_FM_LEARN_SGD_STOC_ELEMENT_H_
#define EXP_FM_LEARN_SGD_STOC_ELEMENT_H_

#include "exp_fm_learn_sgd_stoc.h"

class exp_fm_learn_sgd_stoc_element: public exp_fm_learn_sgd_stoc {
	public:
		virtual void init() {
			exp_fm_learn_sgd_stoc::init();

			if (log != NULL) {
				log->addField("rmse_train", std::numeric_limits<double>::quiet_NaN());
			}
		}
		virtual void learn(DataSubset& train, DataSubset& test) {
			exp_fm_learn_sgd_stoc::learn(train, test);

			std::cout << "SGD: DON'T FORGET TO SHUFFLE THE ROWS IN TRAINING DATA TO GET THE BEST RESULTS." << std::endl;
			// SGD
			for (int i = 0; i < num_iter; i++) {

				double iteration_time = getusertime();
				for (train.data->begin(); !train.data->end(); train.data->next()) {
					double p = fm->predict(train.data->getRow(), sum, sum_sqr);
					//std::cout<<p<<std::endl;
					double mult = 0;
					if (task == 0) {
						//p = std::min(max_target, p);
						//p = std::max(min_target, p);
						//std::cout<<fm->stdev;
						mult = -(train.target(train.data->getRowIndex())-(1/fm->stdev)*p);
					} else if (task == 1) {
						mult = -train.target(train.data->getRowIndex())*(1.0-1.0/(1.0+exp(-train.target(train.data->getRowIndex())*p)));
					}
					else if (task == 2) {
						p = std::min(max_target, p);
						p = std::max(min_target, p);
						mult = -(train.target(train.data->getRowIndex())-exp(p));
					}
					SGD(train.data->getRow(), mult, sum);
				}
				iteration_time = (getusertime() - iteration_time);
 				double rmse_train = evaluate(train);
				double rmse_test = evaluate(test);
				std::cout << "#Iter=" << std::setw(3) << i << "\tTrain=" << rmse_train << "\tTest=" << rmse_test << std::endl;
				if (log != NULL) {
					log->log("rmse_train", rmse_train);
					log->log("time_learn", iteration_time);
					log->newLine();
				}
			}
		}

};

#endif /*FM_LEARN_SGD_ELEMENT_H_*/
