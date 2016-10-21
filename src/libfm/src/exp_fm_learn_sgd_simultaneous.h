// Author Avijit Saha
// stochastic gradient descent for exponential family
//



#ifndef EXP_FM_LEARN_SGD_SIMULTANEOUS_H_
#define EXP_FM_LEARN_SGD_SIMULTANEOUS_H_

#include "exp_fm_learn_sgd.h"


class exp_fm_learn_sgd_simultaneous : public exp_fm_learn_sgd {
	protected:

		virtual void _learn(DataSubset& train, DataSubset& test) {

			uint num_complete_iter = 0;
			//int count=0;
			// make a collection of datasets that are predicted jointly
			int num_data = 2;
			DVector<DataSubset*> main_data(num_data);
			DVector<e_q_term*> main_cache(num_data);
			main_data(0) = &train;
			main_data(1) = &test;
			main_cache(0) = cache;
			main_cache(1) = cache_test;


			predict_data_and_write_to_eterms(main_data, main_cache);
			if (task == TASK_REGRESSION) {
				std::cout<<"stdev in exp_sum"<<fm->stdev<<std::endl;
				// remove the target from each prediction, because: e(c) := \hat{y}(c) - target(c)
				for (uint c = 0; c < train.num_cases; c++) {
					cache[c].e = fm->stdev*cache[c].e - train.target(c);
				}
			}
			 else {
				throw "unknown task";
			}

			/*
			if (count==0) {
				std::cout<<"cache output"<<std::endl;
				for (uint c = 0; c < test.num_cases; c++) {
					std::cout<<cache[c].e<<std::endl;
				}
			}
			*/

			for (uint i = num_complete_iter; i < num_iter; i++) {
				double iteration_time = getusertime();
				clock_t iteration_time3 = clock();
				double iteration_time4 = getusertime4();
				nan_cntr_w0 = 0; inf_cntr_w0 = 0; nan_cntr_w = 0; inf_cntr_w = 0; nan_cntr_v = 0; inf_cntr_v = 0;

				fm_SGD(train,learn_rate);



				if ((nan_cntr_w0 > 0) || (inf_cntr_w0 > 0)) {
					std::cout << "#nans in w0:\t" << nan_cntr_w0 << "\t#inf_in_w0:\t" << inf_cntr_w0 << std::endl;
				}
				if ((nan_cntr_w > 0) || (inf_cntr_w > 0)) {
					std::cout << "#nans in w:\t" << nan_cntr_w << "\t#inf_in_w:\t" << inf_cntr_w << std::endl;
				}
				if ((nan_cntr_v > 0) || (inf_cntr_v > 0)) {
					std::cout << "#nans in v:\t" << nan_cntr_v << "\t#inf_in_v:\t" << inf_cntr_v << std::endl;
				}



				// predict test and train
				//std::cout<<"Updating error term"<<std::endl;
				predict_data_and_write_to_eterms(main_data, main_cache);
				// (prediction of train is not necessary but it increases numerical stability)



				double rmse_train = 0.0;
				if (task == TASK_REGRESSION) {
					// evaluate test and store it
					//std::cout<<"number of test instances in sim "<<test.num_cases<<std::endl;
					for (uint c = 0; c < test.num_cases; c++) {
						double p = cache_test[c].e;
						pred_this(c) = p;
						p = std::min(max_target, p);
						p = std::max(min_target, p);
						pred_sum(c) = p;
					}
					/*
					count =count+1;
					if (count==1) {

						for (uint c = 0; c < test.num_cases; c++) {
							std::cout<<pred_sum(c)<<std::endl;
						}
					}
					*/
					// Evaluate the training dataset and update the e-terms
					for (uint c = 0; c < train.num_cases; c++) {
						double p = cache[c].e;
						p = std::min(max_target, p);
						p = std::max(min_target, p);
						double err = p - train.target(c);
						rmse_train += err*err;
						cache[c].e = fm->stdev * p - train.target(c);
					}
					rmse_train = std::sqrt(rmse_train/train.num_cases);

				} else {
					throw "unknown task";
				}

				iteration_time = (getusertime() - iteration_time);
				iteration_time3 = clock() - iteration_time3;
				iteration_time4 = (getusertime4() - iteration_time4);
				if (log != NULL) {
					log->log("time_learn", iteration_time);
					log->log("time_learn2", (double)iteration_time3 / CLOCKS_PER_SEC);
					log->log("time_learn4", (double)iteration_time4);
				}


				// Evaluate the test data sets
				if (task == TASK_REGRESSION) {
					double rmse_test_this, mae_test_this;
					 _evaluate(pred_sum, test.target, 1.0, rmse_test_this, mae_test_this, num_eval_cases);
					std::cout << "#Iter=" << std::setw(3) << i << "\tTrain=" << rmse_train << "\tTest=" << rmse_test_this << std::endl;

					if (log != NULL) {
						log->log("rmse_mcmc_this", rmse_test_this);

						if (num_eval_cases < test.target.dim) {
							double rmse_test_this, mae_test_this;//, rmse_test2_all_but5, mae_test2_all_but5;
							 _evaluate(pred_this, test.target, 1.0, rmse_test_this, mae_test_this, num_eval_cases, test.target.dim);

							//log->log("rmse_mcmc_test2_this", rmse_test2_this);
							//log->log("rmse_mcmc_test2_all", rmse_test2_all);
						}
						log->newLine();
					}
				} else {
					throw "unknown task";
				}
			}
		}

		void _evaluate(DVector<double>& pred, DVector<DATA_FLOAT>& target, double normalizer, double& rmse, double& mae, uint from_case, uint to_case) {
			assert(pred.dim == target.dim);
			double _rmse = 0;
			double _mae = 0;
			uint num_cases = 0;
			for (uint c = std::max((uint) 0, from_case); c < std::min((uint)pred.dim, to_case); c++) {
				double p = pred(c) * normalizer;
				p = std::min(max_target, p);
				p = std::max(min_target, p);
				double err = p - target(c);
				_rmse += err*err;
				_mae += std::abs((double)err);
				num_cases++;
			}
			//std::cout<<"rmse "<<_rmse<<std::endl;
			//std::cout<<"number of ins "<<num_cases<<std::endl;
			rmse = std::sqrt(_rmse/num_cases);
			mae = _mae/num_cases;

		}

		void _evaluate_class(DVector<double>& pred, DVector<DATA_FLOAT>& target, double normalizer, double& accuracy, double& loglikelihood, uint from_case, uint to_case) {
			double _loglikelihood = 0.0;
			uint _accuracy = 0;
			uint num_cases = 0;
			for (uint c = std::max((uint) 0, from_case); c < std::min((uint)pred.dim, to_case); c++) {
				double p = pred(c) * normalizer;
				if (((p >= 0.5) && (target(c) > 0.0)) || ((p < 0.5) && (target(c) < 0.0))) {
					_accuracy++;
				}
				double m = (target(c)+1.0)*0.5;
				double pll = p;
				if (pll > 0.99) { pll = 0.99; }
				if (pll < 0.01) { pll = 0.01; }
				_loglikelihood -= m*log10(pll) + (1-m)*log10(1-pll);
				num_cases++;
			}
			loglikelihood = _loglikelihood/num_cases;
			accuracy = (double) _accuracy / num_cases;
		}


		void _evaluate(DVector<double>& pred, DVector<DATA_FLOAT>& target, double normalizer, double& rmse, double& mae, uint& num_eval_cases) {
			_evaluate(pred, target, normalizer, rmse, mae, 0, num_eval_cases);
		}

		void _evaluate_class(DVector<double>& pred, DVector<DATA_FLOAT>& target, double normalizer, double& accuracy, double& loglikelihood, uint& num_eval_cases) {
			_evaluate_class(pred, target, normalizer, accuracy, loglikelihood, 0, num_eval_cases);
		}
};

#endif /*FM_LEARN_MCMC_SIMULTANEOUS_H_*/
