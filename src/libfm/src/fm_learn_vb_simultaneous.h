// Author Avijit Saha & Rishabh Misra
// This file contains VB learning
//



#ifndef FM_LEARN_VB_SIMULTANEOUS_H_
#define FM_LEARN_VB_SIMULTANEOUS_H_

#include "fm_learn_vb.h"
#include<ctime>
#include<stdlib.h>
#include<string.h>

class fm_learn_vb_simultaneous : public fm_learn_vb {
	protected:

		virtual void _learn(DataSubset& train, DataSubset& test) {

			uint num_complete_iter = 0;

			// make a collection of datasets that are predicted jointly
			int num_data = 2;
			int num_data_only_test = 1;
			DVector<DataSubset*> main_data(num_data);
			DVector<e_q_term*> main_cache(num_data);
			main_data(0) = &train;
			main_data(1) = &test;
			main_cache(0) = cache;
			main_cache(1) = cache_test;
			DVector<DataSubset*> main_data_only_test(num_data_only_test);
			DVector<e_q_term*> main_cache_only_test(num_data_only_test);
			main_data_only_test(0) = &test;
			main_cache_only_test(0) = cache_test;

			std::cout<<"check in fm_learn_vb_sim"<<std::endl;
			predict_data_and_write_to_eterms(main_data, main_cache);
			predict_t_and_write_to_qterms(&train, cache_t);
			//std::cout<<"check in fm_learn_vb_sim"<<std::endl;
			if (task == TASK_REGRESSION) {
				// remove the target from each prediction, because: e(c) := \hat{y}(c) - target(c)
				for (uint c = 0; c < train.num_cases; c++) {
					cache[c].e = train.target(c) - cache[c].e;
				}

			} else if (task == TASK_CLASSIFICATION) {
				// for Classification: remove from e not the target but a sampled value from a truncated normal
				// for initializing, they are not sampled but initialized with meaningful values:
				// -1 for the negative class and +1 for the positive class (actually these are the values that are already in the target and thus, we can do the same as for regression; but note that other initialization strategies would need other techniques here:
				for (uint c = 0; c < train.num_cases; c++) {
					cache[c].e = train.target(c) - cache[c].e;
				}

			} else {
				throw "unknown task";
			}

			// open file to store test rmse
			std::ofstream file_rmse;
			std::string file;
			std::stringstream convert; // stringstream used for the conversion

			convert << fm->k0<<fm->k1<<fm->num_factor;//add the value of Number to the characters in the stream

			std::string str = convert.str();
			file="test_rmse_" + str + "_vb";
			file_rmse.open(file.c_str());
			file_rmse.close();
			//clear free energy file
			std::ofstream myfile;
			file="free_energy_" + str + "_vb";
			myfile.open(file.c_str());
			myfile.close();

			for (uint i = num_complete_iter; i < num_iter; i++) {
				time_t now=time(0);
				char * temp=ctime(&now);
				std::cout<<temp<<std::endl;
				double iteration_time = getusertime();
				clock_t iteration_time3 = clock();
				double iteration_time4 = getusertime4();
				nan_alpha=0; nan_sigma_0=0; nan_sigma_w=0; nan_sigma_v=0; nan_mu_0_dash=0; nan_sigma_0_dash=0; nan_mu_w_dash=0;
				nan_sigma_w_dash=0; nan_mu_v_dash=0; nan_sigma_v_dash=0; nan_sigma_w=0; nan_sigma_v=0;

				//std::cout<<"check in fm_learn_vb_sim"<<std::endl;
				update_all(train);
				//std::cout<<"check in fm_learn_vb_sim"<<std::endl;

				if ((nan_alpha > 0) || (inf_alpha > 0)) {
					std::cout << "#nans in alpha:\t" << nan_alpha << "\t#inf_in_alpha:\t" << inf_alpha << std::endl;
				}
				if ((nan_sigma_0 > 0) || (inf_sigma_0 > 0)) {
					std::cout << "#nans in alpha:\t" << nan_sigma_0<< "\t#inf_in_alpha:\t" << inf_sigma_0 << std::endl;
				}
				if ((nan_sigma_w > 0) || (inf_sigma_w > 0)) {
					std::cout << "#nans in alpha:\t" << nan_sigma_w << "\t#inf_in_alpha:\t" << inf_sigma_w << std::endl;
				}
				if ((nan_sigma_v > 0) || (inf_sigma_v > 0)) {
					std::cout << "#nans in alpha:\t" << nan_sigma_v << "\t#inf_in_alpha:\t" << inf_sigma_v << std::endl;
				}
				if ((nan_mu_0_dash > 0) || (inf_mu_0_dash > 0)) {
					std::cout << "#nans in alpha:\t" << nan_mu_0_dash << "\t#inf_in_alpha:\t" << inf_mu_0_dash << std::endl;
				}
				if ((nan_sigma_0_dash > 0) || (inf_sigma_0_dash > 0)) {
					std::cout << "#nans in alpha:\t" << nan_sigma_0_dash << "\t#inf_in_alpha:\t" << inf_sigma_0_dash << std::endl;
				}
				if ((nan_mu_w_dash > 0) || (inf_mu_w_dash > 0)) {
					std::cout << "#nans in alpha:\t" << nan_mu_w_dash << "\t#inf_in_alpha:\t" << inf_mu_w_dash << std::endl;
				}
				if ((nan_sigma_w_dash > 0) || (inf_sigma_w_dash > 0)) {
					std::cout << "#nans in alpha:\t" << nan_sigma_w_dash << "\t#inf_in_alpha:\t" << inf_sigma_w_dash << std::endl;
				}
				if ((nan_mu_v_dash > 0) || (inf_mu_v_dash > 0)) {
					std::cout << "#nans in alpha:\t" << nan_mu_v_dash << "\t#inf_in_alpha:\t" << inf_mu_v_dash << std::endl;
				}
				if ((nan_sigma_v_dash > 0) || (inf_sigma_v_dash > 0)) {
					std::cout << "#nans in alpha:\t" << nan_sigma_v_dash << "\t#inf_in_alpha:\t" << inf_sigma_v_dash << std::endl;
				}



				// predict test and train
				//std::cout<<"Updating error term"<<std::endl;
				//predict_data_and_write_to_eterms(main_data, main_cache);
				predict_data_and_write_to_eterms(main_data_only_test, main_cache_only_test);
				// (prediction of train is not necessary but it increases numerical stability)


				std::ofstream file_rmse;
				std::string file;
				std::stringstream convert; // stringstream used for the conversion

				convert << fm->k0<<fm->k1<<fm->num_factor;//add the value of Number to the characters in the stream

				std::string str = convert.str();
				file="test_rmse_" + str + "_vb";
				file_rmse.open(file.c_str(), std::ios_base::app);

				double acc_train = 0.0;
				double rmse_train = 0.0;
				if (task == TASK_REGRESSION) {
					// evaluate test and store it
					for (uint c = 0; c < test.num_cases; c++) {
						double p = cache_test[c].e;
						//std::cout<<p<<std::endl;
						p = std::min(max_target, p);
						p = std::max(min_target, p);
						//std::cout<<p<<std::endl;
						pred_this(c) = p;
					}

					// Evaluate the training dataset and update the e-terms
					for (uint c = 0; c < train.num_cases; c++) {
						double p = cache[c].e;
						p = std::min(max_target, p);
						p = std::max(min_target, p);
						//double err = train.target(c) - p;
						//rmse_train += err*err;
						rmse_train += p*p;
						//cache[c].e = train.target(c) - cache[c].e;
					}
					rmse_train = std::sqrt(rmse_train/train.num_cases);

				} else if (task == TASK_CLASSIFICATION) {
					// evaluate test and store it
					for (uint c = 0; c < test.num_cases; c++) {
						double p = cache_test[c].e;
						p = cdf_gaussian(p);
						pred_this(c) = p;
					}

					// Evaluate the training dataset and update the e-terms
					uint _acc_train = 0;
					for (uint c = 0; c < train.num_cases; c++) {
						double p = cache[c].e;
						p = cdf_gaussian(p);
						if (((p >= 0.5) && (train.target(c) > 0.0)) || ((p < 0.5) && (train.target(c) < 0.0))) {
							_acc_train++;
						}

						double sampled_target;
						if (train.target(c) >= 0.0) {
							{
								// the target is the expected value of the truncated normal
								double mu = cache[c].e;
								double phi_minus_mu = exp(-mu*mu/2.0) / sqrt(3.141*2);
								double Phi_minus_mu = cdf_gaussian(-mu);
								sampled_target = mu + phi_minus_mu / (1-Phi_minus_mu);
							}
						} else {
							{
								// the target is the expected value of the truncated normal
								double mu = cache[c].e;
								double phi_minus_mu = exp(-mu*mu/2.0) / sqrt(3.141*2);
								double Phi_minus_mu = cdf_gaussian(-mu);
								sampled_target = mu - phi_minus_mu / Phi_minus_mu;
							}
						}
						cache[c].e = sampled_target - cache[c].e ;
					}
					acc_train = (double) _acc_train / train.num_cases;

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
					_evaluate(pred_this, test.target, 1.0, rmse_test_this, mae_test_this, num_eval_cases);
					file_rmse<<rmse_test_this<<"\n";
					std::cout << "#Iter=" << std::setw(3) << i << "\tTrain=" << rmse_train << "\tTest=" << rmse_test_this << std::endl;

					if (log != NULL) {
						log->log("rmse_mcmc_this", rmse_test_this);

						if (num_eval_cases < test.target.dim) {
							double rmse_test2_this, mae_test2_this;//, rmse_test2_all_but5, mae_test2_all_but5;
							 _evaluate(pred_this, test.target, 1.0, rmse_test2_this, mae_test2_this, num_eval_cases, test.target.dim);
							//log->log("rmse_mcmc_test2_this", rmse_test2_this);
							//log->log("rmse_mcmc_test2_all", rmse_test2_all);
						}
						log->newLine();
					}
				} else if (task == TASK_CLASSIFICATION) {
					double acc_test_this, ll_test_this;
					 _evaluate_class(pred_this, test.target, 1.0, acc_test_this, ll_test_this, num_eval_cases);

					std::cout << "#Iter=" << std::setw(3) << i << "\tTrain=" << acc_train << "\tTest=" << acc_test_this << "\tTest(ll)=" << ll_test_this << std::endl;

					if (log != NULL) {
						log->log("acc_mcmc_this", acc_test_this);
						log->log("ll_mcmc_this", ll_test_this);

						if (num_eval_cases < test.target.dim) {
							double acc_test2_this,ll_test2_this;
							 _evaluate_class(pred_this, test.target, 1.0, acc_test2_this, ll_test2_this, num_eval_cases, test.target.dim);
							//log->log("acc_mcmc_test2_this", acc_test2_this);
							//log->log("acc_mcmc_test2_all", acc_test2_all);
						}
						log->newLine();
					}

				} else {
					throw "unknown task";
				}
				file_rmse.close();
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
