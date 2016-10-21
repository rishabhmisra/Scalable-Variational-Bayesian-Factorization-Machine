// Author Rishabh Misra
// This file contains VB Online learning

#ifndef FM_LEARN_VB_ONLINE_SIMULTANEOUS_H_
#define FM_LEARN_VB_ONLINE_SIMULTANEOUS_H_

#include "fm_learn_vb_online.h"
#include<ctime>
#include<stdlib.h>
#include<string.h>
#include<math.h>
#include<sstream>
#include<iostream>
#include<algorithm>
#include<iterator>
#include<vector>
using namespace std;
class fm_learn_vb_online_simultaneous : public fm_learn_vb_online {
	protected:

		virtual void _learn(DataSubset& train, DataSubset& test)
		{

			uint num_complete_iter = 0;

			// make a collection of datasets that are predicted jointly
			int num_data = 2;
			int num_data_only_test = 1;

			DVector<DataSubset*> main_data_only_test(num_data_only_test);
			DVector<e_q_term*> main_cache_only_test(num_data_only_test);
			main_data_only_test(0) = &test;
			main_cache_only_test(0) = cache_test;

			std::cout<<"check in fm_learn_vb_online_simultaneous"<<std::endl;

			// open file to store test rmse
			std::ofstream file_rmse;
			std::string file;
			std::stringstream convert; // stringstream used for the conversion

			convert << fm->k0<<fm->k1<<fm->num_factor;//add the value of Number to the characters in the stream

			std::string str = convert.str();
			file="test_rmse_" + str + "_vb_online";
			file_rmse.open(file.c_str());
			file_rmse.close();
			//clear free energy file
			std::ofstream myfile;
			file="free_energy_" + str + "_vb_online";
			myfile.open(file.c_str());
			myfile.close();

			uint tmp;
			total_cases = train.num_cases;
			tmp = ceil((double)(train.num_cases)/(num_batch));
			size_except_last = tmp;
			uint* shuffle = new uint[train.num_cases];
			for(uint i=0;i<train.num_cases;i++)
			{
				shuffle[i] = i+1;
			}

			cout<<"check2\n";

			for (uint k = num_complete_iter; k < num_iter; k++) {		// iterations over data

				double iteration_time = getusertime();
				clock_t iteration_time3 = clock();
				double iteration_time4 = getusertime4();
				nan_alpha=0; nan_sigma_0=0; nan_sigma_w=0; nan_sigma_v=0; nan_mu_0_dash=0; nan_sigma_0_dash=0; nan_mu_w_dash=0;
				nan_sigma_w_dash=0; nan_mu_v_dash=0; nan_sigma_v_dash=0; nan_sigma_w=0; nan_sigma_v=0;

 				random_shuffle(shuffle,shuffle + train.num_cases);
 
                ofstream batch[num_batch];
                for(uint i=0;i<num_batch;i++)
                {
					std::ostringstream sstream;
					sstream << training_file <<"batch" << i+1;
					const char* file = (sstream.str()).c_str();
					batch[i].open(file, ios::out | ios::trunc);
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

				for(uint i=0;i<num_batch;i++)
				{
					batch[i].close();
				}

				for(uint j = 1; j <= num_batch; j++)
				{
					std::ostringstream sstream;
					sstream << training_file <<"batch" << j;
					std::string file = sstream.str();
					DataSubset train1(0,true,true); // no transpose data for sgd, sgda
					train1.load(file,fm->num_attribute);

					MemoryLog::getInstance().logNew("e_q_term", sizeof(e_q_term), train1.num_cases);
                    cache = new e_q_term[train1.num_cases];
                    MemoryLog::getInstance().logNew("t_term", sizeof(t_term), train1.num_cases);
                    cache_t = new t_term[train1.num_cases];

					DVector<DataSubset*> main_data(1);       // to store train data
                    DVector<e_q_term*> main_cache(1);// to store intermediate term for calc. of Rij
		            main_data(0) = &train1;
		            main_cache(0) = cache;
					predict_data_and_write_to_eterms(main_data, main_cache);
					predict_t_and_write_to_qterms(&train1, cache_t);
					if (task == TASK_REGRESSION) 
					{
            	    	// remove the target from each prediction, because: e(c) := \hat{y}(c) - target(c)
                    	for (uint c = 0; c<train1.num_cases; c++) {
                    		cache[c].e = train1.target(c) - cache[c].e;
                    	}

                    } else if (task == TASK_CLASSIFICATION) {
	                	// for Classification: remove from e not the target but a sampled value from a truncated normal^M
    	            	// for initializing, they are not sampled but initialized with meaningful values:^M
        	        	// -1 for the negative class and +1 for the positive class (actually these are the values that are already in the target an    d thus, we can do the same as for regression; but note that other initialization strategies would need other techniques here:
						for (uint c = 0; c < train1.num_cases; c++) {
			           		cache[c].e = train1.target(c) - cache[c].e;
		    	    	}

                    } else {
    		        	throw "unknown task";
	        	    }

                    update_all(train1,train.num_cases);

					if(j==num_batch || j==1)
					{
						free_energy(train1);
					}

					delete[] ((LargeSparseMatrixMemory<float>*) train1.data)->data.value[0].data;
					delete[] ((LargeSparseMatrixMemory<float>*) train1.data)->data.value;
					delete[] ((LargeSparseMatrixMemory<float>*) train1.data_t)->data.value[0].data;
					delete[] ((LargeSparseMatrixMemory<float>*) train1.data_t)->data.value;
					MemoryLog::getInstance().logFree("e_q_term", sizeof(e_q_term), train1.num_cases);
                    delete[] cache;
                    MemoryLog::getInstance().logFree("t_term", sizeof(t_term), train1.num_cases);
                    delete[] cache_t;

                }

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

				predict_data_and_write_to_eterms(main_data_only_test, main_cache_only_test);
				// (prediction of train is not necessary but it increases numerical stability)


				std::ofstream file_rmse;
				std::string file;
				std::stringstream convert; // stringstream used for the conversion

				convert << fm->k0<<fm->k1<<fm->num_factor;//add the value of Number to the characters in the stream

				std::string str = convert.str();
				file="test_rmse_" + str + "_vb_online";
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

				} else if (task == TASK_CLASSIFICATION) {
					// evaluate test and store it
					for (uint c = 0; c < test.num_cases; c++) {
						double p = cache_test[c].e;
						p = cdf_gaussian(p);
						pred_this(c) = p;
					}

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
					std::cout << "#Iter=" << std::setw(3) << k << "\tTest=" << rmse_test_this << std::endl;

					if (log != NULL) {
						log->log("rmse_mcmc_this", rmse_test_this);

						if (num_eval_cases < test.target.dim) {
							double rmse_test2_this, mae_test2_this;
							 _evaluate(pred_this, test.target, 1.0, rmse_test2_this, mae_test2_this, num_eval_cases, test.target.dim);
						}
						log->newLine();
					}
				} else if (task == TASK_CLASSIFICATION) {
					double acc_test_this;
					// _evaluate_class(pred_this, test.target, 1.0, acc_test_this, ll_test_this, num_eval_cases);
					double mapatk;
					uint kk=5;
					_evaluate_class_map(pred_this, test.target, 1.0, acc_test_this, mapatk, kk, num_eval_cases);
					file_rmse<<mapatk<<"\n";
					std::cout << "#Iter=" << std::setw(3) << k << "\tTest=" << acc_test_this << "\tMAP@"<<kk<<"= " << mapatk << std::endl;

					if (log != NULL) {
						log->log("acc_mcmc_this", acc_test_this);
						//log->log("ll_mcmc_this", ll_test_this);

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
				for(uint q=0;q<num_batch;q++)
                                 {
                                     std::ostringstream sstream;
                                     sstream << training_file.c_str() <<"batch" << q+1;
                                     const char* file = (sstream.str()).c_str();
                                     remove(file);
                                 }
			}
			delete [] shuffle;
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


		void _evaluate_class_map(DVector<double>& pred, DVector<DATA_FLOAT>& target, double normalizer, double& accuracy, double& mapatk,uint k, uint& num_eval_cases)
		{
			uint _accuracy = 0;
			uint num_cases = 0;
			for (uint c = 0; c < num_eval_cases; c++) {
				double p = pred(c) * normalizer;
				if (((p >= 0.5) && (target(c) > 0.0)) || ((p < 0.5) && (target(c) < 0.0))) {
					_accuracy++;
				}
				test_user_prediction_item[test_case_user_item(c).id].insert( pair<double,uint>(p,test_case_user_item(c).value));
				num_cases++;
			}

			DVector<double> average_precision;
			average_precision.setSize(test_user_prediction_item.size());
			average_precision.init(0.0);
			uint user_index=0;
			tr(test_user_prediction_item,test_user)
			{
				uint user_id = test_user->first;
				int item_index=0;
				double temp=0.0;
				revtr(test_user->second,traverse_pred)
				{
					if(item_index==k){
						break;
					}
					//double predic = traverse_pred->first;
					uint item_id = traverse_pred->second;
					if(test_user_item_rating[user_id][item_id]==1)
					{
						average_precision(user_index) = ((average_precision(user_index)*(item_index))+1.0)/(item_index+1);
						temp+=average_precision(user_index);
					}
					item_index++;
				}
				if(count_positive_feedback[user_id]!=0)
				{
					average_precision(user_index) = temp/count_positive_feedback[user_id];
				}
				user_index++;
			}
			mapatk = 0.0;
			for(uint i=0; i<average_precision.dim; i++)
			{
				mapatk += average_precision(i);
			}

			mapatk = mapatk/average_precision.dim;

			accuracy = (double) _accuracy / num_cases;
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

#endif
