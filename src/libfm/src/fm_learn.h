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
// fm_learn.h: Generic learning method for factorization machines

#ifndef FM_LEARN_H_
#define FM_LEARN_H_

#include <cmath>
#include "Data.h"
#include "../../fm_core/fm_model.h"
#include "../../util/rlog.h"
#include "../../util/util.h"
#include<map>
#include<set>
#include<algorithm>
#include<vector>
#define revtr(c,it) for(typeof(c.rbegin()) it=c.rbegin();it!=c.rend();it++)
#define tr(c,it) for(typeof(c.begin()) it=c.begin();it!=c.end();it++)
using namespace std;

class fm_learn {
	protected:
		DVector<double> sum, sum_sqr;
		DMatrix<double> pred_q_term;

		// this function can be overwritten (e.g. for MCMC)
		virtual double predict_case(DataSubset& data) {
			return fm->predict(data.data->getRow());
		}
		//-------------------------------------------------------------------
		virtual double predict_case(Data& data, uint c) {
			return fm->predict(data.data->getRow(c));
		}
		//-------------------------------------------------------------------

	public:
		DataMetaInfo* meta;
		fm_model* fm;
		double min_target;
		double max_target;

		int task; // 0=regression, 1=classification
//----------------------------------------------------------
		map< uint, map< uint,int > > test_user_item_rating;
		map< uint, set< pair<double,uint> > > test_user_prediction_item;
		DVector< sparse_entry<uint> > test_case_user_item;
		map< uint,uint > count_positive_feedback;
//----------------------------------------------------------

		const static int TASK_REGRESSION = 0;
		const static int TASK_CLASSIFICATION = 1;
		const static int TASK_REGRESSION_P = 2;


		DataSubset* validation;


		RLog* log;

		fm_learn() { log = NULL; task = 0; meta = NULL;}


		virtual void init() {
			if (log != NULL) {
				if (task == TASK_REGRESSION) {
					log->addField("rmse", std::numeric_limits<double>::quiet_NaN());
					log->addField("mae", std::numeric_limits<double>::quiet_NaN());
				} else if (task == TASK_CLASSIFICATION) {
					log->addField("accuracy", std::numeric_limits<double>::quiet_NaN());
				} else if (task == TASK_REGRESSION_P) {
					log->addField("accuracy", std::numeric_limits<double>::quiet_NaN());
				} else {
					throw "unknown task";
				}
				log->addField("time_pred", std::numeric_limits<double>::quiet_NaN());
				log->addField("time_learn", std::numeric_limits<double>::quiet_NaN());
				log->addField("time_learn2", std::numeric_limits<double>::quiet_NaN());
				log->addField("time_learn4", std::numeric_limits<double>::quiet_NaN());
			}
			std::cout<<"in fm_learn_init"<<std::endl;
			sum.setSize(fm->num_factor);
			sum_sqr.setSize(fm->num_factor);
			pred_q_term.setSize(fm->num_factor, meta->num_relations + 1);
		}

		virtual double evaluate(DataSubset& data) {
			assert(data.data != NULL);
			if (task == TASK_REGRESSION) {
				return evaluate_regression(data);
			} else if (task == TASK_REGRESSION_P) {
				return evaluate_regression(data);
			} else if (task == TASK_CLASSIFICATION) {
				std::cout<<"before test\n";
				return evaluate_classification_map(data);
			} else {
				throw "unknown task";
			}
		}

	public:
		virtual void learn(DataSubset& train, DataSubset& test) {
			if(task == TASK_CLASSIFICATION)
			{
				std::cout<<"in fm learn\n";
				if (task == TASK_CLASSIFICATION){
	                test_case_user_item.setSize(test.data->getNumRows());
	                ifstream test("/home/hduser/avijit/rishabh/data/webscopetestfinal_libfm1");
	                uint test_row = 0;
	                while (!test.eof()) {
	                    uint item_id,user_id,one;
	                    int rating;
	                    char whitespace,colon;
	                    std::string line;
	                    std::getline(test, line);
	                    if (sscanf(line.c_str(), "%d%c%u%c%u%c%u%c%u", &rating, &whitespace, &user_id, &colon, &one, &whitespace, &item_id, &colon, &one) >=9) {
	                        item_id -= 2320895;
	                        test_case_user_item(test_row).id = user_id;
	                        test_case_user_item(test_row).value = item_id;
	                        test_user_item_rating[user_id][item_id] = rating;
	                        if(rating == 1)
	                        {
	                            if(count_positive_feedback.count(user_id)==0)
	                            {
	                                count_positive_feedback[user_id] = 1;
	                            }
	                            else
	                            {
	                                count_positive_feedback[user_id]++;
	                            }
	                        }
	                    }
	                }
	                test.close();
            	}
			}
		}

		virtual void predict(DataSubset& data, DVector<double>& out) = 0;

		virtual void debug() {
			std::cout << "task=" << task << std::endl;
			std::cout << "min_target=" << min_target << std::endl;
			std::cout << "max_target=" << max_target << std::endl;
		}

	protected:
		virtual double evaluate_classification(DataSubset& data) {
			int num_correct = 0;
			double eval_time = getusertime();
			for (data.data->begin(); !data.data->end(); data.data->next()) {
				double p = predict_case(data);
				//std::cout<<p<<"\t";
				if (((p >= 0) && (data.target(data.data->getRowIndex()) >= 0)) || ((p < 0) && (data.target(data.data->getRowIndex()) < 0))) {
					num_correct++;
				}
			}
			std::cout<<"\n";
			eval_time = (getusertime() - eval_time);
			// log the values
			if (log != NULL) {
				log->log("accuracy", (double) num_correct / (double) data.data->getNumRows());
				log->log("time_pred", eval_time);
			}

			return (double) num_correct / (double) data.data->getNumRows();
		}

		virtual double evaluate_classification_map(DataSubset& data) {
			double mapatk;
			uint k=5;
			uint num_cases = 0;
			uint num_eval_cases = data.data->getNumRows();

			for (data.data->begin(); !data.data->end(); data.data->next()) {
				double p = predict_case(data);
				test_user_prediction_item[test_case_user_item(num_cases).id].insert( pair<double,uint>(p,test_case_user_item(num_cases).value));
				num_cases++;
			}
			std::cout<<"\n";
			DVector<double> average_precision;
			average_precision.setSize(test_user_prediction_item.size());
			average_precision.init(0.0);
			uint user_index=0;
			tr(test_user_prediction_item,test_user)
			{
				uint user_id = test_user->first;
				uint item_index=0;
				double temp=0.0;
				revtr(test_user->second,traverse_pred)
				{
					if(item_index==k){
						break;
					}
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
					std::cout<<temp<<"\t";
                    average_precision(user_index) = temp/(double)count_positive_feedback[user_id];
                }
				user_index++;
			}
			mapatk = 0.0;
			for(uint i=0; i<average_precision.dim; i++)
			{
				mapatk += average_precision(i);
			}

			mapatk = mapatk/average_precision.dim;
			// log the values
			if (log != NULL) {
				log->log("map@k", mapatk);
				//log->log("time_pred", eval_time);
			}

			return mapatk;
		}

		virtual double evaluate_regression(DataSubset& data) {
			double rmse_sum_sqr = 0;
			double mae_sum_abs = 0;
			double eval_time = getusertime();
			for (data.data->begin(); !data.data->end(); data.data->next()) {
				double p = predict_case(data);
				p = std::min(max_target, p);
				p = std::max(min_target, p);
				double err = p - data.target(data.data->getRowIndex());
				rmse_sum_sqr += err*err;
				mae_sum_abs += std::abs((double)err);
			}
			eval_time = (getusertime() - eval_time);
			// log the values
			if (log != NULL) {
				log->log("rmse", std::sqrt(rmse_sum_sqr/data.data->getNumRows()));
				log->log("mae", mae_sum_abs/data.data->getNumRows());
				log->log("time_pred", eval_time);
			}

			return std::sqrt(rmse_sum_sqr/data.data->getNumRows());
		}

};

#endif /*FM_LEARN_H_*/
