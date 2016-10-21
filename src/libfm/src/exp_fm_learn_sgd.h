// Author Avijit Saha
// This file contains exponential family stochastic gradient descent
//


#ifndef EXP_FM_LEARN_SGD_H_
#define EXP_FM_LEARN_SGD_H_

#include <sstream>
#include "fm_learn_mcmc.h"


class exp_fm_learn_sgd : public fm_learn {
	public:
		virtual double evaluate(DataSubset& data) { return std::numeric_limits<double>::quiet_NaN(); }
	protected:
		virtual double predict_case(DataSubset& data) {
			throw "not supported for MCMC and ALS";
		}
	public:
		uint num_iter;
		uint num_eval_cases;

		uint sigma;

		uint nan_cntr_v, nan_cntr_w, nan_cntr_w0;
		uint inf_cntr_v, inf_cntr_w, inf_cntr_w0;
		double learn_rate;
		DVector<double> learn_rates;

	protected:
		sparse_row<DATA_FLOAT> empty_data_row;
		DVector<double> pred_sum;
		DVector<double> pred_this;

		e_q_term* cache;
		e_q_term* cache_test;

		virtual void _learn(DataSubset& train, DataSubset& test) {};


		/**
			This function predicts all datasets mentioned in main_data.
			It stores the prediction in the e-term.
		*/
		void predict_data_and_write_to_eterms(DVector<DataSubset*>& main_data, DVector<e_q_term*>& main_cache) {

			assert(main_data.dim == main_cache.dim);
			if (main_data.dim == 0) { return ; }

			// do this using only the transpose copy of the training data:
			for (uint ds = 0; ds < main_cache.dim; ds++) {
				e_q_term* m_cache = main_cache(ds);
				DataSubset* m_data = main_data(ds);
				for (uint i = 0; i < m_data->num_cases; i++) {
					m_cache[i].e = 0.0;
					m_cache[i].q = 0.0;
				}
			}


			// (1) do the 1/2 sum_f (sum_i v_if x_i)^2 and store it in the e/y-term
			// (1.1) e_j = 1/2 sum_f (q_jf+ sum_R q^R_jf)^2
			// (1.2) y^R_j = 1/2 sum_f q^R_jf^2
			// Complexity: O(N_z(X^M) + \sum_{B} N_z(X^B) + n*|B| + \sum_B n^B) = O(\mathcal{C})
			for (int f = 0; f < fm->num_factor; f++) {
				double* v = fm->v.value[f];

				// calculate cache[i].q = sum_i v_if x_i (== q_f-term)
				// Complexity: O(N_z(X^M))
				for (uint ds = 0; ds < main_cache.dim; ds++) {
					e_q_term* m_cache = main_cache(ds);
					DataSubset* m_data = main_data(ds);
					m_data->data_t->begin();
					uint row_index;
					sparse_row<DATA_FLOAT>* feature_data;
					for (uint i = 0; i < m_data->data_t->getNumRows(); i++) {
						{
							row_index = m_data->data_t->getRowIndex();
							feature_data = &(m_data->data_t->getRow());
							m_data->data_t->next();
						}
						double& v_if = v[row_index];

						for (uint i_fd = 0; i_fd < feature_data->size; i_fd++) {
							uint& train_case_index = feature_data->data[i_fd].id;
							FM_FLOAT& x_li = feature_data->data[i_fd].value;
							m_cache[train_case_index].q += v_if * x_li;
						}
					}
				}


				// add 0.5*q^2 to e and set q to zero.
				// O(n*|B|)
				for (uint ds = 0; ds < main_cache.dim; ds++) {
					e_q_term* m_cache = main_cache(ds);
					DataSubset* m_data = main_data(ds);
					for (uint c = 0; c < m_data->num_cases; c++) {
						double q_all = m_cache[c].q;
						m_cache[c].e += 0.5 * q_all*q_all;
						m_cache[c].q = 0.0;
					}
				}

			}

			// (2) do -1/2 sum_f (sum_i v_if^2 x_i^2) and store it in the q-term
			for (int f = 0; f < fm->num_factor; f++) {
				double* v = fm->v.value[f];

				// sum up the q^S_f terms in the main-q-cache: 0.5*sum_i (v_if x_i)^2 (== q^S_f-term)
				// Complexity: O(N_z(X^M))
				for (uint ds = 0; ds < main_cache.dim; ds++) {
					e_q_term* m_cache = main_cache(ds);
					DataSubset* m_data = main_data(ds);

					m_data->data_t->begin();
					uint row_index;
					sparse_row<DATA_FLOAT>* feature_data;
					for (uint i = 0; i < m_data->data_t->getNumRows(); i++) {
						{
							row_index = m_data->data_t->getRowIndex();
							feature_data = &(m_data->data_t->getRow());
							m_data->data_t->next();
						}
						double& v_if = v[row_index];

						for (uint i_fd = 0; i_fd < feature_data->size; i_fd++) {
							uint& train_case_index = feature_data->data[i_fd].id;
							FM_FLOAT& x_li = feature_data->data[i_fd].value;
							m_cache[train_case_index].q -= 0.5 * v_if * v_if * x_li * x_li;
						}
					}
				}
			}

			// (3) add the w's to the q-term
			if (fm->k1) {
				for (uint ds = 0; ds < main_cache.dim; ds++) {
					e_q_term* m_cache = main_cache(ds);
					DataSubset* m_data = main_data(ds);
					m_data->data_t->begin();
					uint row_index;
					sparse_row<DATA_FLOAT>* feature_data;
					for (uint i = 0; i < m_data->data_t->getNumRows(); i++) {
						{
							row_index = m_data->data_t->getRowIndex();
							feature_data = &(m_data->data_t->getRow());
							m_data->data_t->next();
						}
						double& w_i = fm->w(row_index);

						for (uint i_fd = 0; i_fd < feature_data->size; i_fd++) {
							uint& train_case_index = feature_data->data[i_fd].id;
							FM_FLOAT& x_li = feature_data->data[i_fd].value;
							m_cache[train_case_index].q += w_i * x_li;
						}
					}
				}
			}
			// (3) merge both for getting the prediction: w0+e(c)+q(c)
			for (uint ds = 0; ds < main_cache.dim; ds++) {
				e_q_term* m_cache = main_cache(ds);
				DataSubset* m_data = main_data(ds);

				for (uint c = 0; c < m_data->num_cases; c++) {
					double q_all = m_cache[c].q;
					m_cache[c].e = m_cache[c].e + q_all;
					if (fm->k0) {
						m_cache[c].e += fm->w0;
					}
					m_cache[c].q = 0.0;
				}
			}
			/*
			count_in_exp+=1;
			if (count_in_exp==1) {
				e_q_term* m_cache=main_cache(0);
				Data* m_data = main_data(0);
				std::cout<<"error values "<<std::endl;
				for (uint c = 0; c < m_data->num_cases; c++) {
					std::cout<<m_cache[c].e<<std::endl;

				}
			}
			*/
		}

	public:
		virtual void predict(DataSubset& data, DVector<double>& out) {
			assert(data.num_cases == out.dim);

			for (uint i = 0; i < out.dim; i++) {
				if (task == TASK_REGRESSION ) {
					out(i) = std::min(max_target, out(i));
					out(i) = std::max(min_target, out(i));
				} else if (task == TASK_CLASSIFICATION) {
					out(i) = std::min(1.0, out(i));
					out(i) = std::max(0.0, out(i));
				} else {
					throw "task not supported";
				}
			}
		}

	public:
		/*
		virtual void predict(Data& data, DVector<double>& out) {
			assert(data.num_cases == out.dim);
			if (do_sample) {
				assert(data.num_cases == pred_sum_all.dim);
				for (uint i = 0; i < out.dim; i++) {
					out(i) = pred_sum_all(i) / num_iter;
				}
			} else {
				assert(data.num_cases == pred_this.dim);
				for (uint i = 0; i < out.dim; i++) {
					out(i) = pred_this(i);
				}
			}
			for (uint i = 0; i < out.dim; i++) {
				if (task == TASK_REGRESSION ) {
					out(i) = std::min(max_target, out(i));
					out(i) = std::max(min_target, out(i));
				} else if (task == TASK_CLASSIFICATION) {
					out(i) = std::min(1.0, out(i));
					out(i) = std::max(0.0, out(i));
				} else {
					throw "task not supported";
				}
			}
		}*/
	protected:



		void add_main_q(DataSubset& train, uint f) {
			// add the q(f)-terms to the main relation q-cache (using only the transpose data)

			double* v = fm->v.value[f];


			{
				train.data_t->begin();
				uint row_index;
				sparse_row<DATA_FLOAT>* feature_data;
				for (uint i = 0; i < train.data_t->getNumRows(); i++) {
					{
						row_index = train.data_t->getRowIndex();
						feature_data = &(train.data_t->getRow());
						train.data_t->next();
					}
					double& v_if = v[row_index];
					for (uint i_fd = 0; i_fd < feature_data->size; i_fd++) {
						uint& train_case_index = feature_data->data[i_fd].id;
						FM_FLOAT& x_li = feature_data->data[i_fd].value;
						cache[train_case_index].q += v_if * x_li;
					}

				}
			}
		}



		void fm_SGD(DataSubset& train, const double& learn_rate) {
			//std::cout<<"learn rate"<<learn_rate<<std::endl;
			// Update w0
			if (fm->k0) {
				double& w0 = fm->w0;
				double w0_old = w0;
				double w0_sum = 0;
				for (uint i = 0; i < train.num_cases; i++) {
					w0_sum += cache[i].e;
				}
				std::cout<<"w0= "<<w0<<std::endl;
				w0 -= learn_rate * (w0_sum + fm->reg0 * w0)/float(train.num_cases);
				// check for out of bounds values
				std::cout<<"w0= "<<w0<<std::endl;
				if (std::isnan(w0)) {
					nan_cntr_w0++;
					w0 = w0_old;
					assert(! std::isnan(w0_old));
					assert(! std::isnan(w0));
					return;
				}
				if (std::isinf(w0)) {
					inf_cntr_w0++;
					w0 = w0_old;
					assert(! std::isinf(w0_old));
					assert(! std::isinf(w0));
					return;
				}
				// update error
				for (uint i = 0; i < train.num_cases; i++) {
					cache[i].e -= (w0_old - w0);
				}

			}

			// update w_j


			if (fm->k1) {
				uint count_how_many_variables_are_drawn = 0; // to make sure that non-existing ones in the train set are not missed...

				train.data_t->begin();
				uint row_index;
				sparse_row<DATA_FLOAT>* feature_data;
				for (uint i = 0; i < train.data_t->getNumRows(); i++) {
					{
						row_index = train.data_t->getRowIndex();
						feature_data = &(train.data_t->getRow());
						train.data_t->next();
						count_how_many_variables_are_drawn++;
					}
					update_w(train,fm->w(row_index), learn_rate, *feature_data);
				}
				/*
				uint draw_to = fm->num_attribute;
				for (uint i = train.data_t->getNumRows(); i < draw_to; i++) {
					row_index = i;
					feature_data = &(empty_data_row);
					update_w(train,fm->w(row_index), learn_rate,*feature_data);
					count_how_many_variables_are_drawn++;
				}
				*/
				assert(count_how_many_variables_are_drawn == fm->num_attribute);
			}
			// update v_jf


			for (int f = 0; f < fm->num_factor; f++) {
				uint count_how_many_variables_are_drawn = 0; // to make sure that non-existing ones in the train set are not missed...

				for (uint c = 0; c < train.num_cases; c++) {
					cache[c].q = 0.0;
				}

				add_main_q(train, f);

				double* v = fm->v.value[f];

				train.data_t->begin();
				uint row_index;
				sparse_row<DATA_FLOAT>* feature_data;
				for (uint i = 0; i < train.data_t->getNumRows(); i++) {
					{
						row_index = train.data_t->getRowIndex();
						feature_data = &(train.data_t->getRow());
						train.data_t->next();
						count_how_many_variables_are_drawn++;
					}
					update_v(train,v[row_index], learn_rate, *feature_data);
				}
				// draw v's of the main table for which there is no observation in the training data
				/*
				uint draw_to = fm->num_attribute;

				for (uint i = train.data_t->getNumRows(); i < draw_to; i++) {
					row_index = i;
					feature_data = &(empty_data_row);
					update_v(v[row_index], learn_rate, *feature_data);
					count_how_many_variables_are_drawn++;
				}
				*/
				//std::cout<<"count_how_many_variables_are_drawn "<<count_how_many_variables_are_drawn<<std::endl;
				//std::cout<<"attribute "<<fm->num_attribute<<std::endl;
				assert(count_how_many_variables_are_drawn == fm->num_attribute);
			}
		}


		// Find the optimal value for the 1-way interaction w
		void update_w(DataSubset& train,double& w, const double& learn_rate, sparse_row<DATA_FLOAT>& feature_data) {
			double w_sum = 0;
			double w_old = 0;
			for (uint i_fd = 0; i_fd < feature_data.size; i_fd++) {
				uint& train_case_index = feature_data.data[i_fd].id;
				FM_FLOAT x_li = feature_data.data[i_fd].value;
				w_sum += x_li * cache[train_case_index].e;
			}
			w_old = w;
			w -= learn_rate * (w_sum + fm->regw * w)/float(train.num_cases);

			// check for out of bounds values
			if (std::isnan(w)) {
				nan_cntr_w++;
				w = w_old;
				assert(! std::isnan(w_old));
				assert(! std::isnan(w));
				return;
			}
			if (std::isinf(w)) {
				inf_cntr_w++;
				w = w_old;
				assert(! std::isinf(w_old));
				assert(! std::isinf(w));
				return;
			}
			// update error:
			for (uint i_fd = 0; i_fd < feature_data.size; i_fd++) {
				uint& train_case_index = feature_data.data[i_fd].id;
				FM_FLOAT& x_li = feature_data.data[i_fd].value;
				double h = x_li;
				cache[train_case_index].e -= h * (w_old - w);
			}
		}


		// Find the optimal value for the 2-way interaction parameter v
		void update_v(DataSubset& train,double& v, const double& learn_rate, sparse_row<DATA_FLOAT>& feature_data) {
			double v_sum = 0;
			double v_old = 0;
			// v_sigma_sqr = \sum h^2 (always)
			// v_mean = \sum h*e (for non_internlock_interactions)
			for (uint i_fd = 0; i_fd < feature_data.size; i_fd++) {
				uint& train_case_index = feature_data.data[i_fd].id;
				FM_FLOAT& x_li = feature_data.data[i_fd].value;
				e_q_term* cache_li = &(cache[train_case_index]);
				double h = x_li * ( cache_li->q - x_li * v);
				v_sum += h * cache_li->e;
			}
			// update v:
			v_old = v;
			//std::cout<<v_sum<<std::endl;
			v -= learn_rate * (v_sum + fm->regv * v)/float(train.num_cases);
			//std::cout<<v<<std::endl;
			// check for out of bounds values
			if (std::isnan(v)) {
				nan_cntr_v++;
				v = v_old;
				assert(! std::isnan(v_old));
				assert(! std::isnan(v));
				return;
			}
			if (std::isinf(v)) {
				inf_cntr_v++;
				v = v_old;
				assert(! std::isinf(v_old));
				assert(! std::isinf(v));
				return;
			}

			// update error and q:
			for (uint i_fd = 0; i_fd < feature_data.size; i_fd++) {
				uint& train_case_index = feature_data.data[i_fd].id;
				FM_FLOAT& x_li = feature_data.data[i_fd].value;
				e_q_term* cache_li = &(cache[train_case_index]);
				double h = x_li * ( cache_li->q - x_li * v_old);
				cache_li->q -= x_li * (v_old - v);
				cache_li->e -= h * (v_old - v);
			}
		}



	public:
		virtual void init() {
			std::cout<<"in exp_fm_learn_sgd init"<<std::endl;
			fm_learn::init();
			learn_rates.setSize(3);
			//count_in_exp=0;
			empty_data_row.size = 0;
			empty_data_row.data = NULL;



			if (log != NULL) {
				log->addField("alpha", std::numeric_limits<double>::quiet_NaN());
				if (task == TASK_REGRESSION) {
					log->addField("rmse_mcmc_this", std::numeric_limits<double>::quiet_NaN());
					log->addField("rmse_mcmc_all", std::numeric_limits<double>::quiet_NaN());
					log->addField("rmse_mcmc_all_but5", std::numeric_limits<double>::quiet_NaN());

					//log->addField("rmse_mcmc_test2_this", std::numeric_limits<double>::quiet_NaN());
					//log->addField("rmse_mcmc_test2_all", std::numeric_limits<double>::quiet_NaN());
				} else if (task == TASK_CLASSIFICATION) {
					log->addField("acc_mcmc_this", std::numeric_limits<double>::quiet_NaN());
					log->addField("acc_mcmc_all", std::numeric_limits<double>::quiet_NaN());
					log->addField("acc_mcmc_all_but5", std::numeric_limits<double>::quiet_NaN());
					log->addField("ll_mcmc_this", std::numeric_limits<double>::quiet_NaN());
					log->addField("ll_mcmc_all", std::numeric_limits<double>::quiet_NaN());
					log->addField("ll_mcmc_all_but5", std::numeric_limits<double>::quiet_NaN());

					//log->addField("acc_mcmc_test2_this", std::numeric_limits<double>::quiet_NaN());
					//log->addField("acc_mcmc_test2_all", std::numeric_limits<double>::quiet_NaN());
				}

				std::ostringstream ss;
				for (uint g = 0; g < meta->num_attr_groups; g++) {
					ss.str(""); ss << "wmu[" << g << "]"; log->addField(ss.str(), std::numeric_limits<double>::quiet_NaN());
					ss.str(""); ss << "wlambda[" << g << "]"; log->addField(ss.str(), std::numeric_limits<double>::quiet_NaN());
					for (int f = 0; f < fm->num_factor; f++) {
						ss.str(""); ss << "vmu[" << g << "," << f << "]"; log->addField(ss.str(), std::numeric_limits<double>::quiet_NaN());
						ss.str(""); ss << "vlambda[" << g << "," << f << "]"; log->addField(ss.str(), std::numeric_limits<double>::quiet_NaN());
					}
				}
			}
		}


		virtual void learn(DataSubset& train, DataSubset& test) {
			std::cout<<"number of test instances "<<test.num_cases<<std::endl;
			pred_sum.setSize(test.num_cases);
			pred_this.setSize(test.num_cases);
			pred_sum.init(0.0);
			pred_this.init(0.0);

			// init caches data structure
			MemoryLog::getInstance().logNew("e_q_term", sizeof(e_q_term), train.num_cases);
			cache = new e_q_term[train.num_cases];
			MemoryLog::getInstance().logNew("e_q_term", sizeof(e_q_term), test.num_cases);
			cache_test = new e_q_term[test.num_cases];

			// calculate #^R
			std::cout << "learnrate=" << learn_rate << std::endl;
			std::cout << "learnrates=" << learn_rates(0) << "," << learn_rates(1) << "," << learn_rates(2) << std::endl;
			std::cout << "#iterations=" << num_iter << std::endl;
			std::cout << "k1=" << fm->k0 << std::endl;
			std::cout << "k2=" << fm->k1 << std::endl;
			if (train.relation.dim > 0) {
				throw "relations are not supported with SGD";
			}
			_learn(train, test);
			// free data structures
			MemoryLog::getInstance().logFree("e_q_term", sizeof(e_q_term), test.num_cases);
			delete[] cache_test;
			MemoryLog::getInstance().logFree("e_q_term", sizeof(e_q_term), train.num_cases);
			delete[] cache;
		}

		void debug() {
			std::cout << "num_iter=" << num_iter << std::endl;
			fm_learn::debug();
		}

};

#endif /*FM_LEARN_MCMC_H_*/
