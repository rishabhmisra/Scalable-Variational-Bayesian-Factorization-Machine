// Author Avijit Saha & Rishabh Misra
// This file contains VB learning
//


#ifndef FM_LEARN_VB_H_
#define FM_LEARN_VB_H_

#include <sstream>
#include "../../util/matrix.h"
#include "fm_learn.h"
#include <math.h>
#include <fstream>
#include <omp.h>


struct t_term {
	double t;  // for T
	double q;
	double z;
};

bool bin_semaphore=true;

class fm_learn_vb : public fm_learn {
	public:
		virtual double evaluate(DataSubset& data) { return std::numeric_limits<double>::quiet_NaN(); }
	protected:
		virtual double predict_case(DataSubset& data) {
			throw "not supported for VB";
		}
	public:
		uint num_iter;
		uint num_eval_cases;

		// declare posterior parameters
		double alpha, sigma_0;
 		DVectorDoubleVB sigma_w;
		DMatrixDoubleVB sigma_v;

		// declare variational parameters
		double mu_0_dash, sigma_0_dash;

		DVectorDoubleVB mu_w_dash, sigma_w_dash;

		DMatrixDoubleVB mu_v_dash, sigma_v_dash;

		uint nan_alpha, nan_sigma_0, nan_sigma_w, nan_sigma_v, nan_mu_0_dash, nan_sigma_0_dash, nan_mu_w_dash, nan_sigma_w_dash, nan_mu_v_dash, nan_sigma_v_dash;
		uint inf_alpha, inf_sigma_0, inf_sigma_w, inf_sigma_v, inf_mu_0_dash, inf_sigma_0_dash, inf_mu_w_dash, inf_sigma_w_dash, inf_mu_v_dash, inf_sigma_v_dash;

	protected:
		DVector<double> cache_for_group_values;
		sparse_row<DATA_FLOAT> empty_data_row; // this is a dummy row for attributes that do not exist in the training data (but in test data)

		DVector<double> pred_this;

		e_q_term* cache;
		e_q_term* cache_test;
		t_term* cache_t;

		DVector<relation_cache*> rel_cache;

		virtual void _learn(DataSubset& train, DataSubset& test) {};


		/**
			This function predicts all datasets mentioned in main_data.
			It stores the prediction in the e-term.
		*/
		void predict_data_and_write_to_eterms(DVector<DataSubset*>& main_data, DVector<e_q_term*>& main_cache) {

			assert(main_data.dim == main_cache.dim);
			if (main_data.dim == 0) {
				std::cout<<"dimension is = 0"<<std::endl;
				return ;
			}

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
				double* v = mu_v_dash.value[f];

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
				double* v = mu_v_dash.value[f];

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
						double& w_i = mu_w_dash(row_index);

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
						m_cache[c].e += mu_0_dash;
					}
					m_cache[c].q = 0.0;
				}
			}
		}

// calculate T terms terms and store in q for all the training data points
//
		void predict_t_and_write_to_qterms(DataSubset* main_data, t_term* main_cache) {

 			t_term* m_cache=main_cache;
			DataSubset* m_data = main_data;
			// do this using only the transpose copy of the training data:
			for (uint i = 0; i < m_data->num_cases; i++) {
				m_cache[i].q = 0.0;
				m_cache[i].z = 0.0;
				m_cache[i].t = 0.0;
			}

			// (1) do the 1/2 sum_f (sum_i v_if x_i)^2 and store it in the e/y-term
			// (1.1) e_j = 1/2 sum_f (q_jf+ sum_R q^R_jf)^2
			// (1.2) y^R_j = 1/2 sum_f q^R_jf^2
			// Complexity: O(N_z(X^M) + \sum_{B} N_z(X^B) + n*|B| + \sum_B n^B) = O(\mathcal{C})
			for (int f = 0; f < fm->num_factor; f++) {
				double* v = mu_v_dash.value[f];
				double* v_sigma = sigma_v_dash.value[f];
				// calculate cache[i].q = sum_i v_if x_i (== q_f-term)
				// Complexity: O(N_z(X^M))
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
					double& v_if_sigma = v_sigma[row_index];
					for (uint i_fd = 0; i_fd < feature_data->size; i_fd++) {
						uint& train_case_index = feature_data->data[i_fd].id;
						FM_FLOAT& x_li = feature_data->data[i_fd].value;
						m_cache[train_case_index].q += v_if * x_li * v_if * x_li;
						m_cache[train_case_index].z += v_if_sigma * x_li * x_li;
					}
				}


				for (uint c = 0; c < m_data->num_cases; c++) {
					double q_all = m_cache[c].q;
					double z_all = m_cache[c].z;
					m_cache[c].t += (0.5 * z_all * z_all + z_all * q_all);
					m_cache[c].q = 0.0;
					m_cache[c].z = 0.0;
				}
			}

			// (2) do -1/2 sum_f (sum_i v_if^2 x_i^2) and store it in the q-term
			for (int f = 0; f < fm->num_factor; f++) {
				double* v = mu_v_dash.value[f];
				double* v_sigma = sigma_v_dash.value[f];

				// sum up the q^S_f terms in the main-q-cache: 0.5*sum_i (v_if x_i)^2 (== q^S_f-term)
				// Complexity: O(N_z(X^M))
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
					double& v_if_sigma = v_sigma[row_index];
					for (uint i_fd = 0; i_fd < feature_data->size; i_fd++) {
						uint& train_case_index = feature_data->data[i_fd].id;
						FM_FLOAT& x_li = feature_data->data[i_fd].value;
						m_cache[train_case_index].q -= (v_if * v_if * x_li * x_li * x_li * x_li * v_if_sigma +
							0.5 * x_li * x_li * x_li * x_li * v_if_sigma * v_if_sigma);
					}
				}
			}

			// (3) add the w's to the q-term
			if (fm->k1) {
				m_data->data_t->begin();
				uint row_index;
				sparse_row<DATA_FLOAT>* feature_data;
				for (uint i = 0; i < m_data->data_t->getNumRows(); i++) {
					{
						row_index = m_data->data_t->getRowIndex();
						feature_data = &(m_data->data_t->getRow());
						m_data->data_t->next();
					}
					double& w_i = sigma_w_dash(row_index);
					for (uint i_fd = 0; i_fd < feature_data->size; i_fd++) {
						uint& train_case_index = feature_data->data[i_fd].id;
						FM_FLOAT& x_li = feature_data->data[i_fd].value;
						m_cache[train_case_index].q += w_i * x_li * x_li;
					}
				}
			}
			// (3) merge all the values for getting the prediction:

			for (uint c = 0; c < m_data->num_cases; c++) {
				double q_all = m_cache[c].q;
				m_cache[c].t = m_cache[c].t + q_all;
				if (fm->k0) {
					m_cache[c].t += sigma_0_dash;
				}
				m_cache[c].q = 0.0;
			}
		}


	public:




	public:
		virtual void predict(DataSubset& data, DVector<double>& out) {

			/*
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
			*/
		}

	protected:



		void add_main_q(DataSubset& train, uint f) {
			// add the q(f)-terms to the main relation q-cache (using only the transpose data)
			double* v = mu_v_dash.value[f];
			double* v_sigma = sigma_v_dash.value[f];

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
					double& v_if_sigma = v_sigma[row_index];
					for (uint i_fd = 0; i_fd < feature_data->size; i_fd++) {
						uint& train_case_index = feature_data->data[i_fd].id;
						FM_FLOAT& x_li = feature_data->data[i_fd].value;
						cache[train_case_index].q += v_if * x_li;
						cache_t[train_case_index].q += v_if_sigma * x_li * x_li;
						cache_t[train_case_index].z += v_if * v_if * x_li * x_li;
					}

				}
			}
		}

		void update_all(DataSubset& train) {
		// update w_0's parameters
			if (fm->k0) {
				update_w0(train);
			}
		// update w's parameters
			//std::cout<<"check in fm_learn_vb update5"<<std::endl;
			if (fm->k1) {
				//train.data_t->begin();
				uint row_index;
				sparse_row<DATA_FLOAT>* feature_data;
				//#pragma omp parallel for
				for (uint i = 0; i < train.data_t->getNumRows(); i++) {
					{
						//row_index = train.data_t->getRowIndex();
						feature_data = &(train.data_t->getRow(i));
						row_index = i;
						//feature_data = &(train.data_t->data(i));
						//train.data_t->next();
					}
					uint g = meta->attr_group(row_index);
					update_w(mu_w_dash(row_index), sigma_w_dash(row_index), sigma_w(g), *feature_data);
				}
			}
			//std::cout<<"check in fm_learn_vb update6"<<std::endl;
		// update v's parameters
			if (fm->num_attribute>0) {
				for (int f = 0; f < fm->num_factor; f++) {
					for (uint c = 0; c < train.num_cases; c++) {
						cache[c].q = 0.0;
						cache_t[c].q = 0.0;
						cache_t[c].z = 0.0;
					}

					//std::cout<<"check in fm_learn_vb update7"<<std::endl;
					add_main_q(train, f);
					//std::cout<<"check in fm_learn_vb update7.1"<<std::endl;
					double* v = mu_v_dash.value[f];
					double* v1 = sigma_v_dash.value[f];

					//train.data_t->begin();
					uint row_index;
					sparse_row<DATA_FLOAT>* feature_data;
					//#pragma omp parallel for
					for (uint i = 0; i < train.data_t->getNumRows(); i++) {
						{
							//row_index = train.data_t->getRowIndex();
							feature_data = &(train.data_t->getRow(i));
							row_index = i;
							//feature_data = &(train.data_t->data(i));
							//train.data_t->next();
						}
						uint g = meta->attr_group(row_index);
						update_v(f, v[row_index], v1[row_index], sigma_v(g,f), *feature_data);
						//std::cout<<"check in fm_learn_vb update8"<<std::endl;
					}
				}
			}

			//update hyperparameters

			//update alpha

			{
			double alpha_temp=0.0;
			double alpha_old=0.0;
			//predict_t_and_write_to_qterms(&train, cache_t);
			for (uint i = 0; i < train.num_cases; i++) {
				alpha_temp += cache[i].e * cache[i].e + cache_t[i].t;
			}
			alpha_old=alpha;
			alpha=(double) train.num_cases/ alpha_temp;
			// check for out of bounds values
			if (std::isnan(alpha)) {
				nan_alpha++;
				alpha=alpha_old;
				assert(! std::isnan(alpha_old));
				assert(! std::isnan(alpha));
				return;
			}
			if (std::isinf(alpha)) {
				inf_alpha++;
				alpha=alpha_old;
				assert(! std::isinf(alpha_old));
				assert(! std::isinf(alpha));
				return;
			}
			}

			//update sigma_0
			sigma_0=1.0/(mu_0_dash * mu_0_dash+ sigma_0_dash);
			//update sigma_w
			DVector<double>& w_sigma_temp = cache_for_group_values;
			w_sigma_temp.init(0.0);
			for (uint i=0; i<fm->num_attribute;i++) {
				uint g = meta->attr_group(i);
				w_sigma_temp(g)+=mu_w_dash.value[i] * mu_w_dash.value[i] + sigma_w_dash.value[i];
			}
			for (uint i=0; i< meta->num_attr_groups; i++) {
				sigma_w(i)=(double) meta->num_attr_per_group(i)/w_sigma_temp(i);
			}

			//update sigma_v
			DVector<double>& v_sigma_temp = cache_for_group_values;
			for (int f=0;f<fm->num_factor;f++) {
				v_sigma_temp.init(0.0);
				double* v=mu_v_dash.value[f];
				double* v1=sigma_v_dash.value[f];
				for (uint i=0; i<fm->num_attribute;i++) {
					uint g=meta->attr_group(i);
					v_sigma_temp(g)+=v[i] * v[i] + v1[i];
				}
				for (uint g=0; g<meta->num_attr_groups; g++) {
					sigma_v(g,f)=(double) meta->num_attr_per_group(g)/v_sigma_temp(g);
				}
			}

			free_energy(train);
		}


		void update_w0(DataSubset& train) {
			//std::cout<< train.num_cases;
			double sigma_old;
			sigma_old=sigma_0_dash;
			sigma_0_dash=1.0/(sigma_0 + train.num_cases * alpha);
			double w0_temp=0.0;
			double mu_old;
			mu_old=mu_0_dash;
			for (uint i=0; i<train.num_cases;i++) {
				w0_temp+=cache[i].e + mu_0_dash;
			}
			mu_0_dash=sigma_0_dash * alpha * w0_temp;

			for (uint i=0; i<train.num_cases; i++) {
				cache[i].e=cache[i].e + (mu_old-mu_0_dash);
				cache_t[i].t=cache_t[i].t + (sigma_0_dash-sigma_old);
			}
			//std::cout<<"mu_0_dash "<<mu_0_dash<<std::endl;
			//std::cout<<"sigma_0 "<<sigma_0<<std::endl;
			//std::cout<<"alpha "<<alpha<<std::endl;
			//std::cout<<"sigma_0_dash "<<sigma_0_dash<<std::endl;
 		}

		void update_w (double& mu, double& sigma, double& sigma_w, sparse_row<DATA_FLOAT>& feature_data) {
			double w_sigma_sqr = 0;
			double w_mean = 0;
			double mu_old=0.0, sigma_old=0.0;
			mu_old=mu;
			sigma_old=sigma;
			//std::cout<<feature_data.size;
			for (uint i_fd = 0; i_fd < feature_data.size; i_fd++) {
				uint& train_case_index = feature_data.data[i_fd].id;
				FM_FLOAT x_li = feature_data.data[i_fd].value;
				w_mean += x_li * (cache[train_case_index].e + x_li * mu);
				w_sigma_sqr += x_li * x_li;
			}
			sigma = (double) 1.0 / (sigma_w + alpha * w_sigma_sqr);
			mu =  sigma * alpha * w_mean;

			// update w:

			if (std::isnan(sigma) || std::isinf(sigma)) {
				//mu = 0.0;
				nan_sigma_w_dash+=1;
				sigma=sigma_old;
			}

			// check for out of bounds values
			if (std::isnan(mu)) {
				nan_mu_w_dash++;
				mu = mu_old;
				assert(! std::isnan(mu_old));
				assert(! std::isnan(mu));
				return;
			}
			if (std::isinf(mu)) {
				inf_mu_w_dash++;
				mu = mu_old;
				assert(! std::isinf(mu_old));
				assert(! std::isinf(mu));
				return;
			}
			// update error:
			for (uint i_fd = 0; i_fd < feature_data.size; i_fd++) {
				uint& train_case_index = feature_data.data[i_fd].id;
				FM_FLOAT& x_li = feature_data.data[i_fd].value;
				double h = x_li;
				cache[train_case_index].e += h * (mu_old - mu);
				cache_t[train_case_index].t += h * h * (sigma - sigma_old);
			}
		}

		// Find the optimal value for the 2-way interaction parameter v
		void update_v(int& f, double& mu, double& sigma, double& sigma_v_g, sparse_row<DATA_FLOAT>& feature_data) {
			double v_sigma_sqr = 0;
			double v_mean = 0;

			double mu_old,sigma_old;
			mu_old=mu;
			sigma_old=sigma;

			// v_sigma_sqr = \sum h^2 (always)
			// v_mean = \sum h*e (for non_internlock_interactions)
			for (uint i_fd = 0; i_fd < feature_data.size; i_fd++) {
				uint& train_case_index = feature_data.data[i_fd].id;
				FM_FLOAT& x_li = feature_data.data[i_fd].value;
				e_q_term* cache_li = &(cache[train_case_index]);
				t_term* cache_li_t = &(cache_t[train_case_index]);
				double h = cache_li->q - x_li * mu;
				double h1 = cache_li_t->q - x_li * x_li * sigma;
				v_mean += x_li * h * (cache_li->e + x_li * mu * h);  //check this line
				v_sigma_sqr += x_li * x_li * h * h + x_li * x_li * h1;
			}
			sigma = (double) 1.0 / (sigma_v_g + alpha * v_sigma_sqr);
			mu =  sigma * alpha * v_mean;

			if (std::isnan(sigma) || std::isinf(sigma)) {
				sigma=sigma_old;
				//mu = 0.0;
				nan_sigma_v_dash+=1;
			}
			// check for out of bounds values
			if (std::isnan(mu)) {
				nan_mu_v_dash++;
				mu = mu_old;
				assert(! std::isnan(mu_old));
				assert(! std::isnan(mu));
				return;
			}
			if (std::isinf(mu)) {
				inf_mu_v_dash++;
				mu = mu_old;
				assert(! std::isinf(mu_old));
				assert(! std::isinf(mu));
				return;
			}


			// update error and q:
			for (uint i_fd = 0; i_fd < feature_data.size; i_fd++) {
				uint& train_case_index = feature_data.data[i_fd].id;
				FM_FLOAT& x_li = feature_data.data[i_fd].value;
				e_q_term* cache_li = &(cache[train_case_index]);
				t_term* cache_li_t = &(cache_t[train_case_index]);
				double h = x_li * ( cache_li->q - x_li * mu_old);
				double h1 = x_li * x_li * ( cache_li_t->q - x_li * x_li * sigma_old );
				double h2= x_li * x_li * (cache_li_t->z - x_li * x_li * mu_old * mu_old);
				while(bin_semaphore == false){}
				if(bin_semaphore == true)
				{
					bin_semaphore = false;
					cache_li->q += x_li * (mu - mu_old);
					cache_li_t->q += x_li * x_li * (sigma - sigma_old);
					cache_li_t->z += x_li * x_li * (mu * mu - mu_old * mu_old);
					cache_li->e += h* (mu_old - mu);
					cache_li_t->t += (h1 + h2) * (sigma - sigma_old);
					cache_li_t->t += h1 * (mu * mu - mu_old * mu_old);
					bin_semaphore = true;
				}
			}
		}

		void free_energy(DataSubset& train) {
			std::ofstream myfile;
			std::string file;
			std::stringstream convert; // stringstream used for the conversion

			convert << fm->k0<<fm->k1<<fm->num_factor;//add the value of Number to the characters in the stream

			std::string str = convert.str();
			file="free_energy_" + str + "_vb";
			myfile.open(file.c_str(), std::ios_base::app);
			double temp=0.0;
			double free_energy=0.0;
			//predict_t_and_write_to_qterms(&train, cache_t);
			for (uint i = 0; i < train.num_cases; i++) {
				temp += cache[i].e * cache[i].e + cache_t[i].t;
			}
			double temp1=2 * 3.14 * (1.0/alpha);
			free_energy += - 0.5 * alpha * temp - .5 * train.num_cases * std::log (temp1);
			free_energy += - 0.5 * sigma_0 * (mu_0_dash * mu_0_dash + sigma_0_dash) + 0.5 * std::log(sigma_0_dash * sigma_0) +.5;
			for (uint i=0;i<fm->num_attribute;i++) {
				uint g=meta->attr_group(i);
				free_energy += - 0.5 * sigma_w(g) * (mu_w_dash.value[i] * mu_w_dash.value[i] + sigma_w_dash.value[i]) +
					0.5 * std::log(sigma_w_dash.value[i] * sigma_w(g)) +.5;
			}
			for (int f=0;f<fm->num_factor;f++) {
				double* v=mu_v_dash.value[f];
				double* v1=sigma_v_dash.value[f];
				for (uint i=0;i<fm->num_attribute;i++) {
					uint g=meta->attr_group(i);
					free_energy += - 0.5 * sigma_v(g,f) * (v[i] * v[i] + v1[i]) + 0.5 * std::log(v1[i] * sigma_v(g,f))+.5;
				}
			}
			myfile<<-free_energy<<"\n";
			myfile.close();
			std::cout<<"free energy "<<free_energy<<std::endl;
		}


	public:
		virtual void init() {
			fm_learn::init();

			cache_for_group_values.setSize(meta->num_attr_groups);

			empty_data_row.size = 0;
			empty_data_row.data = NULL;
			// hyperprior parameters
			alpha = 1.0;
			sigma_0= 1.0;
			mu_0_dash=0.0;
			sigma_0_dash=0.02;
			// set dimension
			sigma_w.setSize(meta->num_attr_groups);
			sigma_v.setSize(meta->num_attr_groups,fm->num_factor);
			mu_w_dash.setSize(fm->num_attribute);
			sigma_w_dash.setSize(fm->num_attribute);

			mu_v_dash.setSize(fm->num_factor,fm->num_attribute);
			sigma_v_dash.setSize(fm->num_factor,fm->num_attribute);

			// initialize
			sigma_w.init(1);
			sigma_v.init(1);
			mu_w_dash.init_normal(0,1);
			sigma_w_dash.init(.02);
			mu_v_dash.init_normal(0,1);
			sigma_v_dash.init(.02);

			if (log != NULL) {
				log->addField("alpha", std::numeric_limits<double>::quiet_NaN());
				if (task == TASK_REGRESSION) {
					log->addField("rmse_mcmc_this", std::numeric_limits<double>::quiet_NaN());
					log->addField("rmse_mcmc_all", std::numeric_limits<double>::quiet_NaN());

					//log->addField("rmse_mcmc_test2_this", std::numeric_limits<double>::quiet_NaN());
					//log->addField("rmse_mcmc_test2_all", std::numeric_limits<double>::quiet_NaN());
				} else if (task == TASK_CLASSIFICATION) {
					log->addField("acc_mcmc_this", std::numeric_limits<double>::quiet_NaN());
					log->addField("acc_mcmc_all", std::numeric_limits<double>::quiet_NaN());
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
			pred_this.setSize(test.num_cases);
			pred_this.init(0.0);
			// init caches data structure
			MemoryLog::getInstance().logNew("e_q_term", sizeof(e_q_term), train.num_cases);
			cache = new e_q_term[train.num_cases];
			MemoryLog::getInstance().logNew("e_q_term", sizeof(e_q_term), test.num_cases);
			cache_test = new e_q_term[test.num_cases];
			MemoryLog::getInstance().logNew("t_term", sizeof(t_term), train.num_cases);
			cache_t = new t_term[train.num_cases];

			rel_cache.setSize(train.relation.dim);
			for (uint r = 0; r < train.relation.dim; r++) {
				MemoryLog::getInstance().logNew("relation_cache", sizeof(relation_cache), train.relation(r).data->num_cases);
				rel_cache(r) = new relation_cache[train.relation(r).data->num_cases];
				for (uint c = 0; c < train.relation(r).data->num_cases; c++) {
					rel_cache(r)[c].wnum = 0;
				}
			}

			// calculate #^R
			for (uint r = 0; r < train.relation.dim; r++) {
				for (uint c = 0; c < train.relation(r).data_row_to_relation_row.dim; c++) {
					rel_cache(r)[train.relation(r).data_row_to_relation_row(c)].wnum += 1.0;
				}
			}
			std::cout<<"in learn of fm_learn_vb"<<std::endl;
			_learn(train, test);

			// free data structures
			for (uint i = 0; i < train.relation.dim; i++) {
				MemoryLog::getInstance().logFree("relation_cache", sizeof(relation_cache), train.relation(i).data->num_cases);
				delete[] rel_cache(i);
			}
			MemoryLog::getInstance().logFree("e_q_term", sizeof(e_q_term), test.num_cases);
			delete[] cache_test;
			MemoryLog::getInstance().logFree("e_q_term", sizeof(e_q_term), train.num_cases);
			delete[] cache;
			MemoryLog::getInstance().logFree("t_term", sizeof(t_term), train.num_cases);
			delete[] cache_t;
		}


		virtual void debug() {
			fm_learn::debug();
			std::cout << "num_eval_cases=" << num_eval_cases << std::endl;
		}
};

#endif /*FM_LEARN_MCMC_H_*/
