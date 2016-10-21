// Author Rishabh Misra
// This file contains VB learning

#ifndef FM_LEARN_VB_ONLINE_H_
#define FM_LEARN_VB_ONLINE_H_

#include <sstream>
#include "../../util/matrix.h"
#include "fm_learn.h"
#include <math.h>
#include <fstream>
#include <omp.h>


class fm_learn_vb_online : public fm_learn {
	public:
		virtual double evaluate(DataSubset& data) { return std::numeric_limits<double>::quiet_NaN(); }
	protected:
		virtual double predict_case(DataSubset& data) {
			throw "not supported for VB";
		}
	public:
		DVector<uint> col_count;
		uint num_iter;
		uint num_eval_cases;
		uint total_cases,num_batch;
		uint size_except_last;
		std::string training_file,testing_file;
		double new_w0,lamda;
		DVector<double> new_wj,new_vj;
		DVector<uint> t_wj,t_vj;
		uint t_w0,t0_w0,t0_wj,t0_vj;
		DVectorDoubleVB natural_mu_w_dash,natural_sigma_w_dash;
        DMatrixDoubleVB natural_mu_v_dash,natural_sigma_v_dash;
		double natural_mu_0_dash,natural_sigma_0_dash;

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
		virtual void predict(DataSubset& data, DVector<double>& out) {}

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

		void update_all(DataSubset& train,uint _size) {
		// update w_0's parameters
			if (fm->k0) {
				update_w0(train,_size);
			}
		// update w's parameters
			if (fm->k1) {
				uint row_index;
				sparse_row<DATA_FLOAT>* feature_data;
				for (uint i = 0; i < train.data_t->getNumRows(); i++) {
					{
						feature_data = &(train.data_t->getRow(i));
						row_index = i;
						if((*feature_data).size==0)
						continue;
					}
					uint g = meta->attr_group(row_index);
					update_w(mu_w_dash(row_index), sigma_w_dash(row_index), sigma_w(g), *feature_data,_size,i,train.num_cases);
				}
			}
		// update v's parameters
			if (fm->num_attribute>0) {
				for (int f = 0; f < fm->num_factor; f++) {
					
					for (uint c = 0; c < train.num_cases; c++) {
						cache[c].q = 0.0;
						cache_t[c].q = 0.0;
						cache_t[c].z = 0.0;
					}
					add_main_q(train, f);

					double* v = mu_v_dash.value[f];
					double* v1 = sigma_v_dash.value[f];

					uint row_index;
					sparse_row<DATA_FLOAT>* feature_data;
					for (uint i = 0; i < train.data_t->getNumRows(); i++) {
						{
							feature_data = &(train.data_t->getRow(i));
							row_index = i;
							if((*feature_data).size==0)
							continue;
						}
						uint g = meta->attr_group(row_index);
						update_v(f, v[row_index], v1[row_index], sigma_v(g,f), *feature_data,_size,row_index,train.num_cases);
						if ( f==0 )
						{
							t_vj(row_index)+= (*feature_data).size;
						}
					}
				}
				for (uint i = 0; i < train.data_t->getNumRows(); i++) {
					new_vj(i) = pow(double(t0_vj+t_vj(i)),-lamda);
				}
			}

			//update hyperparameters
			//update alpha
			{
			double alpha_temp=0.0;
			double alpha_old=0.0;
			for (uint i = 0; i < train.num_cases; i++) {
				alpha_temp += cache[i].e * cache[i].e + cache_t[i].t;
			}
			alpha_old=alpha;
			alpha=(1-new_w0)*alpha_old + new_w0*((double) train.num_cases/ alpha_temp);
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
			sigma_0=(1-new_w0)*sigma_0 + new_w0*(1.0/(mu_0_dash * mu_0_dash+ sigma_0_dash));

			//update sigma_w
			DVector<double>& w_sigma_temp = cache_for_group_values;
			w_sigma_temp.init(0.0);
			for (uint i=0; i<fm->num_attribute;i++) {
				uint g = meta->attr_group(i);
				w_sigma_temp(g)+=mu_w_dash.value[i] * mu_w_dash.value[i] + sigma_w_dash.value[i];
			}
			for (uint i=0; i< meta->num_attr_groups; i++) {
				sigma_w(i)=(1-new_w0)*sigma_w(i) + new_w0*((double) meta->num_attr_per_group(i)/w_sigma_temp(i));
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
					sigma_v(g,f)=(1-new_w0)*sigma_v(g,f) + new_w0*((double) meta->num_attr_per_group(g)/v_sigma_temp(g));
				}
			}

			t_w0+=1;
            new_w0 = pow(double(t0_w0+t_w0),-lamda);
		}


		void update_w0(DataSubset& train,uint _size) {
			double sigma_dash,sigma_old;
			sigma_dash=sigma_0_dash;
			double w0_temp=0.0;
			double mu_old,mu_dash;
			mu_dash=mu_0_dash;
			mu_old = natural_mu_0_dash;
            sigma_old = natural_sigma_0_dash;
			double eta1=0.0,eta2=0.0;
			for (uint i=0; i<train.num_cases;i++)
			{
				w0_temp=cache[i].e + mu_0_dash;
				natural_sigma_0_dash = ((1-new_w0)*sigma_old) + new_w0*(sigma_0 + _size * alpha);
				natural_mu_0_dash = ((1-new_w0)*mu_old) + new_w0*_size*alpha*w0_temp;
				eta1 += natural_mu_0_dash;
				eta2 += natural_sigma_0_dash;
			}
			natural_mu_0_dash = eta1 / train.num_cases;
			natural_sigma_0_dash = eta2 / train.num_cases;
			mu_0_dash = natural_mu_0_dash/natural_sigma_0_dash;
			sigma_0_dash = 1.0/natural_sigma_0_dash;

			for (uint i=0; i<train.num_cases; i++) {
				cache[i].e=cache[i].e + (mu_dash-mu_0_dash);
				cache_t[i].t=cache_t[i].t + (sigma_0_dash-sigma_dash);
			}
 		}

		void update_w (double& mu, double& sigma, double& sigma_w, sparse_row<DATA_FLOAT>& feature_data,uint _size,uint col,uint bs) {
			double w_sigma_sqr = 0;
			double w_mean = 0;
			double mu_old=0.0, sigma_old=0.0,mu_dash,sigma_dash;
			mu_dash=mu;
			sigma_dash=sigma;
			double eta1=0.0,eta2=0.0;
			mu_old = natural_mu_w_dash(col);
			sigma_old = natural_sigma_w_dash(col);
			for (uint i_fd = 0; i_fd < feature_data.size; i_fd++) {

				uint& train_case_index = feature_data.data[i_fd].id;
				FM_FLOAT x_li = feature_data.data[i_fd].value;
				w_mean = x_li * (cache[train_case_index].e + x_li * mu);
				w_sigma_sqr = x_li * x_li;

				natural_sigma_w_dash(col) = ((1-new_wj(col))*sigma_old) + new_wj(col)* (sigma_w + alpha * col_count(col) * w_sigma_sqr);
				natural_mu_w_dash(col) = ((1-new_wj(col))*mu_old) + new_wj(col) * col_count(col) * alpha * w_mean;
				eta1 += natural_mu_w_dash(col);
				eta2 += natural_sigma_w_dash(col);
			}
			t_wj(col)+=feature_data.size;
			new_wj(col) = pow(double(t0_wj+t_wj(col)),-lamda);
			natural_mu_w_dash(col) = eta1/feature_data.size;
			natural_sigma_w_dash(col) = eta2/feature_data.size;
			mu = natural_mu_w_dash(col)/natural_sigma_w_dash(col);
			sigma = 1/natural_sigma_w_dash(col);
			
			if (std::isnan(sigma) || std::isinf(sigma)) {
				//mu = 0.0;
				nan_sigma_w_dash+=1;
				sigma=sigma_dash;
			}

			// check for out of bounds values
			if (std::isnan(mu)) {
				nan_mu_w_dash++;
				mu = mu_dash;
				assert(! std::isnan(mu_dash));
				assert(! std::isnan(mu));
				return;
			}
			if (std::isinf(mu)) {
				inf_mu_w_dash++;
				mu = mu_dash;
				assert(! std::isinf(mu_dash));
				assert(! std::isinf(mu));
				return;
			}
			// update error:
			for (uint i_fd = 0; i_fd < feature_data.size; i_fd++) {
				uint& train_case_index = feature_data.data[i_fd].id;
				FM_FLOAT& x_li = feature_data.data[i_fd].value;
				double h = x_li;
				cache[train_case_index].e += h * (mu_dash - mu);
				cache_t[train_case_index].t += h * h * (sigma - sigma_dash);
			}
		}

		void update_v(int& f, double& mu, double& sigma, double& sigma_v_g, sparse_row<DATA_FLOAT>& feature_data,uint _size,uint col,uint bs) {
			double v_sigma_sqr = 0;
			double v_mean = 0;

			double mu_old,sigma_old,mu_dash,sigma_dash;
			mu_dash=mu;
			sigma_dash=sigma;
			double eta1=0.0,eta2=0.0;
			mu_old = natural_mu_v_dash(f,col);
			sigma_old = natural_sigma_v_dash(f,col);
			for (uint i_fd = 0; i_fd < feature_data.size; i_fd++) {

				uint& train_case_index = feature_data.data[i_fd].id;
				FM_FLOAT& x_li = feature_data.data[i_fd].value;
				e_q_term* cache_li = &(cache[train_case_index]);
				t_term* cache_li_t = &(cache_t[train_case_index]);
				double h = cache_li->q - x_li * mu;
				double h1 = cache_li_t->q - x_li * x_li * sigma;
				v_mean = x_li * h * (cache_li->e + x_li * mu * h);
				v_sigma_sqr = x_li * x_li * h * h + x_li * x_li * h1;

				natural_sigma_v_dash(f,col) = (1-new_vj(col))*sigma_old + new_vj(col)*(sigma_v_g + alpha * col_count(col) *v_sigma_sqr);
				natural_mu_v_dash(f,col) = ((1-new_vj(col))*mu_old) + new_vj(col) * col_count(col) * alpha * v_mean;
				eta1 += natural_mu_v_dash(f,col);
				eta2 += natural_sigma_v_dash(f,col);
			}
			natural_mu_v_dash(f,col) = eta1 / feature_data.size;
			natural_sigma_v_dash(f,col) = eta2 / feature_data.size;
			mu = natural_mu_v_dash(f,col)/natural_sigma_v_dash(f,col);
			sigma = 1/natural_sigma_v_dash(f,col);
			
			if (std::isnan(sigma) || std::isinf(sigma)) {
				sigma=sigma_dash;
				nan_sigma_v_dash+=1;
			}
			// check for out of bounds values
			if (std::isnan(mu)) {
				nan_mu_v_dash++;
				mu = mu_dash;
				assert(! std::isnan(mu_dash));
				assert(! std::isnan(mu));
				return;
			}
			if (std::isinf(mu)) {
				inf_mu_v_dash++;
				mu = mu_dash;
				assert(! std::isinf(mu_dash));
				assert(! std::isinf(mu));
				return;
			}


			// update error and q:
			for (uint i_fd = 0; i_fd < feature_data.size; i_fd++) {
				uint& train_case_index = feature_data.data[i_fd].id;
				FM_FLOAT& x_li = feature_data.data[i_fd].value;
				e_q_term* cache_li = &(cache[train_case_index]);
				t_term* cache_li_t = &(cache_t[train_case_index]);
				double h = x_li * ( cache_li->q - x_li * mu_dash);
				double h1 = x_li * x_li * ( cache_li_t->q - x_li * x_li * sigma_dash );
				double h2= x_li * x_li * (cache_li_t->z - x_li * x_li * mu_dash * mu_dash);
				
				cache_li->q += x_li * (mu - mu_dash);
				cache_li_t->q += x_li * x_li * (sigma - sigma_dash);
				cache_li_t->z += x_li * x_li * (mu * mu - mu_dash * mu_dash);
				cache_li->e += h* (mu_dash - mu);
				cache_li_t->t += (h1 + h2) * (sigma - sigma_dash);
				cache_li_t->t += h1 * (mu * mu - mu_dash * mu_dash);
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

			// set learning rates

			lamda=0.5;
			t0_w0 = 1;
			t0_wj = 1;
			t0_vj = 1;
			t_w0 = 0;
			new_w0= pow(double(t0_w0+t_w0),-lamda);
			new_wj.setSize(fm->num_attribute);
			new_vj.setSize(fm->num_attribute);
			t_wj.setSize(fm->num_attribute);
			t_vj.setSize(fm->num_attribute);
			col_count.setSize(fm->num_attribute);
			col_count.init(0);
			t_wj.init(0);
			t_vj.init(0);
			new_wj.init(pow(double(t0_wj+0),-lamda));
                        new_vj.init(pow(double(t0_vj+0),-lamda));

			natural_mu_0_dash = 0.0;
			natural_sigma_0_dash = 1/sigma_0_dash;

			// find count of each col in the dataset for scaling purpose
			ifstream train_file(training_file.c_str());
			while (!train_file.eof()) {

				int nchar;
	            float _rating;
	            uint _feature;
	            double _value;
	            std::string line;
	            std::getline(train_file, line);
	            const char *pline = line.c_str();
	            while ((*pline == ' ')  || (*pline == 9)) { pline++; } // skip leading spaces
	            if ((*pline == 0)  || (*pline == '#')) { continue; }  // skip empty rows
	            if (sscanf(pline, "%f%n", &_rating, &nchar) >=1) {
	                    pline += nchar;
	                    while (sscanf(pline, "%u:%lf%n", &_feature, &_value, &nchar) >= 2) {
	                            pline += nchar;
	                            col_count(_feature)+=1;
	                    }
	            } else {
	                    throw "cannot parse line \"" + line + "\" at character " + pline[0];
	            }
            }
            train_file.close();
	
			// set dimension
			sigma_w.setSize(meta->num_attr_groups);
			sigma_v.setSize(meta->num_attr_groups,fm->num_factor);
			mu_w_dash.setSize(fm->num_attribute);
			sigma_w_dash.setSize(fm->num_attribute);

			mu_v_dash.setSize(fm->num_factor,fm->num_attribute);
			sigma_v_dash.setSize(fm->num_factor,fm->num_attribute);

			natural_mu_w_dash.setSize(fm->num_attribute);
			natural_sigma_w_dash.setSize(fm->num_attribute);

			natural_mu_v_dash.setSize(fm->num_factor,fm->num_attribute);
			natural_sigma_v_dash.setSize(fm->num_factor,fm->num_attribute);

			// initialize
			sigma_w.init(1);
			sigma_v.init(1);
			mu_w_dash.init_normal(0,1);
			sigma_w_dash.init(.02);
			mu_v_dash.init_normal(0,1);
			sigma_v_dash.init(.02);
			natural_mu_w_dash.assign(mu_w_dash);
			for(uint i=0;i<fm->num_attribute;i++)
			{
				natural_mu_w_dash(i)/=0.02;
				natural_sigma_w_dash(i) = 1/sigma_w_dash(i);
			}

			natural_mu_v_dash.assign(mu_v_dash);
			for(uint j=0;j<fm->num_factor;j++)
			{
				for(uint i=0;i<fm->num_attribute;i++)
				{
					natural_mu_v_dash(j,i)/=0.02;
					natural_sigma_v_dash(j,i) = 1/sigma_v_dash(j,i);
				}
			}


			if (log != NULL) {
				log->addField("alpha", std::numeric_limits<double>::quiet_NaN());
				if (task == TASK_REGRESSION) {
					log->addField("rmse_mcmc_this", std::numeric_limits<double>::quiet_NaN());
					log->addField("rmse_mcmc_all", std::numeric_limits<double>::quiet_NaN());
				} else if (task == TASK_CLASSIFICATION) {
					log->addField("acc_mcmc_this", std::numeric_limits<double>::quiet_NaN());
					log->addField("acc_mcmc_all", std::numeric_limits<double>::quiet_NaN());
					log->addField("ll_mcmc_this", std::numeric_limits<double>::quiet_NaN());
					log->addField("ll_mcmc_all", std::numeric_limits<double>::quiet_NaN());
					log->addField("ll_mcmc_all_but5", std::numeric_limits<double>::quiet_NaN());
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
			MemoryLog::getInstance().logNew("e_q_term", sizeof(e_q_term), test.num_cases);
			cache_test = new e_q_term[test.num_cases];

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
		}


		virtual void debug() {
			fm_learn::debug();
			std::cout << "num_eval_cases=" << num_eval_cases << std::endl;
		}
};

#endif /*FM_LEARN_VB_ONLINE_H_*/