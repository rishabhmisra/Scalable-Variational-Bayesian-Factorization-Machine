1-Compilation
	• To Compile the tools, inside the main directory, give the command- make all

2-Overview of the files:
	• license.txt: license for usage of libFM
	• history.txt: version history and changes
	• readme.txt: this manual
	• Makefile: compiles the executables using make
	• bin: the folder with the executables
		– libFM: the libFM tool
		– convert: a tool for converting text-files into binary format
		– transpose: a tool for transposing binary design matrices
	• scripts
		– triple format to libfm.pl: a Perl script for converting comma/tab-separated datasets into libFM-format.
	• src: the source files of libFM and the tools
	• data: directory contains all the data files

3-Data format
	Each row of data file contains a training case (x, y) for the real-valued feature vector x with the target y. The row states first the value y and then the non-zero values of x.
	
	Example
	4 0:1.5 3:-7.9
	2 1:1e-5 3:2
	-1 1:0.01 6:1
	...
	
	This file contains three cases. The first column states the target of each of the three case: i.e. 4 for the first case, 2 for the second and -1 for the third. After the target, each line contains the non-zero elements of x, where an entry like 0:1.5 reads x0 = 1.5 and 3:-7.9 means x3 = −7.9, etc. That means the left side of INDEX:VALUE states the index within x whereas the right side states the value of xINDEX, i.e. xINDEX = VALUE. In total the data from the example describes the following design matrix X and target vector y:

		| 1.5 0.0 0.0 −7.9 0.0 0.0 0.0 |			|4 |
 	X = | 0.0 0.01 0.0 2.0 0.0 0.0 0.0 |      y = 	|2 |
 		| 0.0 0.01 0.0 0.0 0.0 0.0 1.0 |			|-1|

4-LibFM Parameters

	-cache_size		cache size for data storage (only applicable if data is in binary format), default=infty
	-dim			’k0,k1,k2’: k0=use bias, k1=use 1-way interactions, k2=dim of 2-way interactions; default=1,1,8
	-help			this screen
	-init_stdev		stdev for initialization of 2-way factors; default=0.1
	-iter 			number of iterations; default=100
	-learn_rate		learn_rate for SGD; default=0.1
	-meta 			filename for meta information about data set
	-method 		learning method (SGD, SGDA, ALS, MCMC); default=MCMC
	-out 			filename for output
	-regular 		’r0,r1,r2’ for SGD and ALS: r0=bias regularization, r1=1-way regularization, r2=2-way regularization
	-relation 		BS: filenames for the relations, default=’’
	-rlog 			write measurements within iterations to a file;default=’’
	-task			r=regression, c=binary classification [MANDATORY]
	-test 			filename for test data [MANDATORY]
	-train 			filename for training data [MANDATORY]
	-validation 	filename for validation data (only for SGDA)
	-verbosity 		how much infos to print; default=0
	-batch 			specify number of batches to be considered for online algorithms; default=50

5- Example on How to perform experiments?
	- Suppose we want to run experiments on MovieLens 1M dataset.
	- For that, we first have to convert the training and testing file in libfm format (like mentioned in section 3's example) and then place them in 'data' directory.
	- Let their name be 'sa.train_libfm' and 'sa.test_libfm' respectively.
	- To compile the tool use command 'make all' (without quotes) in the main directory.
	- After that go to 'bin' directory.

	- If you want to apply SGD on the dataset, give command
		*	./libFM -task r -train ../data/sa.train_libfm -test ../data/sa.test_libfm -dim '1,1,8' -method sgd -learn_rate 0.001 -regular '0.01,0.01,0.01'

	- If you want to apply MCMC on the dataset, give command
		*	./libFM -task r -train ../data/sa.train_libfm -test ../data/sa.test_libfm -dim '1,1,8' -method mcmc

	- If you want to apply VBFM on the dataset, give command
		*	./libFM -task r -train ../data/sa.train_libfm -test ../data/sa.test_libfm -dim '1,1,8' -method vb

	- If you want to apply OVBFM on the dataset, give command
		*	./libFM -task r -train ../data/sa.train_libfm -test ../data/sa.test_libfm -dim '1,1,8' -method vb_online -batch 100

	- If you want to apply SGD's version that loads the dataset in batches, give command
		*	./libFM -task r -train ../data/sa.train_libfm -test ../data/sa.test_libfm -dim '1,1,8' -method sgd_online -learn_rate 0.001 -regular '0.01,0.01,0.01' -batch 30