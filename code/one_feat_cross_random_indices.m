function [pos neg] = one_feat_cross_random_indices(k, train_size, test_size)
	%k = k folds
%{
	Positives are:
	Cherry
	Crocus
	Daisy and Daisey
	Flox
	Iris
	Maple
	Orchid
	Peony
	Violet

	Negatives are:
	Apple
	Dafodil
	Lilly and Lily
	Orange
	Rose
	SunflowerA
	Sweetpea

	1) Load in data - two different arrays, positives and controls/negatives
		
	2) Generate indices array

	3) 
%}

	% 10-fold cross-validation

%	positive = ;
%	negative = ;
%{
	positive = load('data/sortedData/Iris.psd.mat');
	negative = load('data/sortedData/Rose.psd.mat');
%}
	%normalize?
	positive = load('data/sortedData/Iris.mfcc.mat');
	negative = load('data/sortedData/Rose.mfcc.mat');

	pos_size = size(positive.mfcc,1);
	neg_size = size(negative.mfcc,1);

	all_pos_indices = 1:pos_size;
	all_neg_indices = 1:neg_size;

	total_pos_rate = zeros(k,1);
	total_neg_rate = zeros(k,1);
%{
	train_size = 10000;
	test_size = 1000;
%}
	parfor i=1:k
		pos_test = randperm(pos_size,test_size);	
		pos_train = randperm(pos_size,train_size);
		neg_test = randperm(neg_size,test_size);
		neg_train = randperm(neg_size,test_size);
		[pos_rate neg_rate] = one_feat(positive.mfcc, negative.mfcc,pos_test,pos_train,neg_test,neg_train);
		total_pos_rate(i) = pos_rate;
		total_neg_rate(i) = neg_rate;
	end
	pos = sum(total_pos_rate)/k;
	neg = sum(total_neg_rate)/k;
end

function [pos_r neg_r] = one_feat(positive, negative, pos_test, pos_train, neg_test, neg_train)
	pos_train_feats = [];
	for i = pos_train(1:end)
		pos_train_feats = [pos_train_feats; positive(i,:)];
	end
	neg_train_feats = [];
	for i = neg_train(1:end)
		neg_train_feats = [neg_train_feats; negative(i,:)];
	end

	key = [repmat(1,length(pos_train(1:end)),1); repmat(0,length(neg_train(1:end)),1)];
    	trained_svm = svmtrain([pos_train_feats; neg_train_feats],key,'kktviolationlevel',0.05,'boxconstraint',0.1,'kernel_function','rbf');

   	pos_r = 0;
	neg_r = 0; 
   
	for i = pos_test(1:end)
		if(svmclassify(trained_svm, positive(i,:)) == 1)
			pos_r = pos_r + 1;
		end
	end 
    
    	for i = neg_test(1:end)
		if(svmclassify(trained_svm, negative(i,:)) == 0)
			neg_r = neg_r + 1;
		end
	end 
 
	pos_r = pos_r/size(pos_test(1:end),2);
	neg_r = neg_r/size(neg_test(1:end),2);
    

end
