function [pos neg] = one_feat_cross_norm(k)
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

	positive = load('data/sortedData/Iris.norm.mat');
	negative = load('data/sortedData/Rose.norm.mat');

	%normalize?

	pos_size = size(positive.norm,1);
	neg_size = size(negative.norm,1);

	all_pos_indices = 1:pos_size;
	all_neg_indices = 1:neg_size;

	total_pos_rate = zeros(k,1);
	total_neg_rate = zeros(k,1);

	parfor i=1:k
		pos_test = all_pos_indices(mod(all_pos_indices,k) == (i-1));
		pos_train = all_pos_indices(mod(all_pos_indices,k) ~= (i-1));
		neg_test = all_neg_indices(mod(all_neg_indices,k) == (i-1));
		neg_train = all_neg_indices(mod(all_neg_indices,k) ~= (i-1));
		[pos_rate neg_rate] = one_feat(positive.norm, negative.norm,pos_test,pos_train,neg_test,neg_train);
		total_pos_rate(i) = pos_rate*(size(pos_test,2)/pos_size(1));
		total_neg_rate(i) = neg_rate*(size(neg_test,2)/neg_size(1));
	end
	pos = sum(total_pos_rate);
	neg = sum(total_neg_rate);
end

function [pos_r neg_r] = one_feat(positive, negative, pos_test, pos_train, neg_test, neg_train)

	train_size = 20000;
	test_size = 2000;


	pos_train_feats = [];
	for i = pos_train(1:train_size)
		pos_train_feats = [pos_train_feats; positive(i,:)];
	end
%	pos_train_feats
	neg_train_feats = [];
	for i = neg_train(1:train_size)
		neg_train_feats = [neg_train_feats; negative(i,:)];
	end
%	neg_train_feats




	%only using first 10 for testing
	key = [repmat(1,length(pos_train(1:train_size)),1); repmat(0,length(neg_train(1:train_size)),1)];
	trained_svm = svmtrain([pos_train_feats; neg_train_feats],key,'kktviolationlevel',0.05,'kernel_function','rbf');
    
   	pos_r = 0;
	neg_r = 0; 
   
	for i = pos_test(1:test_size)
		if(svmclassify(trained_svm, positive(i,:)) == 1)
			pos_r = pos_r + 1;
		end
	end 
    
    	for i = neg_test(1:test_size)
		if(svmclassify(trained_svm, negative(i,:)) == 0)
			neg_r = neg_r + 1;
		end
	end 
    
	pos_r = pos_r/size(pos_test(1:test_size),2);
	neg_r = neg_r/size(neg_test(1:test_size),2);
    

end
