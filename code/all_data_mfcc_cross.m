function [pos neg] = all_data_mfcc_cross(k,train_size, test_size)

%	Positives are:
%	Cherry
%	Crocus
%	Daisy and Daisey
%	Flox
%	Iris
%	Maple
%	Orchid
%	Peony
%	Violet
%
%	Negatives are:
%	Apple
%	Dafodil
%	Lilly and Lily
%	Orange
%	Rose
%	Sunflower
%	Sweetpea


	cherry = load('data/sortedData/Cherry.mfcc.mat');
	crocus = load('data/sortedData/Crocus.mfcc.mat');
	daisey = load('data/sortedData/Daisey.mfcc.mat');
	flox = load('data/sortedData/Flox.mfcc.mat');
	iris = load('data/sortedData/Iris.mfcc.mat');
	maple = load('data/sortedData/Maple.mfcc.mat');
	orchid = load('data/sortedData/Orchid.mfcc.mat');
	peony = load('data/sortedData/Peony.mfcc.mat');
	violet = load('data/sortedData/Violet.mfcc.mat');


	positive_mfcc = [cherry.mfcc ; crocus.mfcc ; daisey.mfcc ; flox.mfcc; iris.mfcc; maple.mfcc ; orchid.mfcc ; peony.mfcc ; violet.mfcc  ];

	apple = load('data/sortedData/Apple.mfcc.mat');
	dafodil = load('data/sortedData/Dafodil.mfcc.mat');
	lilly = load('data/sortedData/Lilly.mfcc.mat');
	orange = load('data/sortedData/Orange.mfcc.mat');
	rose = load('data/sortedData/Rose.mfcc.mat');
	sunflower = load('data/sortedData/Sunflower.mfcc.mat');
	sweetpea = load('data/sortedData/Sweetpea.mfcc.mat');

	negative_mfcc = [apple.mfcc; dafodil.mfcc ; lilly.mfcc; orange.mfcc; rose.mfcc; sunflower.mfcc; sweetpea.mfcc];

	pos_size = size(positive_mfcc,1);
	neg_size = size(negative_mfcc,1);

	total_pos_rate = zeros(k,1);
	total_neg_rate = zeros(k,1);

	parfor i=1:k
		pos_test = randperm(pos_size,test_size);	
		pos_train = randperm(pos_size,train_size);
		neg_test = randperm(neg_size,test_size);
		neg_train = randperm(neg_size,test_size);
		[pos_rate neg_rate] = one_feat(positive_mfcc, negative_mfcc,pos_test,pos_train,neg_test,neg_train);
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


