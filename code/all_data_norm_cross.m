function [pos neg] = all_data_norm_cross(k,train_size, test_size)

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


	cherry = load('data/sortedData/Cherry.norm.mat');
	crocus = load('data/sortedData/Crocus.norm.mat');
	daisey = load('data/sortedData/Daisey.norm.mat');
	flox = load('data/sortedData/Flox.norm.mat');
	iris = load('data/sortedData/Iris.norm.mat');
	maple = load('data/sortedData/Maple.norm.mat');
	orchid = load('data/sortedData/Orchid.norm.mat');
	peony = load('data/sortedData/Peony.norm.mat');
	violet = load('data/sortedData/Violet.norm.mat');


	positive_norm = [cherry.norm ; crocus.norm ; daisey.norm ; flox.norm; iris.norm; maple.norm ; orchid.norm ; peony.norm ; violet.norm  ];

	apple = load('data/sortedData/Apple.norm.mat');
	dafodil = load('data/sortedData/Dafodil.norm.mat');
	lilly = load('data/sortedData/Lilly.norm.mat');
	orange = load('data/sortedData/Orange.norm.mat');
	rose = load('data/sortedData/Rose.norm.mat');
	sunflower = load('data/sortedData/Sunflower.norm.mat');
	sweetpea = load('data/sortedData/Sweetpea.norm.mat');

	negative_norm = [apple.norm; dafodil.norm ; lilly.norm; orange.norm; rose.norm; sunflower.norm; sweetpea.norm];

	pos_size = size(positive_norm,1);
	neg_size = size(negative_norm,1);

	total_pos_rate = zeros(k,1);
	total_neg_rate = zeros(k,1);

	parfor i=1:k
		pos_test = randperm(pos_size,test_size);	
		pos_train = randperm(pos_size,train_size);
		neg_test = randperm(neg_size,test_size);
		neg_train = randperm(neg_size,test_size);
		[pos_rate neg_rate] = one_feat(positive_norm, negative_norm,pos_test,pos_train,neg_test,neg_train);
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


