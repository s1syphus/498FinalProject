function [pos neg] = all_feats_cross_validate(k,train_size, test_size)

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


	%looking at psd first - add in other features soon
	%using only first 3 of each group, add more in later
	cherry = load('data/sortedData/Cherry.psd.mat');
	crocus = load('data/sortedData/Crocus.psd.mat');
	daisey = load('data/sortedData/Daisey.psd.mat');
	flox = load('data/sortedData/Daisey.psd.mat');
	iris = load('data/sortedData/Iris.psd.mat');
	maple = load('data/sortedData/Maple.psd.mat');
	orchid = load('data/sortedData/Orchid.psd.mat');
	peony = load('data/sortedData/Peony.psd.mat');
	violet = load('data/sortedData/Violet.psd.mat');


	positive_psd = [cherry.psd ; crocus.psd ; daisey.psd ; flox.psd; iris.psd; maple.psd ; orchid.psd ; peony.psd ; violet.psd  ];

	apple = load('data/sortedData/Apple.psd.mat');
	dafodil = load('data/sortedData/Dafodil.psd.mat');
	lilly = load('data/sortedData/Lilly.psd.mat');
	orange = load('data/sortedData/Lilly.psd.mat');
	rose = load('data/sortedData/Rose.psd.mat');
	sunflower = load('data/sortedData/Sunflower.psd.mat');
	sweetpea = load('data/sortedData/Sweetpea.psd.mat');

	negative_psd = [apple.psd; dafodil.psd ; lilly.psd; orange.psd; rose.psd; sunflower.psd; sweetpea.psd];

	pos_size = size(positive_psd,1);
	neg_size = size(negative_psd,1);

	total_pos_rate = zeros(k,1);
	total_neg_rate = zeros(k,1);

	parfor i=1:k
		pos_test = randperm(pos_size,test_size);	
		pos_train = randperm(pos_size,train_size);
		neg_test = randperm(neg_size,test_size);
		neg_train = randperm(neg_size,test_size);
		[pos_rate neg_rate] = one_feat(positive_psd, negative_psd,pos_test,pos_train,neg_test,neg_train);
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


