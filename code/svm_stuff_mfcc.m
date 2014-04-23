function [overall_rates] = svm_stuff_mfcc()

	%load in the data, examining mfcc - will add more features later


	cherry = load('../sortedAverageData/Cherry.mfcc.mat');
	crocus = load('../sortedAverageData/Crocus.mfcc.mat');
	daisey = load('../sortedAverageData/Daisey.mfcc.mat');
	flox = load('../sortedAverageData/Flox.mfcc.mat');
	iris = load('../sortedAverageData/Iris.mfcc.mat');
	maple = load('../sortedAverageData/Maple.mfcc.mat');
	orchid = load('../sortedAverageData/Orchid.mfcc.mat');
	peony = load('../sortedAverageData/Peony.mfcc.mat');
	violet = load('../sortedAverageData/Violet.mfcc.mat');

	positive_mfcc = [cherry.mfcc ; crocus.mfcc ; daisey.mfcc ; flox.mfcc; iris.mfcc; maple.mfcc ; orchid.mfcc ; peony.mfcc ; violet.mfcc  ];


	apple = load('../sortedAverageData/Apple.mfcc.mat');
	dafodil = load('../sortedAverageData/Dafodil.mfcc.mat');
	lilly = load('../sortedAverageData/Lilly.mfcc.mat');
	orange = load('../sortedAverageData/Orange.mfcc.mat');
	rose = load('../sortedAverageData/Rose.mfcc.mat');
	sunflower = load('../sortedAverageData/Sunflower.mfcc.mat');
	sweetpea = load('../sortedAverageData/Sweetpea.mfcc.mat');
	negative_mfcc = [apple.mfcc; dafodil.mfcc ; lilly.mfcc; orange.mfcc; rose.mfcc; sunflower.mfcc; sweetpea.mfcc];

const = 0.1:0.1:1;
viol = 0:0.01:0.1;
tic
best_rate = zeros(3,1);
for m=1:10
		k = 10;
		best_sum = 0;
		best_k = 0;
		
		tic
		blah = floor(min([size(positive_mfcc,1),size(negative_mfcc,1)])/10);
		rates = zeros(10,3);
	for l=1:10
		indices = crossvalind('Kfold',l*blah,10);
		pos_total = 0;
		neg_total = 0;
		for i = 1:k
			test = (indices == i);
			train = ~test;
			test_pos = positive_mfcc(test,:);
			test_neg = negative_mfcc(test,:);
			train_pos = positive_mfcc(train,:);
			train_neg = negative_mfcc(train,:);
			key = [repmat(1,length(train_pos),1); repmat(0,length(train_neg),1)];
			svm_model = svmtrain([train_pos; train_neg],key,'kernel_function','rbf','boxconstraint',const(m));
			group = svmclassify(svm_model,[test_pos; test_neg]); %change to test soon
			pos_rate = sum(group(1:size(test_pos,1)))/size(test_pos,1);
			neg_rate = 1 - (sum(group(size(test_pos,1) + 1:end))/size(test_neg,1));
			pos_total = pos_total + pos_rate * (1/k);
			neg_total = neg_total + neg_rate * (1/k);
		end
		temp_sum = pos_total+neg_total;
		rates(l,:) = [pos_total neg_total l*blah];
		if temp_sum > best_sum
			best_sum = temp_sum;
			best_k = l*blah;
			best_rate = [pos_total neg_total best_k];
		end
	end
	best_rate
	const_rate(m,:) = best_rate;
	toc
end
toc

overall_rates = const_rate;


end
