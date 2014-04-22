function [rates] = svm_stuff()

	%load in the data, examining psd - will add more features later


	cherry = load('../sortedAverageData/Cherry.psd.mat');
	crocus = load('../sortedAverageData/Crocus.psd.mat');
	daisey = load('../sortedAverageData/Daisey.psd.mat');
	flox = load('../sortedAverageData/Flox.psd.mat');
	iris = load('../sortedAverageData/Iris.psd.mat');
	maple = load('../sortedAverageData/Maple.psd.mat');
	orchid = load('../sortedAverageData/Orchid.psd.mat');
	peony = load('../sortedAverageData/Peony.psd.mat');
	violet = load('../sortedAverageData/Violet.psd.mat');

	positive_psd = [cherry.psd ; crocus.psd ; daisey.psd ; flox.psd; iris.psd; maple.psd ; orchid.psd ; peony.psd ; violet.psd  ];


	apple = load('../sortedAverageData/Apple.psd.mat');
	dafodil = load('../sortedAverageData/Dafodil.psd.mat');
	lilly = load('../sortedAverageData/Lilly.psd.mat');
	orange = load('../sortedAverageData/Orange.psd.mat');
	rose = load('../sortedAverageData/Rose.psd.mat');
	sunflower = load('../sortedAverageData/Sunflower.psd.mat');
	sweetpea = load('../sortedAverageData/Sweetpea.psd.mat');
	negative_psd = [apple.psd; dafodil.psd ; lilly.psd; orange.psd; rose.psd; sunflower.psd; sweetpea.psd];

const = 0.1:0.1:1;
tic
for m=1:10

		k = 10;
		best_sum = 0;
		best_k = 0;
		tic
		blah = floor(min([size(positive_psd,1),size(negative_psd,1)])/10);
		rates = zeros(10,3);
	for l=1:10
		indices = crossvalind('Kfold',l*blah,10);
		pos_total = 0;
		neg_total = 0;
		for i = 1:k
			test = (indices == i);
			train = ~test;
			test_pos = positive_psd(test,:);
			test_neg = negative_psd(test,:);
			train_pos = positive_psd(train,:);
			train_neg = negative_psd(train,:);
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
end
