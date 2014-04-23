function [overall_rates] = svm_stuff_norm()

	%load in the data, examining norm - will add more features later


	cherry = load('../sortedAverageData/Cherry.norm.mat');
	crocus = load('../sortedAverageData/Crocus.norm.mat');
	daisey = load('../sortedAverageData/Daisey.norm.mat');
	flox = load('../sortedAverageData/Flox.norm.mat');
	iris = load('../sortedAverageData/Iris.norm.mat');
	maple = load('../sortedAverageData/Maple.norm.mat');
	orchid = load('../sortedAverageData/Orchid.norm.mat');
	peony = load('../sortedAverageData/Peony.norm.mat');
	violet = load('../sortedAverageData/Violet.norm.mat');

	positive_norm = [cherry.norm ; crocus.norm ; daisey.norm ; flox.norm; iris.norm; maple.norm ; orchid.norm ; peony.norm ; violet.norm  ];


	apple = load('../sortedAverageData/Apple.norm.mat');
	dafodil = load('../sortedAverageData/Dafodil.norm.mat');
	lilly = load('../sortedAverageData/Lilly.norm.mat');
	orange = load('../sortedAverageData/Orange.norm.mat');
	rose = load('../sortedAverageData/Rose.norm.mat');
	sunflower = load('../sortedAverageData/Sunflower.norm.mat');
	sweetpea = load('../sortedAverageData/Sweetpea.norm.mat');
	negative_norm = [apple.norm; dafodil.norm ; lilly.norm; orange.norm; rose.norm; sunflower.norm; sweetpea.norm];

const = 0.1:0.1:1;
viol = 0:0.01:0.1;
tic
for m=1:10

		k = 10;
		best_sum = 0;
		best_k = 0;
		tic
		blah = floor(min([size(positive_norm,1),size(negative_norm,1)])/10);
		rates = zeros(10,3);
	for l=1:10
		indices = crossvalind('Kfold',l*blah,10);
		pos_total = 0;
		neg_total = 0;
		for i = 1:k
			test = (indices == i);
			train = ~test;
			test_pos = positive_norm(test,:);
			test_neg = negative_norm(test,:);
			train_pos = positive_norm(train,:);
			train_neg = negative_norm(train,:);
			key = [repmat(1,length(train_pos),1); repmat(0,length(train_neg),1)];
			svm_model = svmtrain([train_pos; train_neg],key,'kernel_function','linear','boxconstraint',const(m));%kktviolationlevel',viol(m));%boxconstraint',const(m));
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
	const_rate(m,:) = best_rate;
	toc
end
toc

overall_rates = const_rate;


end
