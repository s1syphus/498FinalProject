function [] = svm_stuff()

	%load in the data, examining psd - will add more features later


	cherry = load('../../Data/sortedAverageData/Cherry.psd.mat');
	crocus = load('../../Data/sortedAverageData/Crocus.psd.mat');
	daisey = load('../../Data/sortedAverageData/Daisey.psd.mat');
	flox = load('../../Data/sortedAverageData/Flox.psd.mat');
	iris = load('../../Data/sortedAverageData/Iris.psd.mat');
	maple = load('../../Data/sortedAverageData/Maple.psd.mat');
	orchid = load('../../Data/sortedAverageData/Orchid.psd.mat');
	peony = load('../../Data/sortedAverageData/Peony.psd.mat');
	violet = load('../../Data/sortedAverageData/Violet.psd.mat');

	positive_psd = [cherry.psd ; crocus.psd ; daisey.psd ; flox.psd; iris.psd; maple.psd ; orchid.psd ; peony.psd ; violet.psd  ];


	apple = load('../../Data/sortedAverageData/Apple.psd.mat');
	dafodil = load('../../Data/sortedAverageData/Dafodil.psd.mat');
	lilly = load('../../Data/sortedAverageData/Lilly.psd.mat');
	orange = load('../../Data/sortedAverageData/Orange.psd.mat');
	rose = load('../../Data/sortedAverageData/Rose.psd.mat');
	sunflower = load('../../Data/sortedAverageData/Sunflower.psd.mat');
	sweetpea = load('../../Data/sortedAverageData/Sweetpea.psd.mat');
	negative_psd = [apple.psd; dafodil.psd ; lilly.psd; orange.psd; rose.psd; sunflower.psd; sweetpea.psd];


	pos_class = positive_psd(1:50,:);
	neg_class = negative_psd(1:50,:);
	key = [repmat(1,length(pos_class)); repmat(0,length(neg_class))];

	svm_model = fitcsvm([pos_class; neg_class],key,'KernelFunction','rbf','Standardize',true);
	[~,scores] = predict(svm_model,[pos_class ; neg_class]);
	scores


end
