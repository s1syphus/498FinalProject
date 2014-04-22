function [] = read_in_average_data(directory_name)
		

		listing = dir(['../../Data/seperatedAudio/' directory_name]);
	M = [];
	
	for i=(3:size(listing,1)-1)
%i = 3;
%listing(3).name
		full_path = ['../../Data/seperatedAudio/' directory_name '/' listing(i).name];
%		full_path
		fid = fopen(full_path);
		a = textscan(fid, '%f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %s', 'Delimiter',',','CollectOutput',1,'HeaderLines',1);
		fclose(fid);
		format short g
%		a
%		a{1}
%		ave = mean(a{1},1)
		M = [M; mean(a{1},1) ];
	end
	%split up into different fields	

	%norm absolute - RMS - Peak
	norm = [M(:,2)  M(:,3) M(:,4)];
	save(['../../Data/sortedAverageData/' directory_name '.norm.mat'],'norm');
	%psd 250 - 500 - 1000 - 2000
	psd = [M(:,5) M(:,6) M(:,7) M(:,8)];
	save(['../../Data/sortedAverageData/' directory_name '.psd.mat'],'psd');

	mfcc = [M(:,9) M(:,10) M(:,11) M(:,12) M(:,13) M(:,14) M(:,15) M(:,16) M(:,17) M(:,18) M(:,19) M(:,20)];
	save(['../../Data/sortedAverageData/' directory_name '.mfcc.mat'],'mfcc');


end
