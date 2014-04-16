function [] = read_in_all_data(directory_name)
%{
	Formatting of csv:
	diffsecs (not important), L1.norm (absolute energy), L2.norm (RMS energy), Linf.norm (peak energy),
		PSD.250 (low frequency energy), PSD.500 (low-mid frequency energy), PSD.1000 (Mid-High frequency energy),PSD.2000 (High frequency energy),
		MFCC.1 (Log-energy 0th mel-frequency cepstral coefficient of audio), ... , MFCC.12

	These will be stored as different .mat files
%}

	listing = dir(['data/' directory_name]);
%	size(listing,1)
%	listing.name
	M = [];
	for i=(3:size(listing,1)-1)
		full_path = ['data/' directory_name '/' listing(i).name];
		fid = fopen(full_path);
		a = textscan(fid, '%f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %s', 'Delimiter',',','CollectOutput',1,'HeaderLines',1);
		fclose(fid);
		format short g
		M = [M; a{1} ];
	end

	%split up into different fields	

	%norm absolute - RMS - Peak
	norm = [M(:,2)  M(:,3) M(:,4)];
	save(['data/sortedData/' directory_name '.norm.mat'],'norm');

	%psd 250 - 500 - 1000 - 2000
	psd = [M(:,5) M(:,6) M(:,7) M(:,8)];
	save(['data/sortedData/' directory_name '.psd.mat'],'psd');

	mfcc = [M(:,9) M(:,10) M(:,11) M(:,12) M(:,13) M(:,14) M(:,15) M(:,16) M(:,17) M(:,18) M(:,19) M(:,20)];
	save(['data/sortedData/' directory_name '.mfcc.mat'],'mfcc');



end
