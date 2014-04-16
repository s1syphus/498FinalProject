function [] = read_in_data(filename)

%{
	file_read = csvread(filename,1,0,[1,0,7,17]);
	file_read
%}

%{
	Formatting of csv:
	diffsecs (not important), L1.norm (absolute energy), L2.norm (RMS energy), Linf.norm (peak energy),
		PSD.250 (low frequency energy), PSD.500 (low-mid frequency energy), PSD.1000 (Mid-High frequency energy),PSD.2000 (High frequency energy),
		MFCC.1 (Log-energy 0th mel-frequency cepstral coefficient of audio), ... , MFCC.12

	These will be stored as different .mat files
%}



	fid = fopen(filename,'rt');
	a = textscan(fid, '%f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %s', 'Delimiter',',','CollectOutput',1,'HeaderLines',1);
	fclose(fid);
	format short g
	M = [a{1} ];
	
	
	M(:,1)

	L1norm = M(:,2);
	save('testL1norm.mat','L1norm');
	L2norm = M(:,3);
	save('testL2norm.mat','L2norm');



%	save('test.mat','M');

end
