function [MFCCs,TIMESTAMPS] = getMFCCs_custom(filenames,do_CMVN,usedeltas)

if nargin <2
    do_CMVN = 1;
end

if nargin <3
    usedeltas = 1;
end

N_signals = length(filenames);

[F_FFT] = haeFFTPiirteet(filenames,0.025,0.01,16000);

% corresponding timestamps
TIMESTAMPS = cell(N_signals,1);
for signal = 1:N_signals
    TIMESTAMPS{signal} = 0:1/100:size(F_FFT{signal},1)/100-1/100;
end

% Mel bank from Voicebox
[MEL,MN,MX]= melbankm(24,0.025*16000,16000,0,0.50001,'u');


F_Mel = cell(length(F_FFT),1);
F_MFCC = cell(length(F_FFT),1);
for signal = 1:length(F_FFT)
    F_Mel{signal} = F_FFT{signal}(:,1:size(MEL,2))*MEL';
    tmp = dct(F_Mel{signal}');
    tmp = tmp(1:13,:)';
    F_MFCC{signal} = tmp;
    procbar(signal,length(F_FFT));
end


F_MFCC_delta = getDeltas(F_MFCC);
F_MFCC_deltadelta = getDeltas(F_MFCC_delta);

   
MFCCs = cell(size(F_MFCC));
for signal = 1:length(F_MFCC)
    MFCCs{signal} = [F_MFCC{signal} F_MFCC_delta{signal} F_MFCC_deltadelta{signal}];
end


if(do_CMVN)
    MFCCs = MVN(MFCCs);    
end

if(~usedeltas)
   for k = 1:length(MFCCs)
      MFCCs{k} = MFCCs{k}(:,1:13); 
   end   
end