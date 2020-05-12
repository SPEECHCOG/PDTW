function track2_main(language,alpha)

% Main script to run ZS2020 challenge experiments for Track2 of ZS2017
% data. 
% Calling this with:
%
% languages = {'mandarin','english','french','LANG1','LANG2'};
% alpha = 0.001
% for langiter = 1:length(languages)
%       track2_main(languages{langiter},alpha) 
% end
% 
% should reproduce PDTW1 submission results for the challenge. 
% Change alpha to 0.0001 to re-create PDTW2 submission.

% Change these to match your own environment
audio_location = '/Users/rasaneno/speechdb/zerospeech2020/2017/'; % Where are ZS2017 audio located (root for all languages)?
feature_location = '/Users/rasaneno/rundata/ZS2020_tmp/feats/'; % where will pre-computed MFCCs be stored?

% Feature parameters
if nargin <1
    language = 'mandarin';
end

config = struct();

if nargin <2
    config.alpha = 1e-3; % threshold for matching downsampled segments & (default: 1e-3)
else
    config.alpha = alpha;
end

%% Model parameters
config.seqlen = 20; % initial stage #1 segment length for matching (frames)
config.seqshift = 10; % window shift for stage #1 segments (frames)
config.n_slices = 4; % number of slices per downsampled segment
config.nearest_to_check = 5; % how many nearest neighbors to consider for each segment
config.duration_thr = 5; % minimum duration of detected segments
config.total_ram = 8e9; % how much RAM to use for distance matrices
config.run_parallel = 1; % run using MATLAB parpool? 
config.write_location = '/Users/rasaneno/rundata/ZS2020_tmp/'; % where to save intermediate results 

config.process_id = language; % name of the process (added to saved files)
config.verbose = 1;


%% Get training file names
ZS = loadZSData2017(language,audio_location); % Load list of audio files

% Load or calculate MFCCs for training utterances
filnam = sprintf('%s/MFCC_features_train_%s.mat',feature_location,language);
if(exist(filnam,'file'))
    load(filnam);
else    
    usedeltas = 1; % default = 1
    do_CMVN = 1; % default = 1
    if(config.verbose)
       fprintf('Calulating MFCCs...\n');
    end
    F_train = getMFCCs_custom(ZS.(language).train.filename,do_CMVN,usedeltas);
    save(filnam,'F_train');
end

%% Run PDTW
patterns = PDTW(F_train,config);

%% Create submission structure
submission_name = 'PDTW4'; % Name your submission here
submission_path = '/Users/rasaneno/rundata/ZS2020/';     % Where to store submission files?        

writepath = sprintf('%s/%s/',submission_path,submission_name);
if(~exist(writepath,'dir'))
    createSubmissionTemplateZS2017(submission_name,submission_path)
end

tmp = sprintf('%s',config.alpha);
    thrname = [tmp(1:3) tmp(end-3:end)];
filename = sprintf('%s/%s/2017/track2/%s_full_%s.txt',submission_path,submission_name,language,thrname);
a = 1:length(patterns.dist);

eval(sprintf('filnams = ZS.%s.train.filename;',language));
signal_names = cell(length(a)*2,1);
times = zeros(length(a)*2,2);
classes = zeros(length(a)*2,1);
z = 1;
class_ID = 1;
for k = 1:length(a)
    
    offset = 0.0;
    if(patterns.offset(a(k),1)-patterns.onset(a(k),1) >= config.duration_thr && patterns.offset(a(k),2)-patterns.onset(a(k),2) >= config.duration_thr)
        [~,tmp] = fileparts(filnams{patterns.signal(a(k),1)});
        signal_names{z} = tmp;
        times(z,1) = patterns.onset(a(k),1)/100+offset;
        times(z,2) = patterns.offset(a(k),1)/100+offset;
        classes(z) = class_ID;
        z = z+1;
        [~,tmp] = fileparts(filnams{patterns.signal(a(k),2)});
        signal_names{z} = tmp;
        times(z,1) = patterns.onset(a(k),2)/100+offset;
        times(z,2) = patterns.offset(a(k),2)/100+offset;
        classes(z) = class_ID;
        class_ID = class_ID +1;
        z = z+1;
    end
    
end
    
writeTrack2Outputs(filename,signal_names,times,classes)
    


    
    
    
    
    
    
    
    
    
    
    
    









