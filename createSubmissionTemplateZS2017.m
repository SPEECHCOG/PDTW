function createSubmissionTemplateZS2017(submission_name,path_in)
% function createSubmissionTemplateZS2017(submission_name,path_in)

authors = {'Maria Andrea Cruz Blandon','Okko Rasanen'};
affiliation = 'Tampere University';

if nargin <2
    ZSpath = sprintf('/Users/rasaneno/rundata/ZS2020/%s/',submission_name);
else
    ZSpath = [path_in '/' submission_name '/'];
end

if nargin <3
    ZS = [];
end

languages = {'english','french','mandarin','LANG1','LANG2'};
durations = {'1s','10s','120s'};

if ~exist(ZSpath,'dir')
    mkdir(ZSpath);
    mkdir([ZSpath '/2017/']);
    mkdir([ZSpath '/2017/code/']);
    mkdir([ZSpath '/2017/track1/']);
    for k = 1:length(languages)
        for j = 1:length(durations)
            mkdir([ZSpath sprintf('/2017/track1/%s/%s/',languages{k},durations{j})]);
        end
    end
    mkdir([ZSpath '/2017/track2/']);
else
    fprintf('Proposed submission template/folder structure already exists at %s\nOverwriting will destroy all existing data.\n',ZSpath);
    %resp = input('Continue? (y/n)','s');
    resp = 'y';
    if(strcmp(resp,'y'))
        fprintf('overwriting...\n');
        rmdir(ZSpath,'s');
        mkdir(ZSpath);
        mkdir([ZSpath '/2017/']);
        mkdir([ZSpath '/2017/code/']);
        fid = fopen([ZSpath '/2017/code/code.m'],'w');
        fprintf(fid,'placeholder for code.txt');
        fclose(fid);
        
        mkdir([ZSpath '/2017/track1/']);
        
        system('find /Users/rasaneno/rundata/ZS2020/ -name ".DS_Store" -delete');
        
        
        for k = 1:length(languages)
            for j = 1:length(durations)
                mkdir([ZSpath sprintf('/2017/track1/%s/%s/',languages{k},durations{j})]);
            end
        end
        mkdir([ZSpath '/2017/track2/']);
    else
        fprintf('aborting ...\n');
        return;
    end
end

% Create general metadata file for the submission
fid = fopen([ZSpath '/metadata.yaml'],'w');
fprintf(fid,'author: ');
fprintf(fid,'\t');
for j = 1:length(authors)-1
    fprintf(fid,[authors{j} ', ']);
end
fprintf(fid,[authors{end}]);        
fprintf(fid,'\n');
fprintf(fid,'affiliation: ');
fprintf(fid,['\t' affiliation '\n']);
fprintf(fid,'open source: ');
fprintf(fid,'\ttrue\n');
fclose(fid);

% Create ZS2017 challenge metadata file
fid = fopen([ZSpath '/2017/metadata.yaml'],'w');
fprintf(fid,'system description: ');
fprintf(fid,'  a brief description of your system, pointing to a paper where possible\n');
fprintf(fid,'hyperparameters: ');
fprintf(fid,'  values of all the hyperparameters\n');
fprintf(fid,'track1 supervised: ');
fprintf(fid,'  false\n');
fprintf(fid,'track2 supervised: ');
fprintf(fid,'  false\n');
fclose(fid);
% 
% 
% % If input data exists
% if(~isempty(ZS))
%     
%     languages = fields(ZS);
%     
%     for langiter = 1:length(languages)
%         
%         if(isfield(ZS.(languages{1}).test,'features_1') && isfield(ZS.(languages{1}).test,'features_1_t'))
%             filepath = [ZSpath sprintf('/2017/track1/%s/1s/',languages{k})];
%             writeTrack1Outputs(ZS.(languages{1}).test.filename_120,ZS.(languages{1}).test.features_1,ZS.(languages{1}).test.features_1_t,filepath);
%         end
%         
%         if(isfield(ZS.(languages{1}).test,'features_10') && isfield(ZS.(languages{1}).test,'features_10_t'))
%             filepath = [ZSpath sprintf('/2017/track1/%s/10s/',languages{k})];
%             writeTrack1Outputs(ZS.(languages{1}).test.filename_10,ZS.(languages{1}).test.features_10,ZS.(languages{1}).test.features_10_t,filepath);
%         end
%         
%         if(isfield(ZS.(languages{1}).test,'features_120') && isfield(ZS.(languages{1}).test,'features_120_t'))
%             filepath = [ZSpath sprintf('/2017/track1/%s/120s/',languages{k})];
%             writeTrack1Outputs(ZS.(languages{1}).test.filename_120,ZS.(languages{1}).test.features_120,ZS.(languages{1}).test.features_120_t,filepath);            
%         end
%         
%     end
%     
%     
% end




