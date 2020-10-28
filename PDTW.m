function patterns = PDTW(F,config)
% function patterns = PDTW(F,config)
%
% Probabilistic PDTW pattern matching from paper O. Rasanen & M.A. Cruz
% Blandon: "Unsupervised discovery of recurring speech patterns using 
% probabilistic adaptive metrics", In Proc.
% Interspeech-2020, Shanghai, China. 
%
% Please cite the above paper if you use this code or its derivatives.
%
% PDTW:
% Performs discovery of pairwise pattern matches from a corpus of speech.
%
%   Inputs:
%       F:  Nx1 cell array, where each cell contains features of training
%           utterance n from {1... N} (frames as rows, feature dim as columns).
%  config:  A struct containing model parameters, including:
%
%       Key parameters:
%
%       .alpha;             Significance threshold for detecting patterns
%                           (smaller is more stringent; default = 0.001)
%
%       .total_ram          How much RAM (in bytes) available for distance metric
%                           calculations (doesn't include overheads from 
%                           input data size etc.; default = 8e9)
% 
%       .write_location     Which directory to store intermediate results? 
%                           Set to [] to disable saving. Default: PDTW.m folder.
%
%       .process_id         Name of the process used in saving (default = 'default'). 
%                          
%
%       Other parameters:
% 
%       .seqlen:            Stage #1 segment length for matching ("L" in
%                           paper, default = 20)
%       .config.seqshift:   window shift for segments ("S" in paper,
%                           default = 10)
%       .nearest_to_check:  how many nearest neighbors to consider for each
%                           segment ("k" in paper, default = 5)
%       .duration_thr:      minimum duration of detected segments (default = 5) 
%       .n_slices:          number of frames per downsampled segment
%                           ("M" in paper, default = 4)
%       .expansion;         +- how many frames to expand segments for stage #2
%                           #2 DTW alignments (i.e., to length 2*expansion+seqlen; 
%                           "E" in paper default = 25)
% 
%   Outputs:
%           patterns:       a struct containing detected patterns. All but
%                           one field come in form [Nx2], where the two columns 
%                           correspond to two matcing patterns discovered
%
%                .signal: signal IDs for detected pairs (Nx2)
%                .onset:  pattern onset frames in the given signals (Nx2)
%                .offset: pattern offset frames in the given signals (Nx2)
%                .dist:   average alignment path probability 
%                         (smaller is better) (Nx1)
%                .id:     consequtive numbering of detected patterns (Nx2)
%
%                
% For questions and bug reports, contact okko.rasanen@tuni.fi. 

% Add paths
curdir = fileparts(which('PDTW.m'));
addpath([curdir '/aux/']);
addpath([curdir '/voicebox/'])
addpath([curdir '/dtw_ellis/'])

if(isfield(config,'nearest_to_check'))
    nearest_to_check = config.nearest_to_check; % how many nearest neighbors to consider for each segment
else
    nearest_to_check = 5;
end

if(isfield(config,'duration_thr'))
    duration_thr = config.duration_thr; % minimum duration of detected segments
else
    duration_thr = 5;
end

if(isfield(config,'n_slices'))
    n_slices = config.n_slices; % number of slices per downsampled segment
else
    n_slices = 4;
end

if(isfield(config,'seqshift'))
    seqshift = config.seqshift; % window shift for segments
else
    seqshift = 10;
end

if(isfield(config,'seqlen'))
    seqlen = config.seqlen; % initial segment length for matching
else
    seqlen = 20;
end
   

if(isfield(config,'expansion'))
    expansion = config.expansion; % how much to enlarge segments for stage #2
else
    expansion = 25;
end

if(isfield(config,'alpha'))
    alpha = config.alpha;
else
    alpha = 0.001;
    fprintf('PARAMETERS: alpha not provided in config. Using default alpha = 0.001.\n');
end

if(isfield(config,'total_ram'))
    total_ram = config.total_ram;
else
    total_ram = 8e9;
    fprintf('PARAMETERS: Amount of RAM not provided in config. Using default total_ram = 8GB.\n');
end

if(isfield(config,'write_location'))
    write_location = config.write_location;
else
    write_location = [fileparts(which('PDTW.m')) '/'];    
end

if(isfield(config,'run_parallel'))
    run_parallel = config.run_parallel;    
else
    run_parallel = 0;
end

if(isfield(config,'process_id'))
    process_id = config.process_id;
else
    process_id = 'default';
end
    
% Hard-coded parameters    
comps_GMM = 1;

if(isfield(config,'verbose'))
    verbose = config.verbose;
else    
    verbose = 1;
end

%% Actual process start


% Concatenate all signal features into one long "corpus"
totlen = sum(cellfun(@length,F));
F_all = zeros(totlen,size(F{1},2)); % feature matrix

F_ind = zeros(totlen,2); % [signal, frame] -pointers to allow recovery
wloc = 1;
for k = 1:length(F)   
    F_all(wloc:wloc+size(F{k},1)-1,:) = F{k};
    F_ind(wloc:wloc+size(F{k},1)-1,1) = k;
    F_ind(wloc:wloc+size(F{k},1)-1,2) = 1:length(F{k});
    wloc = wloc+size(F{k},1);
end

% Make sure there are no NaNs or Infs
F_all(isnan(F_all)) = 0;
F_all(isinf(F_all)) = 0;

% Slice into fixed-length segments
X = zeros(floor(totlen/seqshift)-10,seqlen,size(F_all,2));
X_ind = zeros(floor(totlen/seqshift)-10,seqlen,2);

wloc = 1;
k = 1;
while wloc < size(F_all,1)-seqlen+1
    X(k,:,:) = F_all(wloc:wloc+seqlen-1,:); % segments
    X_ind(k,:,:) = F_ind(wloc:wloc+seqlen-1,:);  % maintain indexing info on the side
    k = k+1;
    wloc = wloc+seqshift;
end

% Downsample each segment for faster processing
dim = size(X,3);
bounds = round(linspace(1,seqlen,n_slices+1));

X_ds = zeros(size(X,1),n_slices*dim);
for slice = 1:n_slices
    X_ds(:,(slice-1)*dim+1:slice*dim) = squeeze(mean(X(:,bounds(slice):bounds(slice+1),:),2));
end


%% Voice activity detection

% assumes that first feature coeff is correlated with signal energy, e.g.,
% MFCC0 or PCA1 of more advanced features
sildims = 1:size(F_all,2):size(X_ds,2);  

E = sum(X_ds(:,sildims),2); % get energy or proxy for energy for each segment

% cluster energies to two classes where the smaller cluster is assumed to be silence
w = 0; 
while(min(w) < 0.05) % ensure that both classes are representative of the data
    [c,v,w]=gaussmix(E(~isnan(E)),[],[],2);
end
d = pdist2(E,c);
[~,idx] = min(d,[],2);
counts = zeros(2,1);
counts(1) = sum(idx == 1);
counts(2) = sum(idx == 2);
[~,i] = max(counts);
prob_sil = 1-normcdf(E,c(i),sqrt(v(i))).*w(i);
X_ds(prob_sil > 0.99,:) = NaN; % set silent segments to NaNs if p(silence) > thr

if(verbose)
    fprintf('Calculating random distances...\n\n');
end

% Make a copy without NaNs to measure distance distributions
X_ds_tmp = X_ds;
X_ds_tmp(sum(isnan(X_ds_tmp),2) > 0,:) = [];

% How many computing nodes (count needed for RAM allocation per node)
if(~run_parallel)
    n_nodes = 1;
else
    p = gcp;
    n_nodes = p.NumWorkers;
end

% Data usually needs to be processed in chunks, so calculate chunk size
% according to available RAM and number of parallel nodes used.
tmp = total_ram./8/n_nodes; % this many numbers can be stored
chunksize = round(tmp/size(X_ds_tmp,1));
n_chunks = ceil(size(X_ds_tmp,1)/chunksize);

% Measure pairwise distances between all possible segments and collect
% random sample of distances

if(~run_parallel)    
    D_r = [];
    for chunk = 1:n_chunks
        wloc = (chunk-1)*chunksize+1;
        endpoint = min(wloc+chunksize-1,size(X_ds_tmp,1));
        warning off all
        tmp = pdist2(X_ds_tmp(wloc:endpoint,:),X_ds_tmp,'cosine');
        warning on all        
        r = randi(numel(tmp),size(tmp,1).*nearest_to_check,1);
        D_r = [D_r;tmp(r)];        
        procbar(chunk,n_chunks);
    end    
else    
    D_r = [];
    parfor chunk = 1:n_chunks
        wloc = (chunk-1)*chunksize+1;
        endpoint = min(wloc+chunksize-1,size(X_ds_tmp,1));
        warning off all
        tmp = pdist2(X_ds_tmp(wloc:endpoint,:),X_ds_tmp,'cosine');
        warning on all        
        r = randi(numel(tmp),size(tmp,1).*nearest_to_check,1);
        D_r = [D_r;tmp(r)];        
        procbar(chunk,n_chunks);
    end
end

if(verbose)
    fprintf('\nModeling random distance distribution with a GMM...\n\n');
end

% Fit a GMM to the distance distribution
D_r(isnan(D_r)) = [];
[c,v,w]=gaussmix(D_r,[],[],comps_GMM);
[c,i] = sort(c,'ascend');
v = v(i);
w = w(i);

% [a,b] = hist(D_r,150);
% figure(5);clf;hold on;hist(D_r,150);
%
% tmp1 = normpdf(b,c(1),sqrt(v(1))).*max(a).*0.2.*w(1);
% if(comps_GMM == 2)
%     tmp2 = normpdf(b,c(2),sqrt(v(2))).*max(a).*0.2.*w(2);
%     kik = (tmp1+tmp2).*max(a)./max(tmp1+tmp2);
% else
%     kik = (tmp1).*max(a)./max(tmp1);
% end
% plot(b,kik,'LineWidth',2,'Color','red');
% drawnow;


if(verbose)
    fprintf('Matching downsampled segments...\n\n');
end

% Find nearest segments to each segment using probabilistic distance

D = ones(size(X_ds,1),nearest_to_check).*NaN; % this stores distance probs
I = ones(size(X_ds,1),nearest_to_check).*NaN; % this stores candidate indices

% Data usually needs to be processed in chunks, so calculate chunk size
% according to available RAM and number of parallel nodes used.
tmp = total_ram./8/n_nodes/2; % this many numbers can be stored
chunksize = round(tmp/size(X_ds,1));
n_chunks = ceil(size(X_ds,1)/chunksize);

% How many neighboring segments to discard from each matching process
olap_frames = ceil(seqlen/seqshift/2+expansion/seqlen+5);

if(~run_parallel)    
    wloc = 1;
    for chunk = 1:n_chunks
        endpoint = min(wloc+chunksize-1,size(X_ds,1));
        warning off all        
        dist_near = pdist2(X_ds(wloc:endpoint,:),X_ds,'cosine'); % get distance
        warning on all
        
        % Prevent neighboring segments from being matched to each other
        for ll = -olap_frames:olap_frames
            for j = 1:size(dist_near,1)
                if(wloc-1+j+ll > 0 && wloc-1+j+ll < size(X_ds,1))
                    dist_near(j,wloc-1+j+ll) = NaN;
                end
            end
        end
                
        [dist_near,ind_near] = sort(dist_near,2,'ascend'); % find nearest
        
        dist_near = dist_near(:,1:nearest_to_check);
        ind_near = ind_near(:,1:nearest_to_check);
        
        ind_near(isnan(dist_near)) = NaN; 
        
        % Calculate likelihood that the data comes from the overall
        % distribution of distances
        
        % convert distances to probability that such a small distance
        % value is observed in the corpus 
        if(comps_GMM == 2)
            dataprob = normcdf(dist_near,c(1),sqrt(v(1))).*w(1)+normcdf(dist_near,c(2),sqrt(v(2))).*w(2);
        elseif(comps_GMM == 1)
            dataprob = normcdf(dist_near,c(1),sqrt(v(1)));
        else
            error('wrong number of GMM components');
        end
        
        % filter out those candidates that are too far
        dist_near(dataprob > alpha) = NaN;
        ind_near(dataprob > alpha) = NaN;
        
        % store info
        D(wloc:endpoint,:) = dist_near;
        I(wloc:endpoint,:) = ind_near;
        
        wloc = wloc+chunksize;
        
        procbar(chunk,n_chunks);
    end
    
else
    % same stuff as above but using parfor loops. Needs some special
    % indexing.
    D_c = cell(n_chunks,1);
    I_c = cell(n_chunks,1);
    
    chunk_indices = cell(n_chunks,1);
    for chunk = 1:n_chunks
        wloc = (chunk-1)*chunksize+1;
        endpoint = min(wloc+chunksize-1,size(X_ds,1));
        chunk_indices{chunk} = wloc:endpoint;
        D_c{chunk} = cell(length(chunk_indices{chunk}),nearest_to_check);
        I_c{chunk} = cell(length(chunk_indices{chunk}),nearest_to_check);
    end
         
    
    parfor chunk = 1:n_chunks
        inds = chunk_indices{chunk};
        warning off all
        dist_near = pdist2(X_ds(inds,:),X_ds,'cosine');        
        wloc = inds(1);
        % Prevent frames overlapping with the current frame to be matched
        for ll = -olap_frames:olap_frames
            for j = 1:size(dist_near,1)
                if(wloc-1+j+ll > 0 && wloc-1+j+ll < size(X_ds,1))            
                    dist_near(j,wloc-1+j+ll) = NaN;
                end
            end
        end
        
        [dist_near,ind_near] = sort(dist_near,2,'ascend');
        
        dist_near = dist_near(:,1:nearest_to_check);
        ind_near = ind_near(:,1:nearest_to_check);
        
        ind_near(isnan(dist_near)) = NaN;
        
        % Calculate likelihood that the data comes from the overall
        % distribution of distances
        
        if(comps_GMM == 2)
            dataprob = normcdf(dist_near,c(1),sqrt(v(1))).*w(1)+normcdf(dist_near,c(2),sqrt(v(2))).*w(2);
        elseif(comps_GMM == 1)
            dataprob = normcdf(dist_near,c(1),sqrt(v(1)));
        else
            error('wrong number of GMM components');
        end
        
        dist_near(dataprob > alpha) = NaN;
        ind_near(dataprob > alpha) = NaN;
        warning on all
        D_c{chunk} = dist_near;
        I_c{chunk} = ind_near;
        
        procbar(chunk,n_chunks);
    end
    
    for chunk = 1:n_chunks
        inds = chunk_indices{chunk};
        D(inds,:) = D_c{chunk};
        I(inds,:) = I_c{chunk};
    end       
end

% Make sure that the same pair does not occur twice in the alternative order
for k = 1:size(I,1)
    for j = 1:size(I,2)
       tmp = I(k,j);
       if(~isnan(tmp))
       I(tmp,I(tmp,:) == k) = NaN;
       end
    end    
end

if(verbose)
fprintf('\nFound %d pattern candidates.\n',sum(~isnan(I(:))));
end


% Save intermediate state
tmp = sprintf('%s',alpha);
thrname = [tmp(1:3) tmp(end-3:end)];

if(~isempty(write_location))
    save([write_location sprintf('%s_pattern_candidates_%s.mat',process_id,thrname)],'process_id','I','D');
end


%% Stage 2: High-resolution alignment with probabilistic DTW
% Calculate statistics on pairwise frame-level distances
if(verbose)
    fprintf('Calculating chance-level for high resolution alignments...\n\n');
end

alldist = zeros(2000000,1); % take ~2M random samples from distances
wloc = 1;

while wloc < 2000000
    
    k = randi(size(X_ind,1));
    
    a = X_ind(k,:,1);
    b = X_ind(k,:,2);
    signal = mode(a);
    b(a ~= signal) = [];
    
    b = max(1,min(b)-expansion):min(max(b)+expansion,size(F{signal},1));
    
    Y1 = F{signal}(b,:);
        
    k2 = k;
    while k2 == k
        k2 = randi(size(X_ind,1),1);
    end

    a2 = X_ind(k2,:,1);
    b2 = X_ind(k2,:,2);
    signal2 = mode(a2);
    b2(a2 ~= signal2) = [];
    b2 = max(1,min(b2)-expansion):min(max(b2)+expansion,size(F{signal2},1));
    Y2 = F{signal2}(b2,:);
    
    M = pdist2(Y1,Y2,'cosine');
    
    alldist(wloc:wloc+numel(diag(M))-1) = diag(M);
    wloc = wloc+numel(diag(M));
            
end

alldist(isnan(alldist)) = [];

% Model distances with a GMM
[mu,sigma,weight]=gaussmix(alldist,[],[],comps_GMM);

[mu,i] = sort(mu,'ascend');
sigma = sigma(i);
weight = weight(i);

%% Calculate real alignment paths for the candidates from Stage #1.

if(verbose)
    fprintf('Aligning well-matching segments with a higher resolution...\n\n\n\n');
end

% Surrogate path probabilities using the given significance value for
% different length paths
chance_probs = zeros(1000,1);
for k = 1:1000
   chance_probs(k) = sum(log(alpha^k));
end

patterns = struct();

patterns.onset = zeros(1e7,2);
patterns.offset = zeros(1e7,2);
patterns.signal = zeros(1e7,2);
patterns.dist = zeros(1e7,1);
patterns.id = zeros(1e7,2);

patcount = 1;
idcount = 1;

if(run_parallel == 0)
    for k = 1:size(X_ind,1)
        
        % Find segment #1 from the original features
        a = X_ind(k,:,1);
        b = X_ind(k,:,2);
        signal1 = mode(a);
        b(a ~= signal1) = [];
        
        % Expand in time
        b = max(1,min(b)-expansion):min(max(b)+expansion,size(F{signal1},1));
        
        Y1 = F{signal1}(b,:); % segment #1 after expansion
        
        for n = 1:nearest_to_check % go through all candidate pairs for segment #1
            if(~isnan(I(k,n)))
                % find segment #2 similarly to above
                k2 = I(k,n);
                a2 = X_ind(k2,:,1);
                b2 = X_ind(k2,:,2);
                signal2 = mode(a2);
                b2(a2 ~= signal2) = [];
                b2 = max(1,min(b2)-expansion):min(max(b2)+expansion,size(F{signal2},1));
                Y2 = F{signal2}(b2,:); % segment #2 after expansion
                
                M = pdist2(Y1,Y2,'cosine'); % frame-wise affinity matrix
                
                % convert to probabilities
                if(comps_GMM == 2)
                    M = normcdf(M,mu(1),sqrt(sigma(1))).*weight(1)+normcdf(M,mu(2),sqrt(sigma(2))).*weight(2);
                else
                    M = normcdf(M,mu(1),sqrt(sigma(1))).*weight(1);
                end
                
                % find alignment path through the affinity-probability matrix
                [p,q] = dp2(M); % do DTW
                
                
                % Path alignment cost across the entire path
                shortpath = zeros(length(p),1);
                for t = 1:length(p)
                    shortpath(t) = M(p(t),q(t));
                end
                
                % Calculate likelihood ratio for all possible sub-paths
                % along the entire path
                c = zeros(length(shortpath),length(shortpath));                
                shortpath_log = log(shortpath);
                
                for leftbound = 1:length(shortpath)
                    for rightbound = leftbound+1:length(shortpath)
                        prob_path = sum(shortpath_log(leftbound:rightbound));
                        c(leftbound,rightbound) = prob_path-chance_probs(rightbound-leftbound);
                    end
                end
                
                % Best limit minimizes the likelihood ratio
                [leftbound,rightbound] = find(c == min(c(:)));
                
                % If path is long enough, save.
                if(rightbound-leftbound+1 >= duration_thr)
                    
                    onset_1 = b(p(leftbound));
                    end_1 = b(p(rightbound));
                    
                    onset_2 = b2(q(leftbound));
                    end_2 = b2(q(rightbound));
                    % Visualize paths?
                    %
                    %                 figure(2);clf;
                    %                 imagesc(M);
                    %                 hold on;
                    %                 for t = leftbound:rightbound
                    %                     if(t < length(p))
                    %                         plot(q(t:t+1),p(t:t+1),'Color','red','LineWidth',2);
                    %                     end
                    %                 end
                    %                 drawnow;pause;
                    %
                    patterns.onset(patcount,1) = onset_1;
                    patterns.offset(patcount,1) = end_1;
                    patterns.signal(patcount,1) = signal1;
                    patterns.id(patcount,1) = idcount;
                    patterns.onset(patcount,2) = onset_2;
                    patterns.offset(patcount,2) = end_2;
                    patterns.signal(patcount,2) = signal2;
                    patterns.id(patcount,2) = idcount+1;
                    patterns.dist(patcount,1) = mean(shortpath(leftbound:rightbound));
                    patcount = patcount+1;
                    idcount = idcount+2;
                end
            end
        end
        if(mod(k,100) == 0)
            procbar(k,size(X_ind,1));
        end
    end
else
    % Same stuff as above, but again with parfor loops. Needs slightly
    % different way of handling variables. Otherwise the same principle.
    PONSET1 = zeros(size(X_ind,1),nearest_to_check);
    PONSET2 = zeros(size(X_ind,1),nearest_to_check);
    POFFSET1 = zeros(size(X_ind,1),nearest_to_check);
    POFFSET2 = zeros(size(X_ind,1),nearest_to_check);
    SIGNAL1 = zeros(size(X_ind,1),nearest_to_check);
    SIGNAL2 = zeros(size(X_ind,1),nearest_to_check);
    DIST = zeros(size(X_ind,1),nearest_to_check);
    parfor k = 1:size(X_ind,1)
        
        a = X_ind(k,:,1);
        b = X_ind(k,:,2);
        signal1 = mode(a);
        b(a ~= signal1) = [];
        
        b = max(1,min(b)-expansion):min(max(b)+expansion,size(F{signal1},1));
        
        Y1 = F{signal1}(b,:);
        
        for n = 1:nearest_to_check
            if(~isnan(I(k,n)))
                k2 = I(k,n);
                a2 = X_ind(k2,:,1);
                b2 = X_ind(k2,:,2);
                signal2 = mode(a2);
                b2(a2 ~= signal2) = [];
                b2 = max(1,min(b2)-expansion):min(max(b2)+expansion,size(F{signal2},1));
                Y2 = F{signal2}(b2,:);
                
                M = pdist2(Y1,Y2,'cosine');
                if(comps_GMM == 2)
                    M = normcdf(M,mu(1),sqrt(sigma(1))).*weight(1)+normcdf(M,mu(2),sqrt(sigma(2))).*weight(2);
                else
                    M = normcdf(M,mu(1),sqrt(sigma(1))).*weight(1);
                end
                
                [p,q] = dp2(M); % do DTW
                
                shortpath = zeros(length(p),1);
                for t = 1:length(p)
                    shortpath(t) = M(p(t),q(t));
                end
                
                c = zeros(length(shortpath),length(shortpath));
                
                shortpath_log = log(shortpath);
                
                for leftbound = 1:length(shortpath)
                    for rightbound = leftbound+1:length(shortpath)
                        prob_path = sum(shortpath_log(leftbound:rightbound));
                        c(leftbound,rightbound) = prob_path-chance_probs(rightbound-leftbound);
                    end
                end
                
                [leftbound,rightbound] = find(c == min(c(:)));
                
                if(rightbound-leftbound+1 >= duration_thr)
                    
                    onset_1 = b(p(leftbound));
                    end_1 = b(p(rightbound));
                    
                    onset_2 = b2(q(leftbound));
                    end_2 = b2(q(rightbound));
                    % Visualize paths?
                    %
                    %                 figure(2);clf;
                    %                 imagesc(M);
                    %                 hold on;
                    %                 for t = leftbound:rightbound
                    %                     if(t < length(p))
                    %                         plot(q(t:t+1),p(t:t+1),'Color','red','LineWidth',2);
                    %                     end
                    %                 end
                    %                 drawnow;pause;
                    %
                    PONSET1(k,n) = onset_1;
                    PONSET2(k,n) = onset_2;
                    POFFSET1(k,n) = end_1;
                    POFFSET2(k,n) = end_2;
                    SIGNAL1(k,n) = signal1;
                    SIGNAL2(k,n) = signal2;
                    DIST(k,n) = mean(shortpath(leftbound:rightbound));
                else
                    DIST(k,n) = NaN;                
                end
            else
                DIST(k,n) = NaN;
            end
        end
        if(mod(k,100) == 0)
            procbar(k,size(X_ind,1));
        end
    end
    
    patcount = 1;
    idcount = 1;
    for k = 1:size(PONSET1,1)        
        for n = 1:size(PONSET1,2)
            if(~isnan(DIST(k,n)))
                if(DIST(k,n) ~= 0)
                patterns.onset(patcount,1) = PONSET1(k,n);
                patterns.onset(patcount,2) = PONSET2(k,n);
                patterns.offset(patcount,1) = POFFSET1(k,n);
                patterns.offset(patcount,2) = POFFSET2(k,n);
                patterns.signal(patcount,1) = SIGNAL1(k,n);
                patterns.signal(patcount,2) = SIGNAL2(k,n);
                patterns.dist(patcount,1) = DIST(k,n);
                patterns.id(patcount,1) = idcount;
                patterns.id(patcount,2) = idcount+1;
                idcount = idcount+2;
                patcount = patcount+1;
                end
            end
        end        
    end    
end

% Prune unused values from pre-allocated variables
ff = fields(patterns);
for f = 1:length(ff)
    patterns.(ff{f}) = patterns.(ff{f})(1:patcount-1,:);    
end

% Save results
if(~isempty(write_location))    
    tmp = sprintf('%s',alpha);
    thrname = [tmp(1:3) tmp(end-3:end)];
    save([write_location sprintf('%s_patterns_%s.mat',process_id,thrname)],'patterns','process_id','I');
end
