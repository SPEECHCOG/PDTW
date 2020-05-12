function F_MVN = MVN(F)
% function F_MVN = MVN(F)
% Mean and variance normalization

for k = 1:length(F)
    F{k} = F{k}-repmat(nanmean(F{k}),size(F{k},1),1);
    F{k} = F{k}./repmat(nanstd(F{k}),size(F{k},1),1);    
end

F_MVN = F;

