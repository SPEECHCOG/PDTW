function F_delta = getDeltas(F,deltawidth)
if nargin <2
    deltawidth = 2;
end


% Get deltas and deltadeltas
F_delta = cell(size(F));

for signal = 1:length(F)
    F_delta{signal} = zeros(size(F{signal},1),13);
    for k = 3:size(F{signal},1)-2;
         
        divisor = 0;
        dt = zeros(1,size(F{signal},2));
        for n = 1:deltawidth
           dt = dt+n.*(F{signal}(k+n,:)-F{signal}(k-n,:));
           divisor = divisor+n.^2;
        end
        dt = dt./(2*divisor);               
        F_delta{signal}(k,:) = dt;
        
    end
    procbar(signal,length(F));
end