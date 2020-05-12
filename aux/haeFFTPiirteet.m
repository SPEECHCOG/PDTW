function  F = haeFFTPiirteet(data_train,wl,ws,opfreq)

if nargin <4    
opfreq = 8000;
end

if nargin <3
    ws = 0.0125*opfreq;
else
    ws = round(ws*opfreq);
end
if nargin <3
    wl = 0.025*opfreq;
else
    wl = round(wl*opfreq);
end

ww = hamming(wl);
F = cell(length(data_train),1);

for signal = 1:length(data_train)
    %data_train{signal}
   %[x,fs] = wavread(data_train{signal});
   [x,fs] = audioread(data_train{signal});
   
   if(fs ~= opfreq)
       x = resample(x,opfreq,fs);
   end
      
   
   S = zeros(round(length(x)/ws)-2,wl/2+1);
   j = 1;
   for loc = 1:ws:length(x)-wl+1
       y = x(loc:loc+wl-1).*ww;
       
       y = 20*log10(abs(fft(y)));
       y = y(1:wl/2+1);
       
       S(j,:) = y;
       j = j+1;       
   end
   
   F{signal} = S;
       
    
end