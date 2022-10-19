function [m, ci] = jmedian(data,N)
% [m ci] = jmedian(data,N)

    if(nargin<2)
%         N = 100000;
        N = 10000;
    end
    
    data = data(~isnan(data));
    if(isempty(data))
        m = NaN;
        ci = [NaN NaN];
    else
        m = nanmedian(data);

        idx = ceil(length(data)*rand(length(data),N));
        data2 = data(idx);

        m2 = nanmedian(data2);

        ci = quantile(m2,[0.025 0.975]);
    end
end