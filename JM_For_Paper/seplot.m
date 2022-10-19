function [h] = seplot(dt,data,color,ebars)
    %[h] = splot(dt,data,color,ebars)
    if(nargin<4)
        ebars = 0;
    end

    if(nargin<3)
        color = 'b';
    end

    
    if(numel(dt)==1)
        t = dt*(1:size(data,2));
    else
        t = dt;
    end
    
    
    h(1) = plot(t,nanmean(data,2),'Linewidth',2,'Color',color);
    hold on;

    N = sum(~isnan(data),2);
    if(ebars==1)
        h(2) = plot(t,nanmean(data,2)+nanstd(data,[],2)./sqrt(N),'Linewidth',1,'Color',color);
        h(3) = plot(t,nanmean(data,2)-nanstd(data,[],2)./sqrt(N),'Linewidth',1,'Color',color);
    end

end