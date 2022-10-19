function [T] = markSig2(p,xxx,SPACING1,SPACING2)   
    %[] = markSig2(p,xxx,SPACING1,SPACING2)   
    if(nargin<2)
        xxx = 1:size(p,2);
    end
    xxx = xxx+0.2;
    yt = get(gca,'ytick');
    YL = ylim;
    YR = range(YL);
    N = length(xxx);
%     ncomparisons = (N^2-N)/2;
    nrows = N-1;
    
    if(nargin<3)
        SPACING1 = 0.06;
    end
    if(nargin<4)
        SPACING2 = 0.09;
    end
%     YR2 = YR*(1+SPACING1*(ncomparisons+1));

%     YR2 = YR*(1+SPACING1*(nrows+1));
%     YL2 = [YL(1) YL(1)+YR2];

%     ylim(YL2);
  
    tops = (0.99*max(ylim)-SPACING1*YR*(nrows-1)):(SPACING1*YR):(0.99*max(ylim));
    bottoms = tops-SPACING2*SPACING1*YR;
    
%     steps = YR*SPACING1*(0:ncomparisons);
%     steps = YR*SPACING1*(0:nrows);
%     bottoms = steps(1:end-1)+SPACING2*mean(diff(steps));
%     tops = steps(2:end)-SPACING2*mean(diff(steps));
    
    
    fz = get(gca,'fontsize');
    
    count = 1;
    count2 = 1;
    for h=1:N-1
        elements = diag(p,h);
        for i=1:length(elements)
            plot([xxx(i)+0.05 xxx(i)+0.05],[bottoms(count) tops(count)],'k','Linewidth',0.5);
            plot([xxx(i+h)-0.05 xxx(i+h)-0.05],[bottoms(count) tops(count)],'k','Linewidth',0.5);
            plot([xxx(i)+0.05 xxx(i+h)-0.05],[tops(count) tops(count)],'k','Linewidth',0.5);
            
            if(isnan(p(i,i+h)))
                T{count2} = text(x,max(ylim),'NAN P-VALUE','Fontsize',5*fz,'HorizontalAlign','Center','VerticalAlign','Bottom');
                count2 = count2+1;
            elseif(p(i,i+h)<0.05)
                T{count2} = text(mean([xxx(i) xxx(i+h)]),tops(count),'*','Fontsize',2.5*fz,'HorizontalAlign','Center','VerticalAlign','Middle');
                count2 = count2+1;
            else
                T{count2} = text(mean([xxx(i) xxx(i+h)]),tops(count),'n.s.','Fontsize',fz,'HorizontalAlign','Center','VerticalAlign','Bottom');
                count2 = count2+1;
            end                        
        end
        count = count+1;
    end
    
%     count = 1;
%     for i=1:N
%         for j=i+1:N
%             plot([xxx(i) xxx(i)],YL(2)+[bottoms(count) tops(count)],'k');
%             plot([xxx(j) xxx(j)],YL(2)+[bottoms(count) tops(count)],'k');
%             plot([xxx(i) xxx(j)],YL(2)+[tops(count) tops(count)],'k');
%             
%             if(p(i,j)<0.05)
%                 text(mean([xxx(i) xxx(j)]),YL(2)+tops(count),'*','Fontsize',2*fz,'HorizontalAlign','Center','VerticalAlign','Middle');
%             else
%                 text(mean([xxx(i) xxx(j)]),YL(2)+tops(count),'n.s.','Fontsize',fz,'HorizontalAlign','Center','VerticalAlign','Middle')
%             end
%             count = count+1;
%         end
%     end
    
%     x = mean(xlim);
%     if(nargin<3)
%         fz = 10;
%     end
%     if(p<0.05)
%         text(x,max(ylim),'*','Fontsize',2*fz,'HorizontalAlign','Center','VerticalAlign','Top')
%     else
%         text(x,max(ylim),'n.s.','Fontsize',fz,'HorizontalAlign','Center','VerticalAlign','Top')
%     end
% 
%     ylim(YL2);
%     set(gca,'ytick',yt);
    
end