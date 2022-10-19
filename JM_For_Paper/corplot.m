function [C,p,P,ttl,h] = corplot(x,y,P_opt,R_opt,col,col2)
%[C,p,P] = corplot(x,y,P_opt,R_opt,col,col2)
    if(nargin<6)
        col2 = 'r';
    end

    if(nargin<5)
        col = 'b';
    end

    if(nargin<4)
        R_opt = 1;
    end

    if(nargin<3)
        P_opt = 0;
    end
    
    hold off;
    h(1) = plot(x,y,'.','Color',col);
    x = reshape(x,1,[]);
    y = reshape(y,1,[]);
    [C,P,RLO,RUP] = corrcoef(x,y,'rows','complete');
    good = ~isnan(x) & ~isnan(y);
    p = polyfit(x(good),y(good),1);
    
    jAXIS2;
    if(P_opt==0)
        if(R_opt==1)
            ttl = sprintf('R = %.02f',C(2));
            title(ttl);
        else
            ttl = '';
        end
    else
        if(P(2)<1e-3)
            ttl = sprintf('R = %.02f, p = %.02e',C(2),P(2));
        else
            ttl = sprintf('R = %.02f, p = %.02g',C(2),P(2));
        end
        title(ttl);
    end
    if(R_opt==1)
        hold on;
        xl = xlim;
        h(2) = plot(xl,p(1)*xl+p(2),'Color',col2);
    end
    
    C = C(2);
    P = P(2);
end