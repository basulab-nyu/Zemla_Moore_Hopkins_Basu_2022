function [V, M, CI, P] = violinStats(dataCell, dotSize, violinColors, lw, sigFun)

    if(nargin<5)
        sigFun = @(x,y) ranksum(x,y);
    end

    if(nargin<4)
        lw = 2;
    end    
    
    n = length(dataCell);
    group = cell2mat(arrayfun(@(x) x*ones(size(dataCell{x})), 1:n, 'UniformOutput', false));    

    V = violinplot(cell2mat(dataCell),group);
    for i_violin = 1:n
        V(i_violin).ScatterPlot.SizeData = dotSize;
        V(i_violin).ViolinColor = violinColors{i_violin};
    end
    
    M = NaN(n,1);
    CI = NaN(n,2);
    for i_group = 1:n
        [m,ci] = jmedian(dataCell{i_group});
        M(i_group) = m;
        CI(i_group,:) = ci;
    end

    P = NaN(n,n);
    for i_A = 1:n-1
        for i_B = i_A+1:n
            P(i_A, i_B) = sigFun(dataCell{i_A}, dataCell{i_B});
        end
    end

    for i_group = 1:n
        plot((i_group-0.4)*[1 1],CI(i_group,:),'k','Linewidth',lw);
    end 
end