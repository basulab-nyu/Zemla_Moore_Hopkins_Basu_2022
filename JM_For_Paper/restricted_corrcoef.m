function [C] = restricted_corrcoef(A,B)
    C = NaN(size(A,1),size(B,1));
    

    for i_A = 1:size(A,1)
        thisA = A(i_A,:);        
        for i_B = i_A
            thisB = B(i_B,:);        
        
            valid = thisA>0 & thisB>0;
            if(sum(valid)>2)
                c = corrcoef(thisA(valid), thisB(valid));
                C(i_A,i_B) = c(1,2);
            end
        end
    end

end