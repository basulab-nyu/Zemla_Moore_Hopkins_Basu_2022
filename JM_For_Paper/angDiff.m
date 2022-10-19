function [d] = angDiff(ang1,ang2)
    %[d] = angDiff(ang1,ang2)
    
    d = angle(exp(1i*(ang1-ang2)));

end