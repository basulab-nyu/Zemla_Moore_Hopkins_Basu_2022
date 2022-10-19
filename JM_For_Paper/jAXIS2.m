jAXIS;
xmin = min(xlim());
xmax = max(xlim());
factor = [0 0];
if(xmin<0)
    factor(1) = 1.1;
else
    factor(1) = 0.9;
end

if(xmax<0)
    factor(2) = 0.9;
else
    factor(2) = 1.1;
end
xlim(factor.*xlim());