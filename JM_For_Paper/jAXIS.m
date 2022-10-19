axis tight;
temp_jAXIS = range(ylim);
if(min(ylim)~=0)
    ylim([min(ylim)-temp_jAXIS/20 max(ylim)+temp_jAXIS/20]);
else
    ylim([min(ylim) max(ylim)+temp_jAXIS/20]);
end
