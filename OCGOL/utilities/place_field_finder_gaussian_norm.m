function [Place_cell] = place_field_finder_gaussian_norm(Place_cell,options)

%% Disable min peak height warning for peak finder fxn

warning_id = 'signal:findpeaks:largeMinPeakHeight';
warning('off',warning_id);

%% Set parameters

%find place fields based on Zaremba et al. 2017
%gaussian kernel with sigma = 3 spatial bins

%use rate map - number of event onsets/ occupancy across all laps
gSigma = options.gSigma;
%rename option variable to use for input for define_Gaussian_kernel fxn
options.sigma_filter = options.gSigma;

% %if to exclude based on center of the located peak
% centerExclude = options.centerExclude;
% 
% %peak distance separation (in bins)
% %not used if centerExclude is set to 0
% peakDistance = options.peakDistance;
% 
% %fraction of each peak to define as left and right edge of each Gaussian
% %fit
% peakFraction = options.peakFraction;
% 
% %how large the area has to be of other fields in comparison to the the
% %place field with the largest area;
% gaussAreaThreshold = options.gaussAreaThreshold;
% 
% %whether to plot the individual place fields
% plotFields = options.plotFields;
% 
% %offset necessary from using extended rate map
% offset =50;

%which place cell cell to run extraction on
place_struct_nb = options.place_struct_nb;

%% Non-extended Gaussian smoothed map

%% Gaussian kernel smoothing filter
%using convolution with custom window
%size of window
gaussFilter = define_Gaussian_kernel(options);

%% Smooth extended_rate_map (input to determining place field

ex_rate_map = Place_cell{place_struct_nb}.Spatial_Info.extended_rate_map;

%smooth extended rate map for each ROI
for rr = 1:size(ex_rate_map,2)
    ex_rate_map_sm(:,rr) = conv(ex_rate_map(:,rr),gaussFilter, 'same');
end

%normalized smoothed gaussian to 1
%get min and max matrix for each gaussian smoothed curve
min_smooth_mat = repmat(min(ex_rate_map_sm,[],1),size(ex_rate_map_sm,1),1);
max_smooth_mat = repmat(max(ex_rate_map_sm,[],1),size(ex_rate_map_sm,1),1);

%normalized smoothed matrix
ex_rate_map_sm_norm = (ex_rate_map_sm - min_smooth_mat)./(max_smooth_mat-min_smooth_mat);

%clip the extended map to get regular length map for visualization after
%processing
rate_map_sm = ex_rate_map_sm(51:150,:);


%% !Replace subsequent input with normalized smoothing!

rate_map_sm = ex_rate_map_sm_norm(51:150,:);
ex_rate_map_sm = ex_rate_map_sm_norm;

%% Find local maxima - (maxima must be between bin 51 and 150)

%[pks,lc] = findpeaks(ex_rate_map_sm(:,1),'MinPeakDistance', peakDistance);
%without min separation
%[pks,lc] = findpeaks(ex_rate_map_sm(:,1));

%find local maxima for each ROI
for rr=1:size(ex_rate_map,2)
    [pks{rr},lc{rr},w{rr}] = findpeaks(ex_rate_map_sm(:,rr),'WidthReference','halfprom',...
        'MinPeakHeight',0.15,'Threshold',0,'MinPeakWidth',4);
end


%% Discard peaks that are within (<=) 15 bins of edge of extended rate map
for rr=1:size(ex_rate_map,2)
    discard_pks = lc{rr} <= 15;
    lc{rr}(discard_pks) =[];
    pks{rr}(discard_pks) =[];
    w{rr}(discard_pks) =[];
end

%% Remove pks with width < 1.5 (~3 bins)
for rr=1:size(ex_rate_map,2)
    discard_pks = w{rr} <= 1.5;
    lc{rr}(discard_pks) =[];
    pks{rr}(discard_pks) =[];
    w{rr}(discard_pks) =[];
end

%% Exclude peaks that are beyond the boundaries of the complete laps (not in extention zone)
for rr=1:size(ex_rate_map,2)
    discard_pks = lc{rr} < 51 | lc{rr} > 150;
    lc{rr}(discard_pks) =[];
    pks{rr}(discard_pks) =[];
    w{rr}(discard_pks) =[];
end


%% Plot to visualize maxima - skip; move advanced plotting beloq
if 0
    figure;
    for rr=282%1:size(ex_rate_map,2)
        hold on;
        %plot smoothed rate map
        plot(ex_rate_map_sm(:,rr),'k');
        %plot start and end point of bin
        stem([51,150],[1 1],'b');
        %plot peaks
        stem(lc{rr},pks{rr},'r')
        %plot edges based on width return
        %start
        stem(lc{rr}-(w{rr}./2),ones(size(lc{rr},1),1),'g');
        %end
        stem(lc{rr}+(w{rr}./2),ones(size(lc{rr},1),1),'g');
        pause;
        clf;
    end
end

%% Select ROIS signficant tuned by SI score

SI_tuned_ROIs = find(Place_cell{place_struct_nb}.Spatial_Info.significant_ROI == 1);
%do for all and then select tuned ones at the end

%% Fit single term gaussian to id'd local maxima

%number of bins to extend plotting of fit gauss beyond the intial width
%from findpeaks
%need to make this as some sort of fraction of peak
%try as FWHM of each gaussian
%c parame
gauss_extend = 15;
tic;
%for rr = SI_tuned_ROIs
%for all ROIs
for rr=1:size(pks,2)

    if ~isempty(pks{rr})
        
        %for each id'd peak
        for peak_nb =1:size(pks{rr})
            %x, y input
            %for each local maximum
            %original width for input to fit
            loc_range{rr}{peak_nb} = [round(lc{rr}(peak_nb)-(w{rr}(peak_nb)./2)):round(lc{rr}(peak_nb)+(w{rr}(peak_nb)./2))];
            %loc_range{rr}{peak_nb} = [round(lc{rr}(peak_nb)-(w{rr}(peak_nb)):round(lc{rr}(peak_nb)+(w{rr}(peak_nb))))];
            curve_range{rr}{peak_nb} = ex_rate_map_sm(loc_range{rr}{peak_nb},rr);
            
            %fit gaussian to peak
            [f{rr}{peak_nb},gof{rr}{peak_nb},output{rr}{peak_nb}] = fit(loc_range{rr}{peak_nb}', ex_rate_map_sm(loc_range{rr}{peak_nb},rr),'gauss1');
            gauss_fit{rr}{peak_nb} = f{rr}{peak_nb}.a1*exp(-(([loc_range{rr}{peak_nb}]-f{rr}{peak_nb}.b1)./f{rr}{peak_nb}.c1).^2);
            gauss_fit{rr}{peak_nb} =f{rr}{peak_nb}.a1*exp(-(([loc_range{rr}{peak_nb}(1)-gauss_extend:loc_range{rr}{peak_nb}(end)+gauss_extend]-f{rr}{peak_nb}.b1)./f{rr}{peak_nb}.c1).^2);
            %full width at half maximum for each peak
            gauss_fwhm{rr}(peak_nb) = round(2*sqrt(2*log(2))*f{rr}{peak_nb}.c1);
            gauss_fit_bin_range{rr}{peak_nb} = [loc_range{rr}{peak_nb}(1)-gauss_extend:loc_range{rr}{peak_nb}(end)+gauss_extend];
        end
    else %fill with empty values
        loc_range{rr} = [];
        curve_range{rr} = [];
        
        f{rr} = []; gof{rr} = []; output{rr} = [];
        gauss_fit{rr} = [];
        gauss_fwhm{rr} = [];

    end
end
toc;

%[x{rr}{peak_nb},y{rr}{peak_nb},z{rr}{peak_nb}] = fit(loc_range{rr}{peak_nb}(2:end)', ex_rate_map_sm(loc_range{rr}{peak_nb}(2:end),rr),'gauss1');

%% Plot - split into 2 subplots with smoothed rate and non-smoothed rate
%skip for now - add as option later for display
if 0
    figure;
    for rr =90 %SI_tuned_ROIs%1:100
        subplot(2,1,1)
        if ~isempty(pks{rr})
            %for each id'd peak
            for peak_nb =1:size(pks{rr})
                %plot range
                hold on
                title(num2str(rr));
                ylim([0 1.2])
                %plot extended rate map (smoothed)
                plot(ex_rate_map_sm(:,rr),'k')
                
                %plot extended rate map (non-smoothed)
                %plot(ex_rate_map(:,rr),'k-');
                %plot peak and width ends
                %plot peak center
                stem(lc{rr}(peak_nb),pks{rr}(peak_nb),'r')
                %plot width around peak
                %start
                stem(lc{rr}(peak_nb)-(w{rr}(peak_nb)./2),pks{rr}(peak_nb)*ones(size(lc{rr}(peak_nb),1),1),'g');
                %end
                stem(lc{rr}(peak_nb)+(w{rr}(peak_nb)./2),pks{rr}(peak_nb)*ones(size(lc{rr}(peak_nb),1),1),'g');
                %plot gaussian fit to ROI peaks
                plot([loc_range{rr}{peak_nb}(1)-gauss_extend:loc_range{rr}{peak_nb}(end)+gauss_extend],gauss_fit{rr}{peak_nb},'m')
            end
        end
        
        %plot start and end point of bin
        stem([51,150],[1 1],'b');
        %cutoff transient rate refline
        cutoff_line = refline(0,0.1);
        cutoff_line.Color = [0.5 0.5 0.5];
        cutoff_line.LineStyle = '--';
        %second cut off line
        cutoff_line2 = refline(0,0.05);
        cutoff_line2.Color = [0.5 0.5 0.5];
        cutoff_line2.LineStyle = '--';
        
        subplot(2,1,2)
        
        hold on
        if ~isempty(pks{rr})
            %for each id'd peak
            for peak_nb =1:size(pks{rr})
                %plot range
                hold on
                title(num2str(rr));
                ylim([0 1.2])
                %plot extended rate map (smoothed)
                %plot(ex_rate_map_sm(:,rr),'k')
                
                %plot extended rate map (non-smoothed)
                plot(ex_rate_map(:,rr),'k-');
                %plot peak and width ends
                %plot peak center
                stem(lc{rr}(peak_nb),pks{rr}(peak_nb),'r')
                %plot width around peak
                %start
                stem(lc{rr}(peak_nb)-(w{rr}(peak_nb)./2),pks{rr}(peak_nb)*ones(size(lc{rr}(peak_nb),1),1),'g');
                %end
                stem(lc{rr}(peak_nb)+(w{rr}(peak_nb)./2),pks{rr}(peak_nb)*ones(size(lc{rr}(peak_nb),1),1),'g');
                %plot gaussian fit to ROI peaks
                plot([loc_range{rr}{peak_nb}(1)-gauss_extend:loc_range{rr}{peak_nb}(end)+gauss_extend],gauss_fit{rr}{peak_nb},'m')
            end
        end
        
        %plot start and end point of bin
        stem([51,150],[1 1],'b');
        %cutoff transient rate refline
        cutoff_line = refline(0,0.1);
        cutoff_line.Color = [0.5 0.5 0.5];
        cutoff_line.LineStyle = '--';
        %pause
        %clf
    end
end

%% Need a check here to a see if all curves have a proper fit; if not, narrow range of fit for that curve by 25%
%check that gaussian curves fit - check that the maximum of the derivative
%of the fitted curve is 
%RESUME HERE
count_nb = 0;
%how many bins to extend on each end when trying to re-adjust Gaussian fit
adjust_width = -2;

for rr = 1:size(pks,2) %SI_tuned_ROIs%1:100
    if ~isempty(pks{rr})
        %for each id'd peak
        for peak_nb =1:size(pks{rr})
            if max(diff(gauss_fit{rr}{peak_nb})) < 0.001
                count_nb = count_nb+1;
                disp(rr)
                disp(peak_nb);
                %re-fit Gaussian using narrowed range
                gauss_extend = 15;
                %same code as above with the width of peak narrow by 2 each end
                disp('extended')
                %x, y input
                %for each local maximum
                %original width for input to fit
                loc_range{rr}{peak_nb} = [round(lc{rr}(peak_nb)-((w{rr}(peak_nb))./2)-adjust_width):round(lc{rr}(peak_nb)+((w{rr}(peak_nb))./2)+adjust_width)];
                %loc_range{rr}{peak_nb} = [round(lc{rr}(peak_nb)-(w{rr}(peak_nb)):round(lc{rr}(peak_nb)+(w{rr}(peak_nb))))];
                curve_range{rr}{peak_nb} = ex_rate_map_sm(loc_range{rr}{peak_nb},rr);
                
                %add 0.01 to peak value(avoid error if neighboring max
                %values are the same - return flat line fit)
                [~,i_max] =max(curve_range{rr}{peak_nb});
                curve_range{rr}{peak_nb}(i_max) = curve_range{rr}{peak_nb}(i_max) + 0.01;
                
                %fit gaussian to peak
                [f{rr}{peak_nb},gof{rr}{peak_nb},output{rr}{peak_nb}] = fit(loc_range{rr}{peak_nb}', curve_range{rr}{peak_nb},'gauss1');
                %calculate gaussian at near range for fgir
                gauss_fit{rr}{peak_nb} = f{rr}{peak_nb}.a1*exp(-(([loc_range{rr}{peak_nb}]-f{rr}{peak_nb}.b1)./f{rr}{peak_nb}.c1).^2);
                
                gauss_fit{rr}{peak_nb} = f{rr}{peak_nb}.a1*exp(-(([loc_range{rr}{peak_nb}(1)-gauss_extend:loc_range{rr}{peak_nb}(end)+gauss_extend]-f{rr}{peak_nb}.b1)./f{rr}{peak_nb}.c1).^2);
                %full width at half maximum for each peak
                gauss_fwhm{rr}(peak_nb) = round(2*sqrt(2*log(2))*f{rr}{peak_nb}.c1);
                gauss_fit_bin_range{rr}{peak_nb} = [loc_range{rr}{peak_nb}(1)-gauss_extend:loc_range{rr}{peak_nb}(end)+gauss_extend];
            end
        end
    end
end

%display how many ROIs needed adjustment
%% Debug fitting param - solution to diff with fitting gaussian - if neighboring peak values are the same, git will return flat curve
%solution add 0.01 (or slightly different from neighboring value for fit
%work)
% [~,i_max] =max(y)
% y(i_max) = y(i_max)+0.01
% rr=58; peak_nb = 1
% x = loc_range{rr}{peak_nb};
% y= ex_rate_map_sm(loc_range{rr}{peak_nb},rr)
% %fit gaussian to peak
% [f_p,gof_p,output_p] = fit(x', y,'gauss1');
% %calculate gaussian at near range for fgir
% gauss_fit_p = f_p.a1*exp(-(([x]-f_p.b1)./f_p.c1).^2);

%% Check intersecting gaussian fit curves and merge

%add minimum smoothed transient rate for cross (set 0 for now adjust later)
%may change this on percentage of highest/lowest curve or decide this value
%empirically
minCurveCross = 0.05;

%check sequential peaks

%if there is a peak (gauss fit that spills beyond end lap, shift by bin lap length
%(100 bins) - do this separately

%threshold for edge of gaussian curve as to where end of field is
%fixed values of percentage of the height of the id'd peak - decide

%%% use ROI 17 as a starting point to test out merging
%check all neighboring peaks first
%then look for anyone with curve extending beyong start or end, if so shift
%and compare to peak next to it
rr=17; %single peak merger test
rr=36; %edge merger test (forward shift peak)
%rr = ;%(rearward shift peak)

%for all spatially tuned ROIs
%for rr =SI_tuned_ROIs
%for all ROIs
%preallocate merge_end to 0
merge_end = zeros(1,size(gauss_fit,2));
%preallocate empty cell for merge middle
merge_middle = cell(1,size(gauss_fit,2));

for rr=1:size(gauss_fit,2)
    %if more than 1 peak
    if size(gauss_fit{rr},2) > 1
        %for each peak comparison
        for pp=1:size(gauss_fit{rr},2)-1
            int_pt{rr}{pp} = InterX([gauss_fit_bin_range{rr}{pp};gauss_fit{rr}{pp}],[gauss_fit_bin_range{rr}{pp+1};gauss_fit{rr}{pp+1}]);
        end
        
        %create logical with intersection
        %for each comparison, check if there is an intersection and is above
        %minimum value
        for cc=1:size(int_pt{rr},2)
            if isempty(int_pt{rr}{cc})
                %no crossing of curves
                merge_middle{rr}(cc) = 0;
            else
                %if above minimuj threshold for crossing
                if int_pt{rr}{cc}(2) > minCurveCross
                    merge_middle{rr}(cc) = 1;
                else %if does not exceed threshold - no merge flag
                    merge_middle{rr}(cc) = 0;
                end
            end
            
        end
        
        %check if first or last peak crosses edge of lap bin (<51 or >150) -
        %only 1 test should be true when tested
        %first peak
        if sum(gauss_fit_bin_range{rr}{1} < 51) | sum(gauss_fit_bin_range{rr}{1} > 150)
            %shift forward by 100 bins and check intersection with last peak
            int_edge{rr} = InterX([gauss_fit_bin_range{rr}{1}+100;gauss_fit{rr}{1}],[gauss_fit_bin_range{rr}{end};gauss_fit{rr}{end}]);
            %assign 1 flag
            crossed_edge(rr) = 1;
            %if intersects,merge (update logical with intersection
            %last peak
        elseif (gauss_fit_bin_range{rr}{end} < 51) | sum(gauss_fit_bin_range{rr}{end} > 150)
            %shift backward by 100 bins and check intersection with first peak
            int_edge{rr} = InterX([gauss_fit_bin_range{rr}{end}-100;gauss_fit{rr}{end}],[gauss_fit_bin_range{rr}{1};gauss_fit{rr}{1}]);
            %assign -1 flag
            crossed_edge(rr) = -1;
        else
            %set to empty for conditional check below
            int_edge{rr} = [];
        end
        
        %check if there is intersection at edges and if yes, then set merge
        %flag
        if ~isempty(int_edge{rr})
            %check flag for which end curve is shifted
            if crossed_edge(rr) == 1
                %check if minimum for curve crossing is met
                if int_edge{rr}(2) > minCurveCross
                    %set merge to +1
                    merge_end(rr) = 1;
                end
            elseif crossed_edge(rr) == -1
                if int_edge{rr}(2) > minCurveCross
                    %set merge to -1
                    merge_end(rr) = -1;
                end
            end
        end
    end
end

%% Get endpoints of each gaussian based on either fraction of peak or set threshold 
%use fraction of peak for now; implement fixed threshold later

%fraction of peak to use as endpoint
fracPeak = 0.5;

for rr=1:size(gauss_fit,2)
    %for each peak if not empty
    if ~isempty(gauss_fit{rr})
        for pp=1:size(gauss_fit{rr},2)
            %gives the bin edge and value at thresgold
            gauss_fit_edges{rr}{pp} = InterX([gauss_fit_bin_range{rr}{pp}; ones(1,size(gauss_fit_bin_range{rr}{pp},2))*fracPeak*max(gauss_fit{rr}{pp})],...
                [gauss_fit_bin_range{rr}{pp};gauss_fit{rr}{pp}]);
            %if no edges detected, check if outlier broad gaussain fit and set fixed thres at 0.1 and
            %calculate edge using this thres (or only one edge found)
            if isempty(gauss_fit_edges{rr}{pp}) || (size(gauss_fit_edges{rr}{pp},2) == 1)
                %disp('empty')
                gauss_fit_edges{rr}{pp} = InterX([gauss_fit_bin_range{rr}{pp}; ones(1,size(gauss_fit_bin_range{rr}{pp},2))*0.1],...
                    [gauss_fit_bin_range{rr}{pp};gauss_fit{rr}{pp}]);
            end
            %if still empty, raise thres to 0.15
            if isempty(gauss_fit_edges{rr}{pp}) || (size(gauss_fit_edges{rr}{pp},2) == 1)
                %disp('empty')
                gauss_fit_edges{rr}{pp} = InterX([gauss_fit_bin_range{rr}{pp}; ones(1,size(gauss_fit_bin_range{rr}{pp},2))*0.15],...
                    [gauss_fit_bin_range{rr}{pp};gauss_fit{rr}{pp}]);
            end
            
        end
    end
end

%% Set merge input such into groups of which #'ed peaks to merge
%alternative is to re-fit a single term gaussian into the smoothed space
%now defined by the new endpoints - likely to be broader and overestimate
%field

%for middle peaks
%flag for whether previous peak was merged
prevMerge = 0;

for rr=1:size(merge_middle,2)
    
    %if more than 1 peak
    if size(gauss_fit{rr},2) > 1
        %for each merge comparison
        for mm=1:size(merge_middle{rr},2)
            if merge_middle{rr}(mm) == 0 && prevMerge == 0
                merge_peak_nb{rr}{mm} = mm;
                prevMerge = 0;
            elseif merge_middle{rr}(mm) == 1 && prevMerge == 0
                merge_peak_nb{rr}{mm} = [mm,mm+1];
                prevMerge = 1;
                %remove previous single peak if current merge
                if mm >1
                     if size(merge_peak_nb{rr}{mm-1},1) ==1 & merge_peak_nb{rr}{mm-1} == mm
                         %delete
                         merge_peak_nb{rr}{mm-1} = [];
                     end
                end
            elseif merge_middle{rr}(mm) == 1 && prevMerge == 1
                merge_peak_nb{rr}{end} = [merge_peak_nb{rr}{end},mm+1];
                prevMerge = 1;
           elseif merge_middle{rr}(mm) == 0 && prevMerge == 1 
                merge_peak_nb{rr}{mm} = mm+1;
                prevMerge = 0;
            end
        end
        %outside check - if last was no merge, add single peak to end
        if merge_middle{rr}(mm) == 0 && prevMerge == 0
            %if not already added
            if merge_peak_nb{rr}{end} ~= size(merge_middle{rr},2)+1
            merge_peak_nb{rr}{end+1} = size(merge_middle{rr},2)+1;
            end
        end
        %add last peak if standalone
    %if 1 peak
    elseif size(gauss_fit{rr},2) == 1
        merge_peak_nb{rr} = 1;
        %if no peaks
    elseif isempty(gauss_fit{rr})
        merge_peak_nb{rr} = [];
    end
    %reset prevMerge flag for next ROI
    prevMerge = 0;
end

%for edges
for rr=1:size(merge_end,2)
    if merge_end(rr) == 1 || merge_end(rr) == -1 %forward shift and integrate with end
        %designate mere peaks
        merge_peak_nb_edge{rr} = [size(gauss_fit{rr},2),1];
    else
        merge_peak_nb_edge{rr} = [];
    end
end

%make combined merging of middle and edge points into 1 cell
%take first and end point (1) (end)
%see if there are single or combined vectors
%if both single, delete, both, add shared
%if either 1 has more than 1 element; add to bigger 1, delete smaller
%if both more than one elements, merge into 1

for rr =1:size(merge_peak_nb_edge,2)
    %if there are peaks to merge
    if ~isempty(merge_peak_nb_edge{rr})
        if size(merge_peak_nb{rr}{end},2)==1 & size(merge_peak_nb{rr}{1},2)==1
            %set first index to merge
            merge_peak_nb{rr}{1} = merge_peak_nb_edge{rr};
            %delete last
            merge_peak_nb{rr}{end} = [];
        
        elseif size(merge_peak_nb{rr}{end},2)>1 & size(merge_peak_nb{rr}{1},2)>1
            %merge into first position
            merge_peak_nb{rr}{1} = [merge_peak_nb{rr}{end}, merge_peak_nb{rr}{1}];
            %delete last
            merge_peak_nb{rr}{end} = [];
        else %if either end contains standalone peak
            %first has multiple, take last and end to first, delete last
            if size(merge_peak_nb{rr}{1},2)>1
               merge_peak_nb{rr}{1} = [merge_peak_nb{rr}{end}, merge_peak_nb{rr}{1}];
               %delete last
               merge_peak_nb{rr}{end} = [];
            else size(merge_peak_nb{rr}{end},2)>1
               merge_peak_nb{rr}{end} = [merge_peak_nb{rr}{end},merge_peak_nb{rr}{1}];
               %delete first
               merge_peak_nb{rr}{1} = [];
            end
        end
    end
end

%% Check that all place fields have both edges defined (not just 1 end) 
%if 1 edge missing, determine which relative to center peak
%for each ROI
%for each ID'd place field
for rr=1:size(gauss_fit_edges,2)
    %for each identified peak
    for pp=1:size(gauss_fit_edges{rr},2)
        %if not empty
        if ~isempty(gauss_fit_edges{rr}{pp})
        %if only 1 edge of place field detected
        if size(gauss_fit_edges{rr}{pp},2) == 1
            %check which edge is missing by comparing it to peak curve
            %if existing edge greater than center of peak
            if lc{rr}(pp) > gauss_fit_edges{rr}{pp}(1)
                %fill in lower edge (by taking bin edge of fit)
                 gauss_fit_edges{rr}{pp} = [[gauss_fit_bin_range{rr}{pp}(1); 0.05], gauss_fit_edges{rr}{pp}];
            else
                %fill in farther edge
                gauss_fit_edges{rr}{pp} = [gauss_fit_edges{rr}{pp},[gauss_fit_bin_range{rr}{pp}(end); 0.05]];
            end
            
        end
        else %if empty at both ends fill in with find peak edges - not spatially tuned neurons
            disp('Filled in non-fitted Gaussian neuron')
             gauss_fit_edges{rr}{pp} = [loc_range{rr}{pp}(1), loc_range{rr}{pp}(end); 0.05 0.05];             
            
        end
    end
end

%% Future check - see if any edge values are greater than 0.1 which may affect downstream mering
%should be less than 0.1 for edge value corresponding to bin
%necesssary for very wide fits 

%% Set place field centers, edges
%(take endpoints of fitting Gaussians)

%correct edges for track offset for border analysis
%gauss_fit_edges;
%adjust bin in the end by shifting to by 50 bins less or equal to 150
%adjust by 100 those greater than 150

%delete empty merge cells inside inside final merge cell
for rr=1:size(merge_peak_nb,2)
    if iscell(merge_peak_nb{rr})
        merge_peak_nb{rr}(cellfun(@isempty,merge_peak_nb{rr})) = [];
    end
end

%temp_ROI = [1:259,261:481]

%rr=90
%check for peaks
%for rr = 1:size(merge_peak_nb,2)
    for rr = 91:size(merge_peak_nb,2)
    if ~isempty(merge_peak_nb{rr})
        %check if cell (more than 1 intial peaks)
        
        if iscell(merge_peak_nb{rr})
            %if only 1 peak
            %may need checking or simple make a corrected edge cell above
%             if size(merge_peak_nb{rr},2) == 1
%                 placeField.edge{rr} = round(gauss_fit_edges{rr}{1}(1,:));
%                 %take difference
%                 placeField.width{rr} = diff(placeField.edge{rr});
%                 placeField.center{rr} = round(placeField.width{rr}/2);
                
            %else %for multiple peaks; for each merge decision
                for mm=1:size(merge_peak_nb{rr},2)
                    %if single peak, do as above
                    if size(merge_peak_nb{rr}{mm},2) == 1
                        placeField.edge{rr}(mm,:) = round(gauss_fit_edges{rr}{merge_peak_nb{rr}{mm}}(1,:));
                        %take difference
                        placeField.width{rr}(mm) = diff(placeField.edge{rr}(mm,:));
                        placeField.center{rr}(mm) = round(placeField.width{rr}(mm)/2);
                        %if more than 1 peak, merge first and last peak of that
                        %merge
                    else
                        %edges of place field
                        placeField.edge{rr}(mm,1) = round(gauss_fit_edges{rr}{merge_peak_nb{rr}{mm}(1)}(1,1));
                        placeField.edge{rr}(mm,2) = round(gauss_fit_edges{rr}{merge_peak_nb{rr}{mm}(end)}(1,2));
                        
                        placeField.width{rr}(mm) = diff(placeField.edge{rr}(mm,:));
                        placeField.center{rr}(mm) = round(placeField.width{rr}(mm)/2);
                        
                    end
                end
            %end
        else
            %need this conditional for single peaks w/0 merging
            placeField.edge{rr} = round(gauss_fit_edges{rr}{1}(1,:));
            %take difference
            placeField.width{rr} = diff(placeField.edge{rr});
            %this is wrong! - CHANGE, but in consequential b/c doing
            %adjusted of bin values below
            placeField.center{rr} = round(placeField.width{rr}/2);
        end
        
    else
        placeField.edge{rr} = [];
        placeField.width{rr} = [];
        placeField.center{rr} = [];
    end
    
end



%% Plot putative place fields for each ROI
if 0 %skip - make option
    figure;
    for rr =SI_tuned_ROIs%1:100
        subplot(2,1,1)
        if ~isempty(pks{rr})
            %for each id'd peak
            for peak_nb =1:size(pks{rr})
                %plot range
                hold on
                title(num2str(rr));
                ylim([0 1.2])
                %plot extended rate map (smoothed)
                plot(ex_rate_map_sm(:,rr),'k')
                
                %plot extended rate map (non-smoothed)
                %plot(ex_rate_map(:,rr),'k-');
                %plot peak and width ends
                %plot peak center
                stem(lc{rr}(peak_nb),pks{rr}(peak_nb),'r')
                %plot width around peak
                %start
                stem(lc{rr}(peak_nb)-(w{rr}(peak_nb)./2),pks{rr}(peak_nb)*ones(size(lc{rr}(peak_nb),1),1),'g');
                %end
                stem(lc{rr}(peak_nb)+(w{rr}(peak_nb)./2),pks{rr}(peak_nb)*ones(size(lc{rr}(peak_nb),1),1),'g');
                %plot gaussian fit to ROI peaks
                plot([loc_range{rr}{peak_nb}(1)-gauss_extend:loc_range{rr}{peak_nb}(end)+gauss_extend],gauss_fit{rr}{peak_nb},'m')
            end
        end
        
        %plot start and end point of bin
        stem([51,150],[1 1],'b');
        %cutoff transient rate refline
        cutoff_line = refline(0,0.1);
        cutoff_line.Color = [0.5 0.5 0.5];
        cutoff_line.LineStyle = '--';
        %second cut off line
        cutoff_line2 = refline(0,0.05);
        cutoff_line2.Color = [0.5 0.5 0.5];
        cutoff_line2.LineStyle = '--';
        
        %plot place field edges
        for pl=1:size(placeField.edge{rr},1)
            stem(placeField.edge{rr}(pl,:),[0.5 0.5],'*c')
        end
        
        subplot(2,1,2)
        
        hold on
        if ~isempty(pks{rr})
            %for each id'd peak
            for peak_nb =1:size(pks{rr})
                %plot range
                hold on
                title(num2str(rr));
                ylim([0 1.2])
                %plot extended rate map (smoothed)
                %plot(ex_rate_map_sm(:,rr),'k')
                
                %plot extended rate map (non-smoothed)
                plot(ex_rate_map(:,rr),'k-');
                %plot peak and width ends
                %plot peak center
                stem(lc{rr}(peak_nb),pks{rr}(peak_nb),'r')
                %plot width around peak
                %start
                stem(lc{rr}(peak_nb)-(w{rr}(peak_nb)./2),pks{rr}(peak_nb)*ones(size(lc{rr}(peak_nb),1),1),'g');
                %end
                stem(lc{rr}(peak_nb)+(w{rr}(peak_nb)./2),pks{rr}(peak_nb)*ones(size(lc{rr}(peak_nb),1),1),'g');
                %plot gaussian fit to ROI peaks
                plot([loc_range{rr}{peak_nb}(1)-gauss_extend:loc_range{rr}{peak_nb}(end)+gauss_extend],gauss_fit{rr}{peak_nb},'m')
            end
        end
        
        %plot start and end point of bin
        stem([51,150],[1 1],'b');
        %cutoff transient rate refline
        cutoff_line = refline(0,0.1);
        cutoff_line.Color = [0.5 0.5 0.5];
        cutoff_line.LineStyle = '--';
        
                %plot place field edges
        for pl=1:size(placeField.edge{rr},1)
            stem(placeField.edge{rr}(pl,:),[0.5 0.5],'*c')
        end
        pause
        clf
    end

end

%% Area under curve of gaussians for comparison - skip for now; come back later



%% Adjust the values to fit with the 1-100 bin range

%bins between 51 -150 --> subtract 50
%bins less than 51 --> 100 - (50 -x) = 50+x
%bin greater than 150 --> x-150

%make copy for adjusted values
placeField.edge_adj = placeField.edge;
placeField.center_adj = placeField.center;
placeField.width_adj = placeField.width;

for rr=1:size(placeField.edge,2)
    %if not empty
    if ~isempty(placeField.edge{rr})
        %for each edge of place field
        %find each class and make adjustment
        field_log{1} = placeField.edge{rr} >= 51 & placeField.edge{rr} <=150;
        field_log{2} = placeField.edge{rr} < 51;
        field_log{3} = placeField.edge{rr} > 150;
        
        %check if any fit class, and adjust the bins
        for cc=1:3
            if any(field_log{cc}(:))
                if cc==1
                    placeField.edge_adj{rr}(field_log{cc}) = placeField.edge{rr}(field_log{cc}) - 50;
                elseif cc==2
                    placeField.edge_adj{rr}(field_log{cc}) = placeField.edge{rr}(field_log{cc}) + 50;
                elseif cc==3
                    placeField.edge_adj{rr}(field_log{cc}) = placeField.edge{rr}(field_log{cc}) - 150;
                end
            end
        end
        
                %for a for each place field here
%         if placeField.width_adj{rr} <= 0
%             placeField.width_adj{rr} = placeField.width_adj{rr} + 100;
%             placeField.center_adj{rr} = placeField.edge_adj{rr}(:,1) + round(placeField.width_adj{rr}./2)';
%         end
        %if negative, add 100 bins
        %placeField.width_adj{rr} = placeField.width_adj{rr}
        
        %recalculate the center and width
        placeField.width_adj{rr} = diff(placeField.edge_adj{rr}',1);
        placeField.center_adj{rr} = placeField.edge_adj{rr}(:,1) + round(placeField.width_adj{rr}./2)';
        
        %adjust if negative (crosses past lap border)
        for pp=1:size(placeField.width_adj{rr},2)
            if placeField.width_adj{rr}(pp) <= 0
                %adjust width
                placeField.width_adj{rr}(pp) = placeField.width_adj{rr}(pp) + 100;
                %adjust center; if center beyond 100, subtract 100
                if (placeField.edge_adj{rr}(pp,1) + round(placeField.width_adj{rr}(pp)./2)) > 100
                    placeField.center_adj{rr}(pp) = placeField.edge_adj{rr}(pp,1) + round(placeField.width_adj{rr}(pp)./2)-100;
                else
                     placeField.center_adj{rr}(pp) = placeField.edge_adj{rr}(pp,1) + round(placeField.width_adj{rr}(pp)./2);
                end
                %placeField.center_adj{rr} = placeField.edge_adj{rr}(:,1) + round(placeField.width_adj{rr}./2)';
            end
        end
        
    else %set to empty
        placeField.edge_adj{rr} = placeField.edge{rr};
        placeField.center_adj{rr} = placeField.center{rr};
        placeField.width_adj{rr} = placeField.width{rr};
    end
    
end

%% Extract place field info for those that are SI tuned or TS tuned and save to struct
SI_tuned = Place_cell{place_struct_nb}.Spatial_Info.significant_ROI;
TS_tuned = Place_cell{place_struct_nb}.Tuning_Specificity.significant_ROI;

%all field
Place_cell{place_struct_nb}.placeField.edge = placeField.edge_adj;
Place_cell{place_struct_nb}.placeField.center = placeField.center_adj;
Place_cell{place_struct_nb}.placeField.width = placeField.width_adj;
%spatial info tuned
Place_cell{place_struct_nb}.Spatial_Info.placeField.edge = placeField.edge_adj(SI_tuned);
Place_cell{place_struct_nb}.Spatial_Info.placeField.center = placeField.center_adj(SI_tuned);
Place_cell{place_struct_nb}.Spatial_Info.placeField.width = placeField.width_adj(SI_tuned);
%tuning spec. tuned
Place_cell{place_struct_nb}.Tuning_Specificity.placeField.edge = placeField.edge_adj(TS_tuned);
Place_cell{place_struct_nb}.Tuning_Specificity.placeField.center = placeField.center_adj(TS_tuned);
Place_cell{place_struct_nb}.Tuning_Specificity.placeField.width = placeField.width_adj(TS_tuned);


%% Plot final values
if 0
figure;
for rr =SI_tuned_ROIs%1:100
    
    %adjusted values
    subplot(2,1,1)
    hold on
    title(num2str(rr));
    ylim([0 0.8])
    %plot extended rate map (smoothed)
    plot(rate_map_sm(:,rr),'k')
    for pl=1:size(placeField.edge_adj{rr},1)
        %place field edges
        stem(placeField.edge_adj{rr}(pl,:),[0.5 0.5],'*c')
        %place centers
        stem(placeField.center_adj{rr}(pl),[0.5],'*r')
    end
    
    subplot(2,1,2)
    if ~isempty(pks{rr})
        %for each id'd peak
        for peak_nb =1:size(pks{rr})
            %plot range
            hold on
            title(num2str(rr));
            ylim([0 0.8])
            %plot extended rate map (smoothed)
            plot(ex_rate_map_sm(:,rr),'k')
            
            %plot extended rate map (non-smoothed)
            %plot(ex_rate_map(:,rr),'k-');
            %plot peak and width ends
            %plot peak center
            stem(lc{rr}(peak_nb),pks{rr}(peak_nb),'r')
            %plot width around peak
            %start
            stem(lc{rr}(peak_nb)-(w{rr}(peak_nb)./2),pks{rr}(peak_nb)*ones(size(lc{rr}(peak_nb),1),1),'g');
            %end
            stem(lc{rr}(peak_nb)+(w{rr}(peak_nb)./2),pks{rr}(peak_nb)*ones(size(lc{rr}(peak_nb),1),1),'g');
            %plot gaussian fit to ROI peaks
            plot([loc_range{rr}{peak_nb}(1)-gauss_extend:loc_range{rr}{peak_nb}(end)+gauss_extend],gauss_fit{rr}{peak_nb},'m')
        end
    end
    
    %plot start and end point of bin
    stem([51,150],[1 1],'b');
    %cutoff transient rate refline
    cutoff_line = refline(0,0.1);
    cutoff_line.Color = [0.5 0.5 0.5];
    cutoff_line.LineStyle = '--';
    %second cut off line
    cutoff_line2 = refline(0,0.05);
    cutoff_line2.Color = [0.5 0.5 0.5];
    cutoff_line2.LineStyle = '--';
    
    %plot place field edges
    for pl=1:size(placeField.edge{rr},1)
        stem(placeField.edge{rr}(pl,:),[0.5 0.5],'*c')
    end
    
    %pause;
    %clf;
end
end




end


