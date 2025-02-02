function [task_selective_ROIs] = split_remapping_category(cent_diff_AandB, tuned_logical, pf_vector_max, session_vars,max_transient_peak, options)
%split mutually tuned neurons by remapping category: 
%common (less than certain centroid difference between max
%tuned_log = tunedLogical.ts.AandB_tuned;

%% Get ROI indices of A selective, B selective, A&B selective neurons

%choose if SI or TS tuned
switch options.tuning_criterion
    case 'si' %spatial information
        AandB_tuned_idx = find(tuned_logical.si.AandB_tuned == 1);
        Aonly_tuned_idx = find(tuned_logical.si.onlyA_tuned == 1);
        Bonly_tuned_idx = find(tuned_logical.si.onlyB_tuned == 1);
        
    case 'ts' %tuning specificity
        %both A and B tuned by TS
        AandB_tuned_idx = find(tuned_logical.ts.AandB_tuned ==1);
        %only A tuned by TS
        Aonly_tuned_idx = find(tuned_logical.ts.onlyA_tuned == 1);
        %only B tuned by tS
        Bonly_tuned_idx = find(tuned_logical.ts.onlyB_tuned == 1);
        
        %all A tuned by SI
        A_tuned_si_idx = find(tuned_logical.si.Atuned == 1);
        %all B tuned by SI
        B_tuned_si_idx = find(tuned_logical.si.Btuned == 1);
        
        %only A tuned by TS, but also not SI B tuned
        Aonly_notSIb_idx = setdiff(Aonly_tuned_idx,B_tuned_si_idx);
        
        %only B tuned by TS, but also not SI A tuned
        Bonly_notSIa_idx = setdiff(Bonly_tuned_idx,A_tuned_si_idx);

end


%% Define/load variables for each session

%for each session
for ii = 1:size(session_vars,2)
    % behavior and imaging related variables
    Behavior_split_lap{ii} = session_vars{ii}.Behavior_split_lap;
    Events_split_lap{ii} = session_vars{ii}.Events_split_lap;
    Behavior_split{ii} = session_vars{ii}.Behavior_split;
    Event_split{ii} = session_vars{ii}.Events_split;
    Imaging_split{ii} = session_vars{ii}.Imaging_split;
    Place_cell{ii} = session_vars{ii}.Place_cell;
    Behavior_full{ii} = session_vars{ii}.Behavior;
    
    %all within run domain
    position{ii}  = Behavior_split_lap{ii}.Run.position;
    time{ii} = Behavior_split_lap{ii}.Run.time;
    events_full{ii} = Events_split_lap{ii}.Run.run_onset_binary;
    run_intervals{ii} = Behavior_split_lap{ii}.run_ones;
    
    %global trial type order across restricted laps
    trialOrder{ii} = Behavior_full{ii}.performance.trialOrder;
end

%for each session
for ss=1:size(session_vars,2)
    %for each lap
    for ii=1:size(run_intervals{ss},2)
        events{ss}{ii} = events_full{ss}{ii}(logical(run_intervals{ss}{ii}),:);
    end
end

%% Get lap indices for each lap in all B or B trials

%only correct

%get unique lap indices
lapA_idxs = unique(Behavior_split{1}{1}.resampled.lapNb);
lapB_idxs = unique(Behavior_split{1}{2}.resampled.lapNb);

%get lap start and end indices for all A or B trials
%all A
for ll=1:size(lapA_idxs,1)
    lap_idxs.A(ll,1) = find(Behavior_split{1}{1}.resampled.lapNb == lapA_idxs(ll),1,'first');
    lap_idxs.A(ll,2) = find(Behavior_split{1}{1}.resampled.lapNb == lapA_idxs(ll),1,'last');
end

%all B
for ll=1:size(lapB_idxs,1)
    lap_idxs.B(ll,1) = find(Behavior_split{1}{2}.resampled.lapNb == lapB_idxs(ll),1,'first');
    lap_idxs.B(ll,2) = find(Behavior_split{1}{2}.resampled.lapNb == lapB_idxs(ll),1,'last');
end


%% Split run indices/intervals by lap
%get run on/off binary for 


%% Event onsets in run interval
%only correct trials (1,2)
%for each ROI
for rr=1:size(events{ss}{1},2)
    %time of significant run events in A
    event_norm_time.A{rr} = Imaging_split{1}{1}.time_restricted(find(Event_split{1}{1}.Run.run_onset_binary(:,rr) == 1))/60;
    %normalizesd position of significant run events in A
    event_norm_pos_run.A{rr} = Behavior_split{1}{1}.resampled.position_norm(find(Event_split{1}{1}.Run.run_onset_binary(:,rr) == 1));
    %lap assignment for A
    event_lap_idx.A{rr} = Behavior_split{1}{1}.resampled.lapNb(logical(Event_split{1, 1}{1, 1}.Run.run_onset_binary(:,rr)));
    
    %time of significant run events in B
    event_norm_time.B{rr} = Imaging_split{1}{2}.time_restricted(find(Event_split{1}{2}.Run.run_onset_binary(:,rr) == 1))/60;
    %normalizesd position of significant run events in B
    event_norm_pos_run.B{rr} = Behavior_split{1}{2}.resampled.position_norm(find(Event_split{1}{2}.Run.run_onset_binary(:,rr) == 1));
    %lap assignment for B
    event_lap_idx.B{rr} = Behavior_split{1}{2}.resampled.lapNb(logical(Event_split{1}{2}.Run.run_onset_binary(:,rr)));
end


%% Plot the run epochs, corresponding position bin edges of place field

%TS tuning problems: 36 69 72 102 157 170 179 227
%72 

%overlay area of max place field bin

%spatial bin assignment for each run-epoch frame (100 bins) 
%A trials
session_vars{1}.Place_cell{1}.Bin{8};
%B trials
session_vars{1}.Place_cell{2}.Bin{8};

%get the run epoch binaries for each set of trials - A session - 5339 run
%frames

%plot the run epochs as patches
session_vars{1}.Behavior_split{1}.run_ones;

%get the run epoch binaries for each set of trials - B session - 4547 run
%frames
session_vars{1}.Behavior_split{2}.run_ones; 


%run events - binary onset across entire run/no run interval
Event_split{1}{1}.Run.run_onset_ones;

%get bins

%get edges for corresponding bins!! - find place in spatial info where 
%correct A trials
run_position_norm{1} = Behavior_split{1}{1}.resampled.run_position_norm;
%correct B trials
run_position_norm{2} = Behavior_split{1}{2}.resampled.run_position_norm;

%Bin running position in 100 bins and get edges for each set of laps:
%for each number of bins, bin the normalized position during run epochs

%for correct A trials
[count_bin{1},edges{1},bin{1}] = histcounts(run_position_norm{1}, 100);
%for correct B trials
[count_bin{2},edges{2},bin{2}] = histcounts(run_position_norm{2}, 100);


%% Remove ROIs idx's without a id'd place field

%edges of all identified
%correct A
placeFieldEdges{1} = Place_cell{1}{1}.placeField.edge(Aonly_notSIb_idx);
%correct B
placeFieldEdges{2} = Place_cell{1}{2}.placeField.edge(Bonly_notSIa_idx);

%in A trials
idx_wo_placeFields{1} = Aonly_notSIb_idx(find(cellfun(@isempty,placeFieldEdges{1}) ==1));
%in B trials
idx_wo_placeFields{2} = Bonly_notSIa_idx(find(cellfun(@isempty,placeFieldEdges{2}) ==1));

%remove ROIs for A and B that do not have place fields
%for A trials
if ~isempty(idx_wo_placeFields{1})
    %copy
    Aonly_field_filtered = setdiff(Aonly_notSIb_idx,idx_wo_placeFields{1});

else %keep the previous ROIs (copy only)
    Aonly_field_filtered = Aonly_notSIb_idx;
end
%for B trials
if ~isempty(idx_wo_placeFields{2})
    %copy
    Bonly_field_filtered = setdiff(Bonly_notSIa_idx,idx_wo_placeFields{2});

else %keep the previous ROIs (copy only)
    Bonly_field_filtered = Bonly_notSIa_idx;
end

%% Determine the equivalent normalized position range of the max place field

%extract the max place field index for A and B trial using filtered A tuned
%and B tuned ROIs
max_field_idx{1} = max_transient_peak{1}{1}(Aonly_field_filtered);
max_field_idx{2} = max_transient_peak{1}{2}(Bonly_field_filtered);

%get the edges of the max transient place field for each set of idxs
%edges of all identified
%correct A
placeField_filtered{1} = Place_cell{1}{1}.placeField.edge(Aonly_field_filtered);
%correct B
placeField_filtered{2} = Place_cell{1}{2}.placeField.edge(Bonly_field_filtered);

%check which field has more than 1 field and select edges of the one with
%higher transient rate

%for each trial (A and B )
for tt=1:2
    for rr=1:size(placeField_filtered{tt},2)
        if size(placeField_filtered{tt}{rr},1) > 1
            placeField_filtered_max{tt}{rr} = placeField_filtered{tt}{rr}(max_field_idx{tt}(rr),:);
        else
            placeField_filtered_max{tt}{rr} = placeField_filtered{tt}{rr};
        end
    end
end

%convert edges from relevant place field to normalized postion edges
for tt=1:2
    for rr=1:size(placeField_filtered_max{tt},2)
        %start position of PF
        placeField_filtered_max_posnorm{tt}{rr}(1) = edges{1}(placeField_filtered_max{tt}{rr}(1));
        %end position of PF
        placeField_filtered_max_posnorm{tt}{rr}(2) = edges{1}(placeField_filtered_max{tt}{rr}(2)+1)-0.01;
    end
end


%% Get normalized position distance converstion factor

%get median lap length based on the registered length of each lap
median_track_len = median(Behavior_full{1}.position_lap(:,2));

%conversion factor (norm_pos/cm length) - 100 bins
norm_conv_factor = median_track_len/100;
%median_track_len/1;

%% Filter out neurons that do not have at least 5 sig events in max place field in at least 5 distinct laps

%calcium event position and absolute restrict time
%all neuron idx space
event_norm_pos_run.A;
event_norm_time.A;  
event_lap_idx.A;

Aonly_field_filtered;
Bonly_field_filtered;

%find events occuring within max place field for each ROI
for tt=1:2
    for rr=1:size(placeField_filtered_max_posnorm{tt},2)
        %get idxs of events with max place field
        if tt == 1 %correct A trials
            events_in_field{tt}{rr} = find(event_norm_pos_run.A{Aonly_field_filtered(rr)} >= placeField_filtered_max_posnorm{tt}{rr}(1) & ...
                event_norm_pos_run.A{Aonly_field_filtered(rr)} <= placeField_filtered_max_posnorm{tt}{rr}(2));
            %register the corresponding lap of in-field filtered event
            event_in_field_laps{tt}{rr} = event_lap_idx.A{Aonly_field_filtered(rr)}(events_in_field{tt}{rr});
            %get number of unique events (those occuring on each lap)
            event_in_field_nb{tt}{rr} = size(unique(event_in_field_laps{tt}{rr}),1);
            %get position of in-field events
            events_in_field_pos{tt}{rr} = event_norm_pos_run.A{Aonly_field_filtered(rr)}(events_in_field{tt}{rr});
            
        elseif tt == 2 %correct B trials
            events_in_field{tt}{rr} = find(event_norm_pos_run.B{Bonly_field_filtered(rr)} >= placeField_filtered_max_posnorm{tt}{rr}(1) & ...
                event_norm_pos_run.B{Bonly_field_filtered(rr)} <= placeField_filtered_max_posnorm{tt}{rr}(2));
            %register the corresponding lap of in-field filtered event
            event_in_field_laps{tt}{rr} = event_lap_idx.B{Bonly_field_filtered(rr)}(events_in_field{tt}{rr});
            %get number of unique events (those occuring on each lap)
            event_in_field_nb{tt}{rr} = size(unique(event_in_field_laps{tt}{rr}),1);
            %get position of in-field events
            events_in_field_pos{tt}{rr} = event_norm_pos_run.B{Bonly_field_filtered(rr)}(events_in_field{tt}{rr});
        end
    end
end

%check which task-selective ROIs have less than 5 events
%correct A trials
event_thres_exclude_log.A  = cell2mat(event_in_field_nb{1}) < 5;
%correct B trials
event_thres_exclude_log.B  = cell2mat(event_in_field_nb{2}) < 5;

%update indices with event 
ROI_field_filtered_event.A = Aonly_field_filtered(~event_thres_exclude_log.A);
ROI_field_filtered_event.B = Bonly_field_filtered(~event_thres_exclude_log.B);

%update assn place fields
placeField_eventFilt{1} = placeField_filtered_max_posnorm{1}(~event_thres_exclude_log.A);
placeField_eventFilt{2} = placeField_filtered_max_posnorm{2}(~event_thres_exclude_log.B);

%update event position (normalized)
event_pos_inField{1} = events_in_field_pos{1}(~event_thres_exclude_log.A);
event_pos_inField{2} = events_in_field_pos{2}(~event_thres_exclude_log.B);

%% Make sure the animal was in a run epoch in the min/max range of space on opposing laps (al least 80%) of space on at least 6 laps

% Check that in running epoch within 3 bins to the left or right of each
ROI_field_filtered_event.A
ROI_field_filtered_event.B

%take the median position of the events in field and min/max position
for tt=1:2 %for correct A and B trials
    for rr=1:size(event_pos_inField{tt},2)
        med_pos_event{tt}(rr) = median(event_pos_inField{tt}{rr}); 
        %into one matrix min and max of each event
        min_max_pos_event{tt}(rr,1) = min(event_pos_inField{tt}{rr}); 
        min_max_pos_event{tt}(rr,2) = max(event_pos_inField{tt}{rr}); 
    end
end

%get correct A and B laps idx
corr_lap_idx{1} = unique(Behavior_split{1}{1}.resampled.lapNb);
corr_lap_idx{2} = unique(Behavior_split{1}{2}.resampled.lapNb);

%get indices across all laps of the ranges
for tt=1:2 %for correct A and B trials
    for rr=1:size(event_pos_inField{tt},2)
        %get indices that match the position range (ALL LAPS)
        pos_range_indices{tt}{rr} = find( Behavior_full{1}.resampled.normalizedposition >= min_max_pos_event{tt}(rr,1) & ...
            Behavior_full{1}.resampled.normalizedposition <= min_max_pos_event{tt}(rr,2));
        %get lap idx of corresponding idxs
        lap_idx_range{tt}{rr} = Behavior_full{1}.resampled.lapNb(pos_range_indices{tt}{rr});
    end
end

%for correct A or B trials
for tt=1:2
    %for each ROI in correct A trials 
    for rr=1:size(lap_idx_range{tt},2)
        %extact logical with only laps correponding to opposing trial laps- dependent on B parameter
        if tt == 1 %if looking on run status in B trials for correct A trials
            lap_opposed_idx{tt}{rr} = ismember(lap_idx_range{tt}{rr},corr_lap_idx{2});
        elseif tt == 2 %if looking on run status in A trials for correct B trials
            lap_opposed_idx{tt}{rr} = ismember(lap_idx_range{tt}{rr},corr_lap_idx{1});
        end
        %get the lap number associated with each frame in the opposing trials
        lap_label_opposed{tt}{rr} = lap_idx_range{tt}{rr}(lap_opposed_idx{tt}{rr});
        %the binary indicating in animal in run epoch with that range
        lap_runEpoch_opposed{tt}{rr} = Behavior_full{1}.run_ones(pos_range_indices{tt}{rr}(lap_opposed_idx{tt}{rr}));
        %extract the associated positions
        lap_pos_opposed{tt}{rr} = Behavior_full{1}.resampled.normalizedposition(pos_range_indices{tt}{rr}(lap_opposed_idx{tt}{rr}));
    end
end

%split into individual laps for A events, look in B laps
%for correct A or B trials
for tt=1:2
    %for each ROI in correct A trials
    for rr=1:size(lap_idx_range{tt},2)
        %for each opposing lap
        for ll=1:size(corr_lap_idx{2},1)
            % - depdendent on B parameter
            if tt == 1 %if looking on run status in B trials for correct A trials
                split_lap_idxs{tt}{rr}{ll} = find(lap_label_opposed{tt}{rr} == corr_lap_idx{2}(ll));
            elseif tt == 2
                split_lap_idxs{tt}{rr}{ll} = find(lap_label_opposed{tt}{rr} == corr_lap_idx{1}(ll));
            end
            
            split_lap_pos{tt}{rr}{ll} = lap_pos_opposed{tt}{rr}(split_lap_idxs{tt}{rr}{ll});
            split_lap_runEpoch{tt}{rr}{ll} = lap_runEpoch_opposed{tt}{rr}(split_lap_idxs{tt}{rr}{ll});
        end
    end
end

%for correct A or B trials
for tt=1:2
    %for each ROI in correct A trials
    for rr=1:size(lap_idx_range{tt},2)
        %for each lap
        for ll=1:size(split_lap_pos{tt}{rr},2)
            unique_pos{tt}{rr}{ll} = unique(split_lap_pos{tt}{rr}{ll});
            %for each unique position, check if entirety in run epoch
            for pos_idx=1:size(unique_pos{tt}{rr}{ll},1)
                %get idx associated with given unique pos in the lap
                pos_idxs_each{tt}{rr}{ll}{pos_idx} = find(split_lap_pos{tt}{rr}{ll} == unique_pos{tt}{rr}{ll}(pos_idx));
                run_Epoch_each{tt}{rr}{ll}{pos_idx} = split_lap_runEpoch{tt}{rr}{ll}(pos_idxs_each{tt}{rr}{ll}{pos_idx});
                %check which position did animal spend all of it in run state
                
            end
            %calculate logical of  run at each position and get fraction in run
            %state for each lap
            frac_run_lap{tt}{rr}(ll) = sum(cellfun(@prod,run_Epoch_each{tt}{rr}{ll}))/size(run_Epoch_each{tt}{rr}{ll},2);
        end
    end
end

%for all laps check how above 80% of space in run epoch and check if this
%occurs in at least 6 laps
%for correct A or B trials
for tt=1:2
    %for each ROI in correct A trials
    for rr=1:size(lap_idx_range{tt},2)
        if sum(frac_run_lap{tt}{rr} >= 0.8) >= 6
            %generate logical with 1's include ROI, and 0 exclude ROI
            run_epoch_filt_include_log{tt}(rr) = 1;
        else
            run_epoch_filt_include_log{tt}(rr) = 0;
        end
    end
end

%apply final filter to ROI indices

%select ROI indices (from original)
final_filtered_ROI.A = ROI_field_filtered_event.A(logical(run_epoch_filt_include_log{1}));
final_filtered_ROI.B = ROI_field_filtered_event.B(logical(run_epoch_filt_include_log{2}));

%place field width positions
placeField_final{1} = placeField_eventFilt{1}(logical(run_epoch_filt_include_log{1}));
placeField_final{2} = placeField_eventFilt{2}(logical(run_epoch_filt_include_log{2}));

%event position (normalized)
event_pos_inField_final{1} = event_pos_inField{1}(logical(run_epoch_filt_include_log{1}));
event_pos_inField_final{2} = event_pos_inField{2}(logical(run_epoch_filt_include_log{2}));


%% Plot as shaded area to verify correct id of place field onto normalized

if options.dispFigure ==1
    %plot normalized position
    %show A selective first
    figure('Position', [1930 130 1890 420])
    hold on
    title('A-selective filtered')
    for rr=1:size(final_filtered_ROI.A,2)%1:size(Aonly_notSIb_idx,2)
        %ROI = rr%AandB_tuned_idx(rr);
        ROI = final_filtered_ROI.A(rr); %Aonly_notSIb_idx(rr);
        hold on
        title(num2str(ROI))
        yticks([0 0.5 1])
        ylabel('Normalized position')
        xlabel('Time [min]');
        xticks(0:3:12);
        %ylim([0 1])
        set(gca,'FontSize',14)
        set(gca,'LineWidth',1)
        %A laps
        for ii=1:size(lap_idxs.A,1)
            plot(Imaging_split{1}{1}.time_restricted(lap_idxs.A(ii,1):lap_idxs.A(ii,2))/60,...
                Behavior_split{1}{1}.resampled.position_norm(lap_idxs.A(ii,1):lap_idxs.A(ii,2)),...
                'Color',[0 0 1 0.6],'LineWidth',1.5)
        end
        %B laps
        for ii=1:size(lap_idxs.B,1)
            plot(Imaging_split{1}{2}.time_restricted(lap_idxs.B(ii,1):lap_idxs.B(ii,2))/60,...
                Behavior_split{1}{2}.resampled.position_norm(lap_idxs.B(ii,1):lap_idxs.B(ii,2)),...
                'Color',[1 0 0 0.6],'LineWidth',1.5)
        end
        %overlay significant calcium run events
        %A
        scatter(event_norm_time.A{ROI},event_norm_pos_run.A{ROI},[],[0 0 1],'*')
        %B
        scatter(event_norm_time.B{ROI},event_norm_pos_run.B{ROI},[],[1 0 0],'*')
        
        %plot horz lines signifying start and end of place field
        %start
        lineS = refline(0,placeField_final{1}{rr}(1))
        lineS.Color = 'g';
        %end
        lineE = refline(0,placeField_final{1}{rr}(2))
        lineE.Color = 'g';
        
        pause
        clf
    end
    
    %show B selective second
    figure('Position', [1930 130 1890 420])
    hold on
    title('B-selective filtered')
    for rr=1:size(final_filtered_ROI.B,2)%1:size(Aonly_notSIb_idx,2)
        %ROI = rr%AandB_tuned_idx(rr);
        ROI = final_filtered_ROI.B(rr); %Aonly_notSIb_idx(rr);
        hold on
        title(num2str(ROI))
        yticks([0 0.5 1])
        ylabel('Normalized position')
        xlabel('Time [min]');
        xticks(0:3:12);
        %ylim([0 1])
        set(gca,'FontSize',14)
        set(gca,'LineWidth',1)
        %A laps
        for ii=1:size(lap_idxs.A,1)
            plot(Imaging_split{1}{1}.time_restricted(lap_idxs.A(ii,1):lap_idxs.A(ii,2))/60,...
                Behavior_split{1}{1}.resampled.position_norm(lap_idxs.A(ii,1):lap_idxs.A(ii,2)),...
                'Color',[0 0 1 0.6],'LineWidth',1.5)
        end
        %B laps
        for ii=1:size(lap_idxs.B,1)
            plot(Imaging_split{1}{2}.time_restricted(lap_idxs.B(ii,1):lap_idxs.B(ii,2))/60,...
                Behavior_split{1}{2}.resampled.position_norm(lap_idxs.B(ii,1):lap_idxs.B(ii,2)),...
                'Color',[1 0 0 0.6],'LineWidth',1.5)
        end
        %overlay significant calcium run events
        %A
        scatter(event_norm_time.A{ROI},event_norm_pos_run.A{ROI},[],[0 0 1],'*')
        %B
        scatter(event_norm_time.B{ROI},event_norm_pos_run.B{ROI},[],[1 0 0],'*')
        
        %plot horz lines signifying start and end of place field
        %start
        lineS = refline(0,placeField_final{2}{rr}(1))
        lineS.Color = 'g';
        %end
        lineE = refline(0,placeField_final{2}{rr}(2))
        lineE.Color = 'g';
        
        pause
        clf
    end
end

%% Export task-selective ROIs in struct

%export indices for task selective neurons
task_selective_ROIs.A.idx = final_filtered_ROI.A;
task_selective_ROIs.B.idx = final_filtered_ROI.B;

%field margin for task selective neurons
task_selective_ROIs.A.field_margin = placeField_final{1};
task_selective_ROIs.B.field_margin = placeField_final{2};

%event associated with selected place field
task_selective_ROIs.A.fieldEvents = event_pos_inField_final{1};
task_selective_ROIs.B.fieldEvents = event_pos_inField_final{2};

%% Patch generator for run epochs
%creates x range of patches
if 0
figure;
hold on
xRange = {5:10; 20; 40:42; 50; 60:75; 90:95};
for k1 = 1:size(xRange,1)
    q = xRange{k1};
    %if very brief period
    if length(q) == 1
        q = [q q+0.1];
    end
    %start x end x; end x start x
    qx = [min(q) max(q)  max(q)  min(q)];
    yl = ylim;
    %start y x2; end y x2
    qy = [[1 1]*yl(1) [1 1]*yl(2)];
    %plot the patches
    %green
    %patch(qx, qy, [0 1 0],'EdgeColor', 'none','FaceAlpha', 0.3)
    %red
    patch(qx, qy, [1 0 0],'EdgeColor', 'none','FaceAlpha', 0.3)
end
end

%% OLD PLOTTING CODE
% %plot normalized position
% figure('Position', [1930 130 1890 420])
% for rr=1:size(Bonly_field_filtered,2)%1:size(Aonly_notSIb_idx,2)
%     %ROI = rr%AandB_tuned_idx(rr);
%     ROI = Bonly_field_filtered(rr); %Aonly_notSIb_idx(rr);
%     hold on
%     title(num2str(ROI))
%     yticks([0 0.5 1])
%     ylabel('Normalized position')
%     xlabel('Time [min]');
%     xticks(0:3:12);
%     %ylim([0 1])
%     set(gca,'FontSize',14)
%     set(gca,'LineWidth',1)
%     %A laps
%     for ii=1:size(lap_idxs.A,1)
%         plot(Imaging_split{1}{1}.time_restricted(lap_idxs.A(ii,1):lap_idxs.A(ii,2))/60,...
%             Behavior_split{1}{1}.resampled.position(lap_idxs.A(ii,1):lap_idxs.A(ii,2)),...
%             'Color',[0 0 1 0.6],'LineWidth',1.5)
%     end
%     %B laps
%     for ii=1:size(lap_idxs.B,1)
%         plot(Imaging_split{1}{2}.time_restricted(lap_idxs.B(ii,1):lap_idxs.B(ii,2))/60,...
%             Behavior_split{1}{2}.resampled.position(lap_idxs.B(ii,1):lap_idxs.B(ii,2)),...
%             'Color',[1 0 0 0.6],'LineWidth',1.5)
%     end
%     %overlay significant calcium run events
%     %A
%     scatter(event_norm_time.A{ROI},event_norm_pos_run.A{ROI},[],[0 0 1],'*')
%     %B
%     scatter(event_norm_time.B{ROI},event_norm_pos_run.B{ROI},[],[1 0 0],'*')
%     
%     %generate run and non-run patched across track
%     if 0
%     xRange = {5:10; 20; 40:42; 50; 60:75; 90:95};
%     
%     for pr =1:size(Behavior_split{1, 1}{1, 1}.run_on_off_idx)
%     
%        xRange{pr,1} =  Imaging_split{1}{1}.time_restricted(Behavior_split{1}{1}.run_on_off_idx(pr,1): Behavior_split{1}{1}.run_on_off_idx(pr,2))./60;
%        
%     end
%     for k1 = 1:size(xRange,1)
%         q = xRange{k1};
%         %if very brief period
%         if length(q) == 1
%             q = [q q+0.1];
%         end
%         %start x end x; end x start x
%         qx = [min(q) max(q)  max(q)  min(q)];
%         yl = ylim;
%         %start y x2; end y x2
%         qy = [[1 1]*yl(1) [1 1]*yl(2)];
%         %plot the patches
%         %green
%         %patch(qx, qy, [0 1 0],'EdgeColor', 'none','FaceAlpha', 0.3)
%         %red
%         patch(qx, qy, [1 0 0],'EdgeColor', 'none','FaceAlpha', 0.3)
%     end
%     end
%     pause
%     clf
% end

end

