%% Set parameters for the pipeline

options.defineDir = 1;

%I45_RT
%setDir = 'G:\OCGOL_learning_long_term\I45_RT\behavior_only\I45_RT_rand_d1_052218';
%setDir = 'G:\OCGOL_learning_long_term\I45_RT\behavior_only\I45_RT_5A5B_053018';
%setDir = 'G:\OCGOL_learning_long_term\I45_RT\behavior_only\I45_RT_3A3B_060518';
%setDir = 'G:\OCGOL_learning_long_term\I45_RT\behavior_only\I45_RT_AB_061418';

%I46
%setDir = 'G:\OCGOL_learning_long_term\I46\behavior_only\I46_rand_d1_052918';
%setDir = 'G:\OCGOL_learning_long_term\I46\behavior_only\I46_5A5B_060118';
%setDir = 'G:\OCGOL_learning_long_term\I46\behavior_only\I46_3A3B_060718';
%setDir = 'G:\OCGOL_learning_long_term\I46\behavior_only\I46_AB_061518';

%I47_RS
%setDir = 'G:\OCGOL_learning_long_term\I47_RS\behavior_only\I47_RS_rand_d2_051518';
%setDir = 'G:\OCGOL_learning_long_term\I47_RS\behavior_only\I47_RS_5AB_d7_052218';
%setDir = 'G:\OCGOL_learning_long_term\I47_RS\behavior_only\I47_RS_3AB_d8_052418';
%setDir = 'G:\OCGOL_learning_long_term\I47_RS\behavior_only\I47_RS_AB_061418';


%%% don't use - not 5A5B, but random
%setDir = 'G:\OCGOL_learning_long_term\I47_RS\behavior_only\I47_RS_5AB_d1_051618';
%setDir = 'G:\OCGOL_learning_long_term\I47_RS\behavior_only\I47_RS_5AB_d1_051618_2';
%%%

%I47_LP
%setDir = 'G:\OCGOL_learning_long_term\I47_LP\behavior_only\I47_LP_rand_d2_051518';
%setDir = 'G:\OCGOL_learning_long_term\I47_LP\behavior_only\I47_LP_5AB_d1_051718';
%setDir = 'G:\OCGOL_learning_long_term\I47_LP\behavior_only\I47_LP_3AB_d8_052418';
setDir = 'G:\OCGOL_learning_long_term\I47_LP\behavior_only\I47_LP_AB_061418';

%whether to load in existing XML and CSV behavioral data save in workspace
%1 - load from saved workspace
%0 - read and load from raw XML and CSV files
options.loadBehaviorData = 1;

%whether to load in previously read imaging data
%options.loadImagingData = 0;

%choose the behavior that the animal ran
% RF, GOL-RF (GOL day 0), GOL, OCGOL
%options.BehaviorType = 'RF';
options.BehaviorType = 'OCGOL';

%type of calcium data
options.calcium_data_input = 'CNMF';

%% TODO
%insert selector here for type of behavior; RF, GOL, OCGOL

%% Define input directory for experiment (.mat, .csv, .xml data)

if options.defineDir == 0
    %select with GUI
    directory_name =  uigetdir;
elseif options.defineDir == 1
    %assign from setDir
    directory_name = setDir;
end
%save dir_name into options struct
options.dir_name = directory_name;

%% Import behavioral data and store as MATLAB workspace or load from workspace

tic;
if options.loadBehaviorData == 0
    [CSV,XML] = readBehaviorData_V1(directory_name);
elseif options.loadBehaviorData == 1
    [CSV,XML] = loadBehaviorData_V1(directory_name);
end
toc;

%% Extract position and laps

%minimum distance between sequential laps tag (cm)
options.mindist=10;

%behavior sampling rate (10 kHz) in Hz
options.acqHz=10000;

options.dispfig=1; % Display figure (lap, RFID)

[Behavior] = extractPositionAndLaps(CSV,options);


%% Extract texture/reward locations - there is an out of bounds issue with this

%whether to extract textures (and other signals)
options.textures = true;

if options.textures == true
    %[Behavior] = extractTextures(CSV, Behavior, options);
    %all signals, not just texture related signals
    switch options.BehaviorType
        case 'RF'
            [Behavior] = extractTextures_GOL_RF_training(CSV, Behavior, options);
        case 'GOL-RF' %special case for day 1 of GOL
            [Behavior] = extractTextures_GOL_RF(CSV, Behavior, options);
        case 'GOL'
            [Behavior] = extractTextures_GOL(CSV, Behavior, options);
        case 'OCGOL'
            [Behavior] = extractTextures_OCGOL(CSV, Behavior, options);
    end
end

%% After here CSV raw data matrix should not be necessary 

%% Extract behavior performance (non-restricted)

%add selector here depending on type of behavior run
switch options.BehaviorType
    case 'RF' %same as GOL for now
        %[Behavior] = GOL_performance(Behavior, CSV);
        [Behavior] = GOL_RF_performance_new_inputs_training(Behavior);
    case 'GOL-RF'
        [Behavior] = GOL_RF_performance_new_inputs(Behavior);
    case 'GOL'
        %update lick struct in behavior to retain info about restricted
        %licks and median reward position
        [Behavior] = GOL_performance_new_inputs(Behavior);
    case 'OCGOL'
        %works well for (I52RT AB PSEM/sal 113018)
        %fault with missed reward (I53LT AB PSEM/sal 113018)
        %in licks plot shade the reward zones
        %for reward collected as well
        [Behavior] = OCGOL_performance_new_inputs(Behavior);
end

%% Read imaging times (XML) and generate Imaging struct

% Restrict calcium data to selected lap -  restrict trace to only full laps
options.restrict=1; 

%Select from which starting lap
options.startlap= 'first'; 

%Select the end lap
options.endlap='last';

%display the calcium trace aligned to behavior figure
options.dispfig=0; % Display figure

%neurons to display
options.ROI =40; 

[Imaging, Behavior] = restrict_data_training(Behavior, XML, options);

%% Determine speed of animal

%averaging window for speed calculation
options.moving_window = 10; 

[Behavior] = resample_behavioral_data_and_speed_training(Behavior, Imaging, options);

%% Export speed data and reward positional data; mean speed by spatial bin

%global variables
% time = Behavior.resampled.time;
% Behavior.resampled.normalizedposition
position = Behavior.resampled.position;

%%speed of animal
speed = Behavior.speed;

%idxs # corresponding to 10s
idx_width = 75;

switch options.BehaviorType
    case 'RF'
        %lap idx corresponding to each imaging timepoint
        lap_idx_resampled = Behavior.resampled.lapNb;
        %get lap idx from 1 ... end
        lap_idxs = unique(lap_idx_resampled);
        %get idxs correspoinding to each lap
        %for A laps
        for ll = 1:size(lap_idxs,1)
            %target A reward zones
            split_lap.res_idx{ll} = find(lap_idx_resampled == lap_idxs(ll));
            %[~,rewards.A.Imin(ll)] = min(abs(position(split_lap.res_idx{ll}) - rewards.A.position(ll)));
            split_lap.lap_position{ll} = position(split_lap.res_idx{ll});
            split_lap.speed{ll} = speed(split_lap.res_idx{ll});
            %non-target B reward zones
            %[~,rewards.A.IminB(ll)] = min(abs(position(rewards.A.res_idx{ll}) - rewards.B.pos_mean));
        end
        
        %bin data by position (cm) into 100 bins
        [pos_bins,pos_edges] = discretize(position,100);
        
        %find each bin and calculate mean velocity in that bin
        for bb=1:100
            %idx's of each bin
            bin_idxs{bb} = find(pos_bins == bb);
            %get velocities in each bin and calcualte mean velocity
            bin_speed(bb) = mean(speed(bin_idxs{bb}));
        end
        
        %export relevant data for cumulative processing
        %export for cumulative analysis
        %make cumul_analysis folder
        mkdir(setDir,'cumul_analysis')
        %save the fractions output data
        save(fullfile(setDir,'cumul_analysis','speed_data.mat'),'split_lap',...
            'Behavior','Imaging','position','speed','lap_idx_resampled','bin_speed',...
            'pos_bins','pos_edges');
        
    case 'OCGOL'
        
        %bin position
        %bin data by position (cm) into 100 bins
        [pos_bins,pos_edges] = discretize(position,100);
        
        %reward zone time onsets for each lap (from absolute Behavior)
        rewards.A.position = Behavior.rewards{2}.full_laps.position;
        rewards.B.position = Behavior.rewards{1}.full_laps.position;
        
        %filter A and B reward positions that are out-of-bounds (mis-id'd)
        rem_A_idx = find(rewards.A.position <131 | rewards.A.position > 150);
        rem_B_idx = find(rewards.B.position <50 | rewards.B.position > 68);
        
        %if mis-idx idx located, remove from list
        if ~isempty(rem_A_idx)
            rewards.A.position(rem_A_idx) = [];
        end
        %for B trials
        if ~isempty(rem_B_idx)
            rewards.B.position(rem_B_idx) = [];
        end
        
        %mean positions for non-target reward zone speed calculation
        rewards.A.pos_mean = mean(rewards.A.position);
        rewards.B.pos_mean = mean(rewards.B.position);
        
        %lap idx by trial type
        lap_idx.A = find(Behavior.lap_id.trial_based == 2);
        lap_idx.B = find(Behavior.lap_id.trial_based == 3);
        
        %load int
        
        %resampled lap idx corrsponding to resampled and restricted time space
        lap_idx_resampled = Behavior.resampled.lapNb;
        
        %get idxs correspoinding to each lap
        %for A laps
        for ll = 1:size(lap_idx.A,2)
            %target A reward zones
            rewards.A.res_idx{ll} = find(lap_idx_resampled == lap_idx.A(ll));
            [~,rewards.A.Imin(ll)] = min(abs(position(rewards.A.res_idx{ll}) - rewards.A.position(ll)));
            rewards.A.lap_position{ll} = position(rewards.A.res_idx{ll});
            rewards.A.speed{ll} = speed(rewards.A.res_idx{ll});
            %non-target B reward zones
            [~,rewards.A.IminB(ll)] = min(abs(position(rewards.A.res_idx{ll}) - rewards.B.pos_mean));
            %extract binned indices corresponding to A laps
            rewards.A.pos_bins{ll} = pos_bins(rewards.A.res_idx{ll});
        end
        
        %for B laps
        for ll = 1:size(lap_idx.B,2)
            rewards.B.res_idx{ll} = find(lap_idx_resampled == lap_idx.B(ll));
            [~,rewards.B.Imin(ll)] = min(abs(position(rewards.B.res_idx{ll}) - rewards.B.position(ll)));
            rewards.B.lap_position{ll} = position(rewards.B.res_idx{ll});
            rewards.B.speed{ll} = speed(rewards.B.res_idx{ll});
            %non-target A reward zones
            [~,rewards.B.IminA(ll)] = min(abs(position(rewards.B.res_idx{ll}) - rewards.A.pos_mean));
            %extract binned indices corresponding to A laps
            rewards.B.pos_bins{ll} = pos_bins(rewards.B.res_idx{ll});
        end
        
        %linearize A laps speed and bin cells
        rewards.A.linear_pos_bins = cell2mat(rewards.A.pos_bins');
        rewards.A.linear_speed = cell2mat(rewards.A.speed');
        
        %linearize B laps speed and bin cells
        rewards.B.linear_pos_bins = cell2mat(rewards.B.pos_bins');
        rewards.B.linear_speed = cell2mat(rewards.B.speed');
                
        %find each bin and calculate mean velocity in that bin
        for bb=1:100
            %A
            %idx's of each bin
            bin_idxs.A{bb} = find(rewards.A.linear_pos_bins == bb);
            %get velocities in each bin and calcualte mean velocity
            bin_speed.A(bb) = mean(rewards.A.linear_speed(bin_idxs.A{bb}));
            %B
            %idx's of each bin
            bin_idxs.B{bb} = find(rewards.B.linear_pos_bins == bb);
            %get velocities in each bin and calcualte mean velocity
            bin_speed.B(bb) = mean(rewards.B.linear_speed(bin_idxs.B{bb}));
        end
        
                %export relevant data for cumulative processing
        %export for cumulative analysis
        %make cumul_analysis folder
        mkdir(setDir,'cumul_analysis')
        %save the fractions output data
        save(fullfile(setDir,'cumul_analysis','speed_data.mat'),...
            'Behavior','Imaging','position','speed','lap_idx_resampled','rewards',...
            'bin_speed','bin_idxs');
end

%7A and 9B


%% Move this to common plot
%plot speed within A range for A laps
figure
subplot(2,2,2)
hold on
title('A zone speed')
for ll=1:size(lap_idx.A,2)
    %plot line plot along range
    plot(rewards.A.speed{ll}((rewards.A.Imin(ll)-idx_width):(rewards.A.Imin(ll)+idx_width)));
    %plot line showing start of reward range
    plot([idx_width idx_width],[0 25],'k')
end
subplot(2,2,1)
hold on
title('B zone speed')
for ll=1:size(lap_idx.A,2)
    %plot line plot along range
    plot(rewards.A.speed{ll}((rewards.A.IminB(ll)-idx_width):(rewards.A.IminB(ll)+idx_width)));
    %plot line showing start of reward range
    plot([idx_width idx_width],[0 25],'k')
end

subplot(2,2,4)
hold on
title('A zone speed')
for ll=1:size(lap_idx.B,2)
    %plot line plot along range
    plot(rewards.B.speed{ll}((rewards.B.IminA(ll)-idx_width):(rewards.B.IminA(ll)+idx_width)));
    %plot line showing start of reward range
    plot([idx_width idx_width],[0 25],'k')
end

subplot(2,2,3)
hold on
title('B zone speed')
for ll=1:size(lap_idx.B,2)
    %plot line plot along range
    plot(rewards.B.speed{ll}((rewards.B.Imin(ll)-idx_width):(rewards.B.Imin(ll)+idx_width)));
    %plot line showing start of reward range
    plot([idx_width idx_width],[0 25],'k')
end


%QC check
%plot
figure
for ll=1:size(lap_idx.A,2)
    subplot(size(lap_idx.A,2),1,ll)
    hold on
    plot(position(rewards.A.res_idx{ll}))
    plot([rewards.A.Imin(ll) rewards.A.Imin(ll)], [0 200])
end

figure
for ll=1:size(lap_idx.B,2)
    subplot(size(lap_idx.B,2),1,ll)
    hold on
    plot(position(rewards.B.res_idx{ll}))
    plot([rewards.B.Imin(ll) rewards.B.Imin(ll)], [0 200])
end

%bin data spatially and take peri 5 bin around reward zone



%% Save Behavior struct temporarily here - later do at end

% fprintf('Saving behavioral data...');
% save(fullfile(directory_name,'output','Behavior.mat'),'Behavior');
% fprintf('Done\n');

%{
%% Restrict data to complete laps

% Restrict calcium data to selected lap -  restrict trace to only full laps
options.restrict=1; 

%Select from which starting lap
options.startlap= 'first'; 

%Select the end lap
options.endlap='last';

%display the calcium trace aligned to behavior figure
options.dispfig=1; % Display figure

%neurons to display
options.ROI =40; 

[Imaging, Behavior, F_vars] = restrict_data(C_df, Behavior, XML, options,F_vars);

%% Event detection -initial detection with 2 sigma onset - WORKS (for restricted as well)!

options.update_events = 0;
options.imaging_rate = 30; %in Hz
options.mindurevent = 1; %in s (250 ms Zaremba 2017 / 1 s Danielson 2016)
options.restrict = true; %only analyze data from complete laps 

%which ROI to plot
options.ROI = 40;
%how many iterations (after initial detection)
options.it = 2;
%set to zero before entering iteration stage in the later cell (initialization)
options.iterationNb = 0;

%go through this and the recalculation function 
%make sure that the recalculation function uses the same inputs as the
%initial function used to compute the dF/F
tic;
[Events] = detect_events(options, Imaging);
toc; 

%% Remove events, recalculate baseline and dF/F, baseline sigma and update events iteratively - WORKS (for restricted as well)!

%add a plot summary for this

%whether to use recalculate sigma and mean to detect events
options.update_events = 1;
%temporary
options.dff_type = 'Rolling median';

%check; imported from OCGOL
%add pass for dF/F calculation parameters to function
tic
for ii =1:options.it
    %which iteration the loop is on - used for duration filter
    %only filters on last iteration of the loop
    options.iterationNb = ii;
    %Update dff and sigma
    updated_dff = update_dff_revised(Events.onset_offset, F_vars, options);
    
    % Update events
    [Events] = detect_events(options, Imaging, updated_dff);
    
end
toc

%% Update imaging struct with recalculated dF/F with event masking - TODO

%% Resample behavioral data
%todo - add resampling of updated dF/F in imaging struct

[Behavior] = resample_behavioral_data(Behavior,Imaging,options);

%% Resample data and determine run epochs

%Danielson or Cossart method
options.method='peak'; %'peak' (Danielson) or 'speed'

% threshold running epochs based on :
% min peak speed(Danielson et al. 2016a, b, 2016) 
% OR average speed (Cossart) 
options.moving_window=10; % window width for moving mean filter and speed filter - Cossart too
%options.minspeed=2; %minimum speed (cm/s)  -- only if 'speed' - Cossart
%too
%merging is also done for Cossart in the script
%minimum duration criterion applies as well

%Danielson criteria
options.minpeak=5;  % minimum peak speed (cm/s) -- only if 'peak' %Danielson 2016b that the animal has to reach within epoch
options.mindur=1; %Minimum duration (s) for running epoch Danielson 2016b
options.merge=0.5; %Merge consecutive running epochs separated by less than (s) Danielson 2016b
options.minspeed = 0; %minimum speed for run epoch threshold (can raise in increments when animal does micromovements around stop point)
options.dispfig=1; % Display figure

%which ROI to display
options.c2plot = 40;

%TODO - merge updated dF/F from above 
[Events, Behavior] = determine_run_epochs(Events,Behavior,Imaging, updated_dff, options);

%% Split into behavior, imaging, event data into individual laps - work on this!

%modify to split laps irrespective of behavior - save into a single struct
%[Behavior_split,Imaging_split,Events_split,Behavior_split_lap,Events_split_lap] = split_laps(Behavior,Imaging,CSV,Events);

%% Split trials based on OCGOL performance

%TODO - split also the updated dF/F traces

switch options.BehaviorType
    case 'RF' %work on this
        %[Behavior_split,Imaging_split,Events_split,Behavior_split_lap,Events_split_lap] = split_laps(Behavior,Imaging,CSV,Events);
    case 'GOL-RF' %work on this
        
    case 'GOL' %work on this
        
    case 'OCGOL'
        [Behavior_split,Imaging_split,Events_split,Behavior_split_lap,Events_split_lap] = split_trials_OCGOL(Behavior,Imaging,Events,options);
end

%% Extract calcium event properties for split and all laps - check this

%for OCGOL data
%for each of 3 conditions = 1 - all, 2 - run, 3 = norun (inside function)
%for each trial type (trial type)


%TODO - update this using the updated dF/F imaging struct
for ii=1:5
    Events_split{ii} = event_properties(updated_dff, Events_split{ii},options);
end

%without split on all lap restricted data
Events = event_properties(updated_dff, Events,options);

%% Identification of spatially-tuned cells - works RZ with new events - split this
% Set parameters
options.sigma_filter=3; % Sigma (in bin) of gaussian filter (Danielson et al.2016 = 3 bins)
options.smooth_span=3; % span for moving average filter on dF/F (Dombeck 2010 = 3) (Sheffield 2015 = 3)
options.minevents=3; % Min nb of events during session
options.Nbin=[2;4;5;8;10;20;25;100]; % Number of bins to test ([2;4;5;8;10;20;25;100] Danielson et al. 2016)
options.bin_spatial_tuning=100; % Number of bins to compute spatial tuning curve (rate map) -value must be in options.Nbin
options.Nshuffle=1000; % Nb of shuffle to perform
options.pvalue=0.05; % Min p value to be considered as significant
options.dispfig=1; % Display figure 

%use binned position rather than raw for tuning specificity calculation
options.binPosition = 1;

tic;
%for all type of trials --> current: 1 -A trials; 2 -B trials; 3 - all
%laps/trials
for ii=1:3
    %work on this part
    %spatial binning, rate maps, and spatial tuning score
    disp('Calculate bin space, events, rate maps, generate STCs, and SI score')
    [Place_cell{ii}] = spatial_properties(Behavior_split{ii}, Events_split{ii}, Imaging_split{ii},options);
    %tuning specific calculation function here
    disp('Calculate turining specificity score')
    [Place_cell{ii}] = tuning_specificity_RZ_V2(Place_cell{ii},Behavior_split{ii},Events_split{ii},options);
    
end
toc; 

%% Save relevant variables for shuffle processing

%save(fullfile(directory_name,))

%% Run place cell shuffle for spatial information
%offload this to HPC for processing
%check what the difference is between 
for ii=1:3
    [Place_cell{ii}] = shuffle_place_cell(Place_cell{ii},Behavior_split{ii},Events_split{ii},options);
end
%alternative way of calculating shuffle for SI and TS - not finalized
%[Place_cell{ii}]=shuffle_place_cell_spatial_V2_RZ(Place_cell{ii},Behavior_split{ii},Events_split{ii},options);

%% Create tuned ROI binary mask for tuned cells and add to Place_cell struct
%add this to shuffle script and remove from there
%nb of ROIs
ROInb = size(Imaging.trace,2);

%for A, B, and all laps
for ii=1:size(Place_cell,2)
    tunedROImask = zeros(1,ROInb);
    tunedROImask(Place_cell{1,ii}.Tuned_ROI) = 1;
    Place_cell{1,ii}.Tuned_ROI_mask =  tunedROImask;
end


%% save the workspace

%make output folder containing the saved variables in the experiments
%directory
save_path = directory_name;

try
    cd(save_path)
    mkdir('output');
    cd(fullfile(save_path, ['\','output']))
catch
    disp('Directory already exists');
    cd(fullfile(save_path, ['\','output']))
end

%get date
currentDate = date;
%replace dashes with underscores in date
dashIdx = regexp(currentDate,'\W');
currentDate(dashIdx) = '_';

%to exclude certain variables
%save([currentDate,'.mat'],'-regexp','^(?!(data)$).','-v7.3');

%save relevant variable for further analysis
tic
save([currentDate,'_ca_analysis.mat'],...
    'directory_name','Place_cell','Events_split','Behavior_split','Imaging_split',...
    'Behavior','Imaging', 'Events', 'updated_dff',...
    'F_vars', 'options', 'Behavior_split_lap',...
    'Events_split_lap','-v7.3');
toc

disp('Done saving.');

%}

%% For modifcation below
%{

%% Place field finder (Dombeck/Barthos 2018)
%Place field finder - Dombeck/Harvey 2010
tic;
%imaging period
options.dt = Imaging.dt;
%number of shuffles
options.nbShuffle = 1000;

%detect place fields and return - real
for ii=1
    [Place_cell{ii}] =  placeFieldDetection_barthosBeta(Place_cell{ii});

    
    if 0
    %shuffled the dF/F traces
    %1 min for 1000 shuffles of indices
    trace_shuffled_mat = shuffle_dFF(Imaging_split{ii},options);
    
    %recalculate the inputs parameters for each shuffle run
    %preallocate shuffle matrix with # of place fields per ROI
    placeField_nb_shuffle = zeros(options.nbShuffle,171);
    PF_inputs = cell(1,options.nbShuffle);
    
    tic;
    parfor n=1:options.nbShuffle
        %generate inputs based on shuffle
        PF_inputs{n} = activity_map_generator_PF(Behavior_split{ii}, Events_split{ii}, trace_shuffled_mat(:,:,n),options);
        %detect place fields and return - shuffled
        [placeField_nb_shuffle(n,:)] =  placeFieldDetection_barthosBeta_shuffle(PF_inputs{n});
    end
    toc;
    
    %get p_value for each ROI
    pVal = sum(logical(placeField_nb_shuffle))/options.nbShuffle;
    
    %assign which ROIs have sig fields
    Place_cell{ii}.placeField.sig_ROI = pVal < 0.05;
    end
    
end
toc;

%% Centroid difference/shift (Danielson 2016)
%use this
%calculate the angle between the the tuning specificity vectors (not
%difference)

%whether to plot figure
options.plotFigure = 0;

%centroid threshold - how much angular separation between tuning vectors
options.centroidThres = pi/10; %(pi/5 - 20 cm); pi/10 - 10 cm

centroids = centroidDiffV2_newEvents(Place_cell,options);


%% Event spiral and mean dF/F by lap plots - Zaremba 2017 based
%modify this for single trials

% plots each event for given ROI during running epochs along lap across laps for

%where to make avi video of selected plotd
options.makeVideo = 0;
options.videoName = 'cellEvents_all';

%speed at which each event transitions to the next
options.plotSpeed = 0.001;
%plot all on top of one another vs each individually
options.hold = 0;

%add silence figure option to code
options.suppressFigure = 1;

%to manually move through the events
options.manualAdvance = 0;

ROIrange = 242;


spiralEvents = spiral_eventsV6_newSplit_PFtest(Behavior_split_lap, Events_split_lap, Behavior_split, Events_split, Imaging_split, Place_cell, Behavior, ROIrange,options);


%% Plot Spatial tuning curves (STC) with selected ROI, PV correlation, TC correlation, PF distribution
%need input from ROI selector function

%spatial tuning curves
%which trials to sort by
options.sortOrder = 1;
plotSTC(selectROI_PF_idx,Place_cell,options)

%place field distributions, place field counts, and PF distributions
plotPF(Place_cell)

%spatial correlation function goes here
[PVcorr,TCcorr] = spatialCorr(Place_cell);

%}

