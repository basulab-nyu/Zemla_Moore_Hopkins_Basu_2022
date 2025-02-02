function OCGOL_global_remapping_V0_update(path_dir,crossdir,options)

%% Import variables and define options

%run componenet registration across sessions
options.register = 1;
% 
% %whether to load place field data processed below
options.loadPlaceField_data = 0;
% 
% %load extracted ROI zooms/outlines
options.load_ROI_zooms_outlines = 1;
% 
% %visualize ROI outlines of matches across sessions
options.visualize_match = 1;
% 
% %load SCE data shuffled n=50/100 (re-shuffle later on cluster with n =1000)
%options.loadSCE = 0;
% 
% %all A and B trials used for learning (sessions determined below)
options.selectTrial = [4 5];

%for use in global workspace
selectTrial = options.selectTrial;

%flag to all A or B trial or only correct A or B trials
%all correct = 0 ==> uses trials 4,5 (set for learning data)
%all correct = 1 ==> uses trials 1,2 (set for recall data)
% options.allCorrect = 0;
% 
% %this is just for controlling the display of labels across sessions/days
% %doesn't control anything else
% options.learning_data = 0;


%ANIMAL #1
%I56_RTLS
%input directories to matching function
%  path_dir = {'G:\OCGOL_learning_short_term\I56_RTLS\I56_RLTS_5AB_041019_1',...
%      'G:\OCGOL_learning_short_term\I56_RTLS\I56_RTLS_5AB_041119_2',...
%      'G:\OCGOL_learning_short_term\I56_RTLS\I56_RTLS_3A3B_041219_3',...
%      'G:\OCGOL_learning_short_term\I56_RTLS\I56_RTLS_3A3B_041319_4',...
%      'G:\OCGOL_learning_short_term\I56_RTLS\I56_RTLS_ABrand_no_punish_041519_5',...
%      'G:\OCGOL_learning_short_term\I56_RTLS\I56_RTLS_ABrand_no_punish_041619_6',...
%      'G:\OCGOL_learning_short_term\I56_RTLS\I56_RTLS_ABrand_punish_041719_7'};
% % %cross session directory
%  crossdir = 'G:\OCGOL_learning_short_term\I56_RTLS\crossSession_update';

%ANIMAL #2
%I57_RTLS
%  path_dir = {'G:\OCGOL_learning_short_term\I57_RTLS\I57_RLTS_5AB_041019_1',...
%      'G:\OCGOL_learning_short_term\I57_RTLS\I57_RTLS_5AB_041119_2',...
%      'G:\OCGOL_learning_short_term\I57_RTLS\I57_RTLS_3A3B_041219_3',...
%      'G:\OCGOL_learning_short_term\I57_RTLS\I57_RTLS_1A1B_041319_4',...
%      'G:\OCGOL_learning_short_term\I57_RTLS\I57_RTLS_ABrand_no_punish_041519_5',...
%      'G:\OCGOL_learning_short_term\I57_RTLS\I57_RTLS_ABrand_no_punish_041619_6',...
%      'G:\OCGOL_learning_short_term\I57_RTLS\I57_RTLS_ABrand_punish_041719_7'};
% 
% %cross session directory
% crossdir = 'G:\OCGOL_learning_short_term\I57_RTLS\crossSession_update';

%ANIMAL #3
%I57_LT
%  path_dir = {'G:\OCGOL_learning_short_term\I57_LT\I57_LT_5A5B_041619_1',...
%      'G:\OCGOL_learning_short_term\I57_LT\I57_LT_5A5B_041719_2',...
%      'G:\OCGOL_learning_short_term\I57_LT\I57_LT_3A3B_041819_3',...
%      'G:\OCGOL_learning_short_term\I57_LT\I57_LT_3A3B_041919_4',...
%      'G:\OCGOL_learning_short_term\I57_LT\I57_LT_ABrand_no_punish_042019_5',...
%      'G:\OCGOL_learning_short_term\I57_LT\I57_LT_ABrand_no_punish_042119_6',...
%      'G:\OCGOL_learning_short_term\I57_LT\I57_LT_ABrand_punish_042219_7',...
%      'G:\OCGOL_learning_short_term\I57_LT\I57_LT_ABrand_punish_042319_8'};
%  
%  crossdir = 'G:\OCGOL_learning_short_term\I57_LT\crossSession_update';

%ANIMAL #4
%I58 RT
%input directories to matching function
%  path_dir = {'E:\OCGOL_learning_short_term\I58_RT\I58_RT_5A5B_073019_1',...
%      'E:\OCGOL_learning_short_term\I58_RT\I58_RT_5A5B_073119_2',...
%      'E:\OCGOL_learning_short_term\I58_RT\I58_RT_3A3B_080119_3',...
%      'E:\OCGOL_learning_short_term\I58_RT\I58_RT_3A3B_080219_4',...
%      'E:\OCGOL_learning_short_term\I58_RT\I58_RT_randAB_no_punish_080319_5',...
%      'E:\OCGOL_learning_short_term\I58_RT\I58_RT_randAB_no_punish_080419_6',...
%      'E:\OCGOL_learning_short_term\I58_RT\I58_RT_ABrand_punish_080519_7',...
%      'E:\OCGOL_learning_short_term\I58_RT\I58_RT_randAB_punish_080619_8',...
%      'E:\OCGOL_learning_short_term\I58_RT\I58_RT_randAB_punish_080719_9'};
% %cross session directory
% crossdir = 'E:\OCGOL_learning_short_term\I58_RT\crossSession_update';

%ANIMAL #5
%I58 LT
% path_dir = {'E:\OCGOL_learning_short_term\I58_LT\I58_LT_5A5B_080419_1',...
%      'E:\OCGOL_learning_short_term\I58_LT\I58_LT_5A5B_080519_2',...
%      'E:\OCGOL_learning_short_term\I58_LT\I58_LT_3A3B_080619_3',...
%      'E:\OCGOL_learning_short_term\I58_LT\I58_LT_3A3B_080719_4',...
%      'E:\OCGOL_learning_short_term\I58_LT\I58_LT_randAB_no_punish_080819_5',...
%      'E:\OCGOL_learning_short_term\I58_LT\I58_LT_randAB_no_punish_080919_6',...
%      'E:\OCGOL_learning_short_term\I58_LT\I58_LT_randAB_punish_081119_7'};
% %cross session directory
% crossdir = 'E:\OCGOL_learning_short_term\I58_LT\crossSession_update';

%ANIMAL #6
% %I58 RTLP
%  path_dir = {'E:\OCGOL_learning_short_term\I58_RTLP\I58_RTLP_5A5B_080419_1',...
%      'E:\OCGOL_learning_short_term\I58_RTLP\I58_RTLP_5A5B_080519_2',...
%      'E:\OCGOL_learning_short_term\I58_RTLP\I58_RTLP_3A3B_080619_3',...
%      'E:\OCGOL_learning_short_term\I58_RTLP\I58_RTLP_3A3B_080719_4',...
%      'E:\OCGOL_learning_short_term\I58_RTLP\I58_RTLP_randAB_no_punish_080819_5',...
%      'E:\OCGOL_learning_short_term\I58_RTLP\I58_RTLP_randAB_no_punish_080919_6'};
% %cross session directory
% crossdir = 'E:\OCGOL_learning_short_term\I58_RTLP\crossSession_update';

%MR1
 path_dir = {'D:\OCGOL_reversal\MR1\MR1_Random_2022_02_28-001_1',...
     'D:\OCGOL_reversal\MR1\MR1_Random_2022_03_01-001_2',...
     'D:\OCGOL_reversal\MR1\MR1_Random_2022_03_02-002_3',...
     'D:\OCGOL_reversal\MR1\MR1_RevAB_2022_03_03-001_4',...
     'D:\OCGOL_reversal\MR1\MR1_RevRandom_2022_03_04-001_5',...
     'D:\OCGOL_reversal\MR1\MR1_RevRandom_2022_03_05-001_6',...
     'D:\OCGOL_reversal\MR1\MR1_RevRandom_2022_03_08-001_7',...
     'D:\OCGOL_reversal\MR1\MR1_RevRandom_2022_03_09-001_8',...
     'D:\OCGOL_reversal\MR1\MR1_RevRandom_2022_03_10-001_9'};
%cross session directory
crossdir = 'D:\OCGOL_reversal\MR1\crossSession_update';

%MR4
%  path_dir = {'D:\OCGOL_reversal\MR4\MR4_Random_2022_03_04-001_1',...
%      'D:\OCGOL_reversal\MR4\MR4_Random_2022_03_05-001_2',...
%      'D:\OCGOL_reversal\MR4\MR4_Random_2022_03_06-001_3',...
%      'D:\OCGOL_reversal\MR4\MR4_RevAB_2022_03_07-002_4',...
%      'D:\OCGOL_reversal\MR4\MR4_RevRandom_2022_03_08-001_5',...
%      'D:\OCGOL_reversal\MR4\MR4_RevRandom_2022_03_09-001_6',...
%      'D:\OCGOL_reversal\MR4\MR4_RevRandom_2022_03_11-001_7',...
%      'D:\OCGOL_reversal\MR4\MR4_RevRandom_2022_03_12-001_8',...
%      'D:\OCGOL_reversal\MR4\MR4_RevRandom_2022_03_13-001_9'};
% %cross session directory
% crossdir = 'D:\OCGOL_reversal\MR4\crossSession_update';

%% Determine number of sessions to analyze for animal (automatically calculated)

%which session to include in calculation
options.sessionSelect = 1:size(path_dir,2);

%for use as var in global workspace
sessionSelect = options.sessionSelect;

%% Override to selection (only 1st for animal 1)
% options.sessionSelect = 1;
% sessionSelect = 1;


%% Load place cell variables for each session
%get mat directories in each output folder
for ii=options.sessionSelect
    %get matfile names for each session
    matfiles{ii} = dir([path_dir{ii},'\output','\*.mat']);
end

%load in place cell variables (and others later)
for ii = options.sessionSelect%1:size(path_dir,2)
    %add event variables
    disp(ii)
    %decide which variables here do not need to be loaded
    fileName = matfiles{ii}(1).name
    if endsWith(fileName,'ca_analysis.mat')
        session_vars{ii} = load(fullfile(matfiles{ii}(1).folder,matfiles{ii}(1).name),'Place_cell', 'Behavior',...
        'Behavior_split_lap','Behavior_split','Events_split','Events_split_lap', 'Imaging_split');
    else
    end
end

%load additional data from struct
for ii = options.sessionSelect
    %add event variables
    disp(ii)
    fileName = matfiles{ii}(1).name
    if endsWith(fileName,'ca_analysis.mat')
        session_vars_append{ii} = load(fullfile(matfiles{ii}(1).folder,matfiles{ii}(1).name),'Imaging','updated_dff','Events');
    else
    end
end

%assign to main session variable struct (additional variables)
for ii = options.sessionSelect
    session_vars{ii}.Imaging = session_vars_append{ii}.Imaging;
    session_vars{ii}.updated_dff = session_vars_append{ii}.updated_dff;
    session_vars{ii}.Events = session_vars_append{ii}.Events;
end


%% Match ROIs from across OCGOL days

if options.register == 1
    %run cross registration
    disp('Running registration of components');
    [registered] = match_ROIs_V2(path_dir,crossdir);
    
    %save registration variable in crosssession
    disp('Saving registered component matchings');
    save(fullfile(crossdir,'registered.mat'),'registered');
    
elseif options.register == 0
    %load the registered struct
    disp('Loading registered component matchings...');
    load(fullfile(crossdir,'registered.mat'),'registered');
    disp('Loaded.');
    
end

%% Load filtered ROI matches into registered struct

%get dir path with wildcard match to .mat files
filtered_ROI_dir_path = subdir(fullfile(crossdir,'filtered_match_ROI','*.mat'));
%load in temp var
match_var = load(filtered_ROI_dir_path.name);
%load in registered struct
registered.multi.assigned_filtered = match_var.ROI_assign_multi_filtered;


%% Get ROI_zooms and ROI_outlines for each neuron on each day
%number of sessions (runs even if not all session vars are loaded)
%already soma parsed
nbSes = size(session_vars,2);

%extract and save ROI zooms/outlines for all neurons
if options.load_ROI_zooms_outlines == 0
    %calculate ROI zooms/outlines
    [ROI_zooms, ROI_outlines] = defineOutlines_eachSes(nbSes,session_vars, path_dir);
    %save to directory
    save(fullfile(crossdir,'ROI_zooms_outlines.mat'),'ROI_zooms','ROI_outlines');
else %load from save files
    load(fullfile(crossdir,'ROI_zooms_outlines.mat'),'ROI_zooms','ROI_outlines');
    disp('Loaded ROI zooms and outlines.')
end

%% Visualize the matching ROIs that were matched above (match on every session only!)
%number of ROIs (rows) by sessions (cols)
rows = 20;
cols = 6; %take # of sessions as input

%number of sessions to look at
nb_ses = cols;

if options.visualize_match ==1
    visualize_matches_filtered(rows,cols,registered,ROI_zooms,ROI_outlines,nb_ses,crossdir);
end

%% Generate correct and incorrect STCs and cross-correlate

[A,B,trial_counts_tbl] = corr_incorr_correlation(session_vars{1});

%% Calculate relevant place fields
%SAME FUNCTION AS USED FOR SINGLE SESSION DATA

if options.loadPlaceField_data == 0
    %use rate map - number of event onsets/ occupancy across all laps
    options.gSigma = 3;
    %which place cell struct to do placefield extraction on
    %iterate through place_cell cells of interest
    %4 - all A regardless if correct
    %5 - all B regardless if correct
    %I57 RTLS - problem with 4,4 - fixed
    %I57 LT - problem with ses 4, trial 5 adjust (set to -2) - narrow as opposed to
    %extend field - apply to rest of animals
    
    for ss =options.sessionSelect%1:size(session_vars,2) %1,2,3,4,5,6 OK
        %for ss= [4]
        disp(['Running session: ', num2str(ss)]);
        for ii = options.selectTrial
            options.place_struct_nb = ii;
            disp(['Running trial type: ', num2str(ii)]);
            [session_vars{ss}.Place_cell] = place_field_finder_gaussian(session_vars{ss}.Place_cell,options);
        end
    end
    
    %save whole place cell struct and load in and replace for each session in
    %the future
    %make post-processing directory (postProcess)
    mkdir(crossdir,'postProcess')
    
    %for each Place_cell session extract placeField struct
    %use trial types here
    for ss = options.sessionSelect
        for tt=options.selectTrial
            session_pf{ss}(tt).placeField = session_vars{ss}.Place_cell{tt}.placeField;
        end
    end
    
    %save Place_cell struct in that directory
    save(fullfile(crossdir,'postProcess','placeField_upd_struct.mat'),'session_pf')
    
else
    tic;
    disp('Loading place field data')
    load(fullfile(crossdir,'postProcess','placeField_upd_struct.mat'));
    toc
    %replace the Place_cell struct in the session_vars cell
    for ss = options.sessionSelect
        %all A and all B
        for tt=options.selectTrial
            session_vars{ss}.Place_cell{tt}.placeField = session_pf{ss}(tt).placeField;
        end
    end
end

%% Define tuned logical vectors
%SAME FUNCTION AS USED FOR SINGLE SESSION DATA

%select which session to use
%options.sessionSelect = [1 2 3 4 5 6];
%returns struct of structs
[tunedLogical] = defineTunedLogicals(session_vars,options);


%% Calculate the transient rates in each of the place fields (integrate later) and recalculate centroids based on highest transient rate field
%SAME FUNCTION AS USED IN SINGLE SESSION DATA

%funtion to calculate transient rate in field
%take raw event rate and divide by occupancy (s) transients/s
% TODO: see ifrecalculate to see if dividing by normalized occupancy (fractional 0-1)
%yields different result
%[field_event_rates,pf_vector] = transient_rate_in_field(session_vars);

%which trials to use to calculate the in field transient rate
%[1 2] - only correct A B trials
%[4 5] - all A B trials
%A correct/B correct or all
%options.selectTrial = [4 5];
%which sessions to run
%options.sessionSelect = [1 2 3 4 5 6];
%continue to modify 
[field_event_rates,pf_vector,field_total_events, select_fields] = transient_rate_in_field_multi_ses(session_vars,options);

%% Get max transient peak here
%SAME FUNCTION AS USED IN SINGLE SESSION DATA

%get field event rates of max peak
%adjust center vector to point to the field
options.select_adj_vec = 1;
[max_bin_rate,max_transient_peak] = max_transient_rate_multi_ses(session_vars,field_event_rates,pf_vector,options);

%% Filter filtered matching components for SI or TS tuning for at least on id'd place field and 5 events in firld
%QC checked

%which trials to use to calculate the in field transient rate
%options.selectTrial = [4 5];
%which session to include in calculation
%options.sessionSelect = [1 2 3 4 5 6];
%select fields has logical 1 for whichever neurons has a place field at at
%least 5 events on distinct laps within that PF - otherwise not PF
[registered] = filter_matching_components(registered,tunedLogical,select_fields,options);

%% Recurrence and fraction active analysis

%set to blank for learning analysis
removedROI_clean = [];

[recurr,frac_active,recurr_ex,frac_active_ex] = recurrence_analysis(registered,removedROI_clean,session_vars,tunedLogical,select_fields,options);

%save recurrence and fraction active of neurons

save(fullfile(crossdir,'recurrence.mat'),'recurr','frac_active','recurr_ex','frac_active_ex');

%% Export matching STCs for Jason for detecting splitter cells across time
%all neurons first with minimum number of events

[matching_tun_curves] = export_day_matched_STCs(session_vars,session_vars_append,registered);

%export the matching STCs
save(fullfile(crossdir,'matching_tun_curves.mat'),'matching_tun_curves');

%% Centroid difference (max transient rate)
%QC checked

%this parameter doesn't matter as it runs for all neurons
options.tuning_criterion = 'ts';
%which trials to use 
%options.selectTrial = [4 5];
%which session to include in calculation
%options.sessionSelect = [1 2 3 4 5 6];
[cent_diff,pf_vector_max] = centroid_diff_multi_ses(session_vars,tunedLogical, pf_vector,field_event_rates,select_fields,registered,options);

%save pf_vector_max and cent_diff for angle difference analysis

save(fullfile(crossdir,'pf_vector_max.mat'),'pf_vector_max');

%% TC correlation for matching (+/-) tuned ROIs - using Tuning Specificity only
%already filtered for at least 1 sig place field and 5 distinct in-field events
%these outputs are used to matching tuning curve day to day correlations
% tc_corr_match.STC_mat_AB_A
% tc_corr_match.STC_mat_AB_B

% figure
% imagesc(tc_corr_match.STC_mat_AB_A{1, 4})
% hold on
% colorbar

[tc_corr_match] = tc_corr_matching_neurons(session_vars,registered,options);


%% Get the same output from the function above but for SI tuned neurons using a hack on the inputs to the above function

%load the SI values into the existing TS tuned values
[tc_corr_match] = tc_corr_matching_neurons_si_output_hack(session_vars,registered,options,tc_corr_match);

%save to output file for cumulative analysis
save(fullfile(crossdir,'tc_corr_match.mat'),'tc_corr_match')

% isequal(cell2mat(tc_corr_match.ts.matching_ROI_all_day_STC.ts.A{1, 2}'),tc_corr_match.ts.STC_mat_A{1, 2}')

%% PV and TC correlations for all matching neurons (PV) in A and B trials across days (line plot); TC corr (for A tuned or B tuned on both days)

%set option as to how to select neurons for plots
%options.tuning_criterion = 'si'; %si or ts
%options.sessionSelect = [1 2 3 4 5 6 ];
%options.selectSes = [4 5];
%learning or recall datasets

[PV_TC_corr] = PV_TC_corr_across_days(session_vars,tunedLogical,registered,options);

%save to output file for cumulative analysis
save(fullfile(crossdir,'PV_TC_corr.mat'),'PV_TC_corr')


%% Extract performance fractions across sessions (respective laps)
%check if agree with manual analysis
%turn into table with future code upgrade

%which sessions to use
%options.sessionSelect = [1 2 3 4 5 6];

%performance and total laps for each session
[ses_perf,ses_lap_ct] = session_performance(session_vars,options);

%modify into table format with descriptors
%1st row - performance on all laps
%2nd row - A trial performance
%3rd row - B trial performance

for ii=1:size(session_vars,2)
    var_names{ii} = ['Session_',num2str(ii)];
end

%convert into table
ses_perf_table = array2table(ses_perf,'RowNames',{'All trials','A trials','B trials'},'VariableNames',var_names);
ses_lap_ct_table = array2table(ses_lap_ct,'RowNames',{'All trials','A trials','B trials'},'VariableNames',var_names);

%export performance tables for Jason
save(fullfile(crossdir,'perf_lap_tables.mat'),'ses_perf_table','ses_lap_ct_table');

%export session performance data
save(fullfile(crossdir,'ses_perf.mat'),'ses_perf','ses_lap_ct');


%% Split neurons by A or B task selective category - A or B selective (exclusive)
%QC checked - same function as used for single session analysis

%which criterion to use for task-selective ROIs
%ts or both - ts selects only selective neurons based on TS tuning
%criterion
%both - uses both SI and TS criterion to select selectiven neurons
options.tuning_criterion = 'both';
%display events vs position for each task selective neuron in A or B
options.dispFigure = 0;
[task_selective_ROIs] = task_selective_categorize_multi_ses(tunedLogical,session_vars, max_transient_peak,options);



%% Number of place fields and widths for each sub-class of neurons
%QC checked

%analysis is run for all neurons and task-selective neurons
%neurons are already filtered for at least 1 sig place field and 5 distinct
%events in field

%consider modifying this to work with placeField_properties (single_ses)
%not necessary since it seems to run correct

%doesn;t matter
options.tuning_criterion = 'si'; %si or ts

%A correct/B correct or all
%options.selectTrial = [4 5];
%options.sessionSelect = [1 2 3 4 5 6];
[placeField_dist, pf_count_filtered_log, pf_count_filtered] = placeField_properties_multi_ses(session_vars,tunedLogical,select_fields,task_selective_ROIs,options);

%create logical where there is at least 1 field with at least 5
%lap-distinct events (all neurons)
%pf count filtered - 
%pf_count_filtered_log 

%save the place field distributions output data
%save(fullfile(path_dir{1},'cumul_analysis','placeField_dist.mat'),'placeField_dist');

%% Calculate fraction tuned S.I. vs T.S for every session
%QC checked

%output from tuned logicals is already filtered for min 5 events and 1 sig
%place field
[tuned_fractions,tuned_logicals] = fractionTuned_multi_ses(tunedLogical,pf_count_filtered_log,options);

%save fractional count
save(fullfile(crossdir,'tuned_fractions.mat'),'tuned_fractions');
%save tuned logicals
save(fullfile(crossdir,'tuned_logicals.mat'),'tuned_logicals');


%% Task remapping filter - split into remapping categories
%use this for getting partial remappers
%fixed bug with edge selection as for single session data
%same code as for single session except runs through each session inside of
%function

%which criterion to use for task-selective ROIs
options.tuning_criterion = 'ts';
%display events vs position for each task selective neuron in A or B
options.dispFigure = 0;
%number of degrees of centroid difference
%45 deg ~25 cm; 
%36 deg ~20 cm;
%27 deg ~15 cm;
%18 dege ~10 cm
options.deg_thres = 18;
%ranges for splitting the global remappers
%0-10 cm; 10 - 30cm; 30+ cm
options.deg_ranges = [0 18 54];
%degree threshold for partial remappers
options.partial_deg_thres = [18 36];
%choice between KS test of unpaired Mann Whitney U (later)
%either 'ranksum' or ks
options.AUC_test = 'ranksum';
%significance level of test
options.p_sig = 0.05;
%make sure that this function does not overwrite the the previous
%task_selective_ROIs structure
if 0
    [task_remapping_ROIs,partial_field_idx] = remapping_categorize_multi_ses(cent_diff, tunedLogical ,pf_vector, session_vars,...
        max_transient_peak, pf_count_filtered,select_fields,options);
    
    %calculate total A&B neurons for each session
    for ss=options.sessionSelect
        remap_cat_count(ss,1) = length(task_remapping_ROIs{ss}.global_near);
        remap_cat_count(ss,2) = length(task_remapping_ROIs{ss}.global_far);
        remap_cat_count(ss,3) = length(task_remapping_ROIs{ss}.rate);
        remap_cat_count(ss,4) = length(task_remapping_ROIs{ss}.common);
        remap_cat_count(ss,5) = length(task_remapping_ROIs{ss}.partial);
        remap_cat_count(ss,6) = length(task_remapping_ROIs{ss}.mixed);
        task_remapping_ROIs{ss}.nbROI = sum(remap_cat_count(ss,:));
    end
    
    fraction_rel_AB = remap_cat_count./repmat(sum(remap_cat_count,2),1,6);
end

%% Save task-selective and task remapping neurons into struct neurons 

%get # of ROIs in each session
for ss=sessionSelect
    ses_nbROI(ss) = size(session_vars{ss}.Place_cell{selectTrial(1)}.Tuned_ROI_mask,2);
end

if 0
save(fullfile(crossdir,'task_neurons.mat'),'task_selective_ROIs','task_remapping_ROIs','ses_nbROI');
end

%% Load selective neurons
if 0
load(fullfile(crossdir,'task_neurons.mat'),'task_selective_ROIs','task_remapping_ROIs','ses_nbROI');
end

%% Assign each matching neuron to remapping category
if 0
match_mat = registered.multi.assigned_filtered;
%make cell with categorical values
cat_registered_cell = cell(size(match_mat,1),size(match_mat,2));

%for each session determine which category each cell belongs to
for ss=options.sessionSelect
    [~,idx_match] = intersect(match_mat(:,ss),task_selective_ROIs{ss}.A.idx);
    cat_registered_cell(idx_match,ss) = {'A-selective'};
    [~,idx_match] = intersect(match_mat(:,ss),task_selective_ROIs{ss}.B.idx);
    cat_registered_cell(idx_match,ss) = {'B-selective'};
    [~,idx_match] = intersect(match_mat(:,ss),task_remapping_ROIs{ss}.common);
    cat_registered_cell(idx_match,ss) = {'common'};
    [~,idx_match] = intersect(match_mat(:,ss),task_remapping_ROIs{ss}.rate);
    cat_registered_cell(idx_match,ss) = {'rate'};
    [~,idx_match] = intersect(match_mat(:,ss),task_remapping_ROIs{ss}.global_near);
    cat_registered_cell(idx_match,ss) = {'near'};
    [~,idx_match] = intersect(match_mat(:,ss),task_remapping_ROIs{ss}.global_far);
    cat_registered_cell(idx_match,ss) = {'far'};
    [~,idx_match] = intersect(match_mat(:,ss),task_remapping_ROIs{ss}.partial);
    cat_registered_cell(idx_match,ss) = {'partial'};
    [~,idx_match] = intersect(match_mat(:,ss),task_remapping_ROIs{ss}.mixed);
    cat_registered_cell(idx_match,ss) = {'mixed'};
end

end
%extract 2 session index
% ses_comp = [4,5];
% selMatchIdxs = find(sum(~isnan(match_mat(:,ses_comp)),2)==2);
% 
% cat_registered_cell(selMatchIdxs,ses_comp);

%% Extract normalized events
tic;
[norm_events] = normalize_events_pos(session_vars,options);
toc;

%% Get AUC/min calculation for each ROI in A vs. B
%AUC/min and frequency of events events/min in run epoch for A or B trials
%all added no run epochs

[session_vars] = AUC_rate(session_vars,options);

%% Task-selective neurons - AUC/min and event freq distributions

%code for rose plot like event density is here
activity_distributions(session_vars,task_selective_ROIs,options)

%% Raster plot across days (FIGURE 4) (at least 2 match between sessions) raster (non_norm)
%this generates the figure rasters in 4
%currently takes all neurons regardless of tuning and traces them across
%time
%those with nan values on day 1 and excluded


%set option as to how to select neurons for plots
%this doesn't matter as input
options.tuning_criterion = 'si'; %si or ts
%options.sessionSelect = [1 2 3 4 5 6];
%chose all A/B (learning) vs. only correct A/B (recall)
%options.selectTrial = [4,5];

%is it a learning set (for plot/raster annotation); no effect on
%calculations
%options.learning_data = 1;
%non_norm_matching_STC_rasters(session_vars,tunedLogical,registered,options,crossdir)


%% Generate STC maps of neurons tuned in either session and plot side by side
%customize to add options
%tuned in both sessions by SI score
%sorted by A trials

%set option as to how to select neurons for plots
options.tuning_criterion = 'ts'; %si or ts
%options.sessionSelect = [1 2 3 6];
%plot_STC_OCGOL_training(session_vars,tunedLogical,registered,options)


%% FUNCTIONS BELOW ARE USEFUL FOR VISUALIZATION

if 0
    %% Raster spiral - prepare inputs for multi session display
    % save this in the future as well and load
    options.spiral_width = 0.1;
    [plot_raster_vars] = prepare_inputs_raster_spiral_multi_ses(session_vars,options);
    
    %% Raster, STC, ROI image across days - used this to plot in supplement or main figure
    %plots zoom on of ROI, spiral plot,
    %need to add normalized STC below as for
    %single session data
    
    plot_raster_spiral_STC_ROI_multi_ses(plot_raster_vars,session_vars,registered,cat_registered_cell,ROI_zooms,ROI_outlines,options)
    
    
    %% Visualize place fields and events for each neurons - useful for
    %use this to check code detection accuracy
    %displays spiral plot, event map, smoothed rate map, and SI/TS information
    visualize_neuron_characteristics(plot_raster_vars,norm_events,registered,session_vars,task_selective_ROIs,cat_registered_cell,select_fields,options)
    
    
    %% Plot spiral raster using inputs - single spiral plot for each neuron
    %ploy
    ROI_categories.task_selective_ROIs = task_selective_ROIs;
    
    %works with inputs
    plot_raster_spiral_multi_ses(plot_raster_vars,session_vars,registered,ROI_zooms, ROI_outlines,ROI_categories,options)
    
    
    %% Generate dF/F maps of neurons tuned in either session and plot side by side
    %customize to add options
    %tuned in both sessions by SI score
    %sorted by A trials
    
    %set option as to how to select neurons for plots
    %use SI or TS
    options.tuning_criterion = 'ts'; %si or ts
    
    %select which session to use
    %options.sessionSelect = [1 2 3 6];
    
    plot_dFF_OCGOL_training(session_vars,tunedLogical,registered,options)
    
    
    %% All neuron (at least 2 match between sessions) raster (non_norm)
    
    %set option as to how to select neurons for plots
    % options.tuning_criterion = 'si'; %si or ts
    % non_norm_matching_STC_rasters_learning(session_vars,tunedLogical,registered,options)
    
end


end
