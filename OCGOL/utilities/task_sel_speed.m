function [outputArg1,outputArg2] = task_sel_speed(tunedLogical,task_selective_ROIs,session_vars,ROI_idx_tuning_class,options)


%% Select trial tuned classes of neurons (use logicals as input)
%prefiltered for min 5 events on distinct laps

%S.I.
%for each session
for ss =1:size(session_vars,2)
    %spatial information criterion - regardless if tuned in other session
    Atuned.si{ss} = ROI_idx_tuning_class.si.log.Aonly | ROI_idx_tuning_class.si.log.AB;
    Btuned.si{ss} = ROI_idx_tuning_class.si.log.Bonly | ROI_idx_tuning_class.si.log.AB;

    onlyA_tuned.si{ss} = ROI_idx_tuning_class.si.log.Aonly;
    onlyB_tuned.si{ss} = ROI_idx_tuning_class.si.log.Bonly;    
    AandB_tuned.si{ss} = ROI_idx_tuning_class.si.log.AB;
    neither_tuned.si{ss} = ROI_idx_tuning_class.si.log.N;
    
end

%T.S.
for ss =1:size(session_vars,2)
    %regardless if tuned in other session
    Atuned.ts{ss} = ROI_idx_tuning_class.ts.log.Aonly | ROI_idx_tuning_class.ts.log.AB;
    Btuned.ts{ss} = ROI_idx_tuning_class.ts.log.Bonly | ROI_idx_tuning_class.ts.log.AB;

    onlyA_tuned.ts{ss} = ROI_idx_tuning_class.ts.log.Aonly;
    onlyB_tuned.ts{ss} = ROI_idx_tuning_class.ts.log.Bonly;    
    AandB_tuned.ts{ss} = ROI_idx_tuning_class.ts.log.AB;
    neither_tuned.ts{ss} = ROI_idx_tuning_class.ts.log.N;
end


for ss =1:size(session_vars,2)
    %blank all neuron idx logical vector
    all_neurons{ss} = true(size(Atuned.si{ss}));
    %blank logical equal to size of number of neurons
    blank_log = false(1,size(all_neurons{ss},2));
end

%A/B selective logicals; all A, all B, A&B by either criterion
for ss=1:size(session_vars,2)
    
    %A selective and B selective neurons
    A_sel{ss} = blank_log;
    A_sel{ss}(task_selective_ROIs.A.idx) = 1;
    
    B_sel{ss} = blank_log;
    B_sel{ss}(task_selective_ROIs.B.idx) = 1;
    
    %either criterion - A all,B all, A&B all
    Atuned.si_ts{ss} = Atuned.si{ss} | Atuned.ts{ss};
    Btuned.si_ts{ss} = Btuned.si{ss} | Btuned.ts{ss};
    AandB_tuned.si_ts{ss} = Atuned.si_ts{ss} & Btuned.si_ts{ss}; 
end

%% Convert to indices from logicals for input

%selective
A_sel_idx = find(A_sel{1} ==1);
B_sel_idx = find(B_sel{1} ==1);

%all A or B by either criterion or both
%S.I.
Atuned_idx.si = find(Atuned.si{1} ==1);
Btuned_idx.si = find(Btuned.si{1} ==1);
%T.S.
Atuned_idx.ts = find(Atuned.ts{1} ==1);
Btuned_idx.ts = find(Btuned.ts{1} ==1);
%S.I. or T.S
Atuned_idx.si_ts = find(Atuned.si_ts{1} ==1);
Btuned_idx.si_ts = find(Btuned.si_ts{1} ==1);

%AandB - either criterion or both
%S.I.
AandB_tuned_idx.si = find(AandB_tuned.si{1} ==1);
%T.S.
AandB_tuned_idx.ts = find(AandB_tuned.ts{1} ==1);
%S.I. or T.S.
AandB_tuned_idx.si_ts = find(AandB_tuned.si_ts{1} ==1);

%only A or B by either criterion
%S.I.
onlyA_tuned_idx.si = find(onlyA_tuned.si{1} ==1);
onlyB_tuned_idx.si = find(onlyB_tuned.si{1} ==1);
%T.S.
onlyA_tuned_idx.ts = find(onlyA_tuned.ts{1} ==1);
onlyB_tuned_idx.ts = find(onlyB_tuned.ts{1} ==1);


%% Load in run epoch and speed variables

%across entire restricted session
run_epoch_all_laps = session_vars{1}.Behavior.run_ones;

%across every lap
run_epoch_each_lap = session_vars{1}.Behavior_split_lap.run_ones;

%speed across all complete laps (downsampled to match frames)
speed_all_laps = session_vars{1}.Behavior.speed;

%normalized position across all laps
norm_pos_all_laps = session_vars{1}.Behavior.resampled.normalizedposition;


%% All lap speed and norm position
figure
hold on
%norm speed
plot(speed_all_laps', 'g')
%norm position
plot(norm_pos_all_laps-1, 'k')



end

