%% import prism fraction of licks data on A and B trials

dirpath = 'C:\Users\rzeml\Google Drive\task_selective_place_paper\matlab_data\fig1_csv_data_prism_export';

%fraction of licks in reward zone
frac_licks_A = readmatrix(fullfile(dirpath,'A fraction of licks in reward zone.csv'),'Range','B2:E5');
frac_licks_B = readmatrix(fullfile(dirpath,'B fraction of licks in reward zone.csv'),'Range','B2:E5');

%fraction correct trials
frac_corr_A = readmatrix(fullfile(dirpath,'Fraction correct A trials.csv'),'Range','B2:E5');
frac_corr_B = readmatrix(fullfile(dirpath,'Fraction correct B trials.csv'),'Range','B2:E5');

%% import data for figure 2 stats

dirpath = 'C:\Users\rzeml\Google Drive\task_selective_place_paper\matlab_data';

load(fullfile(dirpath,'source_data_fig2.mat'))


%% Mean and sem for licking fraction of correct trials across training sessions

nb_animals = size(frac_licks_A,2);
%licking mean and sem
mean_lick.A = mean(frac_licks_A,1);
mean_lick.B = mean(frac_licks_B,1);

sem_lick.A = std(frac_licks_A,0,1)./sqrt(nb_animals);
sem_lick.B = std(frac_licks_B,0,1)./sqrt(nb_animals);

%fraction correct mean and sem
mean_corr.A = mean(frac_corr_A,1);
mean_corr.B = mean(frac_corr_B,1);

sem_corr.A = std(frac_corr_A,0,1)./sqrt(nb_animals);
sem_corr.B = std(frac_corr_B,0,1)./sqrt(nb_animals);

%% Mean and sem data for figure 2

nb_FOV = size(source_data_task_sel_remap.mean_AUC.run.Asel,1);

mean_AUC.run.A = mean(source_data_task_sel_remap.mean_AUC.run.Asel,1);
mean_AUC.run.B = mean(source_data_task_sel_remap.mean_AUC.run.Bsel,1);
mean_AUC.run.AB = mean(source_data_task_sel_remap.mean_AUC.run.AB,1);

sem_AUC.run.A = std(source_data_task_sel_remap.mean_AUC.run.Asel,0,1)./sqrt(nb_FOV);
sem_AUC.run.B = std(source_data_task_sel_remap.mean_AUC.run.Bsel,0,1)./sqrt(nb_FOV);
sem_AUC.run.AB = std(source_data_task_sel_remap.mean_AUC.run.AB,0,1)./sqrt(nb_FOV);


%% Export the data to word

%% Start Word document that will contain the formatted stats data

WordFileName='fig_1_mean_sem.doc';
CurDir=pwd;
FileSpec = fullfile(CurDir,WordFileName);
%active X handle for manipulating document (ActXWord)
[ActXWord,WordHandle]=StartWord(FileSpec);

fprintf('Document will be saved in %s\n',FileSpec);

%% Licking mean and sem in reward zones

%RF A trials
mean_txt = mean_lick.A(1);
sem_txt = sem_lick.A(1);

txt_input = 'RF A trials licking in reward zone';
writeDefaultWordText(ActXWord,WordHandle,txt_input);
%newline
writeWordEnter(ActXWord,WordHandle,1);
write_mean_sem(ActXWord,WordHandle, mean_txt,sem_txt);
writeWordEnter(ActXWord,WordHandle,1);

%Random AB A trials
mean_txt = mean_lick.A(4);
sem_txt = sem_lick.A(4);

txt_input = 'Random AB A trials licking in reward zone';
writeDefaultWordText(ActXWord,WordHandle,txt_input);
%newline
writeWordEnter(ActXWord,WordHandle,1);
write_mean_sem(ActXWord,WordHandle, mean_txt,sem_txt);
writeWordEnter(ActXWord,WordHandle,1);

%RF B trials
mean_txt = mean_lick.B(1);
sem_txt = sem_lick.B(1);

txt_input = 'RF B trials licking in reward zone';
writeDefaultWordText(ActXWord,WordHandle,txt_input);
%newline
writeWordEnter(ActXWord,WordHandle,1);
write_mean_sem(ActXWord,WordHandle, mean_txt,sem_txt);
writeWordEnter(ActXWord,WordHandle,1);

%Random AB B trials
mean_txt = mean_lick.B(4);
sem_txt = sem_lick.B(4);

txt_input = 'Random AB B trials licking in reward zone';
writeDefaultWordText(ActXWord,WordHandle,txt_input);
%newline
writeWordEnter(ActXWord,WordHandle,1);
write_mean_sem(ActXWord,WordHandle, mean_txt,sem_txt);
writeWordEnter(ActXWord,WordHandle,1);

%% Fraction correct trials mean and sem
writeWordEnter(ActXWord,WordHandle,1);
writeWordEnter(ActXWord,WordHandle,1);

%RF A trials
mean_txt = mean_corr.A(1);
sem_txt = sem_corr.A(1);

txt_input = 'RF A trials fraction of correct trials';
writeDefaultWordText(ActXWord,WordHandle,txt_input);
%newline
writeWordEnter(ActXWord,WordHandle,1);
write_mean_sem(ActXWord,WordHandle, mean_txt,sem_txt);
writeWordEnter(ActXWord,WordHandle,1);

%Random AB A trials
mean_txt = mean_corr.A(4);
sem_txt = sem_corr.A(4);

txt_input = 'Random AB A trials fraction of correct trials';
writeDefaultWordText(ActXWord,WordHandle,txt_input);
%newline
writeWordEnter(ActXWord,WordHandle,1);
write_mean_sem(ActXWord,WordHandle, mean_txt,sem_txt);
writeWordEnter(ActXWord,WordHandle,1);

%RF B trials
mean_txt = mean_corr.B(1);
sem_txt = sem_corr.B(1);

txt_input = 'RF B trials fraction of correct trials';
writeDefaultWordText(ActXWord,WordHandle,txt_input);
%newline
writeWordEnter(ActXWord,WordHandle,1);
write_mean_sem(ActXWord,WordHandle, mean_txt,sem_txt);
writeWordEnter(ActXWord,WordHandle,1);

%Random AB B trials
mean_txt = mean_corr.B(4);
sem_txt = sem_corr.B(4);

txt_input = 'Random AB B trials fraction of correct trials';
writeDefaultWordText(ActXWord,WordHandle,txt_input);
%newline
writeWordEnter(ActXWord,WordHandle,1);
write_mean_sem(ActXWord,WordHandle, mean_txt,sem_txt);
writeWordEnter(ActXWord,WordHandle,1);

%% Close Word document
CloseWord(ActXWord,WordHandle,FileSpec);

