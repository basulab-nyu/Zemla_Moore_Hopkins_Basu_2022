%% import prism fraction of licks data on A and B trials

%% import data for figure 4 stats

dirpath = 'C:\Users\rzeml\Google Drive\task_selective_place_paper\matlab_data';

load(fullfile(dirpath,'source_data_fig4_5_and_sup.mat'))
load(fullfile(dirpath,'fig4_5_table_data.mat'))

%% Mean and sem data for figure 4

%%% All PV data
%PV A learn 2 vs 7
[mean_2_7.PV.learn.A,sem_2_7.learn.A,nb_comp_2_7.PV.learn.A] = learn_2_7_mean_sem_PV(source_data_short_learn_recall.PV.st_learn.A);

%PV A learn 2 vs 7
[mean_2_7.PV.learn.B,sem_2_7.learn.B,nb_comp_2_7.PV.learn.B] = learn_2_7_mean_sem_PV(source_data_short_learn_recall.PV.st_learn.B);

%learn data
s1 = source_data_short_learn_recall.PV.st_learn.A;
%recall data
s2 = source_data_short_learn_recall.PV.st_recall.d4_d5_sub.A;
%mean and sem comp
[mean_7_lr.PV.A, sem_7_lr.PV.A,nb_comp_PV.A] = learn_recall_7_mean_sem_PV(s1,s2);


%% Export the data to word

%% Start Word document that will contain the formatted stats data

WordFileName='fig_4_5_mean_sem.doc';
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

%% AUC for A/B/AB for run and norun epochs
writeWordEnter(ActXWord,WordHandle,1);
writeWordEnter(ActXWord,WordHandle,1);

%A sel RUN
txt_input = 'A sel A vs. B AUC/min RUN';
writeDefaultWordText(ActXWord,WordHandle,txt_input);
%newline
writeWordEnter(ActXWord,WordHandle,1);

mean_txt = mean_AUC.run.A(1);
sem_txt = sem_AUC.run.A(1);
write_mean_sem(ActXWord,WordHandle, mean_txt,sem_txt);

txt_input = ' vs. ';
writeDefaultWordText(ActXWord,WordHandle,txt_input);

mean_txt = mean_AUC.run.A(2);
sem_txt = sem_AUC.run.A(2);
write_mean_sem(ActXWord,WordHandle, mean_txt,sem_txt);

writeWordEnter(ActXWord,WordHandle,1);

%B RUN
txt_input = 'B-sel A vs. B AUC/min RUN';
writeDefaultWordText(ActXWord,WordHandle,txt_input);
%newline
writeWordEnter(ActXWord,WordHandle,1);

mean_txt = mean_AUC.run.B(1);
sem_txt = sem_AUC.run.B(1);
write_mean_sem(ActXWord,WordHandle, mean_txt,sem_txt);

txt_input = ' vs. ';
writeDefaultWordText(ActXWord,WordHandle,txt_input);

mean_txt = mean_AUC.run.B(2);
sem_txt = sem_AUC.run.B(2);
write_mean_sem(ActXWord,WordHandle, mean_txt,sem_txt);

writeWordEnter(ActXWord,WordHandle,1);

%%%%%

%AB RUN
txt_input = 'AB A vs. B AUC/min RUN';
writeDefaultWordText(ActXWord,WordHandle,txt_input);
%newline
writeWordEnter(ActXWord,WordHandle,1);

mean_txt = mean_AUC.run.AB(1);
sem_txt = sem_AUC.run.AB(1);
write_mean_sem(ActXWord,WordHandle, mean_txt,sem_txt);

txt_input = ' vs. ';
writeDefaultWordText(ActXWord,WordHandle,txt_input);

mean_txt = mean_AUC.run.AB(2);
sem_txt = sem_AUC.run.AB(2);
write_mean_sem(ActXWord,WordHandle, mean_txt,sem_txt);

writeWordEnter(ActXWord,WordHandle,1);

%%%%%%%%%%%%% NO RUN %%%%%%%%%%%%%%%

%A sel NORUN
txt_input = 'A sel A vs. B AUC/min NORUN';
writeDefaultWordText(ActXWord,WordHandle,txt_input);
%newline
writeWordEnter(ActXWord,WordHandle,1);

mean_txt = mean_AUC.norun.A(1);
sem_txt = sem_AUC.norun.A(1);
write_mean_sem(ActXWord,WordHandle, mean_txt,sem_txt);

txt_input = ' vs. ';
writeDefaultWordText(ActXWord,WordHandle,txt_input);

mean_txt = mean_AUC.norun.A(2);
sem_txt = sem_AUC.norun.A(2);
write_mean_sem(ActXWord,WordHandle, mean_txt,sem_txt);

writeWordEnter(ActXWord,WordHandle,1);

%B NORUN
txt_input = 'B-sel A vs. B AUC/min NORUN';
writeDefaultWordText(ActXWord,WordHandle,txt_input);
%newline
writeWordEnter(ActXWord,WordHandle,1);

mean_txt = mean_AUC.norun.B(1);
sem_txt = sem_AUC.norun.B(1);
write_mean_sem(ActXWord,WordHandle, mean_txt,sem_txt);

txt_input = ' vs. ';
writeDefaultWordText(ActXWord,WordHandle,txt_input);

mean_txt = mean_AUC.norun.B(2);
sem_txt = sem_AUC.norun.B(2);
write_mean_sem(ActXWord,WordHandle, mean_txt,sem_txt);

writeWordEnter(ActXWord,WordHandle,1);

%%%%%

%AB NORUN
txt_input = 'AB A vs. B AUC/min NORUN';
writeDefaultWordText(ActXWord,WordHandle,txt_input);
%newline
writeWordEnter(ActXWord,WordHandle,1);

mean_txt = mean_AUC.norun.AB(1);
sem_txt = sem_AUC.norun.AB(1);
write_mean_sem(ActXWord,WordHandle, mean_txt,sem_txt);

txt_input = ' vs. ';
writeDefaultWordText(ActXWord,WordHandle,txt_input);

mean_txt = mean_AUC.norun.AB(2);
sem_txt = sem_AUC.norun.AB(2);
write_mean_sem(ActXWord,WordHandle, mean_txt,sem_txt);

writeWordEnter(ActXWord,WordHandle,1);

%% Fraction tuned IS
%SI A vs. B
txt_input = 'A vs B SI';
writeDefaultWordText(ActXWord,WordHandle,txt_input);
%newline
writeWordEnter(ActXWord,WordHandle,1);

mean_txt = mean_si(1);
sem_txt = sem_si(1);
write_mean_sem(ActXWord,WordHandle, mean_txt,sem_txt);

txt_input = ' vs. ';
writeDefaultWordText(ActXWord,WordHandle,txt_input);

mean_txt = mean_si(2);
sem_txt = sem_si(2);
write_mean_sem(ActXWord,WordHandle, mean_txt,sem_txt);

writeWordEnter(ActXWord,WordHandle,1);

%SI A vs. AB
txt_input = 'A vs AB SI';
writeDefaultWordText(ActXWord,WordHandle,txt_input);
%newline
writeWordEnter(ActXWord,WordHandle,1);

mean_txt = mean_si(1);
sem_txt = sem_si(1);
write_mean_sem(ActXWord,WordHandle, mean_txt,sem_txt);

txt_input = ' vs. ';
writeDefaultWordText(ActXWord,WordHandle,txt_input);

mean_txt = mean_si(3);
sem_txt = sem_si(3);
write_mean_sem(ActXWord,WordHandle, mean_txt,sem_txt);

writeWordEnter(ActXWord,WordHandle,1);
%SI B vs. AB
txt_input = 'B vs AB SI';
writeDefaultWordText(ActXWord,WordHandle,txt_input);
%newline
writeWordEnter(ActXWord,WordHandle,1);

mean_txt = mean_si(2);
sem_txt = sem_si(2);
write_mean_sem(ActXWord,WordHandle, mean_txt,sem_txt);

txt_input = ' vs. ';
writeDefaultWordText(ActXWord,WordHandle,txt_input);

mean_txt = mean_si(3);
sem_txt = sem_si(3);
write_mean_sem(ActXWord,WordHandle, mean_txt,sem_txt);

writeWordEnter(ActXWord,WordHandle,1);

%%%%%%%%TS %%%%%%%%%%%
%ts A vs. B
txt_input = 'A vs B ts';
writeDefaultWordText(ActXWord,WordHandle,txt_input);
%newline
writeWordEnter(ActXWord,WordHandle,1);

mean_txt = mean_ts(1);
sem_txt = sem_ts(1);
write_mean_sem(ActXWord,WordHandle, mean_txt,sem_txt);

txt_input = ' vs. ';
writeDefaultWordText(ActXWord,WordHandle,txt_input);

mean_txt = mean_ts(2);
sem_txt = sem_ts(2);
write_mean_sem(ActXWord,WordHandle, mean_txt,sem_txt);

writeWordEnter(ActXWord,WordHandle,1);

%ts A vs. AB
txt_input = 'A vs AB ts';
writeDefaultWordText(ActXWord,WordHandle,txt_input);
%newline
writeWordEnter(ActXWord,WordHandle,1);

mean_txt = mean_ts(1);
sem_txt = sem_ts(1);
write_mean_sem(ActXWord,WordHandle, mean_txt,sem_txt);

txt_input = ' vs. ';
writeDefaultWordText(ActXWord,WordHandle,txt_input);

mean_txt = mean_ts(3);
sem_txt = sem_ts(3);
write_mean_sem(ActXWord,WordHandle, mean_txt,sem_txt);

writeWordEnter(ActXWord,WordHandle,1);
%ts B vs. AB
txt_input = 'B vs AB ts';
writeDefaultWordText(ActXWord,WordHandle,txt_input);
%newline
writeWordEnter(ActXWord,WordHandle,1);

mean_txt = mean_ts(2);
sem_txt = sem_ts(2);
write_mean_sem(ActXWord,WordHandle, mean_txt,sem_txt);

txt_input = ' vs. ';
writeDefaultWordText(ActXWord,WordHandle,txt_input);

mean_txt = mean_ts(3);
sem_txt = sem_ts(3);
write_mean_sem(ActXWord,WordHandle, mean_txt,sem_txt);

writeWordEnter(ActXWord,WordHandle,1);

%% Mean TC score between selective neurons  
%TC A vs. B
txt_input = 'A vs B TC mean';
writeDefaultWordText(ActXWord,WordHandle,txt_input);
%newline
writeWordEnter(ActXWord,WordHandle,1);

mean_txt = mean_TC.A;
sem_txt = sem_TC.A;
write_mean_sem(ActXWord,WordHandle, mean_txt,sem_txt);

txt_input = ' vs. ';
writeDefaultWordText(ActXWord,WordHandle,txt_input);

mean_txt = mean_TC.B;
sem_txt = sem_TC.B;
write_mean_sem(ActXWord,WordHandle, mean_txt,sem_txt);

writeWordEnter(ActXWord,WordHandle,1);

% A vs AB TC mean
txt_input = 'A vs AB TC mean';
writeDefaultWordText(ActXWord,WordHandle,txt_input);
%newline
writeWordEnter(ActXWord,WordHandle,1);

mean_txt = mean_TC.A;
sem_txt = sem_TC.A;
write_mean_sem(ActXWord,WordHandle, mean_txt,sem_txt);

txt_input = ' vs. ';
writeDefaultWordText(ActXWord,WordHandle,txt_input);

mean_txt = mean_TC.AB;
sem_txt = sem_TC.AB;
write_mean_sem(ActXWord,WordHandle, mean_txt,sem_txt);

writeWordEnter(ActXWord,WordHandle,1);

% B vs AB TC mean
txt_input = 'B vs AB TC mean';
writeDefaultWordText(ActXWord,WordHandle,txt_input);
%newline
writeWordEnter(ActXWord,WordHandle,1);

mean_txt = mean_TC.B;
sem_txt = sem_TC.B;
write_mean_sem(ActXWord,WordHandle, mean_txt,sem_txt);

txt_input = ' vs. ';
writeDefaultWordText(ActXWord,WordHandle,txt_input);

mean_txt = mean_TC.AB;
sem_txt = sem_TC.AB;
write_mean_sem(ActXWord,WordHandle, mean_txt,sem_txt);

writeWordEnter(ActXWord,WordHandle,1);


%% Fraction remapping neurons 
%order of each class of remappers
%'Common'	'Activity'	'Global'	'Partial'	'Unclassified'

% common vs activity 
txt_input = 'Common vs. activity';
writeDefaultWordText(ActXWord,WordHandle,txt_input);
%newline
writeWordEnter(ActXWord,WordHandle,1);

mean_txt = frac_remap_mean(1);
sem_txt = frac_remap_sem(1);
write_mean_sem(ActXWord,WordHandle, mean_txt,sem_txt);

txt_input = ' vs. ';
writeDefaultWordText(ActXWord,WordHandle,txt_input);

mean_txt = frac_remap_mean(2);
sem_txt = frac_remap_sem(2);
write_mean_sem(ActXWord,WordHandle, mean_txt,sem_txt);

writeWordEnter(ActXWord,WordHandle,1);

% common vs global
txt_input = 'Common vs. global';
writeDefaultWordText(ActXWord,WordHandle,txt_input);
%newline
writeWordEnter(ActXWord,WordHandle,1);

mean_txt = frac_remap_mean(1);
sem_txt = frac_remap_sem(1);
write_mean_sem(ActXWord,WordHandle, mean_txt,sem_txt);

txt_input = ' vs. ';
writeDefaultWordText(ActXWord,WordHandle,txt_input);

mean_txt = frac_remap_mean(3);
sem_txt = frac_remap_sem(3);
write_mean_sem(ActXWord,WordHandle, mean_txt,sem_txt);

writeWordEnter(ActXWord,WordHandle,1);

% common vs global
txt_input = 'Common vs. partial';
writeDefaultWordText(ActXWord,WordHandle,txt_input);
%newline
writeWordEnter(ActXWord,WordHandle,1);

mean_txt = frac_remap_mean(1);
sem_txt = frac_remap_sem(1);
write_mean_sem(ActXWord,WordHandle, mean_txt,sem_txt);

txt_input = ' vs. ';
writeDefaultWordText(ActXWord,WordHandle,txt_input);

mean_txt = frac_remap_mean(4);
sem_txt = frac_remap_sem(4);
write_mean_sem(ActXWord,WordHandle, mean_txt,sem_txt);

writeWordEnter(ActXWord,WordHandle,1);

% zone II mean sem for global remapping neurons
txt_input = 'Zone II A vs. B for global remap';
writeDefaultWordText(ActXWord,WordHandle,txt_input);
%newline
writeWordEnter(ActXWord,WordHandle,1);

mean_txt = zoneII_mean;
sem_txt = zoneII_sem;
write_mean_sem(ActXWord,WordHandle, mean_txt,sem_txt);

writeWordEnter(ActXWord,WordHandle,1);

%% Close Word document
%CloseWord(ActXWord,WordHandle,FileSpec);

