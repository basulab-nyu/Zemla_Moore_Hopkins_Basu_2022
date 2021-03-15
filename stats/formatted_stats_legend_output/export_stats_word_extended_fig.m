%% Import all table formatted entries here for data generated for each figure

%import data for each figure
laptop_access = 0;

%laptop path directory
laptop_path_dir = 'C:\Users\rzeml\Google Drive\task_selective_place_paper\matlab_data';
%desktop path directory
desktop_path_dir = 'G:\Google_drive\task_selective_place_paper\matlab_data';

%load table data
if laptop_access == 1
cd(laptop_path_dir)
else
    cd(desktop_path_dir)
end

%load in figure 4 and 5 data
ex_fig_data = load('ex_fig_table_data.mat');

%% Start Word document that will contain the formatted stats data

WordFileName='legend_stats_formatted_extended_fig.doc';
CurDir=pwd;
FileSpec = fullfile(CurDir,WordFileName);
%active X handle for manipulating document (ActXWord)
[ActXWord,WordHandle]=StartWord(FileSpec);

fprintf('Document will be saved in %s\n',FileSpec);


%% Ex Fig. 4b A-B place field speed difference for and A and B selective neurons
%description of statistics
txt_input = 'Ex. Fig. 4b - A-B in-field speed difference for A and B selective neurons';
writeDefaultWordText(ActXWord,WordHandle,txt_input);
writeWordEnter(ActXWord,WordHandle,1);

%open parenthesis
txt_input = '(';
writeDefaultWordText(ActXWord,WordHandle,txt_input);

%input to 1-sample paired Wilcoxon test
dof = table2array(ex_fig_data.table_list.exFig4.t_paired_wilcox_Asel(1,7));
test_stat = table2array(ex_fig_data.table_list.exFig4.t_paired_wilcox_Asel(1,8));
p_val = table2array(ex_fig_data.table_list.exFig4.t_paired_wilcox_Asel(1,9));
sample_n = table2array(ex_fig_data.table_list.exFig4.t_paired_wilcox_Asel(1,5));

%description of comparison
comp_descrip = 'A-B place field speed difference in A-selective neurons';
writeOneSampleWilcoxPooled_0(ActXWord,WordHandle,comp_descrip,test_stat,p_val, dof, sample_n)

%split entry with semicolon
txt_input = '; ';
writeDefaultWordText(ActXWord,WordHandle,txt_input);

%input to 1-sample paired Wilcoxon test
dof = table2array(ex_fig_data.table_list.exFig4.t_paired_wilcox_Bsel(1,7));
test_stat = table2array(ex_fig_data.table_list.exFig4.t_paired_wilcox_Bsel(1,8));
p_val = table2array(ex_fig_data.table_list.exFig4.t_paired_wilcox_Bsel(1,9));
sample_n = table2array(ex_fig_data.table_list.exFig4.t_paired_wilcox_Bsel(1,5));

%description of comparison
comp_descrip = 'A-B place field speed difference in B-selective neurons';
writeOneSampleWilcoxPooled_0(ActXWord,WordHandle,comp_descrip,test_stat,p_val, dof, sample_n)

%close parenthesis
txt_input = ')';
writeDefaultWordText(ActXWord,WordHandle,txt_input);
writeWordEnter(ActXWord,WordHandle,2);

%% Ex Fig 4c Normalized place field count for task-selective and non-selective neurons

%description of statistics
txt_input = 'Ex. Fig. 4c - Normalized place field count comparisons for task-selective and non-selective neurons';
writeDefaultWordText(ActXWord,WordHandle,txt_input);
writeWordEnter(ActXWord,WordHandle,1);

%open parenthesis
txt_input = '(';
writeDefaultWordText(ActXWord,WordHandle,txt_input);

%paired Wilcoxon test with Holm-Sidak correction
dof = table2array(ex_fig_data.table_list.exFig4.t_out.wilcox_singlePF(1,7));
test_stat = table2array(ex_fig_data.table_list.exFig4.t_out.wilcox_singlePF(1,8));
p_val = table2array(ex_fig_data.table_list.exFig4.t_out.wilcox_singlePF(1,10));
sample_n = table2array(ex_fig_data.table_list.exFig4.t_out.wilcox_singlePF(1,5));

%description of comparison
comp_descrip = 'A- vs. B-selective single place field count';
writePairedWilcoxAnimal(ActXWord,WordHandle,comp_descrip,test_stat,p_val, dof, sample_n)

%split entry with semicolon
txt_input = '; ';
writeDefaultWordText(ActXWord,WordHandle,txt_input);

%paired Wilcoxon test with Holm-Sidak correction
dof = table2array(ex_fig_data.table_list.exFig4.t_out.wilcox_singlePF(2,7));
test_stat = table2array(ex_fig_data.table_list.exFig4.t_out.wilcox_singlePF(2,8));
p_val = table2array(ex_fig_data.table_list.exFig4.t_out.wilcox_singlePF(2,10));
sample_n = table2array(ex_fig_data.table_list.exFig4.t_out.wilcox_singlePF(2,5));

%description of comparison
comp_descrip = 'A-selective vs. A&B-A single place field count';
writePairedWilcoxAnimal(ActXWord,WordHandle,comp_descrip,test_stat,p_val, dof, sample_n)

%split entry with semicolon
txt_input = '; ';
writeDefaultWordText(ActXWord,WordHandle,txt_input);

%paired Wilcoxon test with Holm-Sidak correction
dof = table2array(ex_fig_data.table_list.exFig4.t_out.wilcox_singlePF(3,7));
test_stat = table2array(ex_fig_data.table_list.exFig4.t_out.wilcox_singlePF(3,8));
p_val = table2array(ex_fig_data.table_list.exFig4.t_out.wilcox_singlePF(3,10));
sample_n = table2array(ex_fig_data.table_list.exFig4.t_out.wilcox_singlePF(3,5));

%description of comparison
comp_descrip = 'A-selective vs. A&B-B single place field count';
writePairedWilcoxAnimal(ActXWord,WordHandle,comp_descrip,test_stat,p_val, dof, sample_n)

%split entry with semicolon
txt_input = '; ';
writeDefaultWordText(ActXWord,WordHandle,txt_input);

%%%% double

%paired Wilcoxon test with Holm-Sidak correction
dof = table2array(ex_fig_data.table_list.exFig4.t_out.wilcox_doublePF(1,7));
test_stat = table2array(ex_fig_data.table_list.exFig4.t_out.wilcox_doublePF(1,8));
p_val = table2array(ex_fig_data.table_list.exFig4.t_out.wilcox_doublePF(1,10));
sample_n = table2array(ex_fig_data.table_list.exFig4.t_out.wilcox_doublePF(1,5));

%description of comparison
comp_descrip = 'A- vs. B-selective double place field count';
writePairedWilcoxAnimal(ActXWord,WordHandle,comp_descrip,test_stat,p_val, dof, sample_n)

%split entry with semicolon
txt_input = '; ';
writeDefaultWordText(ActXWord,WordHandle,txt_input);

%paired Wilcoxon test with Holm-Sidak correction
dof = table2array(ex_fig_data.table_list.exFig4.t_out.wilcox_doublePF(2,7));
test_stat = table2array(ex_fig_data.table_list.exFig4.t_out.wilcox_doublePF(2,8));
p_val = table2array(ex_fig_data.table_list.exFig4.t_out.wilcox_doublePF(2,10));
sample_n = table2array(ex_fig_data.table_list.exFig4.t_out.wilcox_doublePF(2,5));

%description of comparison
comp_descrip = 'A-selective vs. A&B-A double place field count';
writePairedWilcoxAnimal(ActXWord,WordHandle,comp_descrip,test_stat,p_val, dof, sample_n)

%split entry with semicolon
txt_input = '; ';
writeDefaultWordText(ActXWord,WordHandle,txt_input);

%paired Wilcoxon test with Holm-Sidak correction
dof = table2array(ex_fig_data.table_list.exFig4.t_out.wilcox_doublePF(3,7));
test_stat = table2array(ex_fig_data.table_list.exFig4.t_out.wilcox_doublePF(3,8));
p_val = table2array(ex_fig_data.table_list.exFig4.t_out.wilcox_doublePF(3,10));
sample_n = table2array(ex_fig_data.table_list.exFig4.t_out.wilcox_doublePF(3,5));

%description of comparison
comp_descrip = 'A-selective vs. A&B-B double place field count';
writePairedWilcoxAnimal(ActXWord,WordHandle,comp_descrip,test_stat,p_val, dof, sample_n)

%split entry with semicolon
txt_input = '; ';
writeDefaultWordText(ActXWord,WordHandle,txt_input);

%%%% triple

%paired Wilcoxon test with Holm-Sidak correction
dof = table2array(ex_fig_data.table_list.exFig4.t_out.wilcox_triplePF(1,7));
test_stat = table2array(ex_fig_data.table_list.exFig4.t_out.wilcox_triplePF(1,8));
p_val = table2array(ex_fig_data.table_list.exFig4.t_out.wilcox_triplePF(1,10));
sample_n = table2array(ex_fig_data.table_list.exFig4.t_out.wilcox_triplePF(1,5));

%description of comparison
comp_descrip = 'A- vs. B-selective triple place field count';
writePairedWilcoxAnimal(ActXWord,WordHandle,comp_descrip,test_stat,p_val, dof, sample_n)

%split entry with semicolon
txt_input = '; ';
writeDefaultWordText(ActXWord,WordHandle,txt_input);

%paired Wilcoxon test with Holm-Sidak correction
dof = table2array(ex_fig_data.table_list.exFig4.t_out.wilcox_triplePF(2,7));
test_stat = table2array(ex_fig_data.table_list.exFig4.t_out.wilcox_triplePF(2,8));
p_val = table2array(ex_fig_data.table_list.exFig4.t_out.wilcox_triplePF(2,10));
sample_n = table2array(ex_fig_data.table_list.exFig4.t_out.wilcox_triplePF(2,5));

%description of comparison
comp_descrip = 'A-selective vs. A&B-A triple place field count';
writePairedWilcoxAnimal(ActXWord,WordHandle,comp_descrip,test_stat,p_val, dof, sample_n)

%split entry with semicolon
txt_input = '; ';
writeDefaultWordText(ActXWord,WordHandle,txt_input);

%paired Wilcoxon test with Holm-Sidak correction
dof = table2array(ex_fig_data.table_list.exFig4.t_out.wilcox_triplePF(3,7));
test_stat = table2array(ex_fig_data.table_list.exFig4.t_out.wilcox_triplePF(3,8));
p_val = table2array(ex_fig_data.table_list.exFig4.t_out.wilcox_triplePF(3,10));
sample_n = table2array(ex_fig_data.table_list.exFig4.t_out.wilcox_triplePF(3,5));

%description of comparison
comp_descrip = 'A-selective vs. A&B-B triple place field count';
writePairedWilcoxAnimal(ActXWord,WordHandle,comp_descrip,test_stat,p_val, dof, sample_n)

%close parenthesis
txt_input = ')';
writeDefaultWordText(ActXWord,WordHandle,txt_input);
writeWordEnter(ActXWord,WordHandle,2);

%% Ex Fig 4d - Place field width differences between task-sel and non-sel neurons

%description of statistics
txt_input = 'Ex. Fig. 4c - Normalized place field count comparisons for task-selective and non-selective neurons';
writeDefaultWordText(ActXWord,WordHandle,txt_input);
writeWordEnter(ActXWord,WordHandle,1);

%open parenthesis
txt_input = '(';
writeDefaultWordText(ActXWord,WordHandle,txt_input);

% KS2 test with p-val correction input
dof = table2array(ex_fig_data.table_list.exFig4.t_ks2_pf_width(1,7));
test_stat = table2array(ex_fig_data.table_list.exFig4.t_ks2_pf_width(1,8));
p_val = table2array(ex_fig_data.table_list.exFig4.t_ks2_pf_width(1,10));
sample_n = table2array(ex_fig_data.table_list.exFig4.t_ks2_pf_width(1,5));

%description of comparison
comp_descrip = 'A- vs. B- selective place field width';
%2KS test with Holm-Sidak correction
write2ksTest_p_cor(ActXWord,WordHandle,comp_descrip,test_stat,p_val, dof, sample_n)

%split entry with semicolon
txt_input = '; ';
writeDefaultWordText(ActXWord,WordHandle,txt_input);

% KS2 test with p-val correction input
dof = table2array(ex_fig_data.table_list.exFig4.t_ks2_pf_width(2,7));
test_stat = table2array(ex_fig_data.table_list.exFig4.t_ks2_pf_width(2,8));
p_val = table2array(ex_fig_data.table_list.exFig4.t_ks2_pf_width(2,10));
sample_n = table2array(ex_fig_data.table_list.exFig4.t_ks2_pf_width(2,5));

%description of comparison
comp_descrip = 'A- vs. A&B-A- selective place field width';
%2KS test with Holm-Sidak correction
write2ksTest_p_cor(ActXWord,WordHandle,comp_descrip,test_stat,p_val, dof, sample_n)

%split entry with semicolon
txt_input = '; ';
writeDefaultWordText(ActXWord,WordHandle,txt_input);

% KS2 test with p-val correction input
dof = table2array(ex_fig_data.table_list.exFig4.t_ks2_pf_width(3,7));
test_stat = table2array(ex_fig_data.table_list.exFig4.t_ks2_pf_width(3,8));
p_val = table2array(ex_fig_data.table_list.exFig4.t_ks2_pf_width(3,10));
sample_n = table2array(ex_fig_data.table_list.exFig4.t_ks2_pf_width(3,5));

%description of comparison
comp_descrip = 'A- vs. A&B-B selective place field width';
%2KS test with Holm-Sidak correction
write2ksTest_p_cor(ActXWord,WordHandle,comp_descrip,test_stat,p_val, dof, sample_n)

%split entry with semicolon
txt_input = '; ';
writeDefaultWordText(ActXWord,WordHandle,txt_input);

% KS2 test with p-val correction input
dof = table2array(ex_fig_data.table_list.exFig4.t_ks2_pf_width(4,7));
test_stat = table2array(ex_fig_data.table_list.exFig4.t_ks2_pf_width(4,8));
p_val = table2array(ex_fig_data.table_list.exFig4.t_ks2_pf_width(4,10));
sample_n = table2array(ex_fig_data.table_list.exFig4.t_ks2_pf_width(4,5));

%description of comparison
comp_descrip = 'B- vs. A&B-A selective place field width';
%2KS test with Holm-Sidak correction
write2ksTest_p_cor(ActXWord,WordHandle,comp_descrip,test_stat,p_val, dof, sample_n)

%split entry with semicolon
txt_input = '; ';
writeDefaultWordText(ActXWord,WordHandle,txt_input);

% KS2 test with p-val correction input
dof = table2array(ex_fig_data.table_list.exFig4.t_ks2_pf_width(5,7));
test_stat = table2array(ex_fig_data.table_list.exFig4.t_ks2_pf_width(5,8));
p_val = table2array(ex_fig_data.table_list.exFig4.t_ks2_pf_width(5,10));
sample_n = table2array(ex_fig_data.table_list.exFig4.t_ks2_pf_width(5,5));

%description of comparison
comp_descrip = 'B- vs. A&B-B selective place field width';
%2KS test with Holm-Sidak correction
write2ksTest_p_cor(ActXWord,WordHandle,comp_descrip,test_stat,p_val, dof, sample_n)

%split entry with semicolon
txt_input = '; ';
writeDefaultWordText(ActXWord,WordHandle,txt_input);

% KS2 test with p-val correction input
dof = table2array(ex_fig_data.table_list.exFig4.t_ks2_pf_width(6,7));
test_stat = table2array(ex_fig_data.table_list.exFig4.t_ks2_pf_width(6,8));
p_val = table2array(ex_fig_data.table_list.exFig4.t_ks2_pf_width(6,10));
sample_n = table2array(ex_fig_data.table_list.exFig4.t_ks2_pf_width(6,5));

%description of comparison
comp_descrip = 'A&B-A vs. A&B-B selective place field width';
%2KS test with Holm-Sidak correction
write2ksTest_p_cor(ActXWord,WordHandle,comp_descrip,test_stat,p_val, dof, sample_n)

%close parenthesis
txt_input = ')';
writeDefaultWordText(ActXWord,WordHandle,txt_input);
writeWordEnter(ActXWord,WordHandle,2);

%% Ex Fig 7a - Common vs globap remapping activity index score

%description of statistics
txt_input = 'Ex. Fig. 7a - Common vs global remapping activity index score';
writeDefaultWordText(ActXWord,WordHandle,txt_input);
writeWordEnter(ActXWord,WordHandle,1);

%open parenthesis
txt_input = '(';
writeDefaultWordText(ActXWord,WordHandle,txt_input);

% KS2 test with p-val correction input
dof = table2array(ex_fig_data.table_list.exFig7.t_kstest_com_vs_act(1,7));
test_stat = table2array(ex_fig_data.table_list.exFig7.t_kstest_com_vs_act(1,8));
p_val = table2array(ex_fig_data.table_list.exFig7.t_kstest_com_vs_act(1,9));
sample_n = table2array(ex_fig_data.table_list.exFig7.t_kstest_com_vs_act(1,5));

%description of comparison
comp_descrip = 'Common vs global remapping activity index score';
%2KS test with Holm-Sidak correction
write2ksTest(ActXWord,WordHandle,comp_descrip,test_stat,p_val, dof, sample_n)

%close parenthesis
txt_input = ')';
writeDefaultWordText(ActXWord,WordHandle,txt_input);
writeWordEnter(ActXWord,WordHandle,2);

%% Ex Fig 7c - Activity rate in run epochs for A vs B laps in activity remapping neurons

%description of statistics
txt_input = 'Ex. Fig. 7c - Activity rate in A vs. B laps during RUN for all activity remapping neurons';
writeDefaultWordText(ActXWord,WordHandle,txt_input);
writeWordEnter(ActXWord,WordHandle,1);

%open parenthesis
txt_input = '(';
writeDefaultWordText(ActXWord,WordHandle,txt_input);

%input for paired Wilcoxon test
dof = table2array(ex_fig_data.table_list.exFig7.t_paired_wilcox_activity_rate_com_activity(1,7));
test_stat = table2array(ex_fig_data.table_list.exFig7.t_paired_wilcox_activity_rate_com_activity(1,8));
p_val = table2array(ex_fig_data.table_list.exFig7.t_paired_wilcox_activity_rate_com_activity(1,9));
sample_n = table2array(ex_fig_data.table_list.exFig7.t_paired_wilcox_activity_rate_com_activity(1,5));

%description of comparison
comp_descrip = 'Activity rate in A vs. B laps during RUN for all activity remapping neurons';
writePairedWilcoxPooled_non_cor_pval(ActXWord,WordHandle,comp_descrip,test_stat,p_val, p_val, dof, sample_n);

%close parenthesis
txt_input = ')';
writeDefaultWordText(ActXWord,WordHandle,txt_input);
writeWordEnter(ActXWord,WordHandle,2);

%% Close Word document
CloseWord(ActXWord,WordHandle,FileSpec);


%order of entry
%test name, test statistic, p value (include rounding and sigstar add),
%test statistic - 2 decimal places
%p-value 3 decimal places and sub to <0.001 for low p values
%nb of samples as a char entry

%% Original function customized
%WriteToWordFromMatlab_testing
