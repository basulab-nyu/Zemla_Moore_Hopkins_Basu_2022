%% Master script for generating stats for paper
%source data, import and format Prism analysis, and export to Excel, and
%Word legend data

%import data for each figure

%figure 2 source data
load('G:\Google_drive\task_selective_place_paper\matlab_data\source_data_fig2.mat')


%% Fig 2c AUC analysis RUN
%AUC run data for A and B selective place cells

%AUC/min RUN A,B, AB
AUC_A_sel_RUN = source_data_task_sel_remap.mean_AUC.run.Asel;
AUC_B_sel_RUN = source_data_task_sel_remap.mean_AUC.run.Bsel;
AUC_AB_RUN = source_data_task_sel_remap.mean_AUC.run.AB;

%paired wilcoxon A, B, AB sel on A vs B laps during RUN epochs
%A selective
stats.run.A = paired_wilcoxon_signrank(AUC_A_sel_RUN(:,1),AUC_A_sel_RUN(:,2));

%B selective
stats.run.B = paired_wilcoxon_signrank(AUC_B_sel_RUN(:,1),AUC_B_sel_RUN(:,2));

%AB
stats.run.AB = paired_wilcoxon_signrank(AUC_AB_RUN(:,1),AUC_AB_RUN(:,2));

%3-way Holm-Sidak correction for AUC RUN comparison
%significance level
alpha = 0.05;
%# of comparisons
c = 3; 
%input p-value vector
p = [stats.run.A(1) stats.run.B(1) stats.run.AB(1)];
%return adjustment p-values for AUC RUN comparison
p_adj = holm_sidak_p_adj(p,c,alpha);

%% create transients/min table for task sel neurons - RUN
%Figure, Subfigure, Data aggregation, Comparison, N, Test, Degrees of Freedom, 
%Test statistic, p-value, p-value adjusted, ad. method, Significance

fig_num = repmat('2',3,1);
fig_sub = repmat('c',3,1);
data_agg = repmat('by animal',3,1);
comp_descrip = {'AUC/min difference btn A vs. B laps in RUN - A sel.';...
                'AUC/min difference btn A vs. B laps in RUN - B sel.';...
                'AUC/min difference btn A vs. B laps in RUN - A&B'};
n_sample = [stats.run.A(3) stats.run.B(3) stats.run.AB(3)]';
test_name = repmat('Paired Wilcoxon Sign Rank',3,1);
n_dof = [stats.run.A(4) stats.run.B(4) stats.run.AB(4)]';
test_statistic = [stats.run.A(2) stats.run.B(2) stats.run.AB(2)]';
adj_method = repmat('Holm-Sidak (3-way)', 3,1);
p = [stats.run.A(1) stats.run.B(1) stats.run.AB(1)]';
p_adj = holm_sidak_p_adj(p,c,alpha);
sig_level = check_p_value_sig(p_adj);

%create RUN AUC/min table
t_auc_run = table(fig_num, fig_sub, data_agg, comp_descrip, n_sample,...
            test_name, n_dof, test_statistic,p,p_adj, adj_method, sig_level,...
            'VariableNames',{'Figure','Subfigure','Data aggregation',...
            'Comparison','N', 'Test', 'Degrees of Freedom', 'Test statistic',...
            'p-value', 'p-value adjusted', 'Adjustment method','Significance'});

%% Fig 2c AUC analysis no RUN
%AUC run data for A and B selective place cells

%AUC/min RUN A,B, AB
AUC_A_sel_noRUN = source_data_task_sel_remap.mean_AUC.norun.Asel;
AUC_B_sel_noRUN = source_data_task_sel_remap.mean_AUC.norun.Bsel;
AUC_AB_noRUN = source_data_task_sel_remap.mean_AUC.norun.AB;

%paired wilcoxon A, B, AB sel on A vs B laps during RUN epochs
%A selective
stats.norun.A = paired_wilcoxon_signrank(AUC_A_sel_noRUN(:,1),AUC_A_sel_noRUN(:,2));

%B selective
stats.norun.B = paired_wilcoxon_signrank(AUC_B_sel_noRUN(:,1),AUC_B_sel_noRUN(:,2));

%AB
stats.norun.AB = paired_wilcoxon_signrank(AUC_AB_noRUN(:,1),AUC_AB_noRUN(:,2));

%3-way Holm-Sidak correction for AUC RUN comparison
%significance level
alpha = 0.05;
%# of comparisons
c = 3; 
%input p-value vector
p = [stats.norun.A(1) stats.norun.B(1) stats.norun.AB(1)];

%create AUC/min table
%Figure, Subfigure, Data aggregation, Comparison, N, Test, Degrees of Freedom, 
%Test statistic, p-value, p-value adjusted, ad. method, Significance

fig_num = repmat(2,3,1);
fig_sub = repmat('c',3,1);
data_agg = repmat('by animal',3,1);
comp_descrip = {'AUC/min difference btn A vs. B laps in NO RUN - A sel.';...
                'AUC/min difference btn A vs. B laps in NO RUN - B sel.';...
                'AUC/min difference btn A vs. B laps in NO RUN - A&B'};
n_sample = [stats.norun.A(3), stats.norun.B(3),stats.norun.AB(3)]';
test_name = repmat('Paired Wilcoxon Sign Rank',3,1);
n_dof = [stats.norun.A(4), stats.norun.B(4),stats.norun.AB(4)]';
test_statistic = [stats.norun.A(2) stats.norun.B(2) stats.norun.AB(2)]';
adj_method = repmat('Holm-Sidak (3-way)', 3,1);
p = [stats.norun.A(1) stats.norun.B(1) stats.norun.AB(1)]';
p_adj = holm_sidak_p_adj(p,c,alpha);
sig_level = check_p_value_sig(p_adj);

%create noRUN AUC/min table
t_auc_norun = table(fig_num, fig_sub, data_agg, comp_descrip, n_sample,...
            test_name, n_dof, test_statistic,p,p_adj, adj_method, sig_level,...
            'VariableNames',{'Figure','Subfigure','Data aggregation',...
            'Comparison','N', 'Test', 'Degrees of Freedom', 'Test statistic',...
            'p-value', 'p-value adjusted', 'Adjustment method','Significance'});
        

%% Figure 2d - fraction tuned by SI

frac_si_tuned = source_data_task_sel_remap.frac_tuned.si;

%si tuned fractions - group comparison
[friedman_stats] = friedman_test(frac_si_tuned);

%AvsB
stats.frac_tuned.AvB = paired_wilcoxon_signrank(frac_si_tuned(:,1),frac_si_tuned(:,2));

%AvsA&B selective
stats.frac_tuned.AvAB = paired_wilcoxon_signrank(frac_si_tuned(:,1),frac_si_tuned(:,3));

%BvsAB
stats.frac_tuned.BvAB = paired_wilcoxon_signrank(frac_si_tuned(:,2),frac_si_tuned(:,3));

%3-way Holm-Sidak correction for AUC RUN comparison (inputs)
%significance level
alpha = 0.05;
%# of comparisons
c = 3; 
%input p-value vector
p = [stats.frac_tuned.AvB(1) stats.frac_tuned.AvAB(1) stats.frac_tuned.BvAB(1)];

%create AUC/min table
%Figure, Subfigure, Data aggregation, Comparison, N, Test, Degrees of Freedom, 
%Test statistic, p-value, p-value adjusted, ad. method, Significance

fig_num = repmat(2,4,1);
fig_sub = repmat('d',4,1);
data_agg = repmat('by animal',4,1);
comp_descrip = {'Fraction of A,B, A&B tuned, neither - Spatial info.';...
                'Fraction of neurons tuned - A vs. B - Spatial info.';...
                'Fraction of neurons tuned - A vs. A&B - Spatial info.';...
                'Fraction of neurons tuned - B vs. A&B - Spatial info.'};
n_sample = [friedman_stats(3), stats.frac_tuned.AvB(3) stats.frac_tuned.AvAB(3) stats.frac_tuned.BvAB(3)]';
test_name = [{'Friedman test'}; repmat({'Paired Wilcoxon Sign Rank'},3,1)];
n_dof = [friedman_stats(4), stats.frac_tuned.AvB(4) stats.frac_tuned.AvAB(4) stats.frac_tuned.BvAB(4)]';
test_statistic = [friedman_stats(2), stats.frac_tuned.AvB(2) stats.frac_tuned.AvAB(2) stats.frac_tuned.BvAB(2)]';
adj_method = ['N/A'; repmat({'Holm-Sidak (3-way)'}, 3,1)];
p_all = [friedman_stats(1) stats.frac_tuned.AvB(1), stats.frac_tuned.AvAB(1), stats.frac_tuned.BvAB(1)]';
p_adj = holm_sidak_p_adj(p,c,alpha);
sig_level = check_p_value_sig([p_all(1), p_adj]);

%create noRUN AUC/min table
t_frac_si = table(fig_num, fig_sub, data_agg, comp_descrip, n_sample,...
            test_name, n_dof, test_statistic, p_all, [nan, p_adj]', adj_method, sig_level,...
            'VariableNames',{'Figure','Subfigure','Data aggregation',...
            'Comparison','N', 'Test', 'Degrees of Freedom', 'Test statistic',...
            'p-value', 'p-value adjusted', 'Adjustment method','Significance'});

%% Figure 2d - fraction tuned by TS

frac_ts_tuned = source_data_task_sel_remap.frac_tuned.ts;

%si tuned fractions - group comparison
[friedman_stats] = friedman_test(frac_ts_tuned);

%AvsB
stats.frac_tuned.AvB = paired_wilcoxon_signrank(frac_ts_tuned(:,1),frac_ts_tuned(:,2));

%AvsA&B selective
stats.frac_tuned.AvAB = paired_wilcoxon_signrank(frac_ts_tuned(:,1),frac_ts_tuned(:,3));

%BvsAB
stats.frac_tuned.BvAB = paired_wilcoxon_signrank(frac_ts_tuned(:,2),frac_ts_tuned(:,3));

%3-way Holm-Sidak correction for AUC RUN comparison (inputs)
%significance level
alpha = 0.05;
%# of comparisons
c = 3; 
%input p-value vector
p = [stats.frac_tuned.AvB(1) stats.frac_tuned.AvAB(1) stats.frac_tuned.BvAB(1)];

%create AUC/min table
%Figure, Subfigure, Data aggregation, Comparison, N, Test, Degrees of Freedom, 
%Test statistic, p-value, p-value adjusted, ad. method, Significance

fig_num = repmat(2,4,1);
fig_sub = repmat('d',4,1);
data_agg = repmat('by animal',4,1);
comp_descrip = {'Fraction of A,B, A&B tuned, neither - Tuning spec.';...
                'Fraction of neurons tuned - A vs. B - Tuning spec.';...
                'Fraction of neurons tuned - A vs. A&B - Tuning spec.';...
                'Fraction of neurons tuned - B vs. A&B - Tuning spec.'};
n_sample = [friedman_stats(3), stats.frac_tuned.AvB(3) stats.frac_tuned.AvAB(3) stats.frac_tuned.BvAB(3)]';
test_name = [{'Friedman test'}; repmat({'Paired Wilcoxon Sign Rank'},3,1)];
n_dof = [friedman_stats(4), stats.frac_tuned.AvB(4) stats.frac_tuned.AvAB(4) stats.frac_tuned.BvAB(4)]';
test_statistic = [friedman_stats(2), stats.frac_tuned.AvB(2) stats.frac_tuned.AvAB(2) stats.frac_tuned.BvAB(2)]';
adj_method = ['N/A'; repmat({'Holm-Sidak (3-way)'}, 3,1)];
p_all = [friedman_stats(1) stats.frac_tuned.AvB(1) stats.frac_tuned.AvAB(1) stats.frac_tuned.BvAB(1)]';
p_adj = holm_sidak_p_adj(p,c,alpha);
sig_level = check_p_value_sig([p_all(1), p_adj]);

%create noRUN AUC/min table
t_frac_ts = table(fig_num, fig_sub, data_agg, comp_descrip, n_sample,...
            test_name, n_dof, test_statistic, p_all, [nan, p_adj]', adj_method, sig_level,...
            'VariableNames',{'Figure','Subfigure','Data aggregation',...
            'Comparison','N', 'Test', 'Degrees of Freedom', 'Test statistic',...
            'p-value', 'p-value adjusted', 'Adjustment method','Significance'});
        
%% Figure 2 - uniformity of place field distribution across track
%Rayleigh test of circular uniformity

%unload data
center_bin_radians = source_data_task_sel_remap.pf_dist.center_bin_radians;
pooled_A_counts = source_data_task_sel_remap.pf_dist.pooled_A_counts;
pooled_B_counts = source_data_task_sel_remap.pf_dist.pooled_B_counts;
bin_spacing = source_data_task_sel_remap.pf_dist.bin_spacing;

%get number of neurons here

%use this result since you downbin to 25 bins
%test for A 
%run rayleigh test - similar result if even bin spacing is input or not
[pval_A, z_A] = circ_rtest(center_bin_radians,pooled_A_counts ,bin_spacing);
%test for B
[pval_B, z_B] = circ_rtest(center_bin_radians,pooled_B_counts ,bin_spacing);

%Test statistic, p-value, p-value adjusted, ad. method, Significance

nb_entries = 2;

fig_num = repmat(2,nb_entries,1);
fig_sub = repmat('f',nb_entries,1);
data_agg = repmat('pooled',nb_entries,1);
comp_descrip = {'Distribution of PF centroid locations - A selective (25 bins)';...
                'Distribution of PF centroid locations - B selective (25 bins)';};
n_sample = [sum(pooled_A_counts),sum(pooled_B_counts)]';
test_name = repmat({'Rayleigh test of circular uniformity'},nb_entries,1);
n_dof = repmat('N/A', nb_entries,1);
test_statistic = [z_A,z_B]';
adj_method = repmat('N/A', nb_entries,1);
p_all = [pval_A, pval_B]';
p_adj = repmat('N/A', nb_entries,1);
sig_level = check_p_value_sig(p_all);

%create table
t_pf_dist = table(fig_num, fig_sub, data_agg, comp_descrip, n_sample,...
            test_name, n_dof, test_statistic, p_all, p_adj, adj_method, sig_level,...
            'VariableNames',{'Figure','Subfigure','Data aggregation',...
            'Comparison','N', 'Test', 'Degrees of Freedom', 'Test statistic',...
            'p-value', 'p-value adjusted', 'Adjustment method','Significance'});
        
        
%% Fig. 2G - place field distributions between A and B 

%unload data
bin_assign = source_data_task_sel_remap.pf_dist_all.bin_assign;
%CONTINUE HERE

%all neurons
[~,p_ks2,ks2stat] = kstest2(cell2mat(bin_assign.A),cell2mat(bin_assign.B));

nb_entries = 1;

fig_num = repmat(2,nb_entries,1);
fig_sub = string(repmat('g',nb_entries,1));
data_agg = string(repmat('pooled',nb_entries,1));
comp_descrip = {'Place field centroid distribution of A selective vs. B selective neurons'};
n_sample = string([num2str(numel(cell2mat(bin_assign.A))),' vs ', num2str(numel(cell2mat(bin_assign.B)))]);
test_name = repmat({'2-sample Kolmogorov–Smirnov test'},nb_entries,1);

n_dof = string(repmat('N/A', nb_entries,1));
test_statistic = [ks2stat]';
adj_method = string(repmat('N/A', nb_entries,1));
p_all = [p_ks2]';
p_adj = string(repmat('N/A', nb_entries,1));
sig_level = check_p_value_sig(p_all);

%create table
t_2ks_pf_dist = table(fig_num, fig_sub, data_agg, comp_descrip, n_sample,...
            test_name, n_dof, test_statistic, p_all, p_adj, adj_method, sig_level,...
            'VariableNames',{'Figure','Subfigure','Data aggregation',...
            'Comparison','N', 'Test', 'Degrees of Freedom', 'Test statistic',...
            'p-value', 'p-value adjusted', 'Adjustment method','Significance'});

%% Fig 2h Tuning curve correlation comparison

%unload data
TC_Asel = source_data_task_sel_remap.mean_TC.Asel;
TC_Bsel = source_data_task_sel_remap.mean_TC.Bsel;
TC_AB = source_data_task_sel_remap.mean_TC.AB;

%paired wilcoxon A, B, AB sel on A vs B laps during RUN epochs

stats.tc.AvB = paired_wilcoxon_signrank(TC_Asel,TC_Bsel);

stats.tc.AvAB = paired_wilcoxon_signrank(TC_Asel,TC_AB);

stats.tc.BvAB = paired_wilcoxon_signrank(TC_Bsel,TC_AB);

%3-way Holm-Sidak correction for AUC RUN comparison
%significance level
alpha = 0.05;
%# of comparisons
c = 3; 
%input p-value vector
p = [stats.tc.AvB(1) stats.tc.AvAB(1) stats.tc.BvAB(1)];

%create AUC/min table
%Figure, Subfigure, Data aggregation, Comparison, N, Test, Degrees of Freedom, 
%Test statistic, p-value, p-value adjusted, ad. method, Significance

nb_entries = 3;

fig_num = repmat(2,nb_entries,1);
fig_sub = repmat('h',nb_entries,1);
data_agg = repmat('by animal',nb_entries,1);

%description of comparison
comp_descrip = {'Tuning curve correlation of A vs. B laps - A sel. Vs. Bsel neurons';...
                'Tuning curve correlation of A vs. B laps - A sel. Vs. AB neurons';...
                'Tuning curve correlation of A vs. B laps - B sel. Vs. AB neurons'};
%number of samples for each test            
n_sample = [stats.tc.AvB(3) stats.tc.AvAB(3) stats.tc.BvAB(3)]';
%test ran
test_name = repmat('Paired Wilcoxon Sign Rank',nb_entries,1);
%degrees of freedom
n_dof = [stats.tc.AvB(4) stats.tc.AvAB(4) stats.tc.BvAB(4)]';
%test statistic
test_statistic = [stats.tc.AvB(2) stats.tc.AvAB(2) stats.tc.BvAB(2)]';
adj_method = repmat('Holm-Sidak (3-way)', nb_entries,1);
p_adj = holm_sidak_p_adj(p,c,alpha);
sig_level = check_p_value_sig(p_adj);

%create noRUN AUC/min table
t_tc_corr = table(fig_num, fig_sub, data_agg, comp_descrip, n_sample,...
            test_name, n_dof, test_statistic,p',p_adj', adj_method, sig_level,...
            'VariableNames',{'Figure','Subfigure','Data aggregation',...
            'Comparison','N', 'Test', 'Degrees of Freedom', 'Test statistic',...
            'p-value', 'p-value adjusted', 'Adjustment method','Significance'});


%% Combine Figure 2 stat tables
t1 = repmat({' '},1,12);

%combined table
t_fig2_all = [t_auc_run; t_auc_norun;];

%exported Excel spreadsheet
%write to Excel spreadsheet
writetable(t_auc_run,'myData.xlsx','Sheet','Main Figure',"AutoFitWidth",true,'WriteMode','append')

writetable(cell2table(t1),'myData.xlsx','Sheet','Main Figure',"AutoFitWidth",true,'WriteMode','append')
writetable(t_auc_norun,'myData.xlsx','Sheet','Main Figure',"AutoFitWidth",true,'WriteMode','append')

writetable(cell2table(t1),'myData.xlsx','Sheet','Main Figure',"AutoFitWidth",true,'WriteMode','append')
writetable(t_frac_si,'myData.xlsx','Sheet','Main Figure',"AutoFitWidth",true,'WriteMode','append')

writetable(cell2table(t1),'myData.xlsx','Sheet','Main Figure',"AutoFitWidth",true,'WriteMode','append')
writetable(t_frac_ts,'myData.xlsx','Sheet','Main Figure',"AutoFitWidth",true,'WriteMode','append')
        
writetable(cell2table(t1),'myData.xlsx','Sheet','Main Figure',"AutoFitWidth",true,'WriteMode','append')
writetable(t_pf_dist,'myData.xlsx','Sheet','Main Figure',"AutoFitWidth",true,'WriteMode','append')        

writetable(cell2table(t1),'myData.xlsx','Sheet','Main Figure',"AutoFitWidth",true,'WriteMode','append')
writetable(t_2ks_pf_dist,'myData.xlsx','Sheet','Main Figure',"AutoFitWidth",true,'WriteMode','append')  

writetable(cell2table(t1),'myData.xlsx','Sheet','Main Figure',"AutoFitWidth",true,'WriteMode','append')
writetable(t_tc_corr,'myData.xlsx','Sheet','Main Figure',"AutoFitWidth",true,'WriteMode','append')  


%% Linear mixed effects model test in matlab - 2way repeated measures mixed effects ANOVA (equivalent to PRISM output)
%test_data  = [];

%corr score vector
temp = test_data';
temp = temp(:);
corr_score = temp;

%subject data
temp = categorical(repmat(1:11,6,1))';
subject = temp(:);

%trial_type vector
trial_type = 3*ones(6,11);
trial_type(:,1:6) = 2;
trial_type = trial_type';
trial_type = categorical(trial_type(:)); 

%time vector
%define ordinal time rankining
time_value = {'1','2','3','4','5','6'};


time_vec = repmat([1:6]',1,11)';
time_vec = categorical(categorical(time_vec(:)),time_value,'Ordinal',true);

% %organize into columns for input to lme
% Corr_score
% Subject 
% Time
% Trial_type

%organize test data into table
lme_tbl = table(corr_score, subject, time_vec, trial_type, 'VariableNames',{'corr', 'subject','time','trial'});

%fit LME with subjects being random factor
lme = fitlme(lme_tbl,'corr ~ 1+ time*trial + (1|subject)','FitMethod','REML','DummyVarCoding','effects');

%fit LME with symmetrical covariance matrix (sphericity adjustment) - no
%sig difference
%compound symmetry structure/isotropic symmetry structure 'CompSymm'
%lme = fitlme(lme_tbl,'corr ~ 1+ time*trial + (1|subject)','FitMethod','REML','DummyVarCoding','effects');

%anova on lme
lme_stats = anova(lme,'DFMethod','satterthwaite')


%% 1-way RM linear mixed model ANOVA (mixed effects analysis)
% matches Prism without sphericity assumption

%si_short_term_recall_test = [];

%fraction across time
frac_score = si_short_term_recall_test';
frac_score = frac_score(:);

%subject data
temp = categorical(repmat(1:6,7,1))';
subject = temp';
subject = subject(:);

%time vector
%define ordinal time ranking
time_value = {'1','2','3','4','5','6','7'};

time_vec = repmat([1:7]',1,6);
time_vec = categorical(categorical(time_vec(:)),time_value,'Ordinal',true);

%create entry table for 
lme1_tbl = table(frac_score, subject, time_vec, 'VariableNames',{'corr', 'subject','time'});

%fit LME with subjects being random factor
lme1 = fitlme(lme1_tbl,'corr ~ 1+ time + (1|subject)','FitMethod','REML','DummyVarCoding','effects');

%anova on lme
lme1_stats = anova(lme1,'DFMethod','satterthwaite');

%fit LME with subjects being random factor
%run with symmetrical covariance matrix - i.e. adjust to make variances
%spherical - no GG correction for lme model in matlab
%does not yield same result of GG correction (GG correction is very subtle
% %however, so assuming data sphericity is reasonable
% lme1_spher = fitlme(lme1_tbl,'corr ~ 1+ time + (1|subject)','FitMethod','REML','DummyVarCoding','effects',...
%                 'CovariancePattern','CompSymm');

% %anova on shericity adjusted covariance data
% lme1_stats_spher = anova(lme1_spher,'DFMethod','satterthwaite');




%% Excel spreadsheet output ...

%% 1-sample Wilcoxon sign rank test  here


%% Holm-Bonferroni adjustment here



%% Holm-Sidak p-value correction

%significance level
alpha = 0.05;
%# of comparisons
c = 4; 
%input p-value vector
p = [0.0271 0.0368 0.0008 0.0015];

p_adj = holm_sidak_p_adj(p,c,alpha);


%% Place code for each test here ...


%% Focus on all the stats functions for generating Figure 2 data

%% 


%% 1-way Repeated Measures ANOVA - deal with this later

%rm_sample_data = [];

% training_groups = repmat([1:4],4,1);
% training_groups = training_groups(:);
% 
% rm_table = table(rm_sample_data(:),training_groups,'VariableNames', {'score','t_grps'});
% 
% 
% rm_model = fitrm(rm_table,'score~t_grps','WithinModel','separatemeans')
% 
% rm_model = fitrm(rm_table,'score~t_grps','WithinDesign',table(rm_sample_data'))
% 
% rm_anova = ranova(rm_model)
% 
% anova_rm(rm_sample_data)

%repeated measures 1-way ANOVA
%paired t-test

%unpaired t-test
%kruskall wallis test
%write up functions for each test
%import all matlab data here

