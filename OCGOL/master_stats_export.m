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

%create excel importable table data

%create AUC/min table
%Figure, Subfigure, Data aggregation, Comparison, N, Test, Degrees of Freedom, 
%Test statistic, p-value, p-value adjusted, ad. method, Significance

fig_num = repmat(2,3,1);
fig_sub = repmat('c',3,1);
data_agg = repmat('by animal',3,1);
comp_descrip = {'AUC/min difference btn A vs. B laps in RUN - A sel.';...
                'AUC/min difference btn A vs. B laps in RUN - B sel.';...
                'AUC/min difference btn A vs. B laps in RUN - A&B'};
n_sample = [stats.run.A(3), stats.run.B(3),stats.run.AB(3)]';
test_name = repmat('Paired Wilcoxon Sign Rank',3,1);
n_dof = [stats.run.A(4), stats.run.B(4),stats.run.AB(4)]';
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
        
%create source data spreadsheet

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

%create excel importable table data

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
        
%% Combine Figure 2 stat tables
t1 = repmat({' '},1,12);

%combined table
t_fig2_all = [t_auc_run; t_auc_norun;];

%exported Excel spreadsheet
%write to Excel spreadsheet
writetable(t_auc_run,'myData.xlsx','Sheet','Main Figure',"AutoFitWidth",true,'WriteMode','append')
writetable(cell2table(t1),'myData.xlsx','Sheet','Main Figure',"AutoFitWidth",true,'WriteMode','append')
writetable(t_auc_norun,'myData.xlsx','Sheet','Main Figure',"AutoFitWidth",true,'WriteMode','append')

%% Figure 2d - fraction tuned by SI and TS

frac_si_tuned = 


[p,tbl,stats] = friedman(friedman_test_mat);


%% Paired Wilcoxon Sign Rank
% returns positive sum of ranks W+ (Prsm returns sum of positive and
% negative ranks)

[p,h,stats] = signrank(mileage(:,2),33)

%% Rayleigh test of circular uniformity

%% 2-sample Kolmogorov-Smirnov Test


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

