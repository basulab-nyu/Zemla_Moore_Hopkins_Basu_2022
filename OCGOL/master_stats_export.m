%% Master script for generating stats for paper
%source data, import and format Prism analysis, and export to Excel, and
%Word legend data

%import data for each figure


source_data_task_sel_remap

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


%% Friedman test
[p,tbl,stats] = friedman(friedman_test_mat);


%% Paired Wilcoxon Sign Rank

[p,h,stats] = signrank(mileage(:,2),33)

%friedman_test_mat = [];

%% 1-sample Wilcoxon sign rank test  here

%% 2-sample Kolmogorov-Smirnov Test


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


%% 1-way Repeated Measures ANOVA


%repeated measures 1-way ANOVA
%paired t-test
%Rayleigh test of circular uniformity
%unpaired t-test
%kruskall wallis test
%write up functions for each test
%import all matlab data here

