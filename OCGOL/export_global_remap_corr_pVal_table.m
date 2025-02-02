
%% Directories paths from all animals

path_dir = {'G:\Figure_2_3_selective_remap\I47_LP_AB_d1_062018_1',...
            'G:\Figure_2_3_selective_remap\I42R_AB_d1_032118_1',...
            'G:\Figure_2_3_selective_remap\I42L_AB_d1_032118_1',...
            'G:\Figure_2_3_selective_remap\I42L_AB_d1_032118_2',...
            'G:\Figure_2_3_selective_remap\I53LT_AB_sal_113018_1',...
            'G:\Figure_2_3_selective_remap\I56_RTLS_AB_prePost_sal_042419_1',...
            'G:\Figure_2_3_selective_remap\I52RT_AB_sal_113018_1',...
            'G:\Figure_2_3_selective_remap\I57_RTLS_AB_prePost_792_042519_1',...
            'G:\Figure_2_3_selective_remap\I45_RT_AB_d1_062018_1',...
            'G:\Figure_2_3_selective_remap\I46_AB_d1_062018_1',...
            'G:\Figure_2_3_selective_remap\I57_LT_ABrand_no_punish_042119_1'};

%% Load the r scores and p values for for correlation score/significance

for ii=1:size(path_dir,2)
    r_p_vals(ii) = load(fullfile(path_dir{ii},'cumul_analysis', 'tun_curve_corr.mat'));
end

%% Load the indices for remapping cell type for each animal

for ii=1:size(path_dir,2)
    remap_idx(ii) = load(fullfile(path_dir{ii},'cumul_analysis', 'remap_corr_idx.mat'));
end

%% Make matrix before converting to table
% "animal#" "neuron-idx", "R-value", "p-value", "common or global place cell"

%create cell of cells and then collpase to matrix
corr_p_cell = cell(11,5);

for ii=1:size(path_dir,2)
    temp_common = numel(remap_idx(ii).remapping_corr_idx.final.common);
    temp_global = numel(remap_idx(ii).remapping_corr_idx.final.global);
    temp_neuron_nb = temp_common + temp_global;
    
    %animal index
    corr_p_cell{ii,1} = ii*ones(temp_neuron_nb,1);
    %neuron idx - common then global
    corr_p_cell{ii,2} =[remap_idx(ii).remapping_corr_idx.final.common; remap_idx(ii).remapping_corr_idx.final.global];
    %r value - common then global
    corr_p_cell{ii,3} = [r_p_vals(ii).tun_curve_corr.r(remap_idx(ii).remapping_corr_idx.final.common)'; ...
        r_p_vals(ii).tun_curve_corr.r(remap_idx(ii).remapping_corr_idx.final.global)'];
    %p value 
    corr_p_cell{ii,4} = [r_p_vals(ii).tun_curve_corr.p_val(remap_idx(ii).remapping_corr_idx.final.common)'; ...
        r_p_vals(ii).tun_curve_corr.p_val(remap_idx(ii).remapping_corr_idx.final.global)'];    
    %remapping type label
    corr_p_cell{ii,5} = [repmat(["common"],temp_common,1); repmat(["global"],temp_global,1)];
    
end

%merge all cells in vertical vectors
for ii=1:5
    corr_p_cell_vert{ii} = vertcat(corr_p_cell{:,ii});
end


%% Convert to formatted table for export to Excel

% Create table with following labels:
% "animal#" "neuron-idx", "R-value", "p-value", "common or global place cell"
export_table = table(corr_p_cell_vert{1},corr_p_cell_vert{2},corr_p_cell_vert{3},...
    corr_p_cell_vert{4}, corr_p_cell_vert{5}, ...
    'VariableNames', {'Animal', 'Neuron', 'R_score','p_value','Cell_type'});

%export as Excel spreadsheet
writetable(export_table, 'common_global_remap_table.xlsx')

