% complete pipeline for calcium imaging data pre-processing
clear;
addpath(genpath('../NoRMCorre'));               % add the NoRMCorre motion correction package to MATLAB path
gcp;                                            % start a parallel engine
foldername = 'E:\I42L_AB_d7_032718_1';
        % folder where all the files are located. Currently supported .tif,
        % .hdf5, .raw, .avi, and .mat files
files = subdir(fullfile(foldername,'*.tif'));   % list of filenames (will search all subdirectories)
FOV = size(read_file(files(1).name,1,1));
numFiles = length(files);

%cross-day same FOV imaging parameters
matchToDay1 = true;

% template_dir = 'E:\I42L_AB_d1_032118_2\output';
% template_files = subdir(fullfile(template_dir,'*.mat'));

if matchToDay1
    template_dir = 'E:\I42L_AB_d1_032118_1\input';
    template_files = subdir(fullfile(template_dir,'*.mat'));
    load(template_files(1).name, 'template');
end

%add the directory to matlab search directory - necessary for classifier in
%batches run
addpath(genpath(foldername));  

%remove first field if reading single tifs;
% files(1) = [];
% numFiles = length(files);

%% motion correct (and save registered h5 files as 2d matrices (to be used in the end)..)
% register files one by one. use template obtained from file n to
% initialize template of file n + 1; 
tic;
motion_correct = true;                                         % perform motion correction
non_rigid = false;                                               % flag for non-rigid motion correction

if non_rigid; append = '_nr'; else; append = '_rig'; end        % use this to save motion corrected files

options_mc = NoRMCorreSetParms('d1',FOV(1),'d2',FOV(2),'grid_size',[128,128],'init_batch',200,...
                'overlap_pre',64,'mot_uf',4,'bin_width',200,'max_shift',24,'max_dev',8,'us_fac',50,...
                'output_type','h5');

%do not use empty template if matching to previous day
if ~matchToDay1
    template = [];
end

col_shift = [];
for i = 1:numFiles
    fullname = files(i).name;
    [folder_name,file_name,ext] = fileparts(fullname);    
    options_mc.h5_filename = fullfile(folder_name,[file_name,append,'.h5']);
    if motion_correct    
        [M,shifts,template,options_mc,col_shift] = normcorre_batch(fullname,options_mc,template);
        save(fullfile(folder_name,[file_name,'_shifts',append,'.mat']),'shifts','-v7.3');           % save shifts of each file at the respective folder        
    else    % if files are already motion corrected convert them to h5
        convert_file(fullname,'h5',fullfile(folder_name,[file_name,'_mc.h5']));
    end
end
toc;

%% Save template for use in later day MC before end of script
cd(foldername)
save('template.mat','template');


%% Convert motion corrected file to tif for viewing in Fiji
tic;
convert_to_tif = false;

if convert_to_tif
    fullname_tif_convert = fullfile(folder_name,[file_name,append,'.h5']);
    convert_file(fullname_tif_convert,'tif',fullfile(folder_name,[file_name,'_mc.tif']));
end
toc;

%% Read in subsamples of motion corrected stack and save as tif stacks for viewing in Fiji

%read in start, middle and last stack of frames
% idxStart = [1, (dims(3)/2)-1000, dims(3)-2000];
% for ii=1:3
%     Ymc{ii} = bigread2(options_mc.h5_filename,idxStart(ii),2000);
%     saveastiff(Ymc{ii},fullfile(foldername,[num2str(ii),'_temp.tif']));
% end



%% downsample h5 files and save into a single memory mapped matlab file

if motion_correct
    h5_files = subdir(fullfile(foldername,['*',append,'.h5']));  % list of h5 files (modify this to list all the motion corrected files you need to process)
    %h5_files = subdir(fullfile(foldername,['3temp.tif']));
else
    h5_files = subdir(fullfile(foldername,'*_mc.h5'));
end
    
fr = 30;                                         % frame rate
tsub = 5;                                        % degree of downsampling (for 30Hz imaging rate you can try also larger, e.g. 8-10)
ds_filename = [foldername,'/ds_data.mat'];
data_type = class(read_file(h5_files(1).name,1,1));
data = matfile(ds_filename,'Writable',true);
data.Y  = zeros([FOV,0],data_type);
data.Yr = zeros([prod(FOV),0],data_type);
data.sizY = [FOV,0];
F_dark = Inf;                                    % dark fluorescence (min of all data)
batch_size = 2000;                               % read chunks of that size
batch_size = round(batch_size/tsub)*tsub;        % make sure batch_size is divisble by tsub
Ts = zeros(numFiles,1);                          % store length of each file
cnt = 0;                                         % number of frames processed so far
tt1 = tic;
for i = 1:numFiles
    name = h5_files(i).name;
    info = h5info(name);
    dims = info.Datasets.Dataspace.Size;
    ndimsY = length(dims);                       % number of dimensions (data array might be already reshaped)
    Ts(i) = dims(end);
    Ysub = zeros(FOV(1),FOV(2),floor(Ts(i)/tsub),data_type);
    data.Y(FOV(1),FOV(2),sum(floor(Ts/tsub))) = zeros(1,data_type);
    data.Yr(prod(FOV),sum(floor(Ts/tsub))) = zeros(1,data_type);
    cnt_sub = 0;
    for t = 1:batch_size:Ts(i)
        Y = bigread2(name,t,min(batch_size,Ts(i)-t+1));    
        F_dark = min(nanmin(Y(:)),F_dark);
        ln = size(Y,ndimsY);
        Y = reshape(Y,[FOV,ln]);
        Y = cast(downsample_data(Y,'time',tsub),data_type);
        ln = size(Y,3);
        Ysub(:,:,cnt_sub+1:cnt_sub+ln) = Y;
        cnt_sub = cnt_sub + ln;
    end
    data.Y(:,:,cnt+1:cnt+cnt_sub) = Ysub;
    data.Yr(:,cnt+1:cnt+cnt_sub) = reshape(Ysub,[],cnt_sub);
    toc(tt1);
    cnt = cnt + cnt_sub;
    data.sizY(1,3) = cnt;
end
data.F_dark = F_dark;

%% now run CNMF on patches on the downsampled file, set parameters first

sizY = data.sizY;                       % size of data matrix
patch_size = [40,40];                   % size of each patch along each dimension (optional, default: [32,32])
overlap = [8,8];                        % amount of overlap in each dimension (optional, default: [4,4])

patches = construct_patches(sizY(1:end-1),patch_size,overlap);
K = 9;                                            % number of components to be found
tau = 6;                                          % std of gaussian kernel (half size of neuron) 
p = 2;                                            % order of autoregressive system (p = 0 no dynamics, p=1 just decay, p = 2, both rise and decay)
merge_thr = 0.8;                                  % merging threshold
sizY = data.sizY;

options = CNMFSetParms(...
    'd1',sizY(1),'d2',sizY(2),...
    'deconv_method','constrained_foopsi',...    % neural activity deconvolution method
    'temporal_iter',2,...                       % number of block-coordinate descent steps 
    'ssub',2,...                                % spatial downsampling when processing
    'tsub',4,...                                % further temporal downsampling when processing
    'merge_thr',merge_thr,...                   % merging threshold
    'gSig',tau,... 
    'max_size_thr',300,'min_size_thr',10,...    % max/min acceptable size for each component
    'spatial_method','regularized',...          % method for updating spatial components
    'df_prctile',50,...                         % take the median of background fluorescence to compute baseline fluorescence 
    'df_window', 1000,...                       % length of running window (default [], no window)
    'fr',fr/tsub,...                            % downsamples
    'space_thresh',0.5,...                      % space correlation acceptance threshold
    'min_SNR',3.0,...                           % trace SNR acceptance threshold
    'cnn_thr',0.2,...                           % cnn classifier acceptance threshold
    'nb',1,...                                  % number of background components per patch
    'gnb',3,...                                 % number of global background components
    'decay_time',0.5...                        % length of typical transient for the indicator used
    );

%% Run on patches (the main work is done here)
tic;
[A,b,C,f,S,P,RESULTS,YrA] = run_CNMF_patches(data,K,patches,tau,0,options);  % do not perform deconvolution here since
                                                                             % we are operating on downsampled data
toc;
%% compute correlation image on a small sample of the data (optional - for visualization purposes) 
Cn = correlation_image_max(data.Y,8);

%% classify components

rval_space = classify_comp_corr(data,A,C,b,f,options);
ind_corr = rval_space > options.space_thresh;           % components that pass the correlation test
                                        % this test will keep processes
                                        
%% further classification with cnn_classifier
try  % matlab 2017b or later is needed
    [ind_cnn,value] = cnn_classifier(A,FOV,'cnn_model',options.cnn_thr);
catch
    ind_cnn = true(size(A,2),1);                        % components that pass the CNN classifier
end     
                            
%% event exceptionality

fitness = compute_event_exceptionality(C+YrA,options.N_samples_exc,options.robust_std);
ind_exc = (fitness < options.min_fitness);

%% select components

keep = (ind_corr | ind_cnn) & ind_exc;

%% Centers of the new components
%center of mass of each identified neuron
center = com(A,dims(1), dims(2));

%% run GUI for modifying component selection (optional, close twice to save values)
run_GUI = false;
if run_GUI
    figure;
    Coor = plot_contours(A(:,keep),Cn,options,1); close;
    GUIout = ROI_GUI(A,options,Cn,Coor,keep);
     %GUIout = ROI_GUI(data.Y,A,P,options,Cn,C,b,f)
    options = GUIout{2};
    keep = GUIout{3};    
end

%% ROI gui selection here against the centers identified with the algorithm
%clean out selected spatial components here from the initial run

%modify this function to:
%1) return list of indices that are removed from the inputs centers list
    %used to modify later keep list
%2) centers that are added (as separate list)

keep_centers = center(keep,:);

if 0
    [newcenters,added,removed] = ROI_GUI_centersV3(Cn,keep_centers(crossDayMatch_ROIs(:,1),:));
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%OPTIONAL%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% 
%{
%% Manually refine components - add componenets here with your auto function
[Am,Cm,center_man] = manually_refine_components(data.Yr,A(:,removed),C(removed,:),center,Cn,tau,options);

tic;
[Aplus,Cplus,centerPlus] = manually_refine_components_RZinputV1(double(data.Y),A(:,1),C(1,:),[dims(1),dims(2)],Cn,tau,options,added);
toc;

%remove the dummy component necessary to initialize
Aplus = Aplus(:,2:end);
Cplus = Cplus(2:end,:);

%merge the downsampled matrices
Acomb = [A, Aplus];
Ccomb = [C; Cplus];

%% update spatial components - update the spatial componenets that were added on the downsampled stack

[A2,b2,C2] = update_spatial_components(data.Yr,Ccomb,f,[Acomb,b],P,options);


%% re-classify components

rval_space = classify_comp_corr(data,A2,C2,b2,f,options);
ind_corr = rval_space > options.space_thresh;           % components that pass the correlation test
                                        % this test will keep processes
                                        
%% further re-classification with cnn_classifier
try  % matlab 2017b or later is needed
    [ind_cnn,value] = cnn_classifier(A2,FOV,'cnn_model',options.cnn_thr);
catch
    ind_cnn = true(size(A2,2),1);                        % components that pass the CNN classifier
end    

%% select components

keep = (ind_corr | ind_cnn);
%%%%%%%%%%%%%%%%OPTIONAL%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%}
%% isolate cells that are manually selected using ginput script - used the removed vector above

% match_components = false;
% 
% if match_components
%     
%     inpolygon(53.2,390.2, Coor_kp{178}(1,:),Coor_kp{178}(2,:));
% end

%% view contour plots of selected and rejected components (optional)

throw = ~keep;
Coor_k = [];
Coor_t = [];
figure;
    hold on
    ax1 = subplot(121); Coor_kp = plot_contours(A(:,keep),Cn,options,0,[],Coor_k,[],1,find(keep)); title('Selected components','fontweight','bold','fontsize',14);
    %scatter(newcenters(:,2),newcenters(:,1),'yo');
    hold off
    ax2 = subplot(122); plot_contours(A(:,throw),Cn,options,0,[],Coor_t,[],1,find(throw));title('Rejected components','fontweight','bold','fontsize',14);
    linkaxes([ax1,ax2],'xy')

    
%% compare manual vs. automatically selected neurons 

compare_components = false;

if compare_components
    figure;
    imagesc(Cn);
    hold on
    %manually selected neurons
    scatter(newcenters(:,2),newcenters(:,1),'g*');
    
    %automatically selected neurons
    scatter(cm(keep,2),cm(keep,1),'ro');
    hold off
end

    %% keep only the active components    
A_keep = A(:,keep);
C_keep = C(keep,:);

%% deconvolve (downsampled) temporal components plot GUI with components (optional)

% tic;
% [C_keep,f_keep,Pk,Sk,YrAk] = update_temporal_components_fast(data,A_keep,b,C_keep,f,P,options);
% toc
% 

plot_components = false;

if plot_components
    plot_components_GUI(data.Y,A_keep,C_keep,b,f,Cn,options)
end

%if exist('YrA','var'); R_keep = YrA; else; R_keep = YrA(keep,:); end

R_keep = YrA(keep,:);

%% extract fluorescence on native temporal resolution

options.fr = options.fr*tsub;                   % revert to origingal frame rate
N = size(C_keep,1);                             % total number of components
T = sum(Ts);                                    % total number of timesteps
C_full = imresize(C_keep,[N,T]);                % upsample to original frame rate
R_full = imresize(R_keep,[N,T]);                % upsample to original frame rate
F_full = C_full + R_full;                       % full fluorescence
f_full = imresize(f,[size(f,1),T]);             % upsample temporal background

S_full = zeros(N,T);

P.p = 0;
ind_T = [0;cumsum(Ts(:))];
options.nb = options.gnb;
for i = 1:numFiles
    inds = ind_T(i)+1:ind_T(i+1);   % indeces of file i to be updated
    [C_full(:,inds),f_full(:,inds),~,~,R_full(:,inds)] = update_temporal_components_fast(h5_files(i).name,A_keep,b,C_full(:,inds),f_full(:,inds),P,options);
    disp(['Extracting raw fluorescence at native frame rate. File ',num2str(i),' out of ',num2str(numFiles),' finished processing.'])
end

%% extract DF/F and deconvolve DF/F traces

[F_dff,F0,F,Fd] = detrend_df_f_RZ_V1(A_keep,[b,ones(prod(FOV),1)],C_full,[f_full;-double(F_dark)*ones(1,T)],R_full,options);

C_dec = zeros(N,T);         % deconvolved DF/F traces
S_dec = zeros(N,T);         % deconvolved neural activity
bl = zeros(N,1);            % baseline for each trace (should be close to zero since traces are DF/F)
neuron_sn = zeros(N,1);     % noise level at each trace
g = cell(N,1);              % discrete time constants for each trace
if p == 1; model_ar = 'ar1'; elseif p == 2; model_ar = 'ar2'; else; error('This order of dynamics is not supported'); end

% deconvolution for each neuron here - optional at present
% sometimes problem with SVD when running - skip 
% Error using svd
% Input to SVD must not contain NaN or Inf.
% Error in pinv (line 18)
% [U,S,V] = svd(A,'econ');
% Error in estimate_time_constant (line 49)
% g = pinv(A)*xc(lags+2:end);
% Error in deconvolveCa (line 74)
%             options.pars = estimate_time_constant(y, 2, options.sn);
% Error in run_pipeline_RZbeta1 (line 365)
%     [cc,spk,opts_oasis] = deconvolveCa(F_dff(i,:),model_ar,'method','thresholded','optimize_pars',true,'maxIter',20,...


for i = 1:N
    spkmin = options.spk_SNR*GetSn(F_dff(i,:));
    lam = choose_lambda(exp(-1/(options.fr*options.decay_time)),GetSn(F_dff(i,:)),options.lam_pr);
    [cc,spk,opts_oasis] = deconvolveCa(F_dff(i,:),model_ar,'method','thresholded','optimize_pars',true,'maxIter',20,...
                                'window',150,'lambda',lam,'smin',spkmin);
    bl(i) = opts_oasis.b;
    C_dec(i,:) = cc(:)' + bl(i);
    S_dec(i,:) = spk(:);
    neuron_sn(i) = opts_oasis.sn;
    g{i} = opts_oasis.pars(:)';
    disp(['Performing deconvolution. Trace ',num2str(i),' out of ',num2str(N),' finished processing.'])
end

%% Exponential filter smoothing and zero to median
%exponential moving average to smooth
expDff = []; 
alpha = 0.05;

for idx=1:size(F_dff,1)
    expDff(idx,:) = filter(alpha, [1 alpha-1], F_dff(idx,:));
end

%substract median from each trace to baseline around zero
for idx=1:size(expDff,1)
    expDffMedZeroed(idx,:) = expDff(idx,:) - median(expDff(idx,:));
end


%% save the workspace

%make output folder containing the saved variables in the experiments
%directory
cd(foldername)
try
    mkdir('input');
    cd(fullfile(foldername, ['\','input']))
catch
    disp('Directory already exists');
end

%get date
currentDate = date;
%replace dashes with underscores in date
dashIdx = regexp(currentDate,'\W');
currentDate(dashIdx) = '_';

%to exclude certain variables
%save([currentDate,'.mat'],'-regexp','^(?!(data)$).','-v7.3');

%save all
save([currentDate,'_CNMF.mat'],'-v7.3');



