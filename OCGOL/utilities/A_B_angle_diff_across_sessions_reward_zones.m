function [theta] = A_B_angle_diff_across_sessions_reward_zones(reg,tuned_log_learning,pf_vector_max_learning,aa,options)

%% Define trials and session range

selectTrial = options.selectTrial;
%set range of sessions to process for each animal
sessionSelect = 1:size(reg{aa}.registered.multi.assigned_filtered,2);

%% Define reward start vectors in complex form as well cartesian coordinates for calculating angles below

%unit vectors
%reward A - ~0.7 - get specific from mean norm start position of animal
%later
rewardA_vec = exp(i*deg2rad(0.7*360));
%0.3
rewardB_vec =  exp(i*deg2rad(0.3*360));

%convert to cartesian points
rewardA_cart = [real(rewardA_vec), imag(rewardA_vec)];
rewardB_cart = [real(rewardB_vec), imag(rewardB_vec)];

%check complex vectors on compass plot
% figure
% compass(rewardA_vec)
% hold on
% pause
% compass(rewardB_vec)
%check complex vectors with scatter plot
% figure
% hold on
% scatter(rewardA_cart(1), rewardA_cart(2))
% scatter(rewardB_cart(1), rewardB_cart(2))

%% Calculate rad angle diff for between A tuned ROI and B tuned ROI across sessions
match_ROI.all = reg{aa}.registered.multi.assigned_filtered(:,1:sessionSelect(end));

%create separate A, B, AB matrices for TS filtering -duplicate
match_ROI.allA = match_ROI.all;
match_ROI.allB = match_ROI.all;
match_ROI.AB = match_ROI.all;


%blank logical for matching ROIs
blank_log.A = zeros(size(match_ROI.all,1),size(match_ROI.all,2));
blank_log.B = zeros(size(match_ROI.all,1),size(match_ROI.all,2));
blank_log.AB = zeros(size(match_ROI.all,1),size(match_ROI.all,2));

for ss=sessionSelect
    %all A
    %get tuned indices for each session
    tuned_idx.allA = find(tuned_log_learning{aa}.tuned_logicals.tuned_log_filt_ts{ss}.allA  ==1 );
    [match_idx_A, column_match_idx.A{ss},~]  =  intersect(match_ROI.allA(:,ss),tuned_idx.allA);
    disp(length(match_idx_A))
    %translate match to logical value
    blank_log.A(column_match_idx.A{ss},ss) = 1;
    %nan non-matching ROIs
    match_ROI.allA(~blank_log.A(:,ss),ss) = nan;
    
    %all B
    tuned_idx.allB = find(tuned_log_learning{aa}.tuned_logicals.tuned_log_filt_ts{ss}.allB  ==1 );
    [match_idx_B, column_match_idx.B{ss},~]  =  intersect(match_ROI.allB(:,ss),tuned_idx.allB);
    disp(length(match_idx_B))
    %translate match to logical value
    blank_log.B(column_match_idx.B{ss},ss) = 1;
    %nan non-matching ROIs
    match_ROI.allB(~blank_log.B(:,ss),ss) = nan;
end

%% Extract the tuning vectors for each class across days

%holder for each category vectors
max_vectors.A = nan(size(match_ROI.all,1),size(match_ROI.all,2));
max_vectors.B = nan(size(match_ROI.all,1),size(match_ROI.all,2));

%do for learning A first
for ss=sessionSelect
    %all A
    max_vectors.A(column_match_idx.A{ss},ss) = pf_vector_max_learning{aa}.pf_vector_max{ss}{selectTrial(1)}( match_ROI.allA(column_match_idx.A{ss},ss));
    %all B
    max_vectors.B(column_match_idx.B{ss},ss) = pf_vector_max_learning{aa}.pf_vector_max{ss}{selectTrial(2)}( match_ROI.allB(column_match_idx.B{ss},ss));
end

%% Calculate anglular difference for all A/B (for all crosses)

%get the vectors for both sessions
for ss=sessionSelect
    for ss2=sessionSelect
        %all A
        compare_vectors.A{ss,ss2} = max_vectors.A(find(sum(~isnan(max_vectors.A(:,[ss,ss2])),2) ==2),[ss,ss2]);
        %all B
        compare_vectors.B{ss,ss2} = max_vectors.B(find(sum(~isnan(max_vectors.B(:,[ss,ss2])),2) ==2),[ss,ss2]);
    end
end

%convert complex vector to Cartesian coordinates
for ss=sessionSelect
    for ss2=sessionSelect
        %all A
        for rr=1:size(compare_vectors.A{ss, ss2},1)
            
            %first ses
            compare_vec_cart.A{ss, ss2}{rr,1} = [real(compare_vectors.A{ss, ss2}(rr,1)), imag(compare_vectors.A{ss, ss2}(rr,1))];
            %second ses
            compare_vec_cart.A{ss, ss2}{rr,2} = [real(compare_vectors.A{ss, ss2}(rr,2)), imag(compare_vectors.A{ss, ss2}(rr,2))];
        end
        
        for rr=1:size(compare_vectors.B{ss, ss2},1)
            %all B
            %first ses
            compare_vec_cart.B{ss, ss2}{rr,1} = [real(compare_vectors.B{ss, ss2}(rr,1)), imag(compare_vectors.B{ss, ss2}(rr,1))];
            %second ses
            compare_vec_cart.B{ss, ss2}{rr,2} = [real(compare_vectors.B{ss, ss2}(rr,2)), imag(compare_vectors.B{ss, ss2}(rr,2))];
        end
    end
end
%% Get angular distance to start of reward zone A and start of reward zone B
for ss=sessionSelect
    for ss2=sessionSelect
        %calculate difference between each vector pair
        %for each ROI get difference between ts vector and reward vector
        %all A
        %for each set of vectors compare distance to start of reward zone A and B
        for rr = 1:size(compare_vec_cart.A{ss, ss2},1)
            %define input vectors
            %RELATIVE TO START OF A REWARD ZONE
            %first vector calculate distance to reward A
            uCar = compare_vec_cart.A{ss, ss2}{rr,1};
            vCar = rewardA_cart;
            %calculate angle for each match
            theta.A.rewA{ss, ss2}(rr,1) = compute_abs_rad_angle_btw_cart_vectors(uCar,vCar);
            %second vector match, calculate distance to reward A
            uCar = compare_vec_cart.A{ss, ss2}{rr,2};
            vCar = rewardA_cart;
            theta.A.rewA{ss, ss2}(rr,2) = compute_abs_rad_angle_btw_cart_vectors(uCar,vCar);
            
            %RELATIVE TO START OF B REWARD ZONE
            uCar = compare_vec_cart.A{ss, ss2}{rr,1};
            vCar = rewardB_cart;
            %calculate angle for each match
            theta.A.rewB{ss, ss2}(rr,1) = compute_abs_rad_angle_btw_cart_vectors(uCar,vCar);
            %second vector match, calculate distance to reward A
            uCar = compare_vec_cart.A{ss, ss2}{rr,2};
            vCar = rewardB_cart;
            theta.A.rewB{ss, ss2}(rr,2) = compute_abs_rad_angle_btw_cart_vectors(uCar,vCar);
            
        end
        
        %all B
        for rr = 1:size(compare_vec_cart.B{ss, ss2},1)
            
            %define input vectors
            %RELATIVE TO START OF A REWARD ZONE
            %first vector calculate distance to reward A
            uCar = compare_vec_cart.B{ss, ss2}{rr,1};
            vCar = rewardA_cart;
            %calculate angle for each match
            theta.B.rewA{ss, ss2}(rr,1) = compute_abs_rad_angle_btw_cart_vectors(uCar,vCar);
            %second vector match, calculate distance to reward A
            uCar = compare_vec_cart.B{ss, ss2}{rr,2};
            vCar = rewardA_cart;
            theta.B.rewA{ss, ss2}(rr,2) = compute_abs_rad_angle_btw_cart_vectors(uCar,vCar);
            
            %RELATIVE TO START OF B REWARD ZONE
            uCar =compare_vec_cart.B{ss, ss2}{rr,1};
            vCar = rewardB_cart;
            %calculate angle for each match
            theta.B.rewB{ss, ss2}(rr,1) = compute_abs_rad_angle_btw_cart_vectors(uCar,vCar);
            %second vector match, calculate distance to reward A
            uCar = compare_vec_cart.B{ss, ss2}{rr,2};
            vCar = rewardB_cart;
            theta.B.rewB{ss, ss2}(rr,2) = compute_abs_rad_angle_btw_cart_vectors(uCar,vCar);
            
        end
    end
end

end

