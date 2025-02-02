function [merge_theta_learn_days] = arrange_learning_session_days(theta_learn)

%FIRST 6 days for now until all of data is processed

%% 
%merge recall as a function relative to d1 - d6
for dd=2:9 %for day 1 vs 2 3 4... or equivalent day distance
    
    switch dd
        case 2
            %for each animal, do same session extraction
            for aa=1:6 %all animals
                %extract all the days that correspond to the time duration
                merge_theta_learn_days.A{aa,dd} = theta_learn{aa}.A{1, 2};
                merge_theta_learn_days.B{aa,dd} = theta_learn{aa}.B{1, 2};
            end
        case 3
            for aa=1:6 %all animals
                %extract all the days that correspond to the time duration
                merge_theta_learn_days.A{aa,dd} = theta_learn{aa}.A{1, 3};
                 merge_theta_learn_days.B{aa,dd} = theta_learn{aa}.B{1, 3};
            end
        case 4 %session 4 vs. 7; session 3 vs. 4
            for aa=1:6 %all animals
                %extract all the days that correspond to the time duration
                merge_theta_learn_days.A{aa,dd} = theta_learn{aa}.A{1, 4};
                merge_theta_learn_days.B{aa,dd} = theta_learn{aa}.B{1, 4};
            end
        case 5 %session 2 vs. 6; session 3 vs. 5
            for aa=1:6 %all animals
                %extract all the days that correspond to the time duration
                if aa==1 %skip
                    merge_theta_learn_days.A{aa,dd} = [];
                    merge_theta_learn_days.B{aa,dd} = [];
                elseif aa==2 %skip
                    merge_theta_learn_days.A{aa,dd} = [];
                    merge_theta_learn_days.B{aa,dd} = [];
                elseif (aa>= 3 && aa<= 6)
                    merge_theta_learn_days.A{aa,dd} = theta_learn{aa}.A{1, 5};
                    merge_theta_learn_days.B{aa,dd} = theta_learn{aa}.B{1, 5}; 
                end
            end
            
        case 6 %session 2 vs. 6; session 3 vs. 5
            for aa=1:6 %all animals
                if aa==1 || aa==2
                    merge_theta_learn_days.A{aa,dd} = theta_learn{aa}.A{1, 5};
                    merge_theta_learn_days.B{aa,dd} = theta_learn{aa}.B{1, 5};

                elseif aa>=3 && aa<=6
                    merge_theta_learn_days.A{aa,dd} = theta_learn{aa}.A{1, 6};
                    merge_theta_learn_days.B{aa,dd} = theta_learn{aa}.B{1, 6};
                end
            end   
            
        case 7 %session 2 vs. 6; session 3 vs. 5
            for aa=1:6 %all animals
                if aa==1 || aa==2
                    merge_theta_learn_days.A{aa,dd} = theta_learn{aa}.A{1, 6};
                    merge_theta_learn_days.B{aa,dd} = theta_learn{aa}.B{1, 6};
                elseif aa==3 || aa==4
                    merge_theta_learn_days.A{aa,dd} = theta_learn{aa}.A{1, 7};
                    merge_theta_learn_days.B{aa,dd} = theta_learn{aa}.B{1, 7};
                elseif aa==5 || aa==6
                    merge_theta_learn_days.A{aa,dd} = [];
                    merge_theta_learn_days.B{aa,dd} = [];
                end
            end
            
        case 8 %session 2 vs. 6; session 3 vs. 5
            for aa=1:6 %all animals
                if aa==1 || aa==2 || aa==5
                    merge_theta_learn_days.A{aa,dd} = theta_learn{aa}.A{1, 7};
                    merge_theta_learn_days.B{aa,dd} = theta_learn{aa}.B{1, 7};
                elseif aa==3 || aa==4
                    merge_theta_learn_days.A{aa,dd} = theta_learn{aa}.A{1, 8};
                    merge_theta_learn_days.B{aa,dd} = theta_learn{aa}.B{1, 8};
                elseif aa==6
                    merge_theta_learn_days.A{aa,dd} = [];
                    merge_theta_learn_days.B{aa,dd} = [];
                end
            end            
        case 9 %session 2 vs. 6; session 3 vs. 5
            for aa=1:6 %all animals
                if aa==1 || aa==2 || aa==3 || aa==5 || aa==6
                    %extract all the days that correspond to the time duration
                    merge_theta_learn_days.A{aa,dd} = [];
                    merge_theta_learn_days.B{aa,dd} = [];
                elseif aa==4
                    merge_theta_learn_days.A{aa,dd} = theta_learn{aa}.A{1, 9};
                    merge_theta_learn_days.B{aa,dd} = theta_learn{aa}.B{1, 9};
                end
            end
            
            
    end
    
end


end

