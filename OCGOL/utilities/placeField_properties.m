function [outputArg1,outputArg2] = placeField_properties(session_vars, tunedLogical,options)

%% Select trial tuned classes of neurons


switch options.tuning_criterion
    case 'si' %spatial information
        %for each session
        for ss =1:size(session_vars,2)
            %spatial information criterion
            Atuned{ss} = tunedLogical(ss).si.Atuned;
            Btuned{ss} = tunedLogical(ss).si.Btuned;
            
            AandB_tuned{ss} =  tunedLogical(ss).si.AandB_tuned;
            AorB_tuned{ss} = tunedLogical(ss).si.AorB_tuned;
            onlyA_tuned{ss} = tunedLogical(ss).si.onlyA_tuned;
            onlyB_tuned{ss} = tunedLogical(ss).si.onlyB_tuned;
            AxorB_tuned{ss} =  AorB_tuned{ss} & ~AandB_tuned{ss};
            %all tuned neurons - logical
            all_neurons{ss} = true(size(Atuned{ss}));
            %tuned to neither environment - logical
            neither_tuned{ss} = ~((onlyA_tuned{ss} | onlyB_tuned{ss}) | AandB_tuned{ss});
            
        end
    case 'ts' %spatial information
        for ss =1:size(session_vars,2)
            %spatial information criterion
            Atuned{ss} = tunedLogical(ss).ts.Atuned;
            Btuned{ss} = tunedLogical(ss).ts.Btuned;
            
            AandB_tuned{ss} =  tunedLogical(ss).ts.AandB_tuned;
            AorB_tuned{ss} = tunedLogical(ss).ts.AorB_tuned;
            onlyA_tuned{ss} = tunedLogical(ss).ts.onlyA_tuned;
            onlyB_tuned{ss} = tunedLogical(ss).ts.onlyB_tuned;
            AxorB_tuned{ss} =  AorB_tuned{ss} & ~AandB_tuned{ss};
            
            all_neurons{ss} = true(size(Atuned{ss}));
            %tuned to neither environment - logical
            neither_tuned{ss} = ~((onlyA_tuned{ss} | onlyB_tuned{ss}) | AandB_tuned{ss});
        end
end

%% Get counts of place field numbers for each sub-class

%place centers of 
centers_Aonly = session_vars{1}.Place_cell{1}.placeField.center(onlyA_tuned{1});
centers_Bonly = session_vars{1}.Place_cell{2}.placeField.center(onlyB_tuned{1});

%get # of place fields for each neuron
nb_fields_A = cellfun('size', centers_Aonly,1);
nb_fields_B = cellfun('size', centers_Bonly,1);

%count # of neurons with 1 2 or 3+ place fields
for ii=1:3
    if ii < 3
        field_count_A(ii) = size(find(nb_fields_A == ii),2);
        field_count_B(ii) = size(find(nb_fields_B == ii),2);
    elseif ii ==3
        field_count_A(ii) = size(find(nb_fields_A >= ii),2);
        field_count_B(ii) = size(find(nb_fields_B >= ii),2);
    end
end


%% Get width distributions here

session_vars{1, 1}.Place_cell{1, 1}.placeField.width  


end
