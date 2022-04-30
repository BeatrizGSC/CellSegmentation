% Integration of the DL segmentation trained network. 
% This function must be with in this following structure. 
%  Create a +vision/+labeler folder within a folder that is already
%  on the MATLAB path. 
%
%  Saving the file to the package directory is required. You can add a folder to the
%  path using the ADDPATH function.
classdef AutophagyDeepLearningSegmentation < vision.labeler.AutomationAlgorithm
    
    
    properties(Constant)
        
        Name = 'Autophagy Deep Learning Segmentation';
        
        Description = 'Autophagy Deep Learning Segmentation by Beatriz Garcia Santa Cruz';
      
        UserDirections = {...
            ['Automation algorithms are a way to automate manual labeling ' ...
            'tasks. This AutomationAlgorithm is a template for creating ' ...
            'user-defined automation algorithms. Below are typical steps' ...
            'involved in running an automation algorithm.'], ...
            ['Run: Press RUN to run the automation algorithm. '], ...
            ['Review and Modify: Review automated labels over the interval ', ...
            'using playback controls. Modify/delete/add ROIs that were not ' ...
            'satisfactorily automated at this stage. If the results are ' ...
            'satisfactory, click Accept to accept the automated labels.'], ...
            ['Change Settings and Rerun: If automated results are not ' ...
            'satisfactory, you can try to re-run the algorithm with ' ...
            'different settings. In order to do so, click Undo Run to undo ' ...
            'current automation run, click Settings and make changes to ' ...
            'Settings, and press Run again.'], ...
            ['Accept/Cancel: If results of automation are satisfactory, ' ...
            'click Accept to accept all automated labels and return to ' ...
            'manual labeling. If results of automation are not ' ...
            'satisfactory, click Cancel to return to manual labeling ' ...
            'without saving automated labels.']};
    end
    
    properties
        
        dl = false;
        predicted_pixel_values = [1 2 3];
        predicted_pixel_labels = {'Background' 'Phagophore' ...
                                  'Autolysosome'};
     
        
        
    end
    
    methods
       
        function isValid = checkLabelDefinition(algObj, labelDef)
            
            disp(['Executing checkLabelDefinition on label definition "' labelDef.Name '"'])
            
            if labelDef.Type ~= labelType.PixelLabel
                isValid = false;
            else
                isValid = true;
            end
            
            
        end
                    

    end
    
    methods
       
        function initialize(algObj, I)
            
            disp('Executing initialize on the first image frame')
            
            algObj.dl = load('net.mat', 'net');
           
        end
        
     
        function autoLabels = run(algObj, I)
            
            disp('Executing run on image frame')
            
            segmentedImage = patchBasedSegmentation(I, algObj.dl.net, [256 256]);
            autoLabels = categorical(segmentedImage, algObj.predicted_pixel_values, algObj.predicted_pixel_labels);
            
        end
        
    end
end