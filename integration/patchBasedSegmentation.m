function seg = patchBasedSegmentation(im, net, patchSize)
% Segments an Image using a patch based approach applaying 
% a SemanticSegmantation Network to image patchs.
%
% Segmentation is done using overlapping patches to avoid boundary
% artifacts
%
% Params: im  - the input image
%         net - trained network for patch based semantic seg
%         patchSize - the patch size as [x y] vector TODO must correspond to
%                      trained network patch size
% Returns: seg - the segmented image as uint8 label image. make sure to 
%                convert to approriate other datatype (e.g. categorical) if needed
% Andreas Husch, 2018

BORDER_SIZE = ceil(patchSize ./ 8); % size of patch overlap TODO: parameter?

patchBasedSegmentationFun = @(blockstruct)(semanticseg(blockstruct.data, net, 'outputtype', 'uint8'));
%%MOCKUP
%patchBasedSegmentationFun = @(blockstruct)(uint8(blockstruct.data > mean(blockstruct.data)));
%im = im(:,:,1);

im = im(:,:,1:2);%drop 3rd channel
imSize = size(im);
imSizeXY = imSize(1:2);
%%
seg = blockproc(im, patchSize , patchBasedSegmentationFun, ...
    'BorderSize', BORDER_SIZE, ...
    'PadPartialBlocks', true, ...
    'PadMethod', 'symmetric', ...
    'TrimBorder', true);

%% trim partialblocks padding again 
seg = seg(1:imSizeXY(1), 1:imSizeXY(2));
end 