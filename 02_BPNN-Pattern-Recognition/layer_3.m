%%==================================================================
% Title: Supervised learning for parameter estimation and prediction
% Author: Pratik M Ramdasi
% Date: 10/22/2015
%===================================================================

clear all;
clc;

x1range = 'B:N';
data = xlsread('wine_data.xlsx',x1range);
class1 = data((1:59),:);
class2 = data((60:130),:);
class3 = data((131:178),:);

%random noise component 
noise = sqrt(0.01)*rand([12 13]);

% make the size of datasets equal(60 samples) by adding random noise components
class1_new = [class1;noise(1,:)];
class2_new = [class2((1:60),:)];
class3_new = [class3;noise((1:12),:)];
data_new = [class1_new;class2_new;class3_new];

% standardize the data
s=[];
for i =1:size(data_new,2)
    A = data_new(:,i);
    s(i)=std(A);
end
for j = 1:size(data_new,2)
    for i=1:size(data_new,1)
        data_new(i,j)= (data_new(i,j)- mean(data_new(:,j)))/s(j);
    end
end

% desired values
desired = zeros(180,3);
desired((1:60),1) = 1;
desired((61:120),2) = 1;
desired((121:180),3) = 1;

% This script assumes these variables are defined:
%   data_new - input data.
%   desired - target data.
x = data_new';
t = desired';

% Choose a Training Function
% 'trainlm' is usually fastest.
% 'traingdx' is slower but considers adaptive learning rate and momentum
%  factor
trainFcn = 'traingdx';  
net.trainparam.lr = 0.07; % learning rate
net.trainparam.mc = 0.9; % momentum factor

% Choose a Performance Function
net.performFcn = 'mse';  % Mean Square error

% Create a Pattern Recognition Network
hiddenLayerSize = [30 15];
net = patternnet(hiddenLayerSize,'traingdx','mse');

% Setup Division of Data for Training, Validation, Testing
net.divideFcn = 'dividerand';  % Divide data randomly
net.divideMode = 'sample';  % Divide up every sample
net.divideParam.trainRatio = 70/100;
net.divideParam.valRatio = 15/100;
net.divideParam.testRatio = 15/100;

% Choose Plot Functions
net.plotFcns = {'plotperform','plottrainstate','ploterrhist', ...
    'plotconfusion', 'plotroc'};

% Train the Network
[net,tr] = train(net,x,t);

% Test the Network
y = net(x);
e = gsubtract(t,y);
performance = perform(net,t,y);
tind = vec2ind(t);
yind = vec2ind(y);
percentErrors = sum(tind ~= yind)/numel(tind);

% Recalculate Training, Validation and Test Performance
trainTargets = t .* tr.trainMask{1};
valTargets = t .* tr.valMask{1};
testTargets = t .* tr.testMask{1};
trainPerformance = perform(net,trainTargets,y);
valPerformance = perform(net,valTargets,y);
testPerformance = perform(net,testTargets,y);

% View the Network
view(net)

% Plots
% Uncomment these lines to enable various plots.
figure, plotperform(tr);
figure, plottrainstate(tr);
figure, ploterrhist(e);
figure, plotconfusion(t,y);
figure, plotroc(t,y);


