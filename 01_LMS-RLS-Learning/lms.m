%%==================================================================
% Title: Supervised learning for parameter estimation and prediction
% Author: Pratik M Ramdasi
% Date: 10/06/2015
%===================================================================

clear all;
clc;

%% getting the financial data
x1range = 'B1:B757';
num = xlsread('HistoricalQuotes.xlsx',x1range);
% use half the data for training
len = length(num(:,1));
len1 = len/2 + ceil(0.2*len)+1;
% separate training,testing and validation matrices 
training  = num((1:len/2),:);
validation = num((len/2+1:len1),:);
testing = num(len1+1:len,:);
%normalize training,tesing and validation data wrt mean
training  = (training - mean(training))./ std(training);
validation = (validation - mean(validation))./ std(validation);
testing = (testing - mean(testing))./ std(testing);
% define driving input
noise = sqrt(0.0001)* randn(size(training));
% number of iterations
num_iter = 100; 

%% perform learning algorithm 
wi = [0 0 0]';
alpha= 0.001;
ip_mat = [];
d = [];
for i = 1:length(training)-3
    temp = [training(i);training(i+1);training(i+2)];
    ip_mat = [ip_mat temp];
    desired = training(i+3);
    d = [d desired]; 
end
p = 1;
for k= 1:num_iter
    for i=1:length(ip_mat)
        net = wi' * ip_mat(:,i)+noise(i); 
        error_training = d(:,i) - net;
        e(:,p) = error_training;
        wi = wi + (alpha * ip_mat(:,i) * error_training);
        p = p + 1;
    end
end

%% plot learning curve
plot(e.^2);
title('learning curve');

%% compare results with the wiener-hopf solution
rxx = ip_mat * ip_mat';
rxd = ip_mat * d';
opt_wts = inv(rxx) * rxd;
disp('final weights using LMS:');
disp(wi);
disp('Optimal weights using direct approach are:');
disp(opt_wts);
compare = [wi opt_wts];
[V,D] = eig(rxx);

%% predictor
for i = 1:length(validation)-3
    x_est = [validation(i) validation(i+1) validation(i+2)]';
    y_est = wi' * x_est;
    y = validation(i+3);
    error = y - y_est;
    e_valid(:,i)= error;
end
MSE = mean(e_valid.^2);
disp('MSE is:');
disp (MSE);

%% testing
for i = 1:length(testing)-3
    y = [testing(i) testing(i+1) testing(i+2)]';
    output = wi' * y;
    desired_testing = testing(i+3);
    error_testing= desired_testing - output;
    e_testing(:,i)= error_testing;
end

