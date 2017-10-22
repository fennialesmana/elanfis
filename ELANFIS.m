function [time, Et]=ELANFIS()
tic;
clc;
clear;
close all;

% load dataset
data = csvread('iris.csv');
input_data = data(:, 1:end-1);
output_data = data(:, end);

% parameter initialization
[center,U] = fcm(input_data, 3, [2 100 1e-6]); %center = center cluster, U = membership level
[total_examples, total_features] = size(input_data);
class = 3; % total classes
epoch = 0;
epochmax = 400;
[Yy, Ii] = max(U); % Yy = max value between both membership function, Ii = the class corresponding to the max value

minimal_error = 1000;
best_a = zeros(class, total_features);
best_b = repmat(2, class, total_features);
best_c = zeros(class, total_features);

while epoch < epochmax
    epoch = epoch + 1;
    a = zeros(class, total_features);
    b = repmat(2, class, total_features);
    c = zeros(class, total_features);
    
    for k =1:class
        for i = 1:total_features % looping for all features
            % premise parameter: a
            aTemp = (max(input_data(:, i))-min(input_data(:, i)))/(2*sum(Ii' == k)-2);
            aLower = aTemp*0.5;
            aUpper = aTemp*1.5;
            a(k, i) = (aUpper-aLower).*rand()+aLower;

            %premise parameter: c
            dcc = (2.1-1.9).*rand()+1.9;
            cLower = center(k,total_features)-dcc/2;
            cUpper = center(k,total_features)+dcc/2;
            c(k,i) = (cUpper-cLower).*rand()+cLower;
        end
    end
    
    H = [];
    Mu = zeros(total_examples, class, total_features); % Mu: miu all samples (jml data x jml class x jml features)
    for i = 1:total_examples %looping setiap samples untuk forward pass
        for k = 1:class
            w1(k) = 1;%ini buat w (bukan w bar)
            for j = 1:total_features
                mu(k,j) = 1/(1 + ((input_data(i,j)-c(k,j))/a(k,j))^(2*b(k,j))); %mu: miu one sample
                w1(k) = w1(k)*mu(k,j); % isi w kelas ke k
                Mu(i, k, j) = mu(k, j);
            end;
        end;
        w = w1/sum(w1); % w = w bar satu baris / satu sample data
        XX = [];
        for k = 1:class
            XX = [XX w(k)*input_data(i,:) w(k)];
        end; % pada akhirnya si XX udah jadi matrix H untuk satu sample di looping itu
        H = [H; XX]; %yg ini gabungin matrix H setiap sample
    end

    % consequent parameter (p, q, r)
    beta = pinv(H) * output_data; % moore pseudo invers

    output = H * beta; % hitung outputnya dari weight yang udah didapet

    E = sum((output_data - output).^2)/total_examples; % hitung error

    if E < minimal_error % update yang terbaik
        minimal_error = E;
        best_a = a;
        best_b = b;
        best_c = c;
    end

    Et(epoch) = minimal_error; %tampung minimal error ke Et
    
    % Draw the SSE plot.....
%     plot(1:epoch, Et);
%     title(['Epoch  ' int2str(epoch) ' -> MSE = ' num2str(Et(epoch))]);
%     grid
%     pause(0.001);
end;

%[output_data output output_data-output]
% ----------------------------------------------------------------
time = toc;
end