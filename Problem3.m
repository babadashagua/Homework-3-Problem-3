% CSE 847 Homework 3

clc;
close all;
clear;

load('diabetes.mat');

lambda = [1e-5,1e-4,1e-3,1e-2,1e-1,1,10];
[row_lambda,col_lambda] = size(lambda);

[U,S,V] = svd(x_train,'econ');
[row_number_training,col_number_training] = size(x_train);
[row_num, col_num] = size(V);
MSE_training = zeros(col_lambda,1);
MSE_test = zeros(col_lambda,1);
for j = 1:col_lambda
    weights = zeros(row_num,1);
    for i = 1:row_num
        weights = weights + S(i,i)*U(:,i)'*y_train*V(:,i)/(S(i,i)^2+lambda(j));
    end

    Error_training = x_train*weights - y_train;

    Error_square_training = 0;
    for i = 1:row_num
        Error_square_training = Error_square_training + Error_training(i)^2;
    end

    MSE_training(j) = Error_square_training/row_number_training;

    Error_square_test = 0;
    Error_test = x_test*weights - y_test;
    [row_num_test, col_num_test] = size(Error_test);
    for i = 1:row_num_test
        Error_square_test = Error_square_test + Error_test(i)^2;
    end

    MSE_test(j) = Error_square_test/row_num_test;
end

figure(1)
plot(lambda,MSE_training,'LineWidth',2); hold on; 
plot(lambda,MSE_test,'LineWidth',2); grid on;
legend('Training MSE','Test MSE');
xlabel('\lambda value');
ylabel('MSE');
