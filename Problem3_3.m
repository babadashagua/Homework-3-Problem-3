% Homework 3 problem 3 3)

clc;
close all;
clear;

load('diabetes.mat');

lambda = 1e-5;
MSE = ridge_regression(x_train,y_train,lambda);
figure(1)
plot(MSE(1,:),'LineWidth',2); hold on;
plot(MSE(2,:),'LineWidth',2); grid on;
legend('Training MSE','Test MSE');
xlabel('\lambda = 1e-5');
ylabel('MSE');

lambda = 1e-4;
MSE = ridge_regression(x_train,y_train,lambda);
figure(2)
plot(MSE(1,:),'LineWidth',2); hold on;
plot(MSE(2,:),'LineWidth',2); grid on;
legend('Training MSE','Test MSE');
xlabel('\lambda = 1e-4');
ylabel('MSE');

lambda = 1e-3;
MSE = ridge_regression(x_train,y_train,lambda);
figure(3)
plot(MSE(1,:),'LineWidth',2); hold on;
plot(MSE(2,:),'LineWidth',2); grid on;
legend('Training MSE','Test MSE');
xlabel('\lambda = 1e-3');
ylabel('MSE');

lambda = 1e-2;
MSE = ridge_regression(x_train,y_train,lambda);
figure(4)
plot(MSE(1,:),'LineWidth',2); hold on;
plot(MSE(2,:),'LineWidth',2); grid on;
legend('Training MSE','Test MSE');
xlabel('\lambda = 1e-2');
ylabel('MSE');

lambda = 1e-1;
MSE = ridge_regression(x_train,y_train,lambda);
figure(5)
plot(MSE(1,:),'LineWidth',2); hold on;
plot(MSE(2,:),'LineWidth',2); grid on;
legend('Training MSE','Test MSE');
xlabel('\lambda = 1e-1');
ylabel('MSE');

lambda = 1;
MSE = ridge_regression(x_train,y_train,lambda);
figure(6)
plot(MSE(1,:),'LineWidth',2); hold on;
plot(MSE(2,:),'LineWidth',2); grid on;
legend('Training MSE','Test MSE');
xlabel('\lambda = 1');
ylabel('MSE');

lambda = 10;
MSE = ridge_regression(x_train,y_train,lambda);
figure(7)
plot(MSE(1,:),'LineWidth',2); hold on;
plot(MSE(2,:),'LineWidth',2); grid on;
legend('Training MSE','Test MSE');
xlabel('\lambda = 10');
ylabel('MSE');