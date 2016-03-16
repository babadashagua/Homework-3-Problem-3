function MSE = ridge_regression(x_train,y_train,lambda)

indices = crossvalind('Kfold',y_train,5);

for j = 1:5
    test = (indices == j); train = ~test;
    training_data_x = x_train(train,:);
    test_data_x = x_train(test,:);
    training_data_y = y_train(train,:);
    test_data_y = y_train(test,:);
    [U,S,V] = svd(training_data_x,'econ');
    [row_number_training,col_number_training] = size(training_data_x); % get number of samples in training dataset
    [row_num, col_num] = size(V);
%     MSE_training = zeros(row_number_training,1);
%     MSE_test = zeros(col_lambda,1);

    weights = zeros(row_num,1);
    for i = 1:row_num
        weights = weights + S(i,i)*U(:,i)'*training_data_y*V(:,i)/(S(i,i)^2+lambda);
    end

    Error_training = training_data_x*weights - training_data_y;

    Error_square_training = 0;
    for i = 1:row_num
        Error_square_training = Error_square_training + Error_training(i)^2;
    end

    MSE_training(j) = Error_square_training/row_number_training;

    Error_square_test = 0;
%     x_test = x_train(test,:);
%     y_test = y_train(test,:);
    Error_test = test_data_x*weights - test_data_y;
    [row_num_test, col_num_test] = size(Error_test);
    for i = 1:row_num_test
        Error_square_test = Error_square_test + Error_test(i)^2;
    end

    MSE_test(j) = Error_square_test/row_num_test;

end

MSE = [MSE_training;MSE_test];
end